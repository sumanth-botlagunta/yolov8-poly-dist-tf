"""Custom training loop for YOLOv8.

YoloV8Trainer drives the epoch/step loop with a tf.function-wrapped inner step,
swaps in EMA weights around validation, manages periodic and best checkpoints
with auto-resume, and writes TensorBoard summaries.
"""

import dataclasses
import logging
import os
import signal
import time
from typing import Optional

import numpy as np
import tensorflow as tf
import yaml

from common.progress import Progress

log = logging.getLogger(__name__)

# Loss components shown on the training progress bar, in order, with short labels.
_PROGRESS_LOSSES = [
    ('total_loss', 'loss'), ('box_loss', 'box'), ('cls_loss', 'cls'),
    ('dfl_loss', 'dfl'), ('poly_loss', 'poly'), ('dist_loss', 'dist'),
]


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class YoloV8Trainer:
    """Custom training loop.

    Args:
        task:       YoloV8Task instance (already configured with ExperimentConfig).
        config:     ExperimentConfig (trainer.* sub-config used here).
        output_dir: Directory for checkpoints and TF summary events.
        strategy:   Distribution strategy.  Defaults to MirroredStrategy.
        debug:      If True, run tf.functions eagerly.
    """

    def __init__(
        self,
        task,
        config,
        output_dir: str,
        strategy: Optional[tf.distribute.Strategy] = None,
        debug: bool = False,
        resume_from: str = None,
    ):
        self._task        = task
        self._config      = config
        self._output_dir  = output_dir
        self._debug       = debug
        self._strategy    = strategy or tf.distribute.MirroredStrategy()
        self._resume_from = resume_from

        # Multi-replica (MirroredStrategy): the train step is dispatched via
        # strategy.run, the loss normalizers (num_objs, target_scores_sum) are
        # all-reduced to global counts, and SGDTorch all-reduces gradients. The
        # single-replica path is a no-op for all of those.
        self._num_replicas = self._strategy.num_replicas_in_sync
        self._distributed  = self._num_replicas > 1
        if self._distributed:
            log.info("Distributed training: %d replicas (MirroredStrategy).",
                     self._num_replicas)

        self._model        = None
        self._optimizer    = None
        self._train_ds     = None
        self._train_iter   = None   # persistent iterator (built lazily in train())
        self._mosaic_closed = False  # close_mosaic: rebuilt mosaic-free stream yet?
        self._val_ds       = None
        self._ckpt         = None
        self._ckpt_manager = None
        self._tb_writer    = None

        # Each training step consumes a merged batch: detection + distance rows.
        td = config.task.train_data
        dist_cfg = getattr(td, 'distance_data', None)
        self._merged_batch_size = td.global_batch_size + (
            dist_cfg.global_batch_size if dist_cfg is not None else 0
        )

        self._global_step   = tf.Variable(0, trainable=False, dtype=tf.int64,
                                          name='global_step')
        self._epoch_var     = tf.Variable(0, trainable=False, dtype=tf.int64,
                                          name='completed_epochs')
        self._best_metric   = tf.Variable(-1e9, trainable=False, dtype=tf.float32,
                                          name='best_metric')

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self, total_epochs: int) -> None:
        """Run the full training loop.

        Epoch semantics: when ``steps_per_loop > 0`` (the normal case, derived as
        train_total_examples // batch) every epoch runs exactly that many steps
        from one persistent iterator over the infinite (repeated) training stream,
        so one epoch equals one nominal data pass and the startup step/LR/ETA
        numbers hold by construction. When ``steps_per_loop == 0`` (synthetic/test
        datasets with no example count) the loop falls back to data-driven epochs.
        """
        self._setup()
        start_epoch = self._auto_resume(self._resume_from)

        trainer_cfg   = self._config.trainer
        # Guard against checkpoint_interval=0 (steps_per_loop not computed).
        ckpt_interval = max(trainer_cfg.checkpoint_interval, 1)
        # Log ~20x per epoch; at least every 50 steps.
        spl           = trainer_cfg.steps_per_loop
        log_interval  = max(50, spl // 20) if spl > 0 else 50

        if spl > 0 and self._train_iter is None:
            self._train_iter = iter(self._train_ds)

        # Pre-training checkpoint so an epoch-1 crash still has a restore point.
        if start_epoch == 0:
            self._save_checkpoint()
            log.info("Initial checkpoint saved at step 0.")

        training_start  = time.time()
        last_saved_step = int(self._global_step)  # prevents double-saves

        do_img_summary = 'image' in getattr(self._config.task, 'summary_types', 'scalar')
        n_summary = getattr(self._config.task, 'summary_image_num', 10)

        for epoch in range(start_epoch, total_epochs):
            log.info("─── Epoch %d / %d  (global_step=%d) ───",
                     epoch + 1, total_epochs, int(self._global_step))
            self._maybe_close_mosaic(epoch, total_epochs)
            epoch_start  = time.time()
            step_times   = []
            step_losses  = {}   # last-step values (for _log_step)
            loss_accum   = {}   # running sum for epoch mean
            loss_count   = 0
            python_step  = int(self._global_step)
            _aug_logged  = False

            if spl > 0:
                # Fixed-count epoch. After a mid-epoch resume global_step is not a
                # multiple of steps_per_loop, so run only the remainder and epoch k
                # still ends at k*spl. A return of 0 means this epoch trained fully
                # but crashed during validation: fall through with no training
                # steps so the validation below runs for it.
                steps_this_epoch = self._steps_for_epoch(python_step, spl, epoch)
                if steps_this_epoch == 0:
                    log.info(
                        "Epoch %d training already complete at global_step=%d — "
                        "running its pending validation.", epoch + 1, python_step)

                def _epoch_inputs():
                    for _ in range(steps_this_epoch):
                        try:
                            yield next(self._train_iter)
                        except StopIteration:
                            # Only reachable if the training stream is finite
                            # (misconfiguration; the detection stream repeats).
                            log.warning(
                                "Training dataset exhausted mid-epoch at "
                                "global_step=%d — epoch truncated. The training "
                                "stream should be infinite (.repeat()).",
                                int(self._global_step),
                            )
                            return

                epoch_inputs = _epoch_inputs()
            else:
                epoch_inputs = self._train_ds

            # The `for` pulls the next batch before the body runs, so
            # (body start - previous body end) is the input-pipeline wait.
            # Reporting compute time alone would overstate throughput when
            # training is input-bound.
            prev_body_end = time.time()
            pbar = None   # created lazily on the first step (needs the loss keys)
            for inputs in epoch_inputs:
                data_dt = time.time() - prev_body_end
                t0 = time.time()
                step_losses = self._compiled_train_step(inputs)
                self._global_step.assign_add(1)
                python_step += 1
                step_dt = time.time() - t0
                step_times.append(step_dt + data_dt)  # wall-clock per step

                if not _aug_logged and do_img_summary:
                    # Under distribution `inputs` is PerReplica; log the first
                    # replica's local batch.
                    aug_inputs = inputs
                    if self._distributed:
                        aug_inputs = self._strategy.experimental_local_results(inputs)[0]
                    self._log_aug_images(aug_inputs, python_step)
                    _aug_logged = True

                # Sync the step's losses to host once, reused for both the epoch
                # accumulator and the progress bar (avoids a second device->host copy).
                step_vals = {k: float(v) for k, v in step_losses.items()}
                for k, v in step_vals.items():
                    loss_accum[k] = loss_accum.get(k, 0.0) + v
                loss_count += 1

                # Live progress bar (TTY -> in-place; cloud log file -> periodic line).
                wall = step_dt + data_dt
                img_s = self._merged_batch_size / max(wall, 1e-9)
                if pbar is None:
                    cols = [(k, s) for k, s in _PROGRESS_LOSSES if k in step_vals]
                    self._pbar_cols = cols
                    header = (f"{'Epoch':>10}{'gpu_GB':>8}"
                              + "".join(f"{s:>9}" for _, s in cols) + f"{'img/s':>9}")
                    total_steps = steps_this_epoch if spl > 0 else None
                    pbar = Progress(total=total_steps, unit='step', header=header)
                try:
                    mem = tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e9
                    mem_s = f"{mem:>7.2f}"
                except Exception:
                    mem_s = f"{'-':>7}"
                row = (f"{epoch + 1}/{total_epochs}".rjust(10) + mem_s + " "
                       + "".join(f"{step_vals[k]:>9.3f}" for k, _ in self._pbar_cols)
                       + f"{img_s:>9.0f}")
                pbar.update(1, desc=row)

                if python_step % log_interval == 0:
                    self._log_step(step_losses, python_step, step_dt, data_dt)

                # Mid-epoch periodic checkpoint; skip if this step was already saved.
                if python_step % ckpt_interval == 0 and python_step != last_saved_step:
                    self._sync_completed_epochs(python_step, spl)
                    self._save_checkpoint()
                    last_saved_step = python_step

                if self._shutdown_requested:
                    if pbar is not None:
                        pbar.close()
                    if python_step != last_saved_step:
                        self._sync_completed_epochs(python_step, spl)
                        # Interruption save goes to resume/ so the main directory
                        # stays a clean epoch-boundary sequence.
                        self._save_resume_checkpoint()
                        last_saved_step = python_step
                    log.info("Graceful shutdown at step %d (epoch %d interrupted).",
                             python_step, epoch + 1)
                    return

                # Reset after logging/checkpointing so their cost is not
                # attributed to the next step's data wait.
                prev_body_end = time.time()

            if pbar is not None:
                pbar.close()

            # ---- validation (EMA weights) ----
            # try/finally so a failure inside validation can never leave the EMA
            # shadow weights swapped in as the live weights, which would corrupt
            # subsequent training and any checkpoint saved after.
            val_start = time.time()
            self._optimizer.swap_in(self._model)
            try:
                val_metrics = self._run_validation(epoch=epoch + 1)
            finally:
                self._optimizer.swap_out(self._model)
            val_time = time.time() - val_start

            # ---- best checkpoint ----
            best_metric_name = trainer_cfg.best_checkpoint_eval_metric
            if best_metric_name not in val_metrics:
                # A missing key (typo, or the metric's head is disabled) would
                # default to 0.0 and never save a best checkpoint; surface it.
                log.warning(
                    "best_checkpoint_eval_metric '%s' not in validation metrics %s; "
                    "no best checkpoint will be saved this epoch.",
                    best_metric_name, sorted(val_metrics.keys()),
                )
            metric_val = val_metrics.get(best_metric_name, 0.0)
            # best_checkpoint_metric_comp: 'higher' (default) keeps the max,
            # 'lower' keeps the min (e.g. an error metric). _best_metric stores the
            # value signed so the comparison is always "greater is better" (negated
            # under 'lower').
            comp = getattr(self._config.trainer, 'best_checkpoint_metric_comp',
                           'higher')
            signed = -metric_val if comp == 'lower' else metric_val
            if best_metric_name in val_metrics and signed > float(self._best_metric):
                self._best_metric.assign(signed)
                self._save_best_checkpoint(epoch=epoch + 1, step=python_step)

            # Mark the epoch complete before the epoch-end checkpoint so a resume
            # skips it and starts from the next one.
            self._epoch_var.assign(epoch + 1)

            # Epoch-end checkpoint, always written even when the in-loop periodic
            # save already wrote this step: that pre-validation save recorded the
            # epoch as not-yet-validated (see _sync_completed_epochs), and this one
            # re-writes the same checkpoint number with validation completed. One
            # extra write per epoch when ckpt_interval == steps_per_loop; buys
            # exact resume semantics.
            self._save_checkpoint()
            last_saved_step = python_step

            # Act on a preemption signal that arrived during validation /
            # checkpointing. Validation can take minutes; without this check a
            # SIGTERM during it waits until the next epoch's first step, past the
            # grace window.
            if self._shutdown_requested:
                log.info("Graceful shutdown after epoch %d validation.", epoch + 1)
                return

            # ---- timing & logging ----
            epoch_time     = time.time() - epoch_start
            epochs_done    = epoch + 1 - start_epoch
            avg_epoch_time = (time.time() - training_start) / max(epochs_done, 1)
            eta_seconds    = avg_epoch_time * (total_epochs - (epoch + 1))
            avg_step_time  = sum(step_times) / max(len(step_times), 1)
            # Each step consumes detection + distance rows (e.g. 128 + 16).
            throughput     = self._merged_batch_size / max(avg_step_time, 1e-9)
            epoch_losses   = {k: v / max(loss_count, 1) for k, v in loss_accum.items()}

            self._log_epoch(
                epoch, epoch_losses, val_metrics,
                epoch_time=epoch_time,
                val_time=val_time,
                eta_seconds=eta_seconds,
                throughput=throughput,
            )

        total_time = time.time() - training_start
        log.info("Training complete. Total time: %s", _fmt_duration(total_time))

    def _maybe_close_mosaic(self, epoch: int, total_epochs: int) -> None:
        """Disable mosaic + mixup for the final ``close_mosaic_epochs`` epochs.

        At the first epoch ``>= total_epochs - close_mosaic_epochs`` the train
        stream is rebuilt once with ``mosaic_frequency`` / ``mixup_frequency`` set
        to 0 (Ultralytics ``close_mosaic``). Step/epoch accounting is unaffected:
        the trainer still runs a fixed step count from the new iterator. No-op when
        ``close_mosaic_epochs`` is 0 (the default).
        """
        import dataclasses
        mosaic_cfg = getattr(getattr(self._config.task.train_data, 'parser', None),
                             'mosaic', None)
        n = getattr(mosaic_cfg, 'close_mosaic_epochs', 0) if mosaic_cfg else 0
        if n <= 0 or self._mosaic_closed or epoch < total_epochs - n:
            return

        td = self._config.task.train_data
        new_mosaic = dataclasses.replace(mosaic_cfg, mosaic_frequency=0.0,
                                         mixup_frequency=0.0)
        new_parser = dataclasses.replace(td.parser, mosaic=new_mosaic)
        new_td = dataclasses.replace(td, parser=new_parser)

        ds = self._task.build_inputs(new_td)
        if self._distributed:
            ds = self._strategy.experimental_distribute_dataset(ds)
        self._train_ds = ds
        self._train_iter = iter(ds)
        self._mosaic_closed = True
        log.info("close_mosaic: disabled mosaic + mixup for the final %d epoch(s) "
                 "(from epoch %d)", n, epoch + 1)

    def _sync_completed_epochs(self, python_step: int, steps_per_loop: int) -> None:
        """Sync ``completed_epochs`` before a save that lands on an epoch boundary.

        The epoch's final step triggers the in-loop periodic save (with all tier
        configs ``checkpoint_interval == steps_per_loop``) before the
        post-validation ``_epoch_var.assign(epoch + 1)``, and the epoch-end save
        is then skipped by the ``last_saved_step`` guard. Without this sync the
        boundary checkpoint would persist ``completed_epochs`` one too low and a
        resume from it would re-run a full epoch.

        The epoch is trained but not yet validated here, so the boundary save
        records it as not completed (``completed_epochs`` = trained epochs - 1).
        A crash during validation then resumes with zero training steps for that
        epoch (``_steps_for_epoch`` returns 0) and runs the pending validation, so
        no boundary checkpoint is left unevaluated. The epoch-end save re-writes
        the same checkpoint number with the true count, so a normal restart does
        not re-validate.
        """
        if steps_per_loop > 0 and python_step % steps_per_loop == 0:
            self._epoch_var.assign(python_step // steps_per_loop - 1)

    @staticmethod
    def _steps_for_epoch(global_step: int, steps_per_loop: int, epoch: int) -> int:
        """Steps to run in 0-based ``epoch`` starting at ``global_step``.

        A full epoch is ``steps_per_loop`` steps; after a mid-epoch resume only
        the remainder to the epoch's end boundary (``(epoch+1) * steps_per_loop``)
        is run, so epoch k always ends at exactly ``k * steps_per_loop`` global
        steps and the LR schedule / checkpoint cadence stay aligned with epochs.

        Returns 0 when the epoch's training already completed but the epoch was
        never marked done, i.e. the previous process died DURING validation.
        The caller then skips training and runs the pending validation directly,
        so no epoch-boundary checkpoint is left unevaluated.
        """
        target = (epoch + 1) * steps_per_loop
        return max(0, min(steps_per_loop, target - global_step))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _will_resume(self) -> bool:
        """Return True if this run already has a checkpoint to resume from.

        When True, seed-init is skipped and the run's own weights win. This makes
        `finetune_from` / `init_checkpoint` a fresh-start-only seed: a restarted
        fine-tune resumes from its own checkpoints and never re-reads the source
        checkpoint.
        """
        if self._resume_from:
            return True
        for d in (self._output_dir, os.path.join(self._output_dir, 'resume')):
            if tf.train.latest_checkpoint(d):
                return True
        return False

    def _setup(self) -> None:
        with self._strategy.scope():
            self._model     = self._task.build_model()
            # Seed-init (finetune_from / init_checkpoint) only on a fresh run. When
            # a resumable checkpoint exists, _auto_resume restores this run's own
            # weights, so skipping init here avoids re-loading the source.
            if self._will_resume():
                log.info("Resumable checkpoint present — skipping seed-init "
                         "(finetune_from / init_checkpoint); restoring this run's own weights.")
            else:
                self._task.initialize(self._model)
            # Freeze before building the optimizer so trainable_variables (and the
            # SGD slots / EMA) already exclude the frozen weights. Applied on resume too.
            self._task.apply_freezing(self._model)
            self._optimizer = self._task.build_optimizer()
            self._task._loss_fn = self._task.build_losses()
            # Gradient-accumulation accumulators (no-op unless grad_accum_steps > 1),
            # created after freezing fixes trainable_variables.
            self._task.prepare_grad_accumulation(self._model)
            # Pre-create optimizer slots in cross-replica context: under
            # strategy.run variables cannot be created inside the replica context.
            if hasattr(self._optimizer, 'build'):
                self._optimizer.build(self._model.trainable_variables)

        task_cfg = self._config.task
        # Build the merged training stream at the global batch size; the strategy
        # splits each global batch into per-replica slices, whose gradients sum to
        # the full-batch gradient (numerically identical to single-device).
        # drop_remainder keeps the batch dim static so the split is even; the
        # global batch should be divisible by the replica count.
        self._train_ds = self._task.build_inputs(task_cfg.train_data)
        if self._distributed:
            self._train_ds = self._strategy.experimental_distribute_dataset(self._train_ds)
        # Validation stays single-device; it is not the bottleneck and keeps the
        # COCO/distance/polygon aggregation simple.
        self._val_ds   = self._task.build_inputs(task_cfg.validation_data)

        _task, _model, _optimizer = self._task, self._model, self._optimizer
        _strategy = self._strategy

        if self._distributed:
            @tf.function
            def _compiled_train_step(inputs):
                per_replica = _strategy.run(
                    _task.train_step, args=(inputs, _model, _optimizer)
                )
                # Losses are normalized by the global object count, so each replica
                # returns its share and a sum-reduce reconstructs the full scalar.
                # The norm diagnostics are not per-replica shares (weight_norm is
                # identical on every replica, grad_norm is a per-replica
                # global-norm), so they mean-reduce for a single-device-equivalent
                # value.
                _mean_keys = ('grad_norm', 'weight_norm')
                return {
                    k: _strategy.reduce(
                        tf.distribute.ReduceOp.MEAN if k in _mean_keys
                        else tf.distribute.ReduceOp.SUM, v, axis=None)
                    for k, v in per_replica.items()
                }
        else:
            @tf.function
            def _compiled_train_step(inputs):
                return _task.train_step(inputs, _model, _optimizer)

        @tf.function
        def _compiled_val_step(inputs):
            return _task.validation_step(inputs, _model)

        self._compiled_train_step = _compiled_train_step
        self._compiled_val_step   = _compiled_val_step

        self._ckpt = tf.train.Checkpoint(
            model=self._model,
            optimizer=self._optimizer,      # full EMA wrapper: saves shadow weights + SGD slots
            global_step=self._global_step,
            completed_epochs=self._epoch_var,
            best_metric=self._best_metric,
        )
        self._ckpt_manager = tf.train.CheckpointManager(
            self._ckpt,
            self._output_dir,
            max_to_keep=self._config.trainer.max_to_keep,
        )
        # Separate manager for interruption saves (SIGTERM/SIGINT, supervisor
        # restarts, preemptions), rotating in resume/ (max 2) so the main
        # directory stays a clean epoch-boundary sequence. _auto_resume picks
        # whichever directory holds the highest global step, so a mid-epoch
        # interruption resumes where it stopped and a stale resume checkpoint is
        # superseded once a newer boundary save exists.
        self._resume_ckpt_manager = tf.train.CheckpointManager(
            self._ckpt,
            os.path.join(self._output_dir, 'resume'),
            max_to_keep=2,
        )

        self._tb_writer = tf.summary.create_file_writer(
            os.path.join(self._output_dir, 'tb_events')
        )

        self._shutdown_requested = False

        def _signal_handler(sig, frame):
            log.warning(
                "Signal %s received — will save checkpoint and exit after current step.", sig
            )
            self._shutdown_requested = True

        signal.signal(signal.SIGINT,  _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        params_path = os.path.join(self._output_dir, 'params.yaml')
        with open(params_path, 'w') as _f:
            yaml.dump(dataclasses.asdict(self._config), _f, default_flow_style=False)
        log.info("Full resolved config saved to %s", params_path)

        # Provenance: git commit + dirty diff, resolved TFDS versions, invocation,
        # and environment, so the run dir alone records what produced it.
        try:
            from datetime import datetime
            from common.run_metadata import write_run_metadata
            write_run_metadata(self._output_dir, self._config,
                               resume_from=self._resume_from,
                               started_at=datetime.now().astimezone().isoformat())
        except Exception as _e:                    # pragma: no cover - never fatal
            log.warning("run metadata skipped: %s", _e)

        self._log_startup_info()

    # ------------------------------------------------------------------
    # Startup logging
    # ------------------------------------------------------------------

    def _log_startup_info(self) -> None:
        """Log model architecture, parameter counts, and key training hyperparameters."""
        import io

        # ---- model architecture via Keras summary ----
        buf = io.StringIO()
        self._model.summary(print_fn=lambda line: buf.write(line + '\n'))
        log.info("Model architecture:\n%s", buf.getvalue())

        # ---- parameter counts ----
        total       = sum(v.numpy().size for v in self._model.variables)
        trainable   = sum(v.numpy().size for v in self._model.trainable_variables)
        n_total     = sum(v.numpy().size for v in self._model.non_trainable_variables)
        n_vars      = len(self._model.variables)
        n_train_v   = len(self._model.trainable_variables)
        n_nontrain_v = len(self._model.non_trainable_variables)
        log.info(
            "Parameters: %s total  (%s trainable, %s non-trainable)",
            f"{total:,}", f"{trainable:,}", f"{n_total:,}",
        )
        log.info(
            "Variables:  %d total  (%d trainable, %d non-trainable)",
            n_vars, n_train_v, n_nontrain_v,
        )

        # ---- key training hyperparameters ----
        t   = self._config.trainer
        tk  = self._config.task
        opt = t.optimizer_config
        lr  = opt.learning_rate
        td  = tk.train_data
        vd  = tk.validation_data
        dist_bs = (
            td.distance_data.global_batch_size
            if getattr(td, 'distance_data', None) else 0
        )
        log.info(
            "\n"
            "=== Training Configuration ===\n"
            "  Epochs            : %d\n"
            "  Steps/epoch       : %d  (fixed; stream repeats — one nominal data pass)\n"
            "  Total steps       : %d\n"
            "  Train batch size  : %d  (+ %d distance)\n"
            "  Val batch size    : %d\n"
            "  Val steps/epoch   : %d\n"
            "  ---\n"
            "  LR initial        : %g\n"
            "  LR alpha (min)    : %g\n"
            "  LR decay steps    : %d\n"
            "  Warmup steps      : %d\n"
            "  Smart bias LR     : %g\n"
            "  Momentum          : %g  (start: %g)\n"
            "  Weight decay      : %g\n"
            "  EMA decay         : %g  (dynamic: %s)\n"
            "  Grad clip norm    : %g\n"
            "  ---\n"
            "  Num classes       : %d\n"
            "  With polygons     : %s\n"
            "  With distance     : %s\n"
            "  Input size        : %s\n"
            "  ---\n"
            "  Loss gains        : iou=%g  cls=%g  dfl=%g  dist=%g  poly=%g\n"
            "  Poly sub-gains    : angle=%g  dist=%g  conf=%g\n"
            "  TAL               : alpha=%g  beta=%g  topk=%d\n"
            "  ---\n"
            "  Output dir        : %s\n"
            "  Init checkpoint   : %s\n"
            "==============================",
            t.train_epochs,
            t.steps_per_loop,
            t.train_steps,
            td.global_batch_size, dist_bs,
            vd.global_batch_size,
            t.validation_steps,
            lr.initial_learning_rate,
            lr.alpha,
            lr.decay_steps,
            opt.warmup_steps,
            tk.smart_bias_lr,
            opt.momentum, opt.momentum_start,
            opt.weight_decay,
            opt.ema.average_decay, opt.ema.dynamic_decay,
            tk.gradient_clip_norm,
            tk.num_classes,
            tk.with_polygons,
            tk.with_distance,
            tk.model.input_size,
            tk.losses.iou_gain, tk.losses.cls_gain, tk.losses.dfl_gain,
            tk.losses.dist_gain, tk.losses.poly_gain,
            tk.losses.poly_angle_gain, tk.losses.poly_dist_gain, tk.losses.poly_conf_gain,
            tk.losses.tal_alpha, tk.losses.tal_beta, tk.losses.topk,
            self._output_dir,
            tk.init_checkpoint or "none (training from scratch)",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, epoch: Optional[int] = None) -> dict:
        task_cfg       = self._config.task
        summary_types  = getattr(task_cfg, 'summary_types', 'scalar')
        do_img_summary = 'image' in summary_types
        n_summary      = getattr(task_cfg, 'summary_image_num', 10) if do_img_summary else 0

        summary_images = []   # list of float32 [H, W, 3]
        summary_preds  = []   # list of per-image prediction numpy dicts
        summary_gts    = []   # list of per-image ground-truth numpy dicts
        n_collected    = 0
        _GT_KEYS       = ('bbox', 'classes', 'n_gt', 'polygons')

        logs = None
        val_total = getattr(self._config.trainer, 'validation_steps', 0) or None
        ep_desc = f"Val epoch {epoch}" if epoch is not None else "Validation"
        vbar = Progress(total=val_total, desc=ep_desc, unit='batch')
        for inputs in self._val_ds:
            images, labels = inputs
            step_out   = self._compiled_val_step(inputs)
            logs       = self._task.aggregate_logs(logs, step_out)
            vbar.update(1)

            if n_collected < n_summary:
                imgs_np = images.numpy()   # [B, H, W, 3] uint8 from the eval parser
                # Renderers expect float [0, 1]; eval parser emits uint8 -> /255.
                if imgs_np.dtype == np.uint8:
                    imgs_np = imgs_np.astype('float32') / 255.0
                preds   = step_out['predictions']
                # Convert prediction + GT tensors to numpy once per batch
                preds_np = {k: (v.numpy() if hasattr(v, 'numpy') else v)
                            for k, v in preds.items()}
                gts_np   = {k: labels[k].numpy() for k in _GT_KEYS if k in labels}
                batch_sz = imgs_np.shape[0]
                take     = min(batch_sz, n_summary - n_collected)
                for i in range(take):
                    summary_images.append(imgs_np[i])
                    summary_preds.append({k: v[i] for k, v in preds_np.items()})
                    summary_gts.append({k: v[i] for k, v in gts_np.items()})
                n_collected += take

        vbar.close()
        val_metrics = self._task.reduce_aggregated_logs(
            logs, global_step=int(self._global_step)
        )

        # Append the per-category F1 report (best-conf + all-conf sweep + headline
        # scalars) as one line to <run>/val_history.jsonl. Extract any epoch back
        # to the ckpt-format txt/csv with utils/reports/val_history.py. Never fatal.
        report = getattr(self._task, '_last_val_report', None)
        if report is not None:
            try:
                from eval.val_history import append_record
                jsonl_path = os.path.join(self._output_dir, 'val_history.jsonl')
                append_record(
                    jsonl_path, report,
                    epoch=int(epoch) if epoch is not None else None,
                    step=int(self._global_step),
                    metrics=val_metrics,
                )
                log.info("Appended validation report -> %s (epoch %s)",
                         jsonl_path, epoch)
            except Exception as e:           # pragma: no cover - defensive
                log.warning("Could not append validation report: %s", e)

        if summary_images:
            self._log_image_summaries(summary_images, summary_preds, summary_gts)

        return val_metrics

    def _log_image_summaries(self, images: list, preds_list: list,
                             gts_list: list = None) -> None:
        """Render prediction- and ground-truth-overlay images to TensorBoard.

        Writes ``val/predictions`` and ``val/ground_truth`` at the same step for a
        direct comparison. Boxes are labelled with the class taxonomy names.
        """
        from common.viz_utils import render_summary_images, render_gt_images
        from configs.class_map import DETECTION_CLASSES
        task_cfg  = self._config.task
        draw_box  = getattr(task_cfg, 'summary_image_draw_box',  True)
        draw_poly = getattr(task_cfg, 'summary_image_draw_poly', True)
        step      = int(self._global_step)

        pred_canvas = render_summary_images(
            images, preds_list, draw_box=draw_box, draw_poly=draw_poly,
            class_names=DETECTION_CLASSES,
        )
        gt_canvas = None
        if gts_list:
            gt_canvas = render_gt_images(
                images, gts_list, draw_box=draw_box, draw_poly=draw_poly,
                class_names=DETECTION_CLASSES,
            )

        if pred_canvas is None and gt_canvas is None:
            return  # opencv not available

        with self._tb_writer.as_default():
            # TensorBoard expects float [N, H, W, C] in [0, 1] or uint8 [N, H, W, C]
            if pred_canvas is not None:
                tf.summary.image('val/predictions', pred_canvas, step=step,
                                 max_outputs=len(pred_canvas))
            if gt_canvas is not None:
                tf.summary.image('val/ground_truth', gt_canvas, step=step,
                                 max_outputs=len(gt_canvas))
        self._tb_writer.flush()

    def _log_aug_images(self, inputs, step: int) -> None:
        """Log augmented training images with ground-truth overlays once per epoch.

        Overlaying the post-augmentation GT boxes/polygons gives a visual check
        that mosaic/affine/flip kept the labels aligned with the pixels. Falls
        back to raw frames if labels are absent or opencv is unavailable.

        These are pre-colour-aug frames (geometry + labels only): colour
        augmentation runs on the accelerator inside the compiled train_step, so
        the images are the parser's uint8 output, scaled to float [0, 1] for the
        renderers.
        """
        try:
            from common.viz_utils import render_gt_images
            from configs.class_map import DETECTION_CLASSES
            task_cfg = self._config.task
            n_cfg    = getattr(task_cfg, 'summary_image_num', 10)

            images  = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            labels  = (inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1
                       else None)
            imgs_np = images.numpy()   # [B, H, W, 3] uint8 (or float [0,1] in tests)
            # Renderers expect float [0, 1]; uint8 parser output -> /255.
            if imgs_np.dtype == np.uint8:
                imgs_np = imgs_np.astype('float32') / 255.0
            n       = min(imgs_np.shape[0], n_cfg)

            canvas = None
            if labels is not None:
                draw_box  = getattr(task_cfg, 'summary_image_draw_box',  True)
                draw_poly = getattr(task_cfg, 'summary_image_draw_poly', True)
                gt_keys   = ('bbox', 'classes', 'n_gt', 'polygons')
                gts_np    = {k: labels[k].numpy() for k in gt_keys if k in labels}
                gts_list  = [{k: v[i] for k, v in gts_np.items()} for i in range(n)]
                canvas    = render_gt_images(
                    list(imgs_np[:n]), gts_list,
                    draw_box=draw_box, draw_poly=draw_poly,
                    class_names=DETECTION_CLASSES,
                )

            if canvas is None:
                # No labels, or opencv unavailable -> log the raw augmented frames.
                canvas = (imgs_np[:n] * 255.0).clip(0, 255).astype('uint8')

            with self._tb_writer.as_default():
                tf.summary.image('train/augmentations', canvas, step=step, max_outputs=n)
            self._tb_writer.flush()
        except Exception as e:
            log.debug("Could not log augmentation images: %s", e)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_step(path: str) -> int:
        """Parse the global step from a 'ckpt-<step>' checkpoint path (-1 if not parseable).

        All managed saves use ``checkpoint_number=global_step``, so the basename
        suffix IS the global step, comparable across the main and resume/ dirs.
        """
        try:
            return int(os.path.basename(path).rsplit('ckpt-', 1)[1])
        except (IndexError, ValueError):
            return -1

    @classmethod
    def _pick_latest_checkpoint(cls, candidates) -> Optional[str]:
        """Pick the candidate path with the highest embedded global step.

        Used to arbitrate between the main (epoch-boundary) directory and the
        resume/ (interruption) directory: a mid-epoch interruption save has a
        higher step than the last boundary save until the next boundary lands,
        after which the boundary checkpoint wins and the stale resume save is
        ignored ("used once" semantics, with no bookkeeping to corrupt).
        """
        candidates = [c for c in candidates if c]
        if not candidates:
            return None
        return max(candidates, key=cls._checkpoint_step)

    def _auto_resume(self, resume_from: str = None) -> int:
        """Restore the newest checkpoint (or an explicit one) and return the starting epoch.

        Considers BOTH the main (epoch-boundary) checkpoints and the resume/
        (interruption) checkpoints and restores whichever has the highest global
        step. Uses the explicit `completed_epochs` counter saved in the
        checkpoint, not global_step // steps_per_loop, so resume is exact even
        after mid-epoch SIGINT (the fixed-count epoch loop then runs only the
        remainder to the next boundary, so no step is skipped or repeated).

        Args:
            resume_from: Optional explicit checkpoint path. If given, it wins
                         over both directories.
        """
        target = resume_from
        if not target:
            candidates = (list(self._ckpt_manager.checkpoints)
                          + list(self._resume_ckpt_manager.checkpoints))
            spl = self._config.trainer.steps_per_loop
            mid_epoch_ok = getattr(self._config.trainer, 'mid_epoch_resume', False)
            if not mid_epoch_ok and spl > 0:
                # Boundary-only resume (default): every epoch is one full,
                # uniform pass of the stream. Mid-epoch interruption saves are
                # ignored (up to one epoch of compute is redone) unless they
                # are the ONLY restore points available.
                boundary = [c for c in candidates
                            if self._checkpoint_step(c) % spl == 0]
                dropped = sorted(set(candidates) - set(boundary))
                if dropped and boundary:
                    log.info(
                        "mid_epoch_resume=false: ignoring %d mid-epoch "
                        "checkpoint(s) %s; resuming from the last epoch "
                        "boundary.", len(dropped),
                        [os.path.basename(d) for d in dropped])
                elif dropped:
                    log.warning(
                        "mid_epoch_resume=false but only mid-epoch checkpoints "
                        "exist — resuming from one anyway rather than "
                        "restarting from scratch.")
                candidates = boundary or candidates
            target = self._pick_latest_checkpoint(candidates)
        if target:
            self._ckpt.restore(target)
            completed  = int(self._epoch_var)
            start_step = int(self._global_step)
            log.info(
                "Resumed from %s  (global_step=%d, completed_epochs=%d → starting epoch %d)",
                target, start_step, completed, completed + 1,
            )
            if completed > 0 and start_step == 0:
                log.warning(
                    "completed_epochs=%d but global_step=0 — checkpoint may be corrupt. Verify.",
                    completed,
                )
            return completed
        log.info("No checkpoint found — starting from epoch 1.")
        return 0

    def _save_checkpoint(self) -> None:
        path = self._ckpt_manager.save(
            checkpoint_number=int(self._global_step)
        )
        log.debug("Checkpoint saved: %s", path)

    def _save_resume_checkpoint(self) -> None:
        """Interruption save -> resume/ (rotating, max 2); main dir stays clean."""
        path = self._resume_ckpt_manager.save(
            checkpoint_number=int(self._global_step)
        )
        log.info("Resume checkpoint saved: %s", path)

    def _save_best_checkpoint(self, epoch: int, step: int) -> None:
        metric_name  = self._config.trainer.best_checkpoint_eval_metric
        metric_val   = float(self._best_metric)
        best_dir     = os.path.join(self._output_dir, f'best_{metric_name}')
        os.makedirs(best_dir, exist_ok=True)
        # Saves the raw (live) model weights plus the optimizer + step counters,
        # like a periodic checkpoint; no EMA swap into `model/`. The EMA shadow
        # weights are tracked inside the wrapper (`optimizer=self._optimizer`), so
        # this checkpoint already serializes them; eval/export recover them via
        # `ema.swap_in(model)`. Swapping EMA into `model/` would pair EMA weights
        # with SGD velocity slots computed against the raw weights, an incoherent
        # (weights, momentum) pair that corrupts a training resume.
        #
        # The optimizer + counters are required for resume: without the SGD
        # `iterations` the cosine LR would snap back to its initial value and the
        # momentum slots would reset. `global_step` / `completed_epochs` keep
        # resume bookkeeping consistent.
        best_ckpt = tf.train.Checkpoint(
            model=self._model,
            optimizer=self._optimizer,
            global_step=self._global_step,
            completed_epochs=self._epoch_var,
            best_metric=self._best_metric,
        )
        best_ckpt.write(os.path.join(best_dir, 'ckpt'))

        # Write a human-readable metadata file alongside the checkpoint.
        meta_path = os.path.join(best_dir, 'best_info.yaml')
        with open(meta_path, 'w') as f:
            f.write(
                f"metric: {metric_name}\n"
                f"value:  {metric_val:.6f}\n"
                f"epoch:  {epoch}\n"
                f"step:   {step}\n"
            )

        log.info(
            "Best checkpoint saved → %s  (%s=%.4f  epoch=%d  step=%d)",
            best_dir, metric_name, metric_val, epoch, step,
        )
        with self._tb_writer.as_default():
            self._scalar('epoch/best_checkpoint_epoch', epoch, step=step,
                         key='best_checkpoint_epoch')

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar(tag: str, value, step: int, key: str = None) -> None:
        """tf.summary.scalar with a markdown name+formula description (TB tooltip).

        ``key`` is the short metric key for the description lookup; defaults to the
        last path segment of ``tag`` (e.g. 'train/lr' -> 'lr').
        """
        from eval.metric_meta import describe
        lookup = key if key is not None else tag.split('/')[-1]
        tf.summary.scalar(tag, value, step=step, description=describe(lookup))

    def _log_step(self, losses: dict, step: int, step_time: float,
                  data_wait: float = 0.0) -> None:
        """Log per-step scalars to TensorBoard and console.

        ``step_time`` is the compute time (the train step itself); ``data_wait``
        is the time spent waiting on the input pipeline for this batch. The
        throughput is computed from their SUM (wall clock); reporting compute
        only would overstate speed whenever training is input-bound.
        """
        sgd        = self._optimizer._optimizer
        # Log the LR that moved the weights for the batch just completed, not next
        # step's. `apply_gradients` increments `iterations` at its end, so `sgd.lr`
        # (read at the current `iterations`) is one step ahead by now. The SGDTorch
        # introspection is hasattr-guarded because keras optimizers (adamw/adam)
        # have neither attribute.
        if hasattr(sgd, 'lr_for_last_step'):
            lr = float(sgd.lr_for_last_step)
        else:
            lr_attr = getattr(sgd, 'learning_rate', 0.0)
            lr = float(lr_attr(sgd.iterations) if callable(lr_attr) else lr_attr)
        momentum   = (float(sgd._current_momentum())
                      if hasattr(sgd, '_current_momentum') else 0.0)
        wall       = step_time + data_wait
        # Merged batch (detection + distance rows), what the step actually consumed.
        throughput = self._merged_batch_size / max(wall, 1e-9)
        # Ask the EMA wrapper itself so the logged value honors the configured
        # average_decay and dynamic_decay.
        ema_decay  = float(self._optimizer._get_decay())

        with self._tb_writer.as_default():
            for k, v in losses.items():
                self._scalar(f'train/{k}', v, step=step, key=k)
            self._scalar('train/lr',                   lr,               step=step)
            self._scalar('train/momentum',             momentum,         step=step)
            self._scalar('train/ema_decay',            ema_decay,        step=step)
            # Update-to-weight ratio (Karpathy): lr * ||grad|| / ||weights||. A healthy SGD
            # run sits around 1e-3; far above -> LR too high, far below -> too low / stuck.
            gnorm = float(losses.get('grad_norm', 0.0))
            wnorm = float(losses.get('weight_norm', 0.0))
            self._scalar('train/update_ratio', lr * gnorm / max(wnorm, 1e-12), step=step,
                         key='update_ratio')
            # Per-param-group effective LR (SGDTorch only) makes the bias/BN-vs-
            # weight warmup ramp visible. Skipped for keras optimizers.
            if hasattr(sgd, 'group_lrs_for_last_step'):
                lr_bias, lr_weight = sgd.group_lrs_for_last_step()
                self._scalar('train/lr_bias',   float(lr_bias),   step=step, key='lr_bias')
                self._scalar('train/lr_weight', float(lr_weight), step=step, key='lr_weight')
            self._scalar('train/step_time_ms',         step_time * 1000, step=step)
            self._scalar('train/data_wait_ms',         data_wait * 1000, step=step)
            self._scalar('train/throughput_img_per_s', throughput,       step=step)
            # GPU memory (best-effort; silently skipped on CPU-only machines)
            try:
                mem = tf.config.experimental.get_memory_info('GPU:0')
                self._scalar('system/gpu_mem_gb',      mem['current'] / 1e9, step=step)
                self._scalar('system/gpu_mem_peak_gb', mem['peak'] / 1e9,    step=step)
            except Exception:
                pass

        # Console output is the live progress bar (common/progress.py); this method
        # only writes the per-step TensorBoard scalars. The compute-vs-data-wait
        # breakdown lives in TB (train/step_time_ms, train/data_wait_ms,
        # train/throughput_img_per_s); the bar shows img/s + losses.

    def _log_epoch(
        self,
        epoch: int,
        epoch_losses: dict,
        val_metrics: dict,
        epoch_time: float,
        val_time: float,
        eta_seconds: float,
        throughput: float,
    ) -> None:
        """Log per-epoch scalars to TensorBoard and a structured console block."""
        step            = int(self._global_step)
        trainer_cfg     = self._config.trainer
        metric_name     = trainer_cfg.best_checkpoint_eval_metric
        best_so_far     = float(self._best_metric)
        train_time      = epoch_time - val_time

        scalar_metrics  = {k: v for k, v in val_metrics.items() if not k.startswith('cls/')}

        val_line = "  ".join(
            f"{k}={v:.4f}"
            for k, v in sorted(scalar_metrics.items())
            if k != 'best_conf_thresh'
        )

        def _loss_str(d):
            parts = [
                f"total={float(d.get('total_loss', 0.0)):.4f}",
                f"box={float(d.get('box_loss', 0.0)):.4f}",
                f"dfl={float(d.get('dfl_loss', 0.0)):.4f}",
                f"cls={float(d.get('cls_loss', 0.0)):.4f}",
                f"dist={float(d.get('dist_loss', 0.0)):.4f}",
                f"poly={float(d.get('poly_loss', 0.0)):.4f}",
            ]
            if 'poly_angle_loss' in d:
                parts.append(f"p_a={float(d['poly_angle_loss']):.4f}")
                parts.append(f"p_d={float(d['poly_dist_loss']):.4f}")
                parts.append(f"p_c={float(d['poly_conf_loss']):.4f}")
            return "  ".join(parts)

        log.info(
            "\n"
            "┌─ Epoch %d / %d ─────────────────────────────────────────\n"
            "│  Train (mean): %s\n"
            "│  Val         : %s\n"
            "│  Conf thresh : %.3f    Best %s: %.4f\n"
            "│  Timing      : epoch=%s  train=%s  val=%s  ETA=%s\n"
            "│  Throughput  : %.0f img/s\n"
            "└────────────────────────────────────────────────────────",
            epoch + 1, trainer_cfg.train_epochs,
            _loss_str(epoch_losses),
            val_line,
            float(scalar_metrics.get('best_conf_thresh', 0.0)),
            metric_name, best_so_far,
            _fmt_duration(epoch_time),
            _fmt_duration(train_time),
            _fmt_duration(val_time),
            _fmt_duration(eta_seconds),
            throughput,
        )

        with self._tb_writer.as_default():
            for k, v in val_metrics.items():
                # Per-category metrics arrive as `cls/<NN_name>/<metric>`. Route
                # them to a separate top-level group keyed by metric
                # (`per_class/<metric>/<NN_name>`) so all classes of one metric sit
                # together and out of the headline `val/` group. `key=k` keeps the
                # per-class tooltip.
                if k.startswith('cls/'):
                    parts = k.split('/')
                    if len(parts) == 3:
                        _, cls_name, metric = parts
                        self._scalar(f'per_class/{metric}/{cls_name}', v, step=step, key=k)
                        continue
                self._scalar(f'val/{k}', v, step=step, key=k)
            for k, v in epoch_losses.items():
                self._scalar(f'train/mean/{k}', v, step=step, key=k)
            self._scalar('epoch/time_s',                epoch_time,  step=epoch + 1)
            self._scalar('epoch/train_time_s',          train_time,  step=epoch + 1)
            self._scalar('epoch/val_time_s',            val_time,    step=epoch + 1)
            self._scalar('epoch/eta_s',                 eta_seconds, step=epoch + 1)
            self._scalar('epoch/throughput_img_per_s',  throughput,  step=epoch + 1)
            self._scalar(f'epoch/best_{metric_name}',   best_so_far, step=epoch + 1,
                         key=f'best_{metric_name}')
            # GPU memory peak over the epoch (best-effort)
            try:
                mem = tf.config.experimental.get_memory_info('GPU:0')
                self._scalar('system/gpu_mem_peak_gb', mem['peak'] / 1e9, step=epoch + 1)
            except Exception:
                pass
        self._tb_writer.flush()
