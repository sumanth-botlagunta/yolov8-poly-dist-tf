"""Custom training loop for YOLOv8.

Does NOT use orbit.Controller. Manages:
    - Epoch + step loop with tf.function-wrapped inner step.
    - EMA weight swap around validation.
    - tf.train.CheckpointManager for periodic saves and auto-resume.
    - TensorBoard logging via tf.summary.
    - Auto-resume: on startup checks output_dir for latest checkpoint.

Classes:
    YoloV8Trainer: Coordinates YoloV8Task with data, checkpoints, and logging.
"""

import dataclasses
import logging
import os
import signal
import time
from typing import Optional

import tensorflow as tf
import yaml

log = logging.getLogger(__name__)


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
    ):
        self._task        = task
        self._config      = config
        self._output_dir  = output_dir
        self._debug       = debug
        self._strategy    = strategy or tf.distribute.MirroredStrategy()

        self._model        = None
        self._optimizer    = None
        self._train_ds     = None
        self._val_ds       = None
        self._ckpt         = None
        self._ckpt_manager = None
        self._tb_writer    = None

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
        """Run the full training loop."""
        self._setup()
        start_epoch = self._auto_resume()

        trainer_cfg   = self._config.trainer
        # Guard against checkpoint_interval=0 (steps_per_loop not computed).
        ckpt_interval = max(trainer_cfg.checkpoint_interval, 1)
        # Log ~20× per epoch; at least every 50 steps.
        spl           = trainer_cfg.steps_per_loop
        log_interval  = max(50, spl // 20) if spl > 0 else 50

        # Save a pre-training checkpoint so epoch 1 crashes always have a restore point.
        if start_epoch == 0:
            self._save_checkpoint()
            log.info("Initial checkpoint saved at step 0.")

        training_start  = time.time()
        last_saved_step = int(self._global_step)  # prevents double-saves

        for epoch in range(start_epoch, total_epochs):
            log.info("─── Epoch %d / %d  (global_step=%d) ───",
                     epoch + 1, total_epochs, int(self._global_step))
            epoch_start  = time.time()
            step_times   = []
            step_losses  = {}   # last-step values (for _log_step)
            loss_accum   = {}   # running sum for epoch mean
            loss_count   = 0
            python_step  = int(self._global_step)

            for inputs in self._train_ds:
                t0 = time.time()
                step_losses = self._compiled_train_step(inputs)
                self._global_step.assign_add(1)
                python_step += 1
                step_dt = time.time() - t0
                step_times.append(step_dt)

                for k, v in step_losses.items():
                    loss_accum[k] = loss_accum.get(k, 0.0) + float(v)
                loss_count += 1

                if python_step % log_interval == 0:
                    self._log_step(step_losses, python_step, step_dt)

                # Mid-epoch periodic checkpoint; skip if this step was already saved.
                if python_step % ckpt_interval == 0 and python_step != last_saved_step:
                    self._save_checkpoint()
                    last_saved_step = python_step

                if self._shutdown_requested:
                    if python_step != last_saved_step:
                        self._save_checkpoint()
                        last_saved_step = python_step
                    log.info("Graceful shutdown at step %d (epoch %d interrupted).",
                             python_step, epoch + 1)
                    return

            # ---- validation (EMA weights) ----
            val_start = time.time()
            self._optimizer.swap_weights(self._model)
            val_metrics = self._run_validation()
            self._optimizer.swap_weights(self._model)
            val_time = time.time() - val_start

            # ---- best checkpoint ----
            metric_val = val_metrics.get(trainer_cfg.best_checkpoint_eval_metric, 0.0)
            if metric_val > float(self._best_metric):
                self._best_metric.assign(metric_val)
                self._save_best_checkpoint(epoch=epoch + 1, step=python_step)

            # Mark this epoch as fully complete BEFORE the epoch-end checkpoint so
            # that on resume we correctly skip this epoch and start from the next one.
            self._epoch_var.assign(epoch + 1)

            # Epoch-end checkpoint — always save, but skip if the last training step
            # already wrote one (avoids double-save when ckpt_interval == steps_per_loop).
            if python_step != last_saved_step:
                self._save_checkpoint()
                last_saved_step = python_step

            # ---- timing & logging ----
            epoch_time     = time.time() - epoch_start
            epochs_done    = epoch + 1 - start_epoch
            avg_epoch_time = (time.time() - training_start) / max(epochs_done, 1)
            eta_seconds    = avg_epoch_time * (total_epochs - (epoch + 1))
            avg_step_time  = sum(step_times) / max(len(step_times), 1)
            batch_size     = self._config.task.train_data.global_batch_size
            throughput     = batch_size / max(avg_step_time, 1e-9)
            epoch_losses   = {k: v / max(loss_count, 1) for k, v in loss_accum.items()}

            self._log_epoch(
                epoch, step_losses, epoch_losses, val_metrics,
                epoch_time=epoch_time,
                val_time=val_time,
                eta_seconds=eta_seconds,
                throughput=throughput,
            )

        total_time = time.time() - training_start
        log.info("Training complete. Total time: %s", _fmt_duration(total_time))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        with self._strategy.scope():
            self._model     = self._task.build_model()
            self._task.initialize(self._model)
            self._optimizer = self._task.build_optimizer()
            self._task._loss_fn = self._task.build_losses()

        task_cfg = self._config.task
        self._train_ds = self._task.build_inputs(task_cfg.train_data)
        self._val_ds   = self._task.build_inputs(task_cfg.validation_data)

        _task, _model, _optimizer = self._task, self._model, self._optimizer

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
            "  Steps/epoch       : %d\n"
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

    def _run_validation(self) -> dict:
        task_cfg       = self._config.task
        summary_types  = getattr(task_cfg, 'summary_types', 'scalar')
        do_img_summary = 'image' in summary_types
        n_summary      = getattr(task_cfg, 'summary_image_num', 10) if do_img_summary else 0

        summary_images = []   # list of float32 [H, W, 3]
        summary_preds  = []   # list of per-image numpy dicts
        n_collected    = 0

        logs = None
        for inputs in self._val_ds:
            images, _ = inputs
            step_out   = self._compiled_val_step(inputs)
            logs       = self._task.aggregate_logs(logs, step_out)

            if n_collected < n_summary:
                imgs_np = images.numpy()   # [B, H, W, 3]
                preds   = step_out['predictions']
                # Convert prediction tensors to numpy once per batch
                preds_np = {k: (v.numpy() if hasattr(v, 'numpy') else v)
                            for k, v in preds.items()}
                batch_sz = imgs_np.shape[0]
                take     = min(batch_sz, n_summary - n_collected)
                for i in range(take):
                    summary_images.append(imgs_np[i])
                    summary_preds.append({k: v[i] for k, v in preds_np.items()})
                n_collected += take

        val_metrics = self._task.reduce_aggregated_logs(
            logs, global_step=int(self._global_step)
        )

        if summary_images:
            self._log_image_summaries(summary_images, summary_preds)

        return val_metrics

    def _log_image_summaries(self, images: list, preds_list: list) -> None:
        """Render and write prediction-overlay images to TensorBoard."""
        from train.viz_utils import render_summary_images
        task_cfg  = self._config.task
        draw_box  = getattr(task_cfg, 'summary_image_draw_box',  True)
        draw_poly = getattr(task_cfg, 'summary_image_draw_poly', True)

        canvas = render_summary_images(
            images, preds_list,
            draw_box=draw_box,
            draw_poly=draw_poly,
        )
        if canvas is None:
            return  # opencv not available

        step = int(self._global_step)
        with self._tb_writer.as_default():
            # TensorBoard expects float [N, H, W, C] in [0, 1] or uint8 [N, H, W, C]
            tf.summary.image('val/predictions', canvas, step=step,
                             max_outputs=len(canvas))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _auto_resume(self) -> int:
        """Restore latest checkpoint and return the starting epoch.

        Uses the explicit `completed_epochs` counter saved in the checkpoint —
        not global_step // steps_per_loop — so resume is exact even after
        mid-epoch SIGINT or when steps_per_loop is zero.
        """
        latest = self._ckpt_manager.latest_checkpoint
        if latest:
            self._ckpt.restore(latest)
            completed  = int(self._epoch_var)
            start_step = int(self._global_step)
            log.info(
                "Resumed from %s  "
                "(global_step=%d, completed_epochs=%d → starting epoch %d)",
                latest, start_step, completed, completed + 1,
            )
            if completed > 0 and start_step == 0:
                log.warning(
                    "completed_epochs=%d but global_step=0 — checkpoint may be "
                    "corrupt or from a different run. Verify before continuing.",
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

    def _save_best_checkpoint(self, epoch: int, step: int) -> None:
        metric_name  = self._config.trainer.best_checkpoint_eval_metric
        metric_val   = float(self._best_metric)
        best_dir     = os.path.join(self._output_dir, f'best_{metric_name}')
        os.makedirs(best_dir, exist_ok=True)
        self._optimizer.swap_weights(self._model)
        best_ckpt = tf.train.Checkpoint(model=self._model)
        best_ckpt.write(os.path.join(best_dir, 'ckpt'))
        self._optimizer.swap_weights(self._model)

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
            tf.summary.scalar('epoch/best_checkpoint_epoch', epoch, step=step)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_step(self, losses: dict, step: int, step_time: float) -> None:
        """Log per-step scalars to TensorBoard and console."""
        sgd        = self._optimizer._optimizer
        lr         = float(sgd.lr)
        momentum   = float(sgd._current_momentum())
        batch_size = self._config.task.train_data.global_batch_size
        throughput = batch_size / max(step_time, 1e-9)
        total_loss = float(losses.get('total_loss', 0.0))

        with self._tb_writer.as_default():
            for k, v in losses.items():
                tf.summary.scalar(f'train/{k}', v, step=step)
            tf.summary.scalar('train/lr',                  lr,         step=step)
            tf.summary.scalar('train/momentum',            momentum,   step=step)
            tf.summary.scalar('train/step_time_ms',        step_time * 1000, step=step)
            tf.summary.scalar('train/throughput_img_per_s', throughput, step=step)
            # GPU memory (best-effort; silently skipped on CPU-only machines)
            try:
                mem = tf.config.experimental.get_memory_info('GPU:0')
                tf.summary.scalar('system/gpu_mem_gb',
                                  mem['current'] / 1e9, step=step)
                tf.summary.scalar('system/gpu_mem_peak_gb',
                                  mem['peak'] / 1e9, step=step)
            except Exception:
                pass

        log.info(
            "Step %7d | loss=%.4f  lr=%.2e  mom=%.4f  "
            "%.0fms/step  %.0f img/s",
            step, total_loss, lr, momentum,
            step_time * 1000, throughput,
        )

    def _log_epoch(
        self,
        epoch: int,
        last_step_losses: dict,
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
        per_cls_metrics = {k: v for k, v in val_metrics.items() if k.startswith('cls/')}

        val_line = "  ".join(
            f"{k}={v:.4f}"
            for k, v in sorted(scalar_metrics.items())
            if k != 'best_conf_thresh'
        )

        def _loss_str(d):
            return (
                f"total={float(d.get('total_loss', 0.0)):.4f}"
                f"  box={float(d.get('box_loss',   0.0)):.4f}"
                f"  dfl={float(d.get('dfl_loss',   0.0)):.4f}"
                f"  cls={float(d.get('cls_loss',   0.0)):.4f}"
                f"  dist={float(d.get('dist_loss', 0.0)):.4f}"
                f"  poly={float(d.get('poly_loss', 0.0)):.4f}"
            )

        log.info(
            "\n"
            "┌─ Epoch %d / %d ─────────────────────────────────────────\n"
            "│  Train (mean): %s\n"
            "│  Train (last): %s\n"
            "│  Val         : %s\n"
            "│  Conf thresh : %.3f    Best %s: %.4f\n"
            "│  Timing      : epoch=%s  train=%s  val=%s  ETA=%s\n"
            "│  Throughput  : %.0f img/s\n"
            "└────────────────────────────────────────────────────────",
            epoch + 1, trainer_cfg.train_epochs,
            _loss_str(epoch_losses),
            _loss_str(last_step_losses),
            val_line,
            float(scalar_metrics.get('best_conf_thresh', 0.0)),
            metric_name, best_so_far,
            _fmt_duration(epoch_time),
            _fmt_duration(train_time),
            _fmt_duration(val_time),
            _fmt_duration(eta_seconds),
            throughput,
        )

        if per_cls_metrics:
            log.info("Per-category AP50: %s",
                     {k: round(v, 4) for k, v in sorted(per_cls_metrics.items())})

        with self._tb_writer.as_default():
            for k, v in val_metrics.items():
                tf.summary.scalar(f'val/{k}', v, step=step)
            for k, v in epoch_losses.items():
                tf.summary.scalar(f'train/mean/{k}', v, step=step)
            for k, v in last_step_losses.items():
                tf.summary.scalar(f'train/last/{k}', float(v), step=step)
            tf.summary.scalar('epoch/time_s',            epoch_time,  step=epoch + 1)
            tf.summary.scalar('epoch/train_time_s',      train_time,  step=epoch + 1)
            tf.summary.scalar('epoch/val_time_s',        val_time,    step=epoch + 1)
            tf.summary.scalar('epoch/eta_s',             eta_seconds, step=epoch + 1)
            tf.summary.scalar('epoch/throughput_img_per_s', throughput, step=epoch + 1)
            tf.summary.scalar(f'epoch/best_{metric_name}', best_so_far, step=epoch + 1)
            # GPU memory peak over the epoch (best-effort)
            try:
                mem = tf.config.experimental.get_memory_info('GPU:0')
                tf.summary.scalar('system/gpu_mem_peak_gb',
                                  mem['peak'] / 1e9, step=epoch + 1)
            except Exception:
                pass
