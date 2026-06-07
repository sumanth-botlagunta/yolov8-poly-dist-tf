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
from typing import Optional

import tensorflow as tf
import yaml

log = logging.getLogger(__name__)


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

        self._global_step = tf.Variable(0, trainable=False, dtype=tf.int64,
                                        name='global_step')
        self._best_metric = tf.Variable(-1e9, trainable=False, dtype=tf.float32,
                                        name='best_metric')

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self, total_epochs: int) -> None:
        """Run the full training loop.

        Args:
            total_epochs: Total number of epochs to train (not additional).
        """
        self._setup()
        start_epoch = self._auto_resume()

        trainer_cfg = self._config.trainer

        for epoch in range(start_epoch, total_epochs):
            log.info("Epoch %d / %d", epoch + 1, total_epochs)

            # ---- training ----
            step_losses = {}
            python_step = int(self._global_step)  # one GPU-CPU sync per epoch
            for inputs in self._train_ds:
                step_losses = self._compiled_train_step(inputs)
                self._global_step.assign_add(1)
                python_step += 1

                if python_step % trainer_cfg.checkpoint_interval == 0:
                    self._log_step(step_losses, python_step)

                if python_step % trainer_cfg.checkpoint_interval == 0:
                    self._save_checkpoint()

            # ---- validation (EMA weights) ----
            self._optimizer.swap_weights(self._model)
            val_metrics = self._run_validation()
            self._optimizer.swap_weights(self._model)

            # ---- checkpoint ----
            self._save_checkpoint()
            metric_val = val_metrics.get(
                trainer_cfg.best_checkpoint_eval_metric, 0.0
            )
            if metric_val > float(self._best_metric):
                self._best_metric.assign(metric_val)
                self._save_best_checkpoint()

            # ---- logging ----
            self._log_epoch(epoch, step_losses, val_metrics)

        log.info("Training complete.")

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
            optimizer=self._optimizer,   # full EMA wrapper: saves shadow weights + SGD momentum
            global_step=self._global_step,
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

        params_path = os.path.join(self._output_dir, 'params.yaml')
        with open(params_path, 'w') as _f:
            yaml.dump(dataclasses.asdict(self._config), _f, default_flow_style=False)
        log.info("Full resolved config saved to %s", params_path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self) -> dict:
        logs = None
        for inputs in self._val_ds:
            step_out = self._compiled_val_step(inputs)
            logs = self._task.aggregate_logs(logs, step_out)
        return self._task.reduce_aggregated_logs(
            logs, global_step=int(self._global_step)
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _auto_resume(self) -> int:
        """Restore latest checkpoint and return starting epoch."""
        latest = self._ckpt_manager.latest_checkpoint
        if latest:
            self._ckpt.restore(latest)
            steps_per_epoch = self._config.trainer.steps_per_loop
            start_epoch = int(self._global_step) // max(steps_per_epoch, 1)
            log.info("Resumed from %s (global_step=%d, epoch~%d)",
                     latest, int(self._global_step), start_epoch)
            return start_epoch
        return 0

    def _save_checkpoint(self) -> None:
        path = self._ckpt_manager.save(
            checkpoint_number=int(self._global_step)
        )
        log.debug("Checkpoint saved: %s", path)

    def _save_best_checkpoint(self) -> None:
        best_dir = os.path.join(self._output_dir, 'best')
        os.makedirs(best_dir, exist_ok=True)
        best_ckpt = tf.train.Checkpoint(model=self._model)
        best_ckpt.write(os.path.join(best_dir, 'ckpt'))
        log.info("Best checkpoint saved (metric=%.4f)", float(self._best_metric))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_step(self, losses: dict, step: int) -> None:
        with self._tb_writer.as_default():
            for k, v in losses.items():
                tf.summary.scalar(f'train/{k}', v, step=step)

    def _log_epoch(self, epoch: int, train_losses: dict, val_metrics: dict) -> None:
        step = int(self._global_step)

        scalar_metrics  = {k: v for k, v in val_metrics.items() if not k.startswith('cls/')}
        per_cls_metrics = {k: v for k, v in val_metrics.items() if k.startswith('cls/')}

        log.info(
            "Epoch %d: total=%.4f  conf_thresh=%.3f  val=%s",
            epoch + 1,
            float(train_losses.get('total_loss', 0.0)),
            float(scalar_metrics.get('best_conf_thresh', 0.0)),
            {k: v for k, v in scalar_metrics.items() if k != 'best_conf_thresh'},
        )
        if per_cls_metrics:
            log.info("Per-category AP50: %s",
                     {k: round(v, 4) for k, v in sorted(per_cls_metrics.items())})

        with self._tb_writer.as_default():
            for k, v in val_metrics.items():
                tf.summary.scalar(f'val/{k}', v, step=step)
