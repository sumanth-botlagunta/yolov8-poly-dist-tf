"""Training task orchestration for YOLOv8 polygon + distance model.

Responsibilities:
    - Build model (backbone, decoder, head, detection generator).
    - Load init checkpoint for backbone + decoder modules.
    - Build train and validation tf.data pipelines via InputReader.
    - Implement train_step and validation_step compatible with the
      TF Model Garden trainer loop.
    - Aggregate and log scalar + image summaries.
    - Manage EMA weight swapping around validation.

Optimizer: SGD-Torch variant with cosine LR decay + linear warmup.
    initial_lr:    0.01
    warmup_steps:  7,164
    total_steps:   716,400
    alpha (min LR ratio): 0.01

Classes:
    YoloV8Task: Training task class.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

log = logging.getLogger(__name__)


class YoloV8Task:
    """Orchestrates training and evaluation of the YOLOv8 model."""

    def __init__(self, config):
        """
        Args:
            config: ExperimentConfig object (parsed from YAML).
        """
        self._config = config
        self._model: Optional[tf.keras.Model] = None
        self._loss_fn = None

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def build_model(self) -> tf.keras.Model:
        """Instantiate backbone, decoder, head, detection generator, and wrap in YoloV8."""
        from models.yolo_v8 import build_yolov8
        model_cfg = self._config.task.model
        model = build_yolov8(model_cfg)
        model.deploy = False  # training mode: raw head outputs
        model.build_and_init(model_cfg.input_size)
        self._model = model
        return model

    def initialize(self, model: tf.keras.Model) -> None:
        """Load init checkpoint for backbone + decoder modules."""
        ckpt_path = self._config.task.init_checkpoint
        if not ckpt_path:
            return
        from tools.checkpoint_migration import migrate_checkpoint
        modules = self._config.task.init_checkpoint_modules
        stats = migrate_checkpoint(
            old_ckpt_path=ckpt_path,
            new_model=model,
            output_ckpt_path=os.path.join(os.path.dirname(ckpt_path), 'migrated', 'ckpt'),
            modules=modules,
        )
        log.info("Checkpoint migration: %s", stats)

    def build_inputs(
        self,
        params,
        input_context: Optional[tf.distribute.InputContext] = None,
    ) -> tf.data.Dataset:
        """Build train or eval tf.data pipeline from params config."""
        from data_pipeline.input_reader import build_input_reader_from_config
        reader = build_input_reader_from_config(
            data_cfg=params,
            task_cfg=self._config.task,
            is_training=params.is_training,
        )
        return reader(input_context)

    def build_optimizer(self) -> Any:
        """Build SGD with cosine LR schedule, warmup, and EMA wrapper.

        Requires build_model() to have been called first (EMA needs model.variables).
        Returns an ExponentialMovingAverage wrapping SGDTorch.
        """
        from optimizers.sgd_warmup import SGDTorch
        from optimizers.ema import ExponentialMovingAverage

        opt_cfg = self._config.trainer.optimizer_config
        lr_cfg  = opt_cfg.learning_rate
        ema_cfg = opt_cfg.ema

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr_cfg.initial_learning_rate,
            decay_steps=lr_cfg.decay_steps,
            alpha=lr_cfg.alpha,
        )

        sgd = SGDTorch(
            lr_fn=lr_schedule,
            momentum=opt_cfg.momentum,
            momentum_start=opt_cfg.momentum_start,
            nesterov=opt_cfg.nesterov,
            weight_decay=opt_cfg.weight_decay,
            warmup_steps=opt_cfg.warmup_steps,
        )

        ema = ExponentialMovingAverage(
            optimizer=sgd,
            model=self._model,
            average_decay=ema_cfg.average_decay,
            dynamic_decay=ema_cfg.dynamic_decay,
        )
        return ema

    def build_losses(self):
        """Instantiate TaskAlignedLossExtended from config."""
        from losses.tal_loss import TaskAlignedLossExtended
        task_cfg = self._config.task
        loss_cfg = task_cfg.losses
        return TaskAlignedLossExtended(
            num_classes=task_cfg.num_classes,
            iou_gain=loss_cfg.iou_gain,
            cls_gain=loss_cfg.cls_gain,
            dfl_gain=loss_cfg.dfl_gain,
            dist_gain=loss_cfg.dist_gain,
            poly_dist_gain=loss_cfg.poly_dist_gain,
            poly_conf_gain=loss_cfg.poly_conf_gain,
            poly_angle_gain=loss_cfg.poly_angle_gain,
            tal_alpha=loss_cfg.tal_alpha,
            tal_beta=loss_cfg.tal_beta,
            topk=loss_cfg.topk,
            reg_max=16,
            with_polygons=task_cfg.with_polygons,
            with_distance=task_cfg.with_distance,
        )

    # ------------------------------------------------------------------
    # Train / val steps
    # ------------------------------------------------------------------

    def train_step(
        self,
        inputs: Tuple[tf.Tensor, Dict],
        model: tf.keras.Model,
        optimizer,
        metrics=None,
    ) -> Dict[str, tf.Tensor]:
        """Single training step: forward, loss, gradients, EMA update.

        Returns:
            dict of scalar metric values for logging.
        """
        if self._loss_fn is None:
            self._loss_fn = self.build_losses()

        images, labels = inputs
        with tf.GradientTape() as tape:
            feats = model(images, training=True)
            total, box, dfl, cls, dist, poly = self._loss_fn(feats, labels)

        grads = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return {
            'total_loss': total,
            'box_loss':   box,
            'dfl_loss':   dfl,
            'cls_loss':   cls,
            'dist_loss':  dist,
            'poly_loss':  poly,
        }

    def validation_step(
        self,
        inputs: Tuple[tf.Tensor, Dict],
        model: tf.keras.Model,
        metrics=None,
    ) -> Dict[str, tf.Tensor]:
        """Single evaluation step using EMA weights.

        EMA swap_weights is managed at the epoch level, not per step.
        Runs the model in deploy=True mode to obtain decoded detections.
        """
        images, labels = inputs
        original_deploy = model.deploy
        model.deploy = True
        predictions = model(images, training=False)
        model.deploy = original_deploy
        return {'predictions': predictions, 'labels': labels}

    def aggregate_logs(self, state, step_outputs):
        """Accumulate per-step prediction/GT dicts for end-of-epoch evaluation."""
        if state is None:
            state = {'predictions': [], 'labels': []}
        state['predictions'].append(step_outputs['predictions'])
        state['labels'].append(step_outputs['labels'])
        return state

    def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
        """Compute mAP, F1@50, distance and polygon metrics from accumulated logs."""
        from eval.coco_metrics import COCOEvaluator
        from eval.distance_metrics import DistanceEvaluator
        from eval.polygon_metrics import PolygonEvaluator
        import numpy as np

        task_cfg  = self._config.task
        img_size  = task_cfg.model.input_size[:2]  # [H, W]

        coco_ev = COCOEvaluator(
            num_classes=task_cfg.num_classes,
            image_size=tuple(img_size),
        )
        val_has_distance = getattr(self._config.task.validation_data, 'with_distance', False)
        dist_ev = DistanceEvaluator() if (task_cfg.with_distance and val_has_distance) else None
        poly_ev = PolygonEvaluator(image_size=tuple(img_size)) if task_cfg.with_polygons else None

        for preds, labels in zip(
            aggregated_logs['predictions'], aggregated_logs['labels']
        ):
            coco_ev.update(preds, labels)

            if dist_ev is not None:
                # Per-image: match first GT distance to highest-confidence detection
                n_gt  = labels['n_gt'].numpy()
                gt_ld = labels['log_distance'].numpy()        # [B, M]
                pd_d  = preds['distance'].numpy()             # [B, max_det]
                nd    = preds['num_detections'].numpy()       # [B]
                for i in range(len(n_gt)):
                    if n_gt[i] > 0 and nd[i] > 0:
                        dist_ev.update(
                            pd_d[i, :nd[i]],
                            gt_ld[i, :n_gt[i]],
                        )

            if poly_ev is not None:
                poly_ev.update(
                    pred_boxes=preds['bbox'].numpy(),
                    pred_polygons=preds['polygons'].numpy(),
                    pred_scores=preds['confidence'].numpy(),
                    num_detections=preds['num_detections'].numpy(),
                    gt_boxes=labels['bbox'].numpy(),
                    gt_polygons=labels['polygons'].numpy(),
                    n_gt=labels['n_gt'].numpy(),
                )

        metrics = coco_ev.evaluate()
        if dist_ev is not None:
            metrics.update(dist_ev.evaluate())
        if poly_ev is not None:
            metrics.update(poly_ev.evaluate())

        if self._config.task.per_category_metrics:
            per_cat = coco_ev.per_category_ap50()
            metrics.update({f'cls/{c}': v for c, v in per_cat.items()})

        return metrics
