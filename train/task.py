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
    warmup_steps:  6,354
    total_steps:   635,400
    alpha (min LR ratio): 0.01

Classes:
    YoloV8Task: Training task class.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

log = logging.getLogger(__name__)


def normalize_images(images: tf.Tensor) -> tf.Tensor:
    """uint8 [0, 255] → float32 [0, 1]; float images pass through unchanged.

    The parsers emit uint8 (colour aug + /255 moved to the batch level), so
    EVERY consumer that calls ``model(images)`` directly — ``validation_step``
    and ``tools/eval.py`` — must normalize through
    this one helper. Feeding raw uint8 to the model raises (float32 conv
    kernels); feeding 0–255 floats would silently produce garbage.
    """
    if images.dtype == tf.uint8:
        return tf.cast(images, tf.float32) / 255.0
    return images


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
        # Stashes the per-category F1 report from the last validation pass so the
        # trainer (which owns the output dir + epoch) can persist it. None until a
        # validation pass with a COCO evaluator has run.
        self._last_val_report = None

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
        """Build the optimizer + LR schedule (config-selectable) wrapped in EMA.

        Requires build_model() to have been called first (EMA needs model.variables).
        The optimizer and schedule are chosen by ``optimizer.type`` / ``learning_rate.type``
        (optimizers/factory.py); the defaults ('sgd' / 'cosine') reproduce the previous
        SGDTorch + CosineDecay path exactly. Returns an ExponentialMovingAverage wrapping
        the chosen optimizer.
        """
        from optimizers.factory import build_core_optimizer, build_lr_schedule
        from optimizers.ema import ExponentialMovingAverage

        opt_cfg = self._config.trainer.optimizer_config
        lr_cfg  = opt_cfg.learning_rate
        ema_cfg = opt_cfg.ema

        lr_schedule = build_lr_schedule(lr_cfg)
        core = build_core_optimizer(
            opt_cfg, lr_schedule, bias_lr_scale=self._config.task.smart_bias_lr)

        ema = ExponentialMovingAverage(
            optimizer=core,
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
            poly_gain=loss_cfg.poly_gain,
            tal_alpha=loss_cfg.tal_alpha,
            tal_beta=loss_cfg.tal_beta,
            topk=loss_cfg.topk,
            reg_max=16,
            with_polygons=task_cfg.with_polygons,
            with_distance=task_cfg.with_distance,
            angle_step=task_cfg.model.angle_step,
            use_acsl=loss_cfg.acsl.use_acsl,
            box_iou_type=loss_cfg.box_iou_type,
            cls_loss_type=loss_cfg.cls_loss_type,
            label_smoothing=loss_cfg.label_smoothing,
            focal_gamma=loss_cfg.focal_gamma,
            focal_alpha=loss_cfg.focal_alpha,
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

        # Per-batch colour augmentation on the accelerator (replaces the parser-
        # side /255 + HSV + albumentations). The parsers emit uint8 so the
        # pipeline carries 4× less memory; this runs HSV on every row and
        # albumentations only on detection rows (ignore_bg == 0). When images
        # already arrive as float (some tests), they're assumed to be in [0, 1]
        # and the /255 is skipped by batch_color_augment.
        from data_pipeline.batch_color_aug import batch_color_augment
        p = self._config.task.train_data.parser
        images = batch_color_augment(
            images,
            hue=p.aug_rand_hue,
            sat=p.aug_rand_saturation,
            val=p.aug_rand_brightness,
            albu_freq=p.albumentations_frequency,
            albu_row_mask=tf.equal(labels['ignore_bg'], 0),
        )

        with tf.GradientTape() as tape:
            feats = model(images, training=True)
            total, box, dfl, cls, dist, poly, poly_a, poly_d, poly_c = self._loss_fn(feats, labels)

        grads = tape.gradient(total, model.trainable_variables)
        # Pass clip_norm INTO the optimizer so clipping happens after the
        # cross-replica gradient sum (clipping here, per-replica, would break
        # single-vs-multi-GPU equivalence). No-op on a single replica.
        clip_norm = self._config.task.gradient_clip_norm
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables), clip_norm=clip_norm
        )

        return {
            'total_loss':      total,
            'box_loss':        box,
            'dfl_loss':        dfl,
            'cls_loss':        cls,
            'dist_loss':       dist,
            'poly_loss':       poly,
            'poly_angle_loss': poly_a,
            'poly_dist_loss':  poly_d,
            'poly_conf_loss':  poly_c,
        }

    def validation_step(
        self,
        inputs: Tuple[tf.Tensor, Dict],
        model: tf.keras.Model,
        metrics=None,
    ) -> Dict[str, tf.Tensor]:
        """Single evaluation step using EMA weights.

        EMA swap_in/swap_out is managed at the epoch level, not per step.
        Runs the model in deploy=True mode to obtain decoded detections.
        """
        images, labels = inputs
        # Parsers now emit uint8; normalize to [0, 1] here. Keep float passthrough
        # for backward compat (some tests feed already-normalized float images).
        images = normalize_images(images)
        original_deploy = model.deploy
        model.deploy = True
        try:
            predictions = model(images, training=False)
        finally:
            model.deploy = original_deploy
        return {'predictions': predictions, 'labels': labels}

    def _build_eval_state(self) -> Dict:
        """Construct the COCO/distance/polygon evaluators for one validation pass."""
        from eval.coco_metrics import COCOEvaluator
        from eval.distance_metrics import DistanceEvaluator
        from eval.polygon_metrics import PolygonEvaluator

        task_cfg = self._config.task
        img_size = tuple(task_cfg.model.input_size[:2])  # (H, W)

        coco_ev = COCOEvaluator(
            num_classes=task_cfg.num_classes,
            image_size=img_size,
            ignore_dontcare=task_cfg.ignore_dontcare,
            ignore_iscrowds=task_cfg.ignore_iscrowds,
            iscrowds_labels=task_cfg.iscrowds_labels,
        )
        val_has_distance = getattr(task_cfg.validation_data, 'with_distance', False)
        dist_ev = DistanceEvaluator() if (task_cfg.with_distance and val_has_distance) else None
        poly_ev = PolygonEvaluator(image_size=img_size) if task_cfg.with_polygons else None
        return {'coco': coco_ev, 'dist': dist_ev, 'poly': poly_ev}

    def _update_evaluators(self, state: Dict, preds: Dict, labels: Dict) -> None:
        """Update the evaluators with one batch (converts to numpy immediately).

        Streaming the per-batch update here — rather than buffering every batch's
        raw prediction/label tensors until end-of-epoch — bounds host memory to the
        evaluators' (much smaller) accumulators on large validation sets.
        """
        import numpy as np

        coco_ev, dist_ev, poly_ev = state['coco'], state['dist'], state['poly']
        coco_ev.update(preds, labels)

        if dist_ev is not None:
            # Match each GT to its highest-IoU detection (bbox IoU >= 0.5), then
            # compare that detection's predicted distance to the GT distance.
            # preds['distance'] is in METRES (already exp'd by the generator);
            # DistanceEvaluator expects log space, so convert pred back to log.
            from eval.polygon_metrics import _bbox_iou_matrix

            n_gt  = labels['n_gt'].numpy()
            gt_ld = labels['log_distance'].numpy()        # [B, M]  log-metres
            gt_bx = labels['bbox'].numpy()                # [B, M, 4] yxyx-norm
            # Clamp padded (0) slots before log so unused detections don't emit
            # log(0)=-inf warnings; only valid slots (>0) are indexed below.
            pd_d  = np.log(np.maximum(preds['distance'].numpy(), 1e-9))  # [B, max_det] log-metres
            pd_bx = preds['bbox'].numpy()                 # [B, max_det, 4]
            nd    = preds['num_detections'].numpy()       # [B]
            for i in range(len(n_gt)):
                ng, ndi = int(n_gt[i]), int(nd[i])
                if ng == 0 or ndi == 0:
                    continue
                iou = _bbox_iou_matrix(gt_bx[i, :ng], pd_bx[i, :ndi])  # [ng, ndi]
                matched_det = set()
                pred_pairs, gt_pairs = [], []
                for g in range(ng):
                    d = int(iou[g].argmax())
                    if iou[g, d] >= 0.5 and d not in matched_det:
                        matched_det.add(d)
                        pred_pairs.append(pd_d[i, d])
                        gt_pairs.append(gt_ld[i, g])
                if pred_pairs:
                    dist_ev.update(
                        np.asarray(pred_pairs, dtype=np.float32),
                        np.asarray(gt_pairs, dtype=np.float32),
                    )

        if poly_ev is not None:
            # Pass crowd/dontcare flags so they're excluded from the recall
            # denominator and matching (they're ignore regions). Eval labels carry
            # these; guard with .get for non-eval label dicts.
            ic = labels.get('is_crowd')
            idc = labels.get('is_dontcare')
            poly_ev.update(
                pred_boxes=preds['bbox'].numpy(),
                pred_polygons=preds['polygons'].numpy(),
                pred_scores=preds['confidence'].numpy(),
                num_detections=preds['num_detections'].numpy(),
                gt_boxes=labels['bbox'].numpy(),
                gt_polygons=labels['polygons'].numpy(),
                n_gt=labels['n_gt'].numpy(),
                gt_is_crowd=(ic.numpy() if ic is not None else None),
                gt_is_dontcare=(idc.numpy() if idc is not None else None),
            )

    def aggregate_logs(self, state, step_outputs):
        """Update evaluators incrementally with one validation batch.

        Builds the evaluators on the first call and streams each batch into them, so
        raw prediction/label tensors are not retained across the epoch.
        """
        if state is None:
            state = self._build_eval_state()
        self._update_evaluators(state, step_outputs['predictions'], step_outputs['labels'])
        return state

    def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
        """Finalize mAP, F1@50, distance and polygon metrics from the evaluators."""
        if aggregated_logs is None:
            # No validation batches were seen.
            return {}

        coco_ev = aggregated_logs['coco']
        dist_ev = aggregated_logs['dist']
        poly_ev = aggregated_logs['poly']

        metrics = coco_ev.evaluate()

        # Build the per-category F1/precision/recall report (best-conf + all-conf
        # sweep) and stash it for the trainer to persist. Tiny + off the train
        # step, so it does not affect training throughput. Never fatal.
        try:
            from eval.metrics_report import build_report
            self._last_val_report = build_report(
                coco_ev, step=int(global_step) if global_step is not None else None)
        except Exception as e:               # pragma: no cover - defensive
            log.warning("Could not build validation metrics report: %s", e)
            self._last_val_report = None
        if dist_ev is not None:
            metrics.update(dist_ev.evaluate())
        if poly_ev is not None:
            metrics.update(poly_ev.evaluate())

        if self._config.task.per_category_metrics:
            from configs.class_map import DETECTION_CLASSES
            per_cat = coco_ev.per_category_full_metrics()
            for cat_id, cat_m in per_cat.items():
                # Tag as cls/<NN>_<name>/<metric>: the zero-padded index keeps
                # TensorBoard's alphabetical ordering numeric, while the class name
                # makes the tag readable (no need to remember the index → name map).
                name = DETECTION_CLASSES.get(cat_id, f'class_{cat_id}')
                label = f'{cat_id:02d}_{name}'
                for mn, mv in cat_m.items():
                    metrics[f'cls/{label}/{mn}'] = mv

        return metrics
