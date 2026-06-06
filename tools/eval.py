"""Standalone evaluation script for YOLOv8 polygon + distance model.

Loads a trained checkpoint (EMA weights preferred), runs inference over the
validation or test split, writes a COCO-format results JSON, and prints a
metric table.

Usage:
    python tools/eval.py \
        --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/ckpt-step \
        --split val \
        --output_json /tmp/results.json

Flags:
    --config       Path to experiment YAML.
    --checkpoint   Checkpoint path prefix (e.g. /output/ckpt-1000).
    --split        Dataset split to evaluate: 'val' or 'test'.
    --output_json  Path to write COCO-format detection results JSON.
    --image_size   Override image size (H,W). Defaults to value in config.
"""

import json
import logging

from absl import app, flags, logging as absl_logging
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',      None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('checkpoint',  None, 'Checkpoint path prefix.',          required=True)
    flags.DEFINE_string('split',       'val', "Eval split: 'val' or 'test'.")
    flags.DEFINE_string('output_json', None, 'Path to write COCO results JSON.')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)


def _load_model_from_checkpoint(config, ckpt_path: str) -> tf.keras.Model:
    """Build model, restore EMA weights from checkpoint if present."""
    from models.yolo_v8 import build_yolov8
    from optimizers.sgd_warmup import SGDTorch
    from optimizers.ema import ExponentialMovingAverage

    model = build_yolov8(config.task.model)
    model.deploy = True
    model.build_and_init(config.task.model.input_size)

    # Try to restore full checkpoint (model + EMA shadows)
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(ckpt_path)
    try:
        status.expect_partial()
        log.info("Checkpoint restored: %s", ckpt_path)
    except Exception as e:
        log.warning("Checkpoint restore warning: %s", e)

    return model


def main(_):
    tf.config.run_functions_eagerly(False)

    from configs.yaml_loader import load_config
    from train.task import YoloV8Task
    from eval.coco_metrics import COCOEvaluator
    from eval.distance_metrics import DistanceEvaluator
    from eval.polygon_metrics import PolygonEvaluator

    config = load_config(FLAGS.config)
    task   = YoloV8Task(config)

    model = _load_model_from_checkpoint(config, FLAGS.checkpoint)

    # Select split
    task_cfg  = config.task
    data_cfg  = (task_cfg.train_data if FLAGS.split == 'train'
                 else task_cfg.validation_data)
    # Override split name so we hit the correct TFDS split
    import dataclasses
    data_cfg = dataclasses.replace(data_cfg, is_training=False)
    val_ds   = task.build_inputs(data_cfg)

    img_size = tuple(task_cfg.model.input_size[:2])
    coco_ev  = COCOEvaluator(num_classes=task_cfg.num_classes, image_size=img_size)
    dist_ev  = DistanceEvaluator() if task_cfg.with_distance else None
    poly_ev  = PolygonEvaluator(image_size=img_size) if task_cfg.with_polygons else None

    # Accumulate raw COCO results for JSON export
    dt_results = []

    for step, (images, labels) in enumerate(val_ds):
        predictions = model(images, training=False)
        coco_ev.update(predictions, labels)

        if dist_ev:
            n_gt  = labels['n_gt'].numpy()
            gt_ld = labels['log_distance'].numpy()
            pd_d  = predictions['distance'].numpy()
            nd    = predictions['num_detections'].numpy()
            for i in range(len(n_gt)):
                if n_gt[i] > 0 and nd[i] > 0:
                    dist_ev.update(pd_d[i, :nd[i]], gt_ld[i, :n_gt[i]])

        if poly_ev:
            poly_ev.update(
                pred_boxes=predictions['bbox'].numpy(),
                pred_polygons=predictions['polygons'].numpy(),
                pred_scores=predictions['confidence'].numpy(),
                num_detections=predictions['num_detections'].numpy(),
                gt_boxes=labels['bbox'].numpy(),
                gt_polygons=labels['polygons'].numpy(),
                n_gt=labels['n_gt'].numpy(),
            )

        # Collect raw detections for JSON export
        B = int(predictions['num_detections'].shape[0])
        H, W = img_size
        for i in range(B):
            n_det = int(predictions['num_detections'][i])
            for j in range(n_det):
                y1, x1, y2, x2 = [float(v) for v in predictions['bbox'][i, j]]
                dt_results.append({
                    'image_id':    step * B + i,
                    'category_id': int(predictions['classes'][i, j]),
                    'bbox':        [x1*W, y1*H, (x2-x1)*W, (y2-y1)*H],
                    'score':       float(predictions['confidence'][i, j]),
                })

        if step % 10 == 0:
            log.info("Evaluated %d batches...", step + 1)

    # ---- Print metrics ----
    metrics = coco_ev.evaluate()
    if dist_ev:
        metrics.update(dist_ev.evaluate())
    if poly_ev:
        metrics.update(poly_ev.evaluate())

    print("\n=== Evaluation Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s}: {v:.4f}")

    # ---- Write COCO JSON ----
    if FLAGS.output_json:
        with open(FLAGS.output_json, 'w') as f:
            json.dump(dt_results, f)
        log.info("Results written to %s", FLAGS.output_json)


if __name__ == '__main__':
    app.run(main)
