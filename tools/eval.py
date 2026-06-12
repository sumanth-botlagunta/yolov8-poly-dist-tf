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
    --config        Path to experiment YAML.
    --checkpoint    Checkpoint path prefix (e.g. /output/ckpt-1000).
    --split         Dataset split to evaluate: 'val' or 'test'.
    --output_json   Path to write COCO-format detection results JSON.
    --per_category  Print per-category AP50/AP/AP75/AR100 table.
    --output_dir    If set, write metrics.json (and per_category_metrics.json) here.
"""

import json
import logging
import os

from absl import app, flags, logging as absl_logging
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',      None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('checkpoint',  None, 'Checkpoint path prefix.',          required=True)
    flags.DEFINE_string('split',       'val', "Eval split: 'val' or 'test'.")
    flags.DEFINE_string('output_json', None, 'Path to write COCO results JSON.')
    flags.DEFINE_bool(  'per_category', False, 'Print per-category metrics table.')
    flags.DEFINE_string('output_dir',  None, 'Directory to write metrics JSON files.')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)


def _load_model_from_checkpoint(config, ckpt_path: str) -> tf.keras.Model:
    """Build model and restore EMA weights (falls back to raw if none present)."""
    from models.yolo_v8 import build_yolov8
    from tools.ckpt_loading import restore_eval_weights

    model = build_yolov8(config.task.model)
    model.deploy = True
    model.build_and_init(config.task.model.input_size)

    kind = restore_eval_weights(model, ckpt_path)
    log.info("Checkpoint restored (%s weights): %s", kind, ckpt_path)
    return model


def main(_):
    tf.config.run_functions_eagerly(False)

    from configs.yaml_loader import load_config
    from train.task import YoloV8Task
    from eval.coco_metrics import COCOEvaluator
    from eval.distance_metrics import DistanceEvaluator
    from eval.polygon_metrics import PolygonEvaluator

    config = load_config(FLAGS.config)

    # Activate the same precision policy the trainer used so a bfloat16-trained
    # checkpoint is evaluated on the bfloat16 compute path (must run BEFORE the
    # model is built).
    from tools.runtime_setup import apply_eval_precision_policy
    apply_eval_precision_policy(config)

    task   = YoloV8Task(config)

    model = _load_model_from_checkpoint(config, FLAGS.checkpoint)

    # Select split. validation_data is the held-out test set here, so both
    # 'val' and 'test' map to it; only 'train' selects the training split.
    task_cfg  = config.task
    data_cfg  = (task_cfg.train_data if FLAGS.split == 'train'
                 else task_cfg.validation_data)
    # Force eval mode (no training-time augmentation) on the selected split.
    import dataclasses
    data_cfg = dataclasses.replace(data_cfg, is_training=False)
    val_ds   = task.build_inputs(data_cfg)

    img_size = tuple(task_cfg.model.input_size[:2])
    coco_ev  = COCOEvaluator(num_classes=task_cfg.num_classes, image_size=img_size)
    # Only evaluate distance when the chosen split actually carries distance GT
    # (distance is a training-only stream — gating on the model flag alone would
    # print a misleading dist_mae=0.0 on a split with no distance labels).
    val_has_distance = getattr(data_cfg, 'with_distance', False)
    dist_ev  = DistanceEvaluator() if (task_cfg.with_distance and val_has_distance) else None
    poly_ev  = PolygonEvaluator(image_size=img_size) if task_cfg.with_polygons else None

    from eval.polygon_metrics import _bbox_iou_matrix

    # Accumulate raw COCO results for JSON export
    dt_results = []
    total_batches = 0
    img_id_base = 0   # running image-id counter (val uses drop_remainder=False)

    from train.task import normalize_images

    for step, (images, labels) in enumerate(val_ds):
        # Eval parser emits uint8; the model needs float32 [0, 1] (feeding
        # uint8 raises on the float32 conv kernels).
        predictions = model(normalize_images(images), training=False)
        coco_ev.update(predictions, labels)

        if dist_ev:
            # Match each GT to its highest-IoU detection (>=0.5) and compare that
            # detection's distance to the GT distance. preds['distance'] is in METRES
            # (already exp'd by the generator); convert to log to match the GT and
            # the DistanceEvaluator's log-space contract.
            n_gt  = labels['n_gt'].numpy()
            gt_ld = labels['log_distance'].numpy()
            gt_bx = labels['bbox'].numpy()
            pd_d  = np.log(predictions['distance'].numpy())
            pd_bx = predictions['bbox'].numpy()
            nd    = predictions['num_detections'].numpy()
            for i in range(len(n_gt)):
                ng, ndi = int(n_gt[i]), int(nd[i])
                if ng == 0 or ndi == 0:
                    continue
                iou = _bbox_iou_matrix(gt_bx[i, :ng], pd_bx[i, :ndi])
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

        if poly_ev:
            ic = labels.get('is_crowd')
            idc = labels.get('is_dontcare')
            poly_ev.update(
                pred_boxes=predictions['bbox'].numpy(),
                pred_polygons=predictions['polygons'].numpy(),
                pred_scores=predictions['confidence'].numpy(),
                num_detections=predictions['num_detections'].numpy(),
                gt_boxes=labels['bbox'].numpy(),
                gt_polygons=labels['polygons'].numpy(),
                n_gt=labels['n_gt'].numpy(),
                gt_is_crowd=(ic.numpy() if ic is not None else None),
                gt_is_dontcare=(idc.numpy() if idc is not None else None),
            )

        # Collect raw detections for JSON export. Use a running image-id base
        # (not step*B): the final val batch is smaller (drop_remainder=False), so
        # step*B would collide image ids across batches in the exported JSON.
        B = int(predictions['num_detections'].shape[0])
        H, W = img_size
        for i in range(B):
            n_det = int(predictions['num_detections'][i])
            for j in range(n_det):
                y1, x1, y2, x2 = [float(v) for v in predictions['bbox'][i, j]]
                dt_results.append({
                    'image_id':    img_id_base + i,
                    'category_id': int(predictions['classes'][i, j]),
                    'bbox':        [x1*W, y1*H, (x2-x1)*W, (y2-y1)*H],
                    'score':       float(predictions['confidence'][i, j]),
                })
        img_id_base += B

        total_batches += 1
        if step % 50 == 0:
            log.info("Evaluated %d / ? batches...", total_batches)

    log.info("Evaluation complete: %d batches total.", total_batches)

    # ---- Compute metrics ----
    metrics = coco_ev.evaluate()
    if dist_ev:
        metrics.update(dist_ev.evaluate())
    if poly_ev:
        metrics.update(poly_ev.evaluate())

    print("\n=== Evaluation Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:25s}: {v:.4f}")

    # Print best confidence threshold if available
    if 'best_conf_thresh' in metrics:
        print(f"\n  best_conf_thresh         : {metrics['best_conf_thresh']:.4f}")

    # ---- Per-category metrics ----
    per_cat = None
    if FLAGS.per_category:
        per_cat = coco_ev.per_category_full_metrics()
        print("\n=== Per-Category Metrics ===")
        print(f"{'Cat':>4}  {'AP50':>6}  {'AP':>6}  {'AP75':>6}  {'AR100':>6}")
        for cat_id, m in sorted(per_cat.items()):
            print(f"{cat_id:>4}  {m['ap50']:>6.4f}  {m['ap']:>6.4f}  {m['ap75']:>6.4f}  {m['ar100']:>6.4f}")

    # ---- Write output files ----
    if FLAGS.output_dir:
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        metrics_path = os.path.join(FLAGS.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        log.info("Metrics written to %s", metrics_path)

        if per_cat is not None:
            pc_path = os.path.join(FLAGS.output_dir, 'per_category_metrics.json')
            with open(pc_path, 'w') as f:
                json.dump({str(k): {mk: float(mv) for mk, mv in m.items()}
                           for k, m in per_cat.items()}, f, indent=2)
            log.info("Per-category metrics written to %s", pc_path)

    # ---- Write COCO JSON ----
    if FLAGS.output_json:
        with open(FLAGS.output_json, 'w') as f:
            json.dump(dt_results, f)
        log.info("Results written to %s", FLAGS.output_json)


if __name__ == '__main__':
    app.run(main)
