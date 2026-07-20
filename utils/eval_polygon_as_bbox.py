"""Score the polygon head as if it were the box head.

Runs one split through a checkpoint and scores a COCO evaluator whose detection
boxes are replaced by the tight axis-aligned bbox enclosing each detection's
decoded polygon vertices. Class, score, and the NMS survivor set are unchanged --
only the box geometry becomes whatever the polygon implies.

The output is the same set of tables the normal box val report prints (headline
metrics, per-category AP50/AP/AP75/AR100, and the per-category best-conf +
all-conf sweep), so it can be diffed directly against an existing box val report
to see how well the polygons localize on their own: where the polygon-as-bbox
numbers track the box report the polygon is tight; where they fall short the
polygon extent is loose even though detection fires.

Polygon decode matches eval/polygon_metrics: the origin is the detection's box
center, vertex = center - dist * (cos, sin) at angle (i + offset) * angle_step,
with a per-bin conf gate (default 0.4). A detection whose gated polygon has fewer
than min_vertices bins keeps its original box; those fallbacks are counted and
reported. Because the polygon is center-anchored on the box head, this scores
polygon extent given the detection center, which is inherent to the radial
polygon format.

Usage:
    python -m utils.eval_polygon_as_bbox --config <cfg> --checkpoint /run/ckpt-N \
        --split val [--limit_batches 0] [--conf_thresh 0.4] [--min_vertices 3]
"""

import dataclasses
import logging
import tempfile

from absl import app, flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config', None, 'Experiment YAML.', required=True)
    flags.DEFINE_string('checkpoint', None, 'Checkpoint prefix.', required=True)
    flags.DEFINE_string('split', 'val', "Split: 'val' / 'test' / 'train'.")
    flags.DEFINE_integer('limit_batches', 0, 'Stop after N batches (0 = full split).')
    flags.DEFINE_float('conf_thresh', 0.4, 'Per-bin polygon conf gate (decode default 0.4).')
    flags.DEFINE_integer('min_vertices', 3, 'Min gated bins to build a poly box; '
                         'below this the detection keeps its model box (fallback).')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)


def _polygons_to_boxes(pred_boxes, pred_polygons, num_detections, H, W,
                       conf_thresh, min_verts):
    """Replace each detection's box with the bbox enclosing its polygon.

    Args:
      pred_boxes: [B, N, 4] yxyx normalized (model boxes; used for the polygon
        origin/center and as the fallback when the polygon is too sparse).
      pred_polygons: [B, N, 24, 3] = (conf, dist, angle); dist normalized, angle
        the sigmoid'd sub-bin offset in [0, 1).
      num_detections: [B] valid detections per image.

    Returns:
      (new_boxes [B, N, 4] yxyx normalized, n_poly used, n_fallback).
    """
    from eval.polygon_metrics import _radial_to_cartesian

    new_boxes = np.array(pred_boxes, dtype=np.float32).copy()
    n_poly = 0
    n_fallback = 0
    for i in range(new_boxes.shape[0]):
        nd = int(num_detections[i])
        for j in range(nd):
            poly = pred_polygons[i, j]          # [24, 3] = (conf, dist, angle)
            keep = np.asarray(poly[:, 0]) >= conf_thresh
            if int(keep.sum()) < min_verts:
                n_fallback += 1
                continue                        # keep the model box
            dist = np.maximum(poly[:, 1], 0.0)
            off = np.clip(poly[:, 2], 0.0, 1.0)
            y1, x1, y2, x2 = pred_boxes[i, j]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            verts = _radial_to_cartesian(cx, cy, dist, W, H, offsets=off)[keep]  # [K,2] px
            xs = np.clip(verts[:, 0] / W, 0.0, 1.0)
            ys = np.clip(verts[:, 1] / H, 0.0, 1.0)
            ymin, ymax = float(ys.min()), float(ys.max())
            xmin, xmax = float(xs.min()), float(xs.max())
            if ymax <= ymin or xmax <= xmin:    # degenerate poly box
                n_fallback += 1
                continue
            new_boxes[i, j] = [ymin, xmin, ymax, xmax]
            n_poly += 1
    return new_boxes, n_poly, n_fallback


def main(_):
    tf.config.run_functions_eagerly(False)
    logging.basicConfig(level=logging.INFO)

    from configs.yaml_loader import load_config
    from common.runtime_setup import apply_eval_precision_policy
    from train.task import YoloV8Task, normalize_images
    from eval.coco_metrics import COCOEvaluator
    from eval import metrics_report
    from utils.eval import _load_model_from_checkpoint
    from common.progress import Progress

    config = load_config(FLAGS.config)
    if not config.task.with_polygons:
        raise app.UsageError("This config has no polygon head (task.with_polygons is False).")

    apply_eval_precision_policy(config)
    task = YoloV8Task(config)
    model = _load_model_from_checkpoint(config, FLAGS.checkpoint)

    task_cfg = config.task
    data_cfg = (task_cfg.train_data if FLAGS.split == 'train'
                else task_cfg.validation_data)
    data_cfg = dataclasses.replace(data_cfg, is_training=False)
    val_ds = task.build_inputs(data_cfg)

    H, W = tuple(task_cfg.model.input_size[:2])
    coco_ev = COCOEvaluator(
        num_classes=task_cfg.num_classes,
        image_size=(H, W),
        ignore_dontcare=task_cfg.ignore_dontcare,
        ignore_iscrowds=task_cfg.ignore_iscrowds,
        iscrowds_labels=task_cfg.iscrowds_labels,
        find_best_score_thresh=task_cfg.find_best_score_thresh,
    )

    n_poly_total = 0
    n_fallback_total = 0
    n_batches = 0

    pbar = Progress(total=None, desc='poly-as-bbox', unit='batch')
    for images, labels in val_ds:
        predictions = model(normalize_images(images), training=False)
        new_boxes, n_poly, n_fb = _polygons_to_boxes(
            predictions['bbox'].numpy(),
            predictions['polygons'].numpy(),
            predictions['num_detections'].numpy(),
            H, W, FLAGS.conf_thresh, FLAGS.min_vertices)
        poly_preds = {
            'bbox': new_boxes,
            'classes': predictions['classes'].numpy(),
            'confidence': predictions['confidence'].numpy(),
            'num_detections': predictions['num_detections'].numpy(),
        }
        coco_ev.update(poly_preds, labels)

        n_poly_total += n_poly
        n_fallback_total += n_fb
        n_batches += 1
        pbar.update(1)
        if FLAGS.limit_batches and n_batches >= FLAGS.limit_batches:
            log.info("Stopping at --limit_batches=%d (sampled probe).", FLAGS.limit_batches)
            break
    pbar.close()

    metrics = coco_ev.evaluate()

    # Headline metrics (polygon-derived boxes).
    print("\n" + "=" * 62)
    print("  POLYGON-AS-BBOX detection metrics (poly vertices -> enclosing box)")
    print("=" * 62)
    for k in ['mAP', 'mAP50', 'AR100', 'F1score50', 'precision50', 'recall50',
              'best_conf_thresh']:
        if k in metrics:
            print(f"  {k:<16}: {metrics[k]:.4f}")
    tot = n_poly_total + n_fallback_total
    frac = (100.0 * n_poly_total / tot) if tot else 0.0
    print(f"\n  poly boxes used : {n_poly_total}/{tot}  ({frac:.1f}%)   "
          f"fallback-to-model box: {n_fallback_total}")

    # Per-category AP table (same columns as `utils.eval --per_category`).
    per_cat = coco_ev.per_category_full_metrics()
    print("\n=== Per-Category Metrics (polygon-as-bbox) ===")
    print(f"{'Cat':>4}  {'AP50':>6}  {'AP':>6}  {'AP75':>6}  {'AR100':>6}")
    for cat_id, m in sorted(per_cat.items()):
        print(f"{cat_id:>4}  {m['ap50']:>6.4f}  {m['ap']:>6.4f}  {m['ap75']:>6.4f}  {m['ar100']:>6.4f}")

    # Per-category best-conf + all-conf sweep (identical format to the val report
    # the trainer writes; reuse metrics_report so it diffs cleanly).
    report = metrics_report.build_report(coco_ev)
    with tempfile.NamedTemporaryFile('w+', suffix='.txt', delete=False) as tf_out:
        tmp_path = tf_out.name
    metrics_report.write_txt(report, tmp_path)
    with open(tmp_path) as f:
        print("\n" + f.read())


if __name__ == '__main__':
    app.run(main)
