"""Side-by-side evaluation of per-class vs class-agnostic NMS on one checkpoint.

Runs the network ONCE per batch (raw head outputs) and feeds the same raw
outputs through two detection generators that differ only in
``nms_class_mode`` ("per_class" vs "agnostic"), so both modes are scored on
byte-identical model predictions. Prints a headline metric comparison, the
per-class best-F1 movers, and writes both full ckpt-format reports.

The trained network is untouched — NMS is pure post-processing — so this
comparison needs no retraining and is valid for any existing checkpoint.

Usage:
    python -m tools.compare_nms_modes \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/run/ckpt-NNNN \
        [--split val] [--output_dir /path/to/out]

Output files (in --output_dir, default <ckpt_dir>/nms_compare):
    <ckpt>_per_class.txt/.json   full report, per-class NMS (current default)
    <ckpt>_agnostic.txt/.json    full report, class-agnostic NMS
"""

import dataclasses
import json
import logging
import os

from absl import app, flags
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',     None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('checkpoint', None, 'Checkpoint path prefix.', required=True)
    flags.DEFINE_string('split',      'val', "Eval split: 'val' or 'test'.")
    flags.DEFINE_string('output_dir', None, 'Where to write the two reports '
                        '(default: <ckpt_dir>/nms_compare).')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)

MODES = ('per_class', 'agnostic')

# Headline metrics, printed first in this order when present.
_HEADLINE = ('F1score50', 'precision50', 'recall50', 'best_conf_thresh',
             'mAP', 'mAP50', 'AR100', 'poly_mIoU', 'poly_recall50')


def _build_generator(model_cfg, mode: str):
    from models.detection_generator import YoloV8Layer
    dg = model_cfg.detection_generator
    return YoloV8Layer(
        input_image_size=model_cfg.input_size[:2],
        num_classes=model_cfg.num_classes,
        max_boxes=dg.max_boxes,
        nms_thresh=dg.nms_thresh,
        score_thresh=dg.score_thresh,
        pre_nms_points=dg.pre_nms_points,
        nms_type=dg.nms_type,
        nms_class_mode=mode,
        reg_max=16,
        output_poly_size=model_cfg.output_poly_size,
        min_distance=dg.min_distance,
        max_distance=dg.max_distance,
    )


def _evaluate_both(config, task, ckpt_path: str, split: str):
    """One pass over the split; returns {mode: (metrics, coco_ev)}."""
    from eval.coco_metrics import COCOEvaluator
    from eval.polygon_metrics import PolygonEvaluator
    from models.yolo_v8 import build_yolov8
    from tools.shared.ckpt_loading import restore_eval_weights
    from train.task import normalize_images

    task_cfg = config.task

    # Raw-output model: one forward per batch feeds BOTH generators.
    model = build_yolov8(task_cfg.model)
    model.deploy = False
    model.build_and_init(task_cfg.model.input_size)
    kind = restore_eval_weights(model, ckpt_path)
    log.info("Checkpoint restored (%s weights): %s", kind, ckpt_path)

    generators = {m: _build_generator(task_cfg.model, m) for m in MODES}

    data_cfg = (task_cfg.train_data if split == 'train'
                else task_cfg.validation_data)
    data_cfg = dataclasses.replace(data_cfg, is_training=False)
    val_ds = task.build_inputs(data_cfg)

    img_size = tuple(task_cfg.model.input_size[:2])
    coco_evs = {m: COCOEvaluator(
        num_classes=task_cfg.num_classes,
        image_size=img_size,
        ignore_dontcare=task_cfg.ignore_dontcare,
        ignore_iscrowds=task_cfg.ignore_iscrowds,
        iscrowds_labels=task_cfg.iscrowds_labels,
    ) for m in MODES}
    poly_evs = ({m: PolygonEvaluator(image_size=img_size) for m in MODES}
                if task_cfg.with_polygons else None)

    from tools.shared.progress import Progress
    pbar = Progress(total=None, desc='Evaluating (both NMS modes)', unit='batch')
    for images, labels in val_ds:
        raw = model(normalize_images(images), training=False)
        for m in MODES:
            predictions = generators[m](raw)
            coco_evs[m].update(predictions, labels)
            if poly_evs is not None:
                ic = labels.get('is_crowd')
                idc = labels.get('is_dontcare')
                poly_evs[m].update(
                    pred_boxes=predictions['bbox'].numpy(),
                    pred_polygons=predictions['polygons'].numpy(),
                    pred_scores=predictions['confidence'].numpy(),
                    num_detections=predictions['num_detections'].numpy(),
                    gt_boxes=labels['bbox'].numpy(),
                    gt_polygons=labels['polygons'].numpy(),
                    n_gt=labels['n_gt'].numpy(),
                    gt_is_crowd=(ic.numpy() if ic is not None else None),
                    gt_is_dontcare=(idc.numpy() if idc is not None else None),
                    pred_classes=predictions['classes'].numpy(),
                    gt_classes=labels['classes'].numpy(),
                )
        pbar.update(1)
    pbar.close()

    results = {}
    for m in MODES:
        metrics = coco_evs[m].evaluate()
        if poly_evs is not None:
            metrics.update(poly_evs[m].evaluate())
        results[m] = (metrics, coco_evs[m])
    return results


def _print_comparison(results) -> None:
    (m_pc, _), (m_ag, _) = results['per_class'], results['agnostic']
    shared = [k for k in _HEADLINE if k in m_pc and k in m_ag]
    shared += sorted(k for k in m_pc
                     if k in m_ag and k not in shared and not k.startswith('_')
                     and isinstance(m_pc[k], (int, float)))

    print("\n=== NMS mode comparison (same checkpoint, same raw predictions) ===")
    print(f"{'metric':<22}{'per_class':>12}{'agnostic':>12}{'delta':>12}")
    print("-" * 58)
    for k in shared:
        a, b = float(m_pc[k]), float(m_ag[k])
        print(f"{k:<22}{a:>12.4f}{b:>12.4f}{b - a:>+12.4f}")


def _print_class_movers(reports, top_n: int = 15) -> None:
    from eval.metrics_report import _name

    f1 = {}
    for m in MODES:
        f1[m] = {int(r['category']): float(r['f1'])
                 for r in reports[m].get('best_conf', [])}
    cats = sorted(set(f1['per_class']) & set(f1['agnostic']))
    movers = sorted(
        ((c, f1['agnostic'][c] - f1['per_class'][c]) for c in cats),
        key=lambda x: abs(x[1]), reverse=True,
    )[:top_n]

    print(f"\n=== top per-class best-F1 movers (agnostic − per_class) ===")
    print(f"{'cat':>4} {'name':<20}{'per_class':>11}{'agnostic':>11}{'delta':>10}")
    print("-" * 58)
    for c, d in movers:
        print(f"{c:>4} {_name(c):<20}{f1['per_class'][c]:>11.4f}"
              f"{f1['agnostic'][c]:>11.4f}{d:>+10.4f}")


def main(_):
    tf.config.run_functions_eagerly(False)

    from configs.yaml_loader import load_config
    from eval import metrics_report
    from tools.shared.runtime_setup import apply_eval_precision_policy
    from train.task import YoloV8Task

    config = load_config(FLAGS.config)
    apply_eval_precision_policy(config)
    task = YoloV8Task(config)

    out_dir = FLAGS.output_dir or os.path.join(
        os.path.dirname(FLAGS.checkpoint.rstrip('/')), 'nms_compare')
    os.makedirs(out_dir, exist_ok=True)

    results = _evaluate_both(config, task, FLAGS.checkpoint, FLAGS.split)

    base = os.path.basename(FLAGS.checkpoint.rstrip('/'))
    reports = {}
    for m in MODES:
        metrics, coco_ev = results[m]
        report = metrics_report.build_report(coco_ev)
        reports[m] = report
        paths = metrics_report.save_canonical(report, out_dir, f'{base}_{m}')
        with open(os.path.join(out_dir, f'{base}_{m}_metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()
                       if not k.startswith('_')}, f, indent=2)
        log.info("[%s] report: %s", m, paths['txt'])

    _print_comparison(results)
    _print_class_movers(reports)
    print(f"\nFull reports written to: {out_dir}")


if __name__ == '__main__':
    app.run(main)
