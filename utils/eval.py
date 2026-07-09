"""Standalone evaluation for the YOLOv8 polygon + distance model.

Loads a trained checkpoint (EMA weights preferred), runs inference over the
validation or test split, computes the metric table, and optionally writes a
COCO-format results JSON.

Three modes share one evaluation code path (`evaluate_checkpoint`):

  * single   (default, ``--checkpoint <ckpt>``): evaluate one checkpoint and print
    the metric table, with the full single-checkpoint reporting (``--split``,
    ``--per_category``, ``--output_json``, ``--output_dir``).
  * all      (``--all --watch_dir <dir>``): evaluate every checkpoint already in
    ``<dir>`` once, appending each result as a JSON line to
    ``<dir>/eval_log.jsonl``.
  * watch    (``--watch --watch_dir <dir>``): poll ``<dir>`` and evaluate each new
    checkpoint as it appears, appending to ``<dir>/eval_log.jsonl`` (``--interval``,
    ``--max_evals``).

Usage:
    # one checkpoint
    python -m utils.eval \
        --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/ckpt-step \
        --split val \
        --output_json /tmp/results.json

    # every existing checkpoint in a run directory, once
    python -m utils.eval --config <cfg> --all   --watch_dir /run

    # keep watching for new checkpoints
    python -m utils.eval --config <cfg> --watch --watch_dir /run --interval 300

Flags:
    --config        Path to experiment YAML.
    --checkpoint    Checkpoint path prefix (single mode; e.g. /output/ckpt-1000).
    --split         Dataset split to evaluate: 'val' or 'test'.
    --output_json   Path to write COCO-format detection results JSON (single mode).
    --per_category  Print per-category AP50/AP/AP75/AR100 table (single mode).
    --output_dir    If set, write metrics.json (and per_category_metrics.json) here
                    (single mode).
    --all           Evaluate every existing checkpoint in --watch_dir once.
    --watch         Poll --watch_dir and evaluate each new checkpoint.
    --watch_dir     Run directory to scan (required for --all / --watch).
    --interval      Seconds between polls in --watch mode.
    --max_evals     Stop after this many evaluations in --watch mode (0 = unlimited).
"""

import json
import logging
import os
import time

from absl import app, flags, logging as absl_logging
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',      None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('checkpoint',  None, 'Checkpoint path prefix (single mode).')
    flags.DEFINE_string('split',       'val', "Eval split: 'val' or 'test'.")
    flags.DEFINE_string('output_json', None, 'Path to write COCO results JSON.')
    flags.DEFINE_bool(  'per_category', False, 'Print per-category metrics table.')
    flags.DEFINE_string('output_dir',  None, 'Directory to write metrics JSON files.')
    flags.DEFINE_bool(  'all',   False, 'Evaluate every existing checkpoint in --watch_dir once.')
    flags.DEFINE_bool(  'watch', False, 'Poll --watch_dir and evaluate each new checkpoint.')
    flags.DEFINE_string('watch_dir', None, 'Run directory to scan (for --all / --watch).')
    flags.DEFINE_integer('interval', 300, 'Seconds between polls (--watch mode).')
    flags.DEFINE_integer('max_evals', 0, 'Max evaluations in --watch mode (0 = unlimited).')
    flags.DEFINE_bool(  'dump_failures', False, 'Mine failure cases (FP / missed-GT / low-IoU) '
                        'and write the worst per class as annotated images. Single mode.')
    flags.DEFINE_string('failures_dir', None, 'Where to write failure images '
                        '(default <output_dir>/failures or /tmp/eval_failures).')
    flags.DEFINE_integer('failures_per_class', 8, 'Worst cases to keep per class per kind.')
    flags.DEFINE_integer('limit_batches', 0, 'Stop after this many batches (0 = full split). '
                         'Makes a train-split probe affordable: ~250 batches gives a '
                         'val-sized sample instead of an hours-long full pass.')
    flags.DEFINE_bool('coco_envelope_sweep', False,
                      'Build the report all-conf table from COCO-interpolated envelope '
                      'precision instead of the raw operating-point sweep (default: raw, '
                      'which agrees with the best-conf table / F1score50).')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)


def _load_model_from_checkpoint(config, ckpt_path: str) -> tf.keras.Model:
    """Build the model and restore EMA weights (falls back to raw if none present)."""
    from models.yolo_v8 import build_yolov8
    from common.ckpt_loading import restore_eval_weights

    model = build_yolov8(config.task.model)
    model.deploy = True
    model.build_and_init(config.task.model.input_size)

    kind = restore_eval_weights(model, ckpt_path)
    log.info("Checkpoint restored (%s weights): %s", kind, ckpt_path)
    return model


def evaluate_checkpoint(config, task, ckpt_path: str, split: str = 'val',
                        collect_json: bool = False, failure_collector=None,
                        limit_batches: int = 0):
    """Evaluate one checkpoint and return (metrics, dt_results).

    Shared by every mode. Builds the model, restores EMA weights, runs inference
    over the selected split, and returns the COCO + polygon + distance metric dict.
    When ``collect_json`` is True, ``dt_results`` is the COCO-format detection list
    (else an empty list).

    The caller is responsible for activating the precision policy
    (``apply_eval_precision_policy``) once before the first model is built.
    """
    from eval.coco_metrics import COCOEvaluator
    from eval.distance_metrics import DistanceEvaluator
    from eval.polygon_metrics import PolygonEvaluator
    from eval.polygon_metrics import _bbox_iou_matrix
    from train.task import normalize_images
    import dataclasses

    model = _load_model_from_checkpoint(config, ckpt_path)

    # Select split. validation_data is the held-out test set here, so both
    # 'val' and 'test' map to it; only 'train' selects the training split.
    task_cfg = config.task
    data_cfg = (task_cfg.train_data if split == 'train'
                else task_cfg.validation_data)
    # Force eval mode (no training-time augmentation) on the selected split.
    data_cfg = dataclasses.replace(data_cfg, is_training=False)
    val_ds = task.build_inputs(data_cfg)

    img_size = tuple(task_cfg.model.input_size[:2])
    # Pass the SAME crowd/dontcare flags the trainer's evaluator uses
    # (train/task.py:_build_eval_state) — building on hardcoded defaults made
    # this tool silently diverge from val_history.jsonl the day a config
    # changed those flags.
    coco_ev = COCOEvaluator(
        num_classes=task_cfg.num_classes,
        image_size=img_size,
        ignore_dontcare=task_cfg.ignore_dontcare,
        ignore_iscrowds=task_cfg.ignore_iscrowds,
        iscrowds_labels=task_cfg.iscrowds_labels,
    )
    # Only evaluate distance when the chosen split actually carries distance GT
    # (distance is a training-only stream — gating on the model flag alone would
    # print a misleading dist_mae=0.0 on a split with no distance labels).
    val_has_distance = getattr(data_cfg, 'with_distance', False)
    dist_ev = DistanceEvaluator() if (task_cfg.with_distance and val_has_distance) else None
    poly_ev = PolygonEvaluator(image_size=img_size) if task_cfg.with_polygons else None

    dt_results = []
    total_batches = 0
    img_id_base = 0   # running image-id counter (val uses drop_remainder=False)

    from common.progress import Progress
    pbar = Progress(total=None, desc='Evaluating', unit='batch')   # val_ds length unknown
    for step, (images, labels) in enumerate(val_ds):
        # Eval parser emits uint8; the model needs float32 [0, 1] (feeding
        # uint8 raises on the float32 conv kernels).
        predictions = model(normalize_images(images), training=False)
        coco_ev.update(predictions, labels)

        if failure_collector is not None:
            imgs_np = images.numpy()      # uint8 [B,H,W,3] RGB from the eval parser
            for i in range(imgs_np.shape[0]):
                failure_collector.update(
                    imgs_np[i],
                    {'bbox': predictions['bbox'][i].numpy(),
                     'classes': predictions['classes'][i].numpy(),
                     'confidence': predictions['confidence'][i].numpy(),
                     'num_detections': int(predictions['num_detections'][i])},
                    {'bbox': labels['bbox'][i].numpy(),
                     'classes': labels['classes'][i].numpy(),
                     'n_gt': int(labels['n_gt'][i])})

        if dist_ev:
            # Match each GT to its highest-IoU detection (>=0.5) and compare that
            # detection's distance to the GT distance. preds['distance'] is in METRES
            # (already exp'd by the generator); convert to log to match the GT and
            # the DistanceEvaluator's log-space contract.
            n_gt  = labels['n_gt'].numpy()
            gt_ld = labels['log_distance'].numpy()
            gt_bx = labels['bbox'].numpy()
            # Padded detections beyond num_detections are 0; clamp before log so the
            # unused slots don't emit log(0)=-inf RuntimeWarnings (only valid slots,
            # which are >0, are ever indexed below).
            pd_d  = np.log(np.maximum(predictions['distance'].numpy(), 1e-9))
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
                pred_classes=predictions['classes'].numpy(),
                gt_classes=labels['classes'].numpy(),
            )

        if collect_json:
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
        pbar.update(1)
        if limit_batches and total_batches >= limit_batches:
            log.info("Stopping at --limit_batches=%d (sampled probe, not the full split).",
                     limit_batches)
            break

    pbar.close()
    log.info("Evaluation complete: %d batches total.", total_batches)

    metrics = coco_ev.evaluate()
    if dist_ev:
        metrics.update(dist_ev.evaluate())
    if poly_ev:
        metrics.update(poly_ev.evaluate())

    # Keep a handle on the evaluator for per-category reporting in single mode.
    metrics['_coco_evaluator'] = coco_ev
    return metrics, dt_results


def _print_metrics(metrics: dict) -> None:
    print("\n=== Evaluation Results ===")
    for k, v in sorted(metrics.items()):
        if k.startswith('_'):
            continue
        print(f"  {k:25s}: {v:.4f}")
    if 'best_conf_thresh' in metrics:
        print(f"\n  best_conf_thresh         : {metrics['best_conf_thresh']:.4f}")


def _run_single(config, task):
    """Single-checkpoint mode: evaluate, print, and write output files."""
    if not FLAGS.checkpoint:
        raise app.UsageError("--checkpoint is required (or use --all / --watch with --watch_dir).")

    failure_collector = None
    if FLAGS.dump_failures:
        from eval.failure_mining import FailureCollector
        try:
            from configs.class_map import DETECTION_CLASSES
            names = [str(DETECTION_CLASSES[i]) for i in sorted(DETECTION_CLASSES)]
        except Exception:
            names = None
        failure_collector = FailureCollector(class_names=names,
                                             per_class=FLAGS.failures_per_class)

    metrics, dt_results = evaluate_checkpoint(
        config, task, FLAGS.checkpoint, split=FLAGS.split,
        collect_json=bool(FLAGS.output_json), failure_collector=failure_collector,
        limit_batches=FLAGS.limit_batches)
    coco_ev = metrics.pop('_coco_evaluator')

    _print_metrics(metrics)

    if failure_collector is not None:
        fdir = FLAGS.failures_dir or os.path.join(FLAGS.output_dir or '/tmp/eval_failures', 'failures')
        n = failure_collector.write(fdir)
        log.info("Failure mining: wrote %d annotated images to %s  (%s)",
                 n, fdir, failure_collector.summary())

    per_cat = None
    if FLAGS.per_category:
        per_cat = coco_ev.per_category_full_metrics()
        print("\n=== Per-Category Metrics ===")
        print(f"{'Cat':>4}  {'AP50':>6}  {'AP':>6}  {'AP75':>6}  {'AR100':>6}")
        for cat_id, m in sorted(per_cat.items()):
            print(f"{cat_id:>4}  {m['ap50']:>6.4f}  {m['ap']:>6.4f}  {m['ap75']:>6.4f}  {m['ar100']:>6.4f}")

    if FLAGS.output_dir:
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        metrics_path = os.path.join(FLAGS.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        log.info("Metrics written to %s", metrics_path)

        # Same per-validation report the trainer drops next to checkpoints: best F1 /
        # precision / recall per category over the 0.10 conf grid, mean line, and the
        # full sweep -> <base>.json + <base>.txt.
        from eval import metrics_report
        base = os.path.basename(FLAGS.checkpoint.rstrip('/')) + '_val'
        report = metrics_report.build_report(
            coco_ev, envelope_sweep=FLAGS.coco_envelope_sweep)
        paths = metrics_report.save_canonical(report, FLAGS.output_dir, base)
        log.info("Validation report written to %s and %s", paths['json'], paths['txt'])

        if per_cat is not None:
            pc_path = os.path.join(FLAGS.output_dir, 'per_category_metrics.json')
            with open(pc_path, 'w') as f:
                json.dump({str(k): {mk: float(mv) for mk, mv in m.items()}
                           for k, m in per_cat.items()}, f, indent=2)
            log.info("Per-category metrics written to %s", pc_path)

    if FLAGS.output_json:
        with open(FLAGS.output_json, 'w') as f:
            json.dump(dt_results, f)
        log.info("Results written to %s", FLAGS.output_json)


def _log_metrics_line(config, task, ckpt_path: str, log_path: str) -> None:
    """Evaluate one checkpoint and append its metrics as a JSON line to log_path."""
    metrics, _ = evaluate_checkpoint(
        config, task, ckpt_path, split=FLAGS.split, collect_json=False)
    metrics.pop('_coco_evaluator', None)
    record = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
              for k, v in metrics.items()}
    record['checkpoint'] = ckpt_path
    record['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    log.info("mAP=%.4f  F1@50=%.4f",
             record.get('mAP', 0), record.get('F1score50', 0))


def _list_checkpoints(watch_dir: str):
    """Return all checkpoint prefixes in watch_dir, in numeric step order."""
    state = tf.train.get_checkpoint_state(watch_dir)
    if state is None or not state.all_model_checkpoint_paths:
        return []
    paths = list(state.all_model_checkpoint_paths)
    # all_model_checkpoint_paths may carry relative names; resolve against watch_dir.
    resolved = []
    for p in paths:
        resolved.append(p if os.path.isabs(p) else os.path.join(watch_dir, os.path.basename(p)))
    return resolved


def _run_all(config, task):
    """Evaluate every existing checkpoint in --watch_dir once."""
    if not FLAGS.watch_dir:
        raise app.UsageError("--all requires --watch_dir.")
    log_path = os.path.join(FLAGS.watch_dir, 'eval_log.jsonl')
    ckpts = _list_checkpoints(FLAGS.watch_dir)
    if not ckpts:
        log.warning("No checkpoints found in %s", FLAGS.watch_dir)
        return
    log.info("Evaluating %d checkpoint(s) in %s. Results -> %s",
             len(ckpts), FLAGS.watch_dir, log_path)
    for ckpt in ckpts:
        log.info("Evaluating %s ...", ckpt)
        try:
            _log_metrics_line(config, task, ckpt, log_path)
        except Exception as e:                       # noqa: BLE001
            log.error("Eval failed for %s: %s", ckpt, e)


def _run_watch(config, task):
    """Poll --watch_dir and evaluate each new checkpoint as it appears."""
    if not FLAGS.watch_dir:
        raise app.UsageError("--watch requires --watch_dir.")
    log_path = os.path.join(FLAGS.watch_dir, 'eval_log.jsonl')
    seen = set()
    eval_count = 0
    log.info("Watching %s every %ds. Results -> %s",
             FLAGS.watch_dir, FLAGS.interval, log_path)

    while True:
        latest = tf.train.latest_checkpoint(FLAGS.watch_dir)
        if latest and latest not in seen:
            seen.add(latest)
            log.info("New checkpoint: %s — evaluating...", latest)
            try:
                _log_metrics_line(config, task, latest, log_path)
                eval_count += 1
            except Exception as e:                   # noqa: BLE001
                log.error("Eval failed for %s: %s", latest, e)

        if FLAGS.max_evals > 0 and eval_count >= FLAGS.max_evals:
            break
        time.sleep(FLAGS.interval)


def main(_):
    tf.config.run_functions_eagerly(False)

    from configs.yaml_loader import load_config
    from train.task import YoloV8Task

    config = load_config(FLAGS.config)

    # Activate the same precision policy the trainer used so a bfloat16-trained
    # checkpoint is evaluated on the bfloat16 compute path (must run BEFORE the
    # model is built; the global policy persists for every checkpoint in a loop).
    from common.runtime_setup import apply_eval_precision_policy
    apply_eval_precision_policy(config)

    task = YoloV8Task(config)

    if FLAGS.watch:
        _run_watch(config, task)
    elif FLAGS.all:
        _run_all(config, task)
    else:
        _run_single(config, task)


if __name__ == '__main__':
    app.run(main)
