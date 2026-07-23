"""Score external predictions.json files against the test-split GT on the exact
F1score50 ruler — a head-to-head detector comparison with zero decode risk.

A `predictions.json` (the schema `utils/export/inference_saved_model.py` writes)
is a model's FINAL output: its own decode, NMS, and box order are already baked
in. Scoring such a file applies only OUR metric to a model's real detections, so
two models exported to this schema — e.g. the current model vs a legacy model —
can be compared apples-to-apples on one ruler, with none of the checkpoint-level
convention risks (box channel order, NMS scope) that block loading foreign
weights. This is the format-agnostic re-baseline: it answers "is legacy's F1
actually higher than ours on the SAME measurement?".

The tool:
  1. Loads the test-split GT via the project's PolygonDecoder (same boxes,
     classes, crowd, and dontcare the trainer's evaluator saw), keyed by the
     image basename — the same key the predictions JSON uses.
  2. For each predictions JSON: aligns detections to GT by basename, converts
     original-pixel xywh boxes to the normalized yxyx the evaluator expects, and
     runs them through the SAME COCOEvaluator (same crowd/dontcare/iscrowds
     policy from the config) -> F1score50 + per-class table.
  3. Prints a LOUD match-coverage report (matched images, GT images with no
     predictions, prediction keys with no GT) so a filename/split misalignment
     screams instead of silently producing a wrong number.

Correctness anchor: scoring the CURRENT model's own predictions.json must
reproduce its known F1score50 (~0.7271). If it does, the whole path (parse ->
align -> convert -> score) is proven, and a legacy JSON on the identical ruler is
trustworthy. If it doesn't, the coverage report and the mismatch tell you why
before any conclusion is drawn.

Requirements on each predictions JSON (both models must match):
  - records: {'image_id'|'file_name': <basename with ext>, 'category_id': int,
    'bbox': [x, y, w, h] in ORIGINAL image pixels, 'score': float}
  - generated on the SAME test split images (matching basenames), conf >= ~0.01
    so the F1 sweep has the full operating curve.

Usage:
    python -m utils.score_predictions \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --predictions /tmp/best_preds/best_predictions.json --names ours \
        --predictions /tmp/legacy/legacy_predictions.json  --names legacy \
        --output_json /tmp/headtohead.json
    python -m utils.score_predictions --self_test        # no config/data/JSON
"""

import json
import logging
import os

from absl import app, flags
import numpy as np

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config', None, 'Experiment YAML (for tfds name/dir + crowd policy).')
    flags.DEFINE_multi_string('predictions', None,
                              'predictions.json path. Repeat to score several side by side.')
    flags.DEFINE_multi_string('names', None,
                              'Optional display label per --predictions (same order).')
    flags.DEFINE_string('split', 'test', 'TFDS split for the GT (matches validation_data).')
    flags.DEFINE_string('output_json', None, 'Optional path to write the head-to-head results.')
    flags.DEFINE_bool('self_test', False, 'Run the built-in synthetic self-test and exit.')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate conversion — original-pixel xywh -> normalized yxyx.
# ---------------------------------------------------------------------------

def _preds_to_arrays(pred_list, H, W):
    """A per-image prediction list -> the batched dict COCOEvaluator.update wants.

    Boxes arrive as [x, y, w, h] in original image pixels; the evaluator expects
    yxyx normalized to [0, 1]. Normalizing pred and GT by the SAME (H, W) keeps
    IoU exact (a common per-image affine cancels in IoU).
    """
    if not pred_list:
        return {'bbox': np.zeros((1, 0, 4), np.float32),
                'classes': np.zeros((1, 0), np.int64),
                'confidence': np.zeros((1, 0), np.float32),
                'num_detections': np.array([0], np.int32)}
    boxes, classes, scores = [], [], []
    for p in pred_list:
        x, y, w, h = [float(v) for v in p['bbox']]
        x1, y1, x2, y2 = x, y, x + w, y + h
        boxes.append([y1 / H, x1 / W, y2 / H, x2 / W])
        classes.append(int(p['category_id']))
        scores.append(float(p['score']))
    return {'bbox': np.asarray(boxes, np.float32)[None],
            'classes': np.asarray(classes, np.int64)[None],
            'confidence': np.asarray(scores, np.float32)[None],
            'num_detections': np.array([len(boxes)], np.int32)}


def _gt_to_arrays(gt):
    """A per-image GT dict -> the batched dict COCOEvaluator.update wants."""
    boxes = np.asarray(gt['boxes'], np.float32)          # [M,4] yxyx normalized
    m = boxes.shape[0]
    return {'bbox': boxes[None],
            'classes': np.asarray(gt['classes'], np.int64)[None],
            'n_gt': np.array([m], np.int32),
            'is_crowd': np.asarray(gt['is_crowd'], bool)[None],
            'is_dontcare': np.asarray(gt['is_dontcare'], np.int64)[None]}


# ---------------------------------------------------------------------------
# Scoring core — TFDS-free, so the whole path is self-testable without data.
# ---------------------------------------------------------------------------

def score_predictions(preds_by_img, gt_by_img, num_classes,
                      ignore_dontcare, ignore_iscrowds, iscrowds_labels,
                      image_size=(672, 672)):
    """Score one predictions set against GT; return metrics + coverage.

    Args:
      preds_by_img: {basename: [ {category_id, bbox=[x,y,w,h] px, score}, ... ]}
      gt_by_img: {basename: {boxes[M,4] yxyx-norm, classes, is_crowd, is_dontcare,
        H, W}}
      num_classes / ignore_* / iscrowds_labels: the config's evaluator policy.

    Returns:
      dict with F1score50, the strict/dropped/maxDets audit, per-class rows, and
      the coverage counts.
    """
    from eval.coco_metrics import COCOEvaluator
    from utils.f1_measurement_audit import audit_evaluator

    ev = COCOEvaluator(
        num_classes=num_classes, image_size=image_size,
        ignore_dontcare=ignore_dontcare, ignore_iscrowds=ignore_iscrowds,
        iscrowds_labels=iscrowds_labels, find_best_score_thresh=True)

    matched, gt_without_preds = 0, 0
    for fname, gt in gt_by_img.items():
        plist = preds_by_img.get(fname)
        if plist:
            matched += 1
        else:
            gt_without_preds += 1
            plist = []
        ev.update(_preds_to_arrays(plist, gt['H'], gt['W']), _gt_to_arrays(gt))

    unmatched_pred_keys = [k for k in preds_by_img if k not in gt_by_img]

    metrics = ev.evaluate()
    audit = audit_evaluator(ev, reported_f1=metrics.get('F1score50'))

    # Per-class table (best-F1 operating point), joined with GT counts + names.
    gt_counts = ev._gt_counts()
    names = _class_names(num_classes)
    rows = []
    for r in ev.per_category_best_f1():   # wrapper: IoU=0.5, area='all', maxDets=10
        cat = int(r['category'])
        rows.append({
            'category': cat, 'name': names[cat] if cat < len(names) else str(cat),
            'n_gt': gt_counts.get(cat, {}).get('num_gt', 0),
            'f1': float(r['f1']), 'precision': float(r['precision']),
            'recall': float(r['recall']), 'best_conf': float(r['conf_threshold']),
            'valid': bool(r['valid']),
        })

    return {
        'f1score50': metrics.get('F1score50'),
        'mAP50': metrics.get('mAP50'),
        'audit': audit,
        'per_class': rows,
        'coverage': {
            'gt_images': len(gt_by_img),
            'pred_images': len(preds_by_img),
            'matched': matched,
            'gt_without_preds': gt_without_preds,
            'unmatched_pred_keys': unmatched_pred_keys,
        },
    }


def _class_names(num_classes):
    try:
        from configs.class_map import DETECTION_CLASSES
        return [str(DETECTION_CLASSES[i]) for i in range(num_classes)]
    except Exception:
        return [str(i) for i in range(num_classes)]


# ---------------------------------------------------------------------------
# Loaders — predictions JSON + test-split GT (real run only).
# ---------------------------------------------------------------------------

def load_predictions(path):
    """predictions.json (COCO-style list) -> {basename: [detection, ...]}."""
    with open(path) as f:
        records = json.load(f)
    by_img = {}
    for rec in records:
        key = rec.get('image_id', rec.get('file_name'))
        if key is None:
            raise ValueError(f"record has no image_id/file_name: {rec}")
        by_img.setdefault(os.path.basename(str(key)), []).append(rec)
    return by_img


def load_gt_from_tfds(config, split):
    """Decode the test split via PolygonDecoder -> {basename: gt-dict}.

    Uses the project decoder so boxes/classes/crowd/dontcare match the trainer's
    evaluator exactly; the image basename comes from image/filename (the key the
    predictions JSON uses).
    """
    import tensorflow_datasets as tfds
    from data_pipeline.tfds_decoders import PolygonDecoder

    val = config.task.validation_data
    decoder = PolygonDecoder(num_classes=config.task.num_classes)
    ds = tfds.load(val.tfds_name, split=split, data_dir=val.tfds_data_dir)

    gt_by_img = {}
    n_no_filename = 0
    for ex in ds:
        if 'image/filename' in ex:
            fname = os.path.basename(ex['image/filename'].numpy().decode('utf-8'))
        else:
            n_no_filename += 1
            fname = str(int(ex['image/id'].numpy()))
        dec = decoder.decode(ex)
        gt_by_img[fname] = {
            'boxes': dec['groundtruth_boxes'].numpy(),
            'classes': dec['groundtruth_classes'].numpy(),
            'is_crowd': dec['groundtruth_is_crowd'].numpy(),
            'is_dontcare': dec['groundtruth_dontcare'].numpy(),
            'H': int(dec['height'].numpy()), 'W': int(dec['width'].numpy()),
        }
    if n_no_filename:
        log.warning("%d GT records had no image/filename; keyed by numeric id — "
                    "predictions must key the same way.", n_no_filename)
    return gt_by_img


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def _print_one(name, res):
    cov = res['coverage']
    a = res['audit']
    print(f"\n=== {name} ===")
    print(f"  F1score50            : {res['f1score50']:.4f}")
    print(f"  mAP50                : {res['mAP50']:.4f}")
    print(f"  strict F1 (silent=0) : {a['grid'][10]['strict']:.4f}   "
          f"maxDets=100: {a['grid'][100]['dropped']:.4f}")
    print(f"  coverage: {cov['matched']}/{cov['gt_images']} GT images have preds; "
          f"{cov['gt_without_preds']} without; {len(cov['unmatched_pred_keys'])} "
          f"pred-keys unmatched to GT")
    if cov['matched'] == 0:
        print("  !! ZERO matched images — filename/split MISALIGNMENT. The score is meaningless.")
    elif cov['unmatched_pred_keys']:
        ex = ', '.join(cov['unmatched_pred_keys'][:5])
        print(f"  !! {len(cov['unmatched_pred_keys'])} prediction keys have no GT (e.g. {ex}) — "
              "check the image set matches the test split.")


def _print_headtohead(results):
    if len(results) < 2:
        return
    names = [n for n, _ in results]
    print("\n=== head-to-head F1score50 ===")
    base = results[0][1]['f1score50']
    for n, r in results:
        d = r['f1score50'] - base
        print(f"  {n:20s}  F1={r['f1score50']:.4f}  (Δ vs {names[0]}: {d:+.4f})")

    print("\n=== per-class F1 (each at its own best conf) ===")
    hdr = f"  {'id':>2} {'class':22s} {'n_gt':>6}  " + "  ".join(f"{n:>8s}" for n, _ in results)
    print(hdr)
    rows0 = results[0][1]['per_class']
    for i, r0 in enumerate(rows0):
        line = f"  {r0['category']:>2} {r0['name']:22s} {r0['n_gt']:>6}  "
        line += "  ".join(f"{res['per_class'][i]['f1']:>8.3f}" for _, res in results)
        print(line)


# ---------------------------------------------------------------------------
# Self-test — exercises parse/align/convert/score with synthetic data.
# ---------------------------------------------------------------------------

def _self_test():
    # One image (100x100), 3 classes:
    #   class 0: GT + exact-match detection      -> TP, F1 = 1.0
    #   class 1: GT, no detection                -> silent-with-GT (dropped by F1score50)
    #   class 2: no GT, one FP detection         -> excluded
    # Plus a stray prediction on an image with NO GT -> must be flagged unmatched.
    gt_by_img = {
        'img1.jpg': {
            'boxes': np.array([[0.1, 0.1, 0.3, 0.3],     # class 0 (yxyx norm)
                               [0.5, 0.5, 0.7, 0.7]],     # class 1
                              np.float32),
            'classes': np.array([0, 1], np.int64),
            'is_crowd': np.array([False, False]),
            'is_dontcare': np.array([0, 0], np.int64),
            'H': 100, 'W': 100,
        },
    }
    preds_by_img = {
        'img1.jpg': [
            {'category_id': 0, 'bbox': [10, 10, 20, 20], 'score': 0.9},  # matches GT0 exactly
            {'category_id': 2, 'bbox': [80, 80, 10, 10], 'score': 0.5},  # FP, no GT for class 2
        ],
        'ghost.jpg': [  # prediction on an image with no GT -> unmatched
            {'category_id': 0, 'bbox': [1, 1, 5, 5], 'score': 0.9},
        ],
    }
    res = score_predictions(
        preds_by_img, gt_by_img, num_classes=3,
        ignore_dontcare=True, ignore_iscrowds=False, iscrowds_labels=None)

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    a, cov = res['audit'], res['coverage']
    check('F1score50 == 1.0 (only class0 valid)', abs(res['f1score50'] - 1.0) < 1e-6)
    check('dropped@10 == 1.0', abs(a['grid'][10]['dropped'] - 1.0) < 1e-6)
    check('strict@10 == 0.5 (class1 silent -> 0)', abs(a['grid'][10]['strict'] - 0.5) < 1e-6)
    check('silent-with-GT == [1]', [c for c, _ in a['silent_with_gt']] == [1])
    check('matched == 1 image', cov['matched'] == 1)
    check('gt_without_preds == 0', cov['gt_without_preds'] == 0)
    check("unmatched pred key == ['ghost.jpg']", cov['unmatched_pred_keys'] == ['ghost.jpg'])
    # class0 detection must land as a TP -> per-class F1 = 1.0 (proves px->norm conversion).
    c0 = next(r for r in res['per_class'] if r['category'] == 0)
    check('class0 per-class F1 == 1.0 (px->norm conversion correct)', abs(c0['f1'] - 1.0) < 1e-6)

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def main(_):
    if FLAGS.self_test:
        raise SystemExit(0 if _self_test() else 1)

    if not FLAGS.config or not FLAGS.predictions:
        raise app.UsageError("--config and at least one --predictions are required "
                             "(or use --self_test).")

    names = FLAGS.names or []
    if names and len(names) != len(FLAGS.predictions):
        raise app.UsageError(f"--names count ({len(names)}) must match --predictions "
                             f"count ({len(FLAGS.predictions)}).")

    import tensorflow as tf
    tf.config.run_functions_eagerly(False)
    from configs.yaml_loader import load_config

    config = load_config(FLAGS.config)
    tcfg = config.task
    log.info("Loading GT for split '%s' from %s ...", FLAGS.split,
             tcfg.validation_data.tfds_name)
    gt_by_img = load_gt_from_tfds(config, FLAGS.split)
    log.info("GT loaded: %d images.", len(gt_by_img))

    results = []
    for i, ppath in enumerate(FLAGS.predictions):
        name = names[i] if i < len(names) else os.path.basename(ppath)
        preds_by_img = load_predictions(ppath)
        res = score_predictions(
            preds_by_img, gt_by_img, num_classes=tcfg.num_classes,
            ignore_dontcare=tcfg.ignore_dontcare, ignore_iscrowds=tcfg.ignore_iscrowds,
            iscrowds_labels=tcfg.iscrowds_labels,
            image_size=tuple(tcfg.model.input_size[:2]))
        _print_one(name, res)
        results.append((name, res))

    _print_headtohead(results)

    if FLAGS.output_json:
        with open(FLAGS.output_json, 'w') as f:
            json.dump({n: r for n, r in results}, f, indent=2)
        log.info("Head-to-head written to %s", FLAGS.output_json)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
