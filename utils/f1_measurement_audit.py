"""F1score50 measurement audit: how much of the headline number is aggregation.

The reported F1score50 makes two aggregation choices that deflate the number for
a model that is silent on some classes or emits many detections per image. This
tool measures each contribution against a fixed detection set, so it separates
"the eval scored us low" from "the model is weak" without changing any model
behavior or retraining:

  1. Silent-class drop. F1score50 macro-averages only classes with a valid
     (>= 0) best-F1, silently EXCLUDING classes the model never scored a single
     TP on. A stricter reading scores those GT-bearing silent classes as 0.0 and
     includes them in the mean -- a fuller number that is not flattering by
     omission. The difference (strict - dropped) is exactly the drop artifact.

  2. maxDets cap. F1score50 is computed at maxDets=10: only the 10
     highest-scored detections per image count toward recall. Crowded indoor
     images can carry more true objects than that, capping recall. The same
     accumulated eval already holds the best-F1 at the maxDets=100 slot, so
     recomputing there lifts the cap with no extra inference.

The tool runs ONE inference pass over the split (reusing utils.eval's
evaluate_checkpoint), then re-reads the single accumulated COCOevalCustom under
each aggregation. It writes nothing the trainer reads and mutates no model.

NOT covered here: the detection-generator top-1 argmax mask (multi-label NMS).
That changes the detections themselves -- a decode-path change, not a
re-aggregation of a fixed detection set -- so it cannot be measured post-hoc from
one inference pass and is out of scope for this read-only audit.

Usage:
    python -m utils.f1_measurement_audit --config <cfg> --checkpoint /run/ckpt-N
    python -m utils.f1_measurement_audit --config <cfg> --checkpoint /run/ckpt-N \
        --split val --limit_batches 0 --output_json /tmp/f1_audit.json
    python -m utils.f1_measurement_audit --self_test    # no model / data needed
"""

import json
import logging

from absl import app, flags
import numpy as np

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',     None, 'Path to experiment YAML config.')
    flags.DEFINE_string('checkpoint', None, 'Checkpoint path prefix to audit.')
    flags.DEFINE_string('split',      'val', "Eval split: 'val' or 'test'.")
    flags.DEFINE_integer('limit_batches', 0, 'Stop after this many batches '
                         '(0 = full split). A sampled probe, not the full number.')
    flags.DEFINE_string('output_json', None, 'Optional path to write audit results JSON.')
    flags.DEFINE_bool('self_test', False, 'Run the built-in synthetic self-test '
                      '(no config / checkpoint / dataset needed) and exit.')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)

# The F1score50 headline operating point.
_IOU_THR = 0.5
_AREA = 'all'


def _macro_best_f1(ev, gt_counts, max_dets, strict):
    """Macro-average per-category best-F1 at (IoU=0.5, area='all', max_dets).

    Args:
      ev: An accumulated COCOevalCustom (best_fiscore filled by accumulate()).
      gt_counts: {category_id: {'num_gt': int, ...}} from the evaluator, used to
        tell a GT-bearing silent class (counted 0 under strict) apart from a
        class with no GT at all (always excluded).
      max_dets: Detection budget slot to read (must be in ev.params.maxDets).
      strict: When True, a class that HAS GT but has no valid best-F1 (the model
        never scored a TP on it) is included as 0.0. When False, it is dropped --
        the current F1score50 convention.

    Returns:
      The macro-averaged best-F1 as a float (0.0 if no class contributes).
    """
    rows = ev.per_category_best_f1(iouThr=_IOU_THR, areaRng=_AREA, maxDets=max_dets)
    vals = []
    for r in rows:
        cat = int(r['category'])
        has_gt = gt_counts.get(cat, {}).get('num_gt', 0) > 0
        if r['valid']:
            vals.append(float(r['f1']))
        elif strict and has_gt:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def _silent_with_gt(ev, gt_counts, max_dets):
    """Return category ids that have GT but no valid best-F1 at this max_dets.

    These are the classes the headline F1score50 silently drops and that a strict
    reading scores as 0.0. Sorted by descending GT count (biggest omissions first).
    """
    rows = ev.per_category_best_f1(iouThr=_IOU_THR, areaRng=_AREA, maxDets=max_dets)
    out = []
    for r in rows:
        cat = int(r['category'])
        n_gt = gt_counts.get(cat, {}).get('num_gt', 0)
        if (not r['valid']) and n_gt > 0:
            out.append((cat, n_gt))
    out.sort(key=lambda x: -x[1])
    return out


def audit_evaluator(coco_ev, reported_f1=None):
    """Compute the measurement audit from an already-evaluated COCOEvaluator.

    Args:
      coco_ev: A COCOEvaluator whose evaluate() has run (its internal _ev holds
        the accumulated COCOevalCustom).
      reported_f1: The F1score50 the pipeline reported for this run, for a sanity
        cross-check against the recomputed dropped@10 value.

    Returns:
      A results dict (see the printed report for the fields).
    """
    ev = getattr(coco_ev, '_ev', None)
    if ev is None:
        raise RuntimeError(
            "COCOEvaluator has no accumulated eval (split had no GT/detections?).")
    gt_counts = coco_ev._gt_counts()

    grid = {}
    for md in (10, 100):
        grid[md] = {
            'dropped': _macro_best_f1(ev, gt_counts, md, strict=False),
            'strict':  _macro_best_f1(ev, gt_counts, md, strict=True),
        }
    silent = _silent_with_gt(ev, gt_counts, 10)

    n_classes_with_gt = sum(1 for c in gt_counts.values() if c.get('num_gt', 0) > 0)

    return {
        'reported_f1score50':   (float(reported_f1)
                                 if reported_f1 is not None else None),
        'recomputed_dropped_10': grid[10]['dropped'],
        'grid':                 grid,
        'silent_with_gt':       silent,             # [(cat_id, n_gt), ...]
        'n_classes_with_gt':    n_classes_with_gt,
        'drop_artifact_pts':    grid[10]['dropped'] - grid[10]['strict'],
        'maxdets_artifact_pts': grid[100]['dropped'] - grid[10]['dropped'],
    }


def _class_name(cat_id):
    try:
        from configs.class_map import DETECTION_CLASSES
        return str(DETECTION_CLASSES[cat_id])
    except Exception:
        return str(cat_id)


def _print_report(res):
    g = res['grid']
    print("\n=== F1score50 measurement audit ===")
    if res['reported_f1score50'] is not None:
        rep, rec = res['reported_f1score50'], res['recomputed_dropped_10']
        flag = '' if abs(rep - rec) < 1e-4 else '  <-- DIVERGES, investigate'
        print(f"  reported F1score50            : {rep:.4f}")
        print(f"  recomputed (dropped, maxDets=10): {rec:.4f}{flag}")
    print(f"  classes with GT in split       : {res['n_classes_with_gt']}")
    print("\n  macro best-F1 by aggregation:")
    print(f"    {'':22s}  maxDets=10   maxDets=100")
    print(f"    {'silent-dropped':22s}  {g[10]['dropped']:.4f}       {g[100]['dropped']:.4f}")
    print(f"    {'strict (silent = 0)':22s}  {g[10]['strict']:.4f}       {g[100]['strict']:.4f}")

    print("\n  artifact contributions (points of F1):")
    print(f"    silent-class drop  (dropped10 - strict10) : {res['drop_artifact_pts']:+.4f}")
    print(f"    maxDets cap        (dropped100 - dropped10): {res['maxdets_artifact_pts']:+.4f}")

    silent = res['silent_with_gt']
    if silent:
        print("\n  silent-with-GT classes dropped by the headline metric "
              "(counted as 0 under strict):")
        for cat, n_gt in silent:
            print(f"    [{cat:2d}] {_class_name(cat):20s}  n_gt={n_gt}")
    else:
        print("\n  no GT-bearing class is fully silent at maxDets=10 -> the "
              "silent-drop artifact is 0 at this checkpoint.")
    print()


def run_audit(config, task, ckpt_path, split, limit_batches):
    """Run one inference pass and return the audit results dict."""
    from utils.eval import evaluate_checkpoint
    metrics, _ = evaluate_checkpoint(
        config, task, ckpt_path, split=split,
        collect_json=False, limit_batches=limit_batches)
    coco_ev = metrics.pop('_coco_evaluator')
    res = audit_evaluator(coco_ev, reported_f1=metrics.get('F1score50'))
    res['checkpoint'] = ckpt_path
    res['split'] = split
    return res


# ---------------------------------------------------------------------------
# Self-test: exercises the real aggregation path on a synthetic evaluator, so
# the strict/dropped/maxDets logic is verified without a model, checkpoint, or
# TFDS. Deterministic.
# ---------------------------------------------------------------------------

def _self_test():
    from eval.coco_metrics import COCOEvaluator

    # One image, 3 classes:
    #   class 0: GT box + an exact-match detection      -> TP, valid F1 = 1.0
    #   class 1: GT box, NO detection                   -> silent-with-GT (drop)
    #   class 2: NO GT, one false-positive detection    -> no GT, excluded
    predictions = {
        'bbox':           np.array([[[0.1, 0.1, 0.3, 0.3],    # class 0 (matches GT0)
                                      [0.8, 0.8, 0.9, 0.9]]],  # class 2 FP
                                    dtype=np.float32),
        'classes':        np.array([[0, 2]], dtype=np.int64),
        'confidence':     np.array([[0.9, 0.5]], dtype=np.float32),
        'num_detections': np.array([2], dtype=np.int32),
    }
    groundtruths = {
        'bbox':    np.array([[[0.1, 0.1, 0.3, 0.3],     # class 0
                              [0.5, 0.5, 0.7, 0.7]]],    # class 1
                            dtype=np.float32),
        'classes': np.array([[0, 1]], dtype=np.int64),
        'n_gt':    np.array([2], dtype=np.int32),
    }

    ev = COCOEvaluator(num_classes=3, image_size=(672, 672))
    ev.update(predictions, groundtruths)
    metrics = ev.evaluate()
    res = audit_evaluator(ev, reported_f1=metrics['F1score50'])

    ok = True

    def check(name, cond):
        nonlocal ok
        status = 'ok ' if cond else 'FAIL'
        print(f"  [{status}] {name}")
        ok = ok and cond

    g = res['grid']
    # class0 alone is valid -> dropped mean = 1.0; matches the pipeline F1score50.
    check('reported F1score50 == 1.0', abs(metrics['F1score50'] - 1.0) < 1e-6)
    check('dropped@10 == reported (sanity)',
          abs(res['recomputed_dropped_10'] - metrics['F1score50']) < 1e-6)
    check('dropped@10 == 1.0 (only class0 counts)', abs(g[10]['dropped'] - 1.0) < 1e-6)
    # strict adds class1 (silent, has GT) as 0.0 -> mean of {1.0, 0.0} = 0.5.
    check('strict@10 == 0.5 (class0=1.0, class1=0.0)', abs(g[10]['strict'] - 0.5) < 1e-6)
    check('strict < dropped (drop artifact positive)', g[10]['strict'] < g[10]['dropped'])
    check('drop artifact == +0.5', abs(res['drop_artifact_pts'] - 0.5) < 1e-6)
    # class1 is the only silent-with-GT class; class2 (no GT) must NOT appear.
    silent_ids = [c for c, _ in res['silent_with_gt']]
    check('silent-with-GT == [1] (class2 no-GT excluded)', silent_ids == [1])
    check('classes-with-GT == 2', res['n_classes_with_gt'] == 2)
    # maxDets=100 slot is readable and matches @10 here (< 10 detections).
    check('maxDets=100 slot readable, dropped == 1.0', abs(g[100]['dropped'] - 1.0) < 1e-6)

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def main(_):
    if FLAGS.self_test:
        raise SystemExit(0 if _self_test() else 1)

    if not FLAGS.config or not FLAGS.checkpoint:
        raise app.UsageError("--config and --checkpoint are required "
                             "(or use --self_test).")

    import tensorflow as tf
    tf.config.run_functions_eagerly(False)

    from configs.yaml_loader import load_config
    from common.runtime_setup import apply_eval_precision_policy
    from train.task import YoloV8Task

    config = load_config(FLAGS.config)
    apply_eval_precision_policy(config)   # Before any model is built.
    task = YoloV8Task(config)

    res = run_audit(config, task, FLAGS.checkpoint, FLAGS.split, FLAGS.limit_batches)
    _print_report(res)

    if FLAGS.output_json:
        with open(FLAGS.output_json, 'w') as f:
            json.dump(res, f, indent=2)
        log.info("Audit results written to %s", FLAGS.output_json)


if __name__ == '__main__':
    app.run(main)
