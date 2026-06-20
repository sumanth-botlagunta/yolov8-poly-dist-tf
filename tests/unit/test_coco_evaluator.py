"""Tests for COCOEvaluator.

Validates:
    - Perfect predictions (bbox == GT) yield mAP50 = 1.0.
    - No detections yields mAP = 0.0 without crashing.
    - reset() clears all accumulated state.
    - update() accepts multiple batches; metrics are consistent.
"""

import unittest
import numpy as np
import tensorflow as tf

from eval.coco_metrics import COCOEvaluator


def _make_batch(B=2, n_gt=2, img_h=100, img_w=100, num_classes=3):
    """Synthetic batch: all detections match GT exactly."""
    # GT boxes: two instances per image, yxyx normalized
    gt_boxes = np.array([[[0.1, 0.1, 0.5, 0.5],
                           [0.6, 0.6, 0.9, 0.9]]] * B, dtype=np.float32)
    gt_classes = np.array([[0, 1]] * B, dtype=np.int64)
    n_gt_arr   = np.array([n_gt] * B, dtype=np.int64)

    labels = {
        'bbox':    tf.constant(gt_boxes),
        'classes': tf.constant(gt_classes),
        'n_gt':    tf.constant(n_gt_arr),
    }

    # Perfect predictions: identical boxes, high confidence
    preds = {
        'bbox':           tf.constant(gt_boxes),
        'classes':        tf.constant(gt_classes),
        'confidence':     tf.ones([B, n_gt], dtype=tf.float32),
        'num_detections': tf.constant([n_gt] * B, dtype=tf.int32),
    }
    return preds, labels


class TestCOCOEvaluator(unittest.TestCase):

    def test_perfect_predictions_map50_is_one(self):
        """Exact bbox matches should yield mAP50 = 1.0."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        metrics = ev.evaluate()
        self.assertAlmostEqual(metrics['mAP50'], 1.0, places=2)

    def test_no_detections_returns_zero(self):
        """Empty detection list should not crash and return mAP = 0.0."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        # Zero out detections
        preds['num_detections'] = tf.zeros([2], dtype=tf.int32)
        ev.update(preds, labels)
        metrics = ev.evaluate()
        self.assertAlmostEqual(metrics['mAP50'], 0.0, places=5)

    def test_no_detections_returns_all_seven_keys(self):
        """Both early-return branches return the full metric key set."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        preds['num_detections'] = tf.zeros([2], dtype=tf.int32)
        ev.update(preds, labels)
        metrics = ev.evaluate()
        for k in ('mAP', 'mAP50', 'AR100', 'F1score50',
                  'precision50', 'recall50', 'best_conf_thresh'):
            self.assertIn(k, metrics)

    def test_macro_means_consistent_when_a_class_has_no_gt(self):
        """F1score50, precision50/recall50, and the saved report's mean F1 must all be
        averaged over the SAME classes (those with a valid PR point). A class absent from
        the GT (here class 2: num_classes=3 but GT only uses 0/1) has no valid PR point and
        must be excluded from every macro mean — not counted as 0 in some and skipped in
        others (the pre-fix inconsistency)."""
        # _make_batch: num_classes=3, GT classes {0,1}, perfect detections -> class 2 has
        # no GT, so its precision is all -1 (no valid PR point).
        preds, labels = _make_batch(num_classes=3)
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        ev.update(preds, labels)
        m = ev.evaluate()

        # Classes 0/1 perfectly detected; class 2 (no GT) excluded from the means.
        self.assertAlmostEqual(m['F1score50'], 1.0, places=5)
        self.assertAlmostEqual(m['precision50'], m['F1score50'], places=6)
        self.assertAlmostEqual(m['recall50'],    m['F1score50'], places=6)

        # The saved report's mean F1 equals F1score50 (same denominator)...
        report = ev.metrics_tables()
        self.assertAlmostEqual(report['mean']['f1'], m['F1score50'], places=6)
        # ...but the undetected class is still LISTED (flagged valid=False).
        best = ev.per_category_best_f1()
        invalid = [b for b in best if not b.get('valid', True)]
        self.assertTrue(any(b['category'] == 2 for b in invalid))

    def test_reset_clears_state(self):
        """After reset(), evaluate() on empty state returns zeros."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        ev.reset()
        metrics = ev.evaluate()
        self.assertAlmostEqual(metrics['mAP50'], 0.0, places=5)

    def test_metrics_dict_has_required_keys(self):
        """evaluate() must return mAP, mAP50, AR100, F1score50."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        metrics = ev.evaluate()
        for key in ('mAP', 'mAP50', 'AR100', 'F1score50'):
            self.assertIn(key, metrics)

    def test_multiple_batches_accumulated(self):
        """Calling update() twice doubles the sample count; result stays valid."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        ev.update(preds, labels)
        metrics = ev.evaluate()
        self.assertGreaterEqual(metrics['mAP50'], 0.0)
        self.assertLessEqual(metrics['mAP50'],    1.0)

    def test_f1score50_between_zero_and_one(self):
        """F1score50 must be in [0, 1]."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        metrics = ev.evaluate()
        self.assertGreaterEqual(metrics['F1score50'], 0.0)
        self.assertLessEqual(metrics['F1score50'],    1.0)


if __name__ == '__main__':
    unittest.main()
