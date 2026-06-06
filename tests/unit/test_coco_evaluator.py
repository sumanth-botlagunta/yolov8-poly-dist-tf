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
