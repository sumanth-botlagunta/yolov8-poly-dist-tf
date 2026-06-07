"""Tests for COCOEvaluator is_crowd / is_dontcare GT handling.

The annotation-building logic (eval/coco_metrics.py) does, per GT:
    - is_crowd & class in iscrowds_labels (ignore_iscrowds) → skip entirely.
    - is_dontcare (ignore_dontcare)                          → emit with iscrowd=1.
    - otherwise                                              → emit with iscrowd=0.

These tests inspect the accumulated GT annotations directly so they are deterministic
(independent of pycocotools mAP edge cases for empty GT sets).
"""

import unittest

import numpy as np
import tensorflow as tf

from eval.coco_metrics import COCOEvaluator


def _labels_with_flags(is_crowd, is_dontcare, classes):
    n = len(classes)
    boxes = np.tile(np.array([[0.1, 0.1, 0.5, 0.5]], np.float32), (n, 1))[None]
    return {
        "bbox":         tf.constant(boxes),                                  # [1, n, 4]
        "classes":      tf.constant([classes], dtype=tf.int64),
        "n_gt":         tf.constant([n], dtype=tf.int64),
        "is_crowd":     tf.constant([is_crowd], dtype=tf.bool),
        "is_dontcare":  tf.constant([is_dontcare], dtype=tf.bool),
    }


_EMPTY_PREDS = {
    "bbox":           tf.zeros([1, 1, 4], tf.float32),
    "classes":        tf.zeros([1, 1], tf.int64),
    "confidence":     tf.zeros([1, 1], tf.float32),
    "num_detections": tf.zeros([1], tf.int32),
}


class TestCrowdDontcare(unittest.TestCase):
    def _run(self, **kw):
        ev = COCOEvaluator(num_classes=7, image_size=(100, 100), **kw)
        labels = _labels_with_flags(
            is_crowd=[False, False, True],     # GT2 is a crowd region
            is_dontcare=[False, True, False],  # GT1 is dontcare
            classes=[0, 1, 6],                 # GT2 class 6 ∈ iscrowds_labels
        )
        ev.update(_EMPTY_PREDS, labels)
        return ev

    def test_crowd_class_gt_skipped(self):
        ev = self._run(ignore_iscrowds=True, ignore_dontcare=True, iscrowds_labels=[6])
        cats = [a["category_id"] for a in ev._gt_anns]
        self.assertIn(0, cats)
        self.assertIn(1, cats)
        self.assertNotIn(6, cats)          # crowd-class GT skipped
        self.assertEqual(len(ev._gt_anns), 2)

    def test_dontcare_marked_iscrowd(self):
        ev = self._run(ignore_iscrowds=True, ignore_dontcare=True, iscrowds_labels=[6])
        by_cat = {a["category_id"]: a["iscrowd"] for a in ev._gt_anns}
        self.assertEqual(by_cat[1], 1)     # dontcare → iscrowd=1
        self.assertEqual(by_cat[0], 0)     # normal → iscrowd=0

    def test_iscrowds_not_ignored_keeps_all_gt(self):
        # With ignore_iscrowds=False the crowd-class GT is kept (as a normal GT).
        ev = self._run(ignore_iscrowds=False, ignore_dontcare=True, iscrowds_labels=[6])
        cats = [a["category_id"] for a in ev._gt_anns]
        self.assertIn(6, cats)
        self.assertEqual(len(ev._gt_anns), 3)


if __name__ == "__main__":
    unittest.main()
