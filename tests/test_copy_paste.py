"""Tests for the Copy-Paste augmentation module.

Validates:
    - Output image shape matches the background image shape.
    - The pasted object's box (and polygon row) is appended to the GT annotations.
    - prob=0.0 produces unchanged output.
"""

import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.copy_paste import CopyAndPasteModule


def _bg(h=100, w=100, n=2):
    return {
        "image":  tf.constant(np.full((h, w, 3), 120, np.uint8)),
        "groundtruth_boxes": tf.constant(
            [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]][:n], dtype=tf.float32
        ),
        "groundtruth_classes":  tf.constant([0, 1][:n], dtype=tf.int64),
        "groundtruth_polygons": tf.fill([n, 8], -1.0),
        "groundtruth_is_crowd": tf.constant([False] * n),
        "groundtruth_area":     tf.ones([n], tf.float32),
        "groundtruth_dontcare": tf.zeros([n], tf.int64),
    }


def _obj(h=40, w=40):
    rgba = np.full((h, w, 4), 200, np.uint8)
    rgba[..., 3] = 255  # fully opaque alpha → object is pasted
    return {
        "image":     tf.constant(rgba),
        "orig_bbox": tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32),
        "label":     tf.constant(7, dtype=tf.int64),
        "points":    tf.constant([0.1, 0.1, 0.9, 0.1, 0.5, 0.9], dtype=tf.float32),
    }


class TestCopyAndPasteModule(unittest.TestCase):
    def test_output_shape_preserved(self):
        mod = CopyAndPasteModule(prob=1.0)
        out = mod.process_fn(is_training=True)(_bg(), _obj())
        self.assertEqual(tuple(out["image"].shape), (100, 100, 3))
        self.assertEqual(out["image"].dtype, tf.uint8)

    def test_box_appended(self):
        mod = CopyAndPasteModule(prob=1.0)
        bg = _bg(n=2)
        out = mod.process_fn(is_training=True)(bg, _obj())
        # One object pasted → exactly one extra box / class / polygon row.
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), 3)
        self.assertEqual(int(out["groundtruth_classes"].shape[0]), 3)
        self.assertEqual(int(out["groundtruth_polygons"].shape[0]), 3)
        # Appended class is the object's label.
        self.assertEqual(int(out["groundtruth_classes"][-1]), 7)

    def test_zero_prob_no_change(self):
        mod = CopyAndPasteModule(prob=0.0)
        bg = _bg(n=2)
        out = mod.process_fn(is_training=True)(bg, _obj())
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), 2)
        np.testing.assert_array_equal(out["image"].numpy(), bg["image"].numpy())


if __name__ == "__main__":
    unittest.main()
