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

    def test_graph_mode_map_traces(self):
        """Regression: copy-paste must trace under Dataset.map(AUTOTUNE).

        The augmentation runs inside a tf.cond branch lambda (process_fn), where
        AutoGraph does NOT convert a Python `if` on a symbolic tensor. A prior bug
        used tf.shape(points)[0] (symbolic) in `if max_v == 0`, raising
        OperatorNotAllowedInGraphError at .map() trace time — i.e. at pipeline
        construction, on every shipped config. The eager-mode tests above could not
        catch it because EagerTensor.__eq__ evaluates numerically. This pins the
        real graph-mode path that the input pipeline uses (input_reader.py:171).
        """
        mod = CopyAndPasteModule(prob=1.0)
        fn = mod.process_fn(is_training=True)

        bg_ds = tf.data.Dataset.from_tensors(_bg(n=2))
        obj_ds = tf.data.Dataset.from_tensors(_obj())
        ds = tf.data.Dataset.zip((bg_ds, obj_ds))
        # tf.cond traces BOTH branches regardless of prob, so this .map() is where
        # the prior bug crashed during tracing.
        ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)

        out = next(iter(ds))
        self.assertEqual(tuple(out["image"].shape), (100, 100, 3))
        # Object pasted → exactly one extra GT row appended.
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), 3)
        self.assertEqual(int(out["groundtruth_polygons"].shape[0]), 3)


if __name__ == "__main__":
    unittest.main()
