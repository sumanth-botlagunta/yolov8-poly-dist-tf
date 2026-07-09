"""Tests for the Copy-Paste augmentation module.

Validates:
    - Output image shape matches the background image shape.
    - The pasted object's box (and polygon row) is appended to the GT annotations.
    - prob=0.0 produces unchanged output.
    - The min_height/min_width gate skips pastes whose full-resolution size
      (obj_dims × resize_ratio) is below the minimum.

Fixtures use small synthetic objects, so tests not about the gate construct the
module with min_height=0, min_width=0 to disable it.
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
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0)
        out = mod.process_fn(is_training=True)(_bg(), _obj())
        self.assertEqual(tuple(out["image"].shape), (100, 100, 3))
        self.assertEqual(out["image"].dtype, tf.uint8)

    def test_box_appended(self):
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0)
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

    def test_resolution_correction_preserves_relative_size(self):
        """Compositing on a pre-resized background must yield the same RELATIVE
        object size as compositing on the original-resolution background.

        The pipeline pre-resizes backgrounds to the model input size before
        copy-paste; _copy_and_paste scales the object by (current/original) per
        axis using the bg's 'height'/'width' fields. With a fixed resize ratio
        (min==max==1.0) the appended box's normalized height/width must match
        the unresized case to within 1px rounding.
        """
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0, min_resize_ratio=1.0, max_resize_ratio=1.0)
        fn = mod.process_fn(is_training=True)

        # Case A: original 200×400 background (no height/width fields → corr=1).
        out_a = fn(_bg(h=200, w=400), _obj(h=40, w=40))
        box_a = out_a["groundtruth_boxes"][-1].numpy()
        h_a, w_a = box_a[2] - box_a[0], box_a[3] - box_a[1]
        self.assertAlmostEqual(h_a, 40 / 200, places=5)
        self.assertAlmostEqual(w_a, 40 / 400, places=5)

        # Case B: same background pre-resized to 100×100, original dims recorded.
        bg_b = _bg(h=100, w=100)
        bg_b["height"] = tf.constant(200, tf.int32)
        bg_b["width"] = tf.constant(400, tf.int32)
        out_b = fn(bg_b, _obj(h=40, w=40))
        box_b = out_b["groundtruth_boxes"][-1].numpy()
        h_b, w_b = box_b[2] - box_b[0], box_b[3] - box_b[1]
        # corr = (100/200, 100/400) → pasted 20×10 px on 100×100 → same fractions.
        self.assertAlmostEqual(h_b, h_a, delta=1.0 / 100)
        self.assertAlmostEqual(w_b, w_a, delta=1.0 / 100)

    def test_no_height_fields_is_backward_compatible(self):
        """Without 'height'/'width' fields the correction must be exactly 1."""
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0, min_resize_ratio=1.0, max_resize_ratio=1.0)
        out = mod.process_fn(is_training=True)(_bg(h=100, w=100), _obj(h=40, w=40))
        box = out["groundtruth_boxes"][-1].numpy()
        self.assertAlmostEqual(box[2] - box[0], 0.4, places=5)
        self.assertAlmostEqual(box[3] - box[1], 0.4, places=5)

    def test_min_size_gate_skips_small_pastes(self):
        """Pastes below min_height×min_width at full resolution are SKIPPED.

        The gate compares obj_dims × resize_ratio (the full-resolution pasted
        size) against the minimum. A too-small object leaves the background
        unchanged (no extra GT row); a large-enough one is pasted. Guards that
        min_height/min_width are enforced, not just stored.
        """
        # 40×40 object at ratio 1.0 < 60×100 minimum → skipped.
        mod = CopyAndPasteModule(prob=1.0, min_height=60, min_width=100,
                                 min_resize_ratio=1.0, max_resize_ratio=1.0)
        out = mod.process_fn(is_training=True)(_bg(n=2), _obj(h=40, w=40))
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), 2)
        self.assertEqual(int(out["groundtruth_polygons"].shape[0]), 2)

        # 80×120 object at ratio 1.0 >= 60×100 → pasted.
        out2 = mod.process_fn(is_training=True)(_bg(n=2), _obj(h=80, w=120))
        self.assertEqual(int(out2["groundtruth_boxes"].shape[0]), 3)

        # Gate uses the FULL-RES size (obj_dims × ratio), independent of the
        # background's pre-resize correction: same skip decision on a
        # pre-resized background carrying original dims.
        bg = _bg(h=100, w=100)
        bg["height"] = tf.constant(400, tf.int32)
        bg["width"] = tf.constant(400, tf.int32)
        out3 = mod.process_fn(is_training=True)(bg, _obj(h=40, w=40))
        self.assertEqual(int(out3["groundtruth_boxes"].shape[0]), 2)

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
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0)
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
