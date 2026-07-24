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
    # Both fixture boxes live in the UPPER 40% of the frame: the paste band is
    # y >= H*(1 - height_limit) = 0.4H, so tests that assert a paste happened
    # can never be skipped by the occlusion gate (see the gate tests below).
    return {
        "image":  tf.constant(np.full((h, w, 3), 120, np.uint8)),
        "groundtruth_boxes": tf.constant(
            [[0.1, 0.1, 0.3, 0.3], [0.1, 0.5, 0.3, 0.7]][:n], dtype=tf.float32
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
        """NEW semantics: copy-paste ALWAYS pastes (no accept/reject gate).

        min_height/min_width now shape the ADAPTIVE ratio bounds instead of
        skipping the paste outright:
            max_ratio = min(orig_h·height_limit/obj_h, orig_w/obj_w, max_resize_ratio)
            min_ratio = max(min_height/obj_h, min_width/obj_w, min_resize_ratio)
            min_ratio = min(min_ratio, max_ratio)
        When the floor is feasible under max_resize_ratio, the pasted object
        meets min_height/min_width at full resolution. When max_ratio caps
        below the floor, the realized ratio equals max_ratio (falls short of
        the floor) but the paste still happens — every case below appends a
        GT row (n+1), never skips.
        """
        # Case 1: floor is feasible (max_resize_ratio is generous) → the
        # pasted object's full-resolution size meets the min_height/min_width
        # floor, and the row is appended.
        mod = CopyAndPasteModule(prob=1.0, min_height=60, min_width=60,
                                 min_resize_ratio=0.2, max_resize_ratio=5.0,
                                 height_limit=0.6)
        out = mod.process_fn(is_training=True)(_bg(h=200, w=200, n=2), _obj(h=40, w=40))
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), 3)
        box = out["groundtruth_boxes"][-1].numpy()
        h_px = (box[2] - box[0]) * 200
        w_px = (box[3] - box[1]) * 200
        self.assertGreaterEqual(h_px, 60 - 1.0)
        self.assertGreaterEqual(w_px, 60 - 1.0)

        # Case 2: max_ratio caps BELOW the floor (min_resize_ratio ==
        # max_resize_ratio == 1.0 forces a deterministic draw == max_ratio).
        # max_ratio = min(100*0.6/80, 100/120, 1.0) = 0.75 → below the
        # 60×100 floor, so the pasted size falls short — but it still pastes.
        mod2 = CopyAndPasteModule(prob=1.0, min_height=60, min_width=100,
                                  min_resize_ratio=1.0, max_resize_ratio=1.0,
                                  height_limit=0.6)
        out2 = mod2.process_fn(is_training=True)(_bg(n=2), _obj(h=80, w=120))
        self.assertEqual(int(out2["groundtruth_boxes"].shape[0]), 3)
        box2 = out2["groundtruth_boxes"][-1].numpy()
        h_px2 = (box2[2] - box2[0]) * 100
        self.assertAlmostEqual(h_px2, 80 * 0.75, delta=1.0)
        self.assertLessEqual(h_px2, 60 + 1e-3)  # at (not above) the floor, by construction

        # Case 3 (formerly "skipped on a pre-resized background carrying
        # original dims"): the full-res gate is now folded into the ratio
        # bounds, not a separate reject path — the paste still happens.
        bg = _bg(h=100, w=100)
        bg["height"] = tf.constant(400, tf.int32)
        bg["width"] = tf.constant(400, tf.int32)
        out3 = mod.process_fn(is_training=True)(bg, _obj(h=40, w=40))
        self.assertEqual(int(out3["groundtruth_boxes"].shape[0]), 3)

    def test_placement_band_lower_region_and_contained(self):
        """The pasted object's top edge lands in the LOWER band of the frame.

        off_h ~ U(0.1, 0.9)·(off_h_max − off_h_min) + off_h_min, with
        off_h_min = H·(1 − height_limit). Draw many pastes with a small
        object (so the resize/containment clipping never binds) and assert
        every pasted box's top edge respects the U(0.1, 0.9) floor and stays
        fully inside the canvas.
        """
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0,
                                 min_resize_ratio=0.1, max_resize_ratio=0.3,
                                 height_limit=0.6)
        fn = mod.process_fn(is_training=True)
        H = 100
        for _ in range(30):
            out = fn(_bg(h=H, w=H, n=2), _obj(h=10, w=10))
            box = out["groundtruth_boxes"][-1].numpy()
            ymin_px = box[0] * H
            # off_h_min = 0.4*H; the U(0.1, 0.9) interpolation's own lower
            # bound is off_h_min + 0.1*(off_h_max - off_h_min) >= off_h_min.
            self.assertGreaterEqual(ymin_px, 0.4 * H - 1.0)
            # Fully inside the canvas.
            self.assertGreaterEqual(box[0], -1e-6)
            self.assertLessEqual(box[2], 1.0 + 1e-6)
            self.assertGreaterEqual(box[1], -1e-6)
            self.assertLessEqual(box[3], 1.0 + 1e-6)

    def test_occlusion_gate_skips_paste_covering_existing_gt(self):
        """A paste that would cover > max_occlusion_frac of an existing GT box
        is skipped entirely (image and annotations unchanged).

        Deterministic setup: obj 60x100 on a 100x100 bg with resize ratio
        pinned to 1.0 and height_limit 0.6 forces off_h_min == off_h_max == 40
        and off_w == 0, so the paste box is exactly [0.4, 0.0, 1.0, 1.0]. The
        existing box [0.5, 0.2, 0.9, 0.6] is fully inside it (occlusion 1.0).
        """
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0,
                                 min_resize_ratio=1.0, max_resize_ratio=1.0,
                                 height_limit=0.6)
        bg = _bg(h=100, w=100, n=1)
        bg["groundtruth_boxes"] = tf.constant([[0.5, 0.2, 0.9, 0.6]], tf.float32)
        out = mod.process_fn(is_training=True)(bg, _obj(h=60, w=100))
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), 1)
        self.assertEqual(int(out["groundtruth_polygons"].shape[0]), 1)
        np.testing.assert_array_equal(out["image"].numpy(), bg["image"].numpy())

    def test_occlusion_gate_allows_paste_clear_of_existing_gt(self):
        """Same deterministic placement, but the existing GT box lies above the
        paste region (zero overlap) -> the paste proceeds and appends its row."""
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0,
                                 min_resize_ratio=1.0, max_resize_ratio=1.0,
                                 height_limit=0.6)
        bg = _bg(h=100, w=100, n=1)
        bg["groundtruth_boxes"] = tf.constant([[0.0, 0.0, 0.35, 0.3]], tf.float32)
        out = mod.process_fn(is_training=True)(bg, _obj(h=60, w=100))
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), 2)
        # Appended box is the full paste region.
        np.testing.assert_allclose(
            out["groundtruth_boxes"][-1].numpy(), [0.4, 0.0, 1.0, 1.0], atol=1e-5
        )

    def test_occlusion_gate_none_disables_skip(self):
        """max_occlusion_frac=None restores the unconditional-paste behavior."""
        mod = CopyAndPasteModule(prob=1.0, min_height=0, min_width=0,
                                 min_resize_ratio=1.0, max_resize_ratio=1.0,
                                 height_limit=0.6, max_occlusion_frac=None)
        bg = _bg(h=100, w=100, n=1)
        bg["groundtruth_boxes"] = tf.constant([[0.5, 0.2, 0.9, 0.6]], tf.float32)
        out = mod.process_fn(is_training=True)(bg, _obj(h=60, w=100))
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), 2)

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
