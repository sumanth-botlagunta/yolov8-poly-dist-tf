"""Tests for polygon format conversion (_preprocess_polygons_v2).

The PolyYOLO target is the interleaved radial format
``[dist, angle, conf] × (360/angle_step)``:

    - PolyYOLO output shape is [N, 360/angle_step * 3].
    - dist is the radial distance from the box center to the bin's vertex.
    - angle is the sub-bin offset (vertex_angle - bin_start)/angle_step in [0, 1)
      on occupied bins, 0.0 on absent bins.
    - conf is 1.0 for bins that received a valid vertex and 0.0 for absent bins;
      absent bins also carry dist == 0.0 (so the dist head learns to collapse them).
"""

import math
import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.yolo_parser import V8ParserExtended


def _make_parser() -> V8ParserExtended:
    return V8ParserExtended(
        output_size=[64, 64],
        expanded_strides={"3": 8, "4": 16, "5": 32},
        levels=["3", "4", "5"],
        angle_step=15,
    )


# One box covering the whole image → center at (0.5, 0.5).
# Two vertices: one due-east (bin 0) and one due-south (bin 6), both at radius 0.5.
_BOX = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)        # yxyx normalized
_POLY = tf.constant([[1.0, 0.5, 0.5, 1.0, -1.0, -1.0, -1.0, -1.0]], dtype=tf.float32)


class TestPreprocessPolygonsV2(unittest.TestCase):
    def setUp(self):
        self.parser = _make_parser()
        out = self.parser._preprocess_polygons_v2(_BOX, _POLY, angle_step=15)
        self.out = out.numpy()
        self.dist = self.out[0, 0::3]   # [24]
        self.angle = self.out[0, 1::3]  # [24] sub-bin offset in [0,1)
        self.conf = self.out[0, 2::3]   # [24]

    def test_output_shape(self):
        """24 bins × 3 channels = 72 values per instance."""
        self.assertEqual(self.out.shape, (1, 72))

    def test_conf_binary(self):
        """conf is strictly 0/1; exactly the two occupied bins (0 and 6) are 1."""
        self.assertTrue(set(np.unique(self.conf)).issubset({0.0, 1.0}))
        self.assertEqual(self.conf[0], 1.0)
        self.assertEqual(self.conf[6], 1.0)
        self.assertEqual(int(self.conf.sum()), 2)

    def test_radial_distance_and_absent_bins(self):
        """Occupied bins carry radius 0.5; absent bins carry dist 0 (and conf 0)."""
        self.assertAlmostEqual(self.dist[0], 0.5, places=5)
        self.assertAlmostEqual(self.dist[6], 0.5, places=5)
        absent = np.ones(24, dtype=bool)
        absent[[0, 6]] = False
        np.testing.assert_array_equal(self.dist[absent], 0.0)
        np.testing.assert_array_equal(self.conf[absent], 0.0)
        # The two vertices sit exactly on bin starts (0°, 90°), so their sub-bin
        # offset is 0; every angle entry is therefore 0 here.
        self.assertAlmostEqual(self.angle[0], 0.0, places=5)
        self.assertAlmostEqual(self.angle[6], 0.0, places=5)
        np.testing.assert_array_equal(self.angle[absent], 0.0)


class TestSubBinAngleOffset(unittest.TestCase):
    """A vertex not on a bin boundary must produce its fractional offset."""

    def test_offset_is_fractional_position_in_bin(self):
        parser = _make_parser()
        box = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)  # centre (0.5,0.5)
        # One vertex at 7.5° (the middle of bin 0) at radius 0.4.
        r, ang = 0.4, math.radians(7.5)
        x = 0.5 + r * math.cos(ang)
        y = 0.5 + r * math.sin(ang)
        poly = tf.constant([[x, y, -1.0, -1.0]], dtype=tf.float32)
        out = parser._preprocess_polygons_v2(box, poly, angle_step=15).numpy()
        dist, angle, conf = out[0, 0::3], out[0, 1::3], out[0, 2::3]
        self.assertEqual(int(conf.sum()), 1)
        self.assertEqual(conf[0], 1.0)                 # bin 0
        self.assertAlmostEqual(dist[0], r, places=4)   # radial distance preserved
        self.assertAlmostEqual(angle[0], 0.5, places=4)  # 7.5° / 15° = 0.5


class TestResamplePolygons(unittest.TestCase):
    """resample_polygons shrinks polygon width while preserving the radial target."""

    def _circle_flat(self, cx, cy, r, n, pad_to):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1).reshape(-1).astype(np.float32)
        return np.concatenate([pts, -np.ones(pad_to - pts.size, np.float32)])

    def test_short_polygon_preserved_exactly(self):
        from data_pipeline.augmentations import resample_polygons
        parser = _make_parser()
        polys = tf.constant([self._circle_flat(0.5, 0.5, 0.3, 8, 200)])  # 8 verts, padded
        boxes = tf.constant([[0.2, 0.2, 0.8, 0.8]], tf.float32)
        orig = parser._preprocess_polygons_v2(boxes, polys, 15).numpy()
        red = resample_polygons(polys, 64)
        self.assertEqual(int(red.shape[1]), 128)   # 2 * 64
        out = parser._preprocess_polygons_v2(boxes, red, 15).numpy()
        # <= K vertices → radial target identical (dist + conf).
        np.testing.assert_allclose(out[:, 0::3], orig[:, 0::3], atol=1e-5)
        np.testing.assert_array_equal(out[:, 2::3], orig[:, 2::3])

    def test_long_polygon_radial_target_close(self):
        from data_pipeline.augmentations import resample_polygons
        parser = _make_parser()
        polys = tf.constant([self._circle_flat(0.5, 0.5, 0.3, 2000, 5469 * 2)])
        boxes = tf.constant([[0.2, 0.2, 0.8, 0.8]], tf.float32)
        orig = parser._preprocess_polygons_v2(boxes, polys, 15).numpy()
        out = parser._preprocess_polygons_v2(boxes, resample_polygons(polys, 128), 15).numpy()
        # downsampled but per-bin distances stay within sampling tolerance.
        np.testing.assert_allclose(out[:, 0::3], orig[:, 0::3], atol=1e-3)

    def test_empty_polygon_is_all_minus_one(self):
        from data_pipeline.augmentations import resample_polygons
        polys = tf.fill([1, 200], -1.0)
        red = resample_polygons(polys, 32).numpy()
        self.assertEqual(red.shape, (1, 64))
        np.testing.assert_array_equal(red, -1.0)


class TestEmptyPolygonAngleTarget(unittest.TestCase):
    """A box with NO valid vertices must produce all-zero polygon targets.

    The sub-bin offset is gated on conf (per-bin validity), so empty polygons get
    angle == 0 everywhere — no spurious offset target drives polygon_angle_loss on
    polygon-less objects (the angle/dist losses are masked by conf anyway).
    """

    def test_no_vertices_targets_all_zero(self):
        parser = _make_parser()
        box = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)
        # All vertices invalid (-1 padded).
        empty_poly = tf.constant([[-1.0] * 8], dtype=tf.float32)
        out = parser._preprocess_polygons_v2(box, empty_poly, angle_step=15).numpy()
        angle = out[0, 1::3]
        conf  = out[0, 2::3]
        dist  = out[0, 0::3]
        np.testing.assert_array_equal(angle, 0.0)
        np.testing.assert_array_equal(conf, 0.0)
        np.testing.assert_array_equal(dist, 0.0)

    def test_present_polygon_keeps_validity(self):
        """A present vertex must still set conf=1 even when its sub-bin offset is 0."""
        parser = _make_parser()
        out = parser._preprocess_polygons_v2(_BOX, _POLY, angle_step=15).numpy()
        self.assertEqual(int(out[0, 2::3].sum()), 2)   # two occupied bins (0 and 6)


if __name__ == "__main__":
    unittest.main()
