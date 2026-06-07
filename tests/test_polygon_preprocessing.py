"""Tests for polygon format conversion (_preprocess_polygons_v2).

The PolyYOLO target is the interleaved radial format
``[dist, angle_norm, conf] × (360/angle_step)``:

    - PolyYOLO output shape is [N, 360/angle_step * 3].
    - dist is the radial distance from the box center to the bin's vertex.
    - conf is 1.0 for bins that received a valid vertex and 0.0 for absent bins;
      absent bins also carry dist == 0.0 (so the dist head learns to collapse them).
"""

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
        self.angle = self.out[0, 1::3]  # [24] one-hot
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
        # angle_norm is a one-hot over the dominant bin.
        self.assertEqual(int(self.angle.sum()), 1)


if __name__ == "__main__":
    unittest.main()
