"""Regression tests that PIN the polygon loss reduction conventions.

These lock the (intentional, but easy-to-break) per-vertex reduction differences so
they cannot change silently:

    - polygon_angle_loss AVERAGES cross-entropy over the V vertices (reduce_mean).
    - polygon_dist_loss  and polygon_conf_loss SUM over the V vertices (reduce_sum).

With V=24 the sum-based terms are ~24× the mean-based one for identical per-vertex
error — the configured poly gains compensate. See losses/polygon_loss.py and plan 2.4.
"""

import math
import unittest

import tensorflow as tf

from losses.polygon_loss import (
    polygon_angle_loss,
    polygon_conf_loss,
    polygon_dist_loss,
)

_V = 24
_LOG2 = math.log(2.0)


def _one_fg():
    # A single image with a single foreground anchor; normalizer = 1.0 so the
    # returned scalar is exactly the per-anchor reduced value.
    fg_mask = tf.constant([[True]])
    ssum = tf.constant(1.0)
    return fg_mask, ssum


class TestPolygonLossConventions(unittest.TestCase):
    def test_angle_loss_is_mean_over_vertices(self):
        fg_mask, ssum = _one_fg()
        logits = tf.zeros([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        # sigmoid_cross_entropy(0, 0) = log(2) per vertex; mean over V → log(2).
        loss = float(polygon_angle_loss(logits, target, fg_mask, ssum))
        self.assertAlmostEqual(loss, _LOG2, places=5)

    def test_dist_loss_is_sum_over_vertices(self):
        fg_mask, ssum = _one_fg()
        pd = tf.ones([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        # |1 - 0| = 1 per vertex; SUM over V → V (=24), not 1.
        loss = float(polygon_dist_loss(pd, target, fg_mask, ssum))
        self.assertAlmostEqual(loss, float(_V), places=5)

    def test_conf_loss_is_sum_over_vertices(self):
        fg_mask, ssum = _one_fg()
        logits = tf.zeros([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        # BCE(0, 0) = log(2) per vertex; SUM over V → V * log(2).
        loss = float(polygon_conf_loss(logits, target, fg_mask, ssum))
        self.assertAlmostEqual(loss, _V * _LOG2, places=4)


if __name__ == "__main__":
    unittest.main()
