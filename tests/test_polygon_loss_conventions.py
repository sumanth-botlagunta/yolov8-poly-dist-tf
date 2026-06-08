"""Regression tests that PIN the polygon loss reduction conventions.

These lock the current (old-codebase-matching) per-vertex reduction behavior:

    - polygon_angle_loss AVERAGES cross-entropy over V vertices (reduce_mean),
      normalized by target_scores_sum.
    - polygon_dist_loss  applies softplus to prediction, computes L2 = (target-softplus(pred))^2,
      AVERAGES over V vertices, normalizes by num_objs.
    - polygon_conf_loss  AVERAGES BCE over V vertices, normalizes by num_objs.

With V=24 and num_objs=1, dist and conf return the mean per-vertex value (not 24x it).
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
    norm = tf.constant(1.0)
    return fg_mask, norm


class TestPolygonLossConventions(unittest.TestCase):
    def test_angle_loss_is_mean_over_vertices(self):
        fg_mask, ssum = _one_fg()
        logits = tf.zeros([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        # sigmoid_cross_entropy(0, 0) = log(2) per vertex; mean over V → log(2).
        loss = float(polygon_angle_loss(logits, target, fg_mask, ssum))
        self.assertAlmostEqual(loss, _LOG2, places=5)

    def test_dist_loss_is_mean_over_vertices(self):
        fg_mask, num_objs = _one_fg()
        pd = tf.ones([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        # softplus(1.0) ≈ 1.31326; l2 = (0 - 1.31326)^2 ≈ 1.72465 per vertex.
        # Mean over V → same value (all equal). Divided by num_objs=1 → 1.72465.
        import math
        sp1 = math.log(1.0 + math.e)   # softplus(1.0)
        expected = sp1 ** 2
        loss = float(polygon_dist_loss(pd, target, fg_mask, num_objs))
        self.assertAlmostEqual(loss, expected, places=4)

    def test_conf_loss_is_mean_over_vertices(self):
        fg_mask, num_objs = _one_fg()
        logits = tf.zeros([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        # BCE(0, 0) = log(2) per vertex; MEAN over V → log(2), not V*log(2).
        loss = float(polygon_conf_loss(logits, target, fg_mask, num_objs))
        self.assertAlmostEqual(loss, _LOG2, places=4)

    def test_angle_uses_target_scores_sum_normalizer(self):
        """Doubling target_scores_sum halves the angle loss."""
        fg_mask = tf.constant([[True]])
        logits = tf.zeros([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        loss1 = float(polygon_angle_loss(logits, target, fg_mask, tf.constant(1.0)))
        loss2 = float(polygon_angle_loss(logits, target, fg_mask, tf.constant(2.0)))
        self.assertAlmostEqual(loss2, loss1 / 2.0, places=5)

    def test_dist_uses_num_objs_normalizer(self):
        """Doubling num_objs halves the dist loss."""
        fg_mask = tf.constant([[True]])
        pd = tf.ones([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        loss1 = float(polygon_dist_loss(pd, target, fg_mask, tf.constant(1.0)))
        loss2 = float(polygon_dist_loss(pd, target, fg_mask, tf.constant(2.0)))
        self.assertAlmostEqual(loss2, loss1 / 2.0, places=5)

    def test_conf_uses_num_objs_normalizer(self):
        """Doubling num_objs halves the conf loss."""
        fg_mask = tf.constant([[True]])
        logits = tf.zeros([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        loss1 = float(polygon_conf_loss(logits, target, fg_mask, tf.constant(1.0)))
        loss2 = float(polygon_conf_loss(logits, target, fg_mask, tf.constant(2.0)))
        self.assertAlmostEqual(loss2, loss1 / 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
