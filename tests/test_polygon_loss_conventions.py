"""Regression tests that PIN the polygon loss reduction conventions.

These lock the current (old-codebase-matching) per-vertex reduction behavior:

    - polygon_dist_loss  applies softplus to the prediction, computes
      L2 = (target - softplus(pred))^2, and AVERAGES over the VALID vertices
      only (masked by vertex_mask), normalized by num_objs.
    - polygon_angle_loss applies BCE on the sub-bin offset target and AVERAGES
      over the VALID vertices only (masked), normalized by num_objs.
    - polygon_conf_loss  AVERAGES BCE over ALL bins (occupied → 1, empty → 0 —
      conf is the decode gate and must see negatives; the masked
      form is preserved in its docstring), normalized by num_objs.

All three normalize by num_objs; angle/dist ignore invalid (empty) bins, and an
anchor with no valid vertex contributes 0 (no NaN).
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
_SOFTPLUS1 = math.log(1.0 + math.e)   # softplus(1.0)


def _fg():
    return tf.constant([[True]]), tf.constant(1.0)


def _half_mask():
    # First 12 vertices valid, last 12 invalid.
    return tf.constant([[[1.0] * 12 + [0.0] * 12]])


class TestPolygonLossConventions(unittest.TestCase):
    # ---- distance: masked mean over valid vertices ----
    def test_dist_averages_over_valid_only(self):
        fg, num_objs = _fg()
        # Valid bins: pred=1,target=0 → err softplus(1)^2. Invalid bins: pred=5
        # (huge err) but masked out — must NOT affect the result.
        pd = tf.constant([[[1.0] * 12 + [5.0] * 12]])
        target = tf.zeros([1, 1, _V])
        loss = float(polygon_dist_loss(pd, target, _half_mask(), fg, num_objs))
        self.assertAlmostEqual(loss, _SOFTPLUS1 ** 2, places=4)

    def test_dist_all_valid_matches_plain_mean(self):
        fg, num_objs = _fg()
        pd = tf.ones([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        loss = float(polygon_dist_loss(pd, target, tf.ones([1, 1, _V]), fg, num_objs))
        self.assertAlmostEqual(loss, _SOFTPLUS1 ** 2, places=4)

    def test_dist_zero_valid_is_zero_not_nan(self):
        fg, num_objs = _fg()
        pd = tf.ones([1, 1, _V])
        target = tf.zeros([1, 1, _V])
        loss = float(polygon_dist_loss(pd, target, tf.zeros([1, 1, _V]), fg, num_objs))
        self.assertEqual(loss, 0.0)

    def test_dist_uses_num_objs_normalizer(self):
        fg = tf.constant([[True]])
        pd = tf.ones([1, 1, _V]); target = tf.zeros([1, 1, _V]); m = tf.ones([1, 1, _V])
        l1 = float(polygon_dist_loss(pd, target, m, fg, tf.constant(1.0)))
        l2 = float(polygon_dist_loss(pd, target, m, fg, tf.constant(2.0)))
        self.assertAlmostEqual(l2, l1 / 2.0, places=5)

    # ---- angle: masked BCE over valid vertices ----
    def test_angle_averages_over_valid_only(self):
        fg, num_objs = _fg()
        # Valid bins: logits=0,target=0 → BCE=log2. Invalid bins: logits=10 (huge
        # BCE) but masked out.
        logits = tf.constant([[[0.0] * 12 + [10.0] * 12]])
        target = tf.zeros([1, 1, _V])
        loss = float(polygon_angle_loss(logits, target, _half_mask(), fg, num_objs))
        self.assertAlmostEqual(loss, _LOG2, places=5)

    def test_angle_uses_num_objs_normalizer(self):
        fg = tf.constant([[True]])
        logits = tf.zeros([1, 1, _V]); target = tf.zeros([1, 1, _V]); m = tf.ones([1, 1, _V])
        l1 = float(polygon_angle_loss(logits, target, m, fg, tf.constant(1.0)))
        l2 = float(polygon_angle_loss(logits, target, m, fg, tf.constant(2.0)))
        self.assertAlmostEqual(l2, l1 / 2.0, places=5)

    # ---- conf: masked BCE over valid vertices only ----
    def test_conf_averages_over_all_bins_including_empty(self):
        """Conf is the decode gate: empty bins MUST contribute (target 0).

        The previous masked form averaged over valid bins
        only, so empty bins never received gradient — their conf output drifted
        above the decode/viz threshold while their dist stayed untrained,
        producing star/spiky polygon artifacts in val overlays. The masked form
        is preserved in polygon_conf_loss's docstring for reference.
        """
        fg, num_objs = _fg()
        # Valid bins: logits=0, target=1 → BCE = log2. Empty bins: logits=10,
        # target=0 → BCE = softplus(10) — and they now COUNT toward the mean.
        logits = tf.constant([[[0.0] * 12 + [10.0] * 12]])
        target = tf.constant([[[1.0] * 12 + [0.0] * 12]])
        loss = float(polygon_conf_loss(logits, target, _half_mask(), fg, num_objs))
        bce_empty = 10.0 + math.log1p(math.exp(-10.0))
        expected = (12 * _LOG2 + 12 * bce_empty) / 24.0
        self.assertAlmostEqual(loss, expected, places=4)
        # The masked form would have returned _LOG2 — make sure we are NOT that.
        self.assertNotAlmostEqual(loss, _LOG2, places=2)

    def test_conf_penalizes_confident_empty_bins(self):
        """Predicting high conf on empty bins must cost more than predicting low."""
        fg, num_objs = _fg()
        target = tf.constant([[[1.0] * 12 + [0.0] * 12]])
        high_on_empty = tf.constant([[[5.0] * 12 + [5.0] * 12]])
        low_on_empty  = tf.constant([[[5.0] * 12 + [-5.0] * 12]])
        l_high = float(polygon_conf_loss(high_on_empty, target, _half_mask(), fg, num_objs))
        l_low  = float(polygon_conf_loss(low_on_empty,  target, _half_mask(), fg, num_objs))
        self.assertGreater(l_high, l_low)

    def test_conf_uses_num_objs_normalizer(self):
        fg = tf.constant([[True]])
        logits = tf.zeros([1, 1, _V]); target = tf.zeros([1, 1, _V]); m = tf.ones([1, 1, _V])
        l1 = float(polygon_conf_loss(logits, target, m, fg, tf.constant(1.0)))
        l2 = float(polygon_conf_loss(logits, target, m, fg, tf.constant(2.0)))
        self.assertAlmostEqual(l2, l1 / 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
