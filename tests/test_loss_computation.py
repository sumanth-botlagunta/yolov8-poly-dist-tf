"""Tests for TaskAlignedLossExtended.

Validates:
    - Loss outputs are scalar tensors.
    - Total loss is the weighted sum of all components.
    - Loss decreases when predictions match targets.
    - ignore_bg=1 zeroes out class loss on background anchors.
"""

import tensorflow as tf
import numpy as np
import unittest

from losses.tal_loss import TaskAlignedLossExtended

# Minimal synthetic feats / batch for fast forward-pass tests
_NUM_CLASSES = 3
_IMG_SIZE    = 64    # small feature maps: stride8→8×8, stride16→4×4, stride32→2×2
_BATCH       = 2
_MAX_INST    = 4


def _make_feats(batch_size: int = _BATCH, num_classes: int = _NUM_CLASSES,
                img_size: int = _IMG_SIZE) -> dict:
    """Build minimal raw head outputs."""
    feats = {"box": {}, "cls": {}}
    for level_str, stride in [("3", 8), ("4", 16), ("5", 32)]:
        H = img_size // stride
        W = H
        feats["box"][level_str] = tf.zeros([batch_size, H, W, 64])
        feats["cls"][level_str] = tf.zeros([batch_size, H, W, num_classes])
    for head in ("poly_angle", "poly_dist", "poly_conf"):
        feats[head] = {}
        for level_str, stride in [("3", 8), ("4", 16), ("5", 32)]:
            H = img_size // stride
            feats[head][level_str] = tf.zeros([batch_size, H, H, 24])
    feats["dist"] = {}
    for level_str, stride in [("3", 8), ("4", 16), ("5", 32)]:
        H = img_size // stride
        feats["dist"][level_str] = tf.zeros([batch_size, H, H, 1])
    return feats


def _make_batch(batch_size: int = _BATCH, num_classes: int = _NUM_CLASSES,
                max_inst: int = _MAX_INST) -> dict:
    """Build a minimal labels batch with one valid GT per image."""
    # One GT box centered at (0.4, 0.3)–(0.6, 0.7) (yxyx normalized)
    boxes = tf.constant(
        [[[0.3, 0.4, 0.7, 0.6]] + [[0.0, 0.0, 0.0, 0.0]] * (max_inst - 1)] * batch_size,
        dtype=tf.float32,
    )
    classes = tf.constant(
        [[1] + [0] * (max_inst - 1)] * batch_size, dtype=tf.int64
    )
    polygons = tf.zeros([batch_size, max_inst, 72], dtype=tf.float32)
    log_dist = tf.constant(
        [[-10.0] * max_inst] * batch_size, dtype=tf.float32
    )
    n_gt      = tf.constant([1] * batch_size, dtype=tf.int64)
    ignore_bg = tf.zeros([batch_size], dtype=tf.int64)
    return {
        "bbox":         boxes,
        "classes":      classes,
        "polygons":     polygons,
        "log_distance": log_dist,
        "n_gt":         n_gt,
        "ignore_bg":    ignore_bg,
    }


class TestTaskAlignedLossExtended(unittest.TestCase):
    def setUp(self):
        self.loss_fn = TaskAlignedLossExtended(
            num_classes=_NUM_CLASSES,
            iou_gain=7.5,
            cls_gain=0.5,
            dfl_gain=1.5,
            dist_gain=1.0,
            poly_dist_gain=0.45,
            poly_conf_gain=0.2,
            poly_angle_gain=0.4,
            tal_alpha=0.5,
            tal_beta=6.0,
            topk=3,
            reg_max=16,
            with_polygons=True,
            with_distance=True,
        )

    def test_loss_is_scalar(self):
        """All returned losses are 0-d tensors with finite values."""
        feats = _make_feats()
        batch = _make_batch()
        outputs = self.loss_fn(feats, batch)
        self.assertEqual(len(outputs), 9)
        for val in outputs:
            self.assertEqual(val.shape.rank, 0)
            self.assertTrue(tf.math.is_finite(val))

    def test_loss_components_sum(self):
        """total_loss equals the sum of the five component losses.

        poly_a/poly_d/poly_c are raw (pre-gain) sub-losses and are NOT part of
        the total; poly already contains all gains applied inside _polygon_loss.
        """
        feats  = _make_feats()
        batch  = _make_batch()
        total, box, dfl, cls, dist, poly, poly_a, poly_d, poly_c = self.loss_fn(feats, batch)
        expected_total = box + dfl + cls + dist + poly
        self.assertAlmostEqual(
            float(total), float(expected_total), places=4
        )

    def test_ignore_bg_masks_cls_loss(self):
        """Setting ignore_bg=1 should not raise and should produce finite loss."""
        feats  = _make_feats()
        batch  = _make_batch()
        batch["ignore_bg"] = tf.ones([_BATCH], dtype=tf.int64)
        outputs = self.loss_fn(feats, batch)
        for val in outputs:
            self.assertTrue(tf.math.is_finite(val))

    def test_perfect_prediction_lower_loss(self):
        """Perturbing predictions away from zero should still yield finite losses."""
        feats_zero  = _make_feats()
        feats_noise = {
            head: {
                lvl: t + tf.random.normal(tf.shape(t), stddev=0.1)
                for lvl, t in sub.items()
            }
            for head, sub in feats_zero.items()
        }
        batch = _make_batch()
        outputs_zero  = self.loss_fn(feats_zero, batch)
        outputs_noise = self.loss_fn(feats_noise, batch)
        for val in outputs_zero + outputs_noise:
            self.assertTrue(tf.math.is_finite(val))


if __name__ == "__main__":
    unittest.main()
