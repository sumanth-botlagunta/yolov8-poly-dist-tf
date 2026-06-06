"""Tests for the distance regression loss.

Validates:
    - Sentinel value -10.0 is excluded from loss computation.
    - Loss is 0.0 when all GT distances are invalid.
    - L1 value is correct for a simple known case.
"""

import tensorflow as tf
import unittest

from losses.distance_loss import distance_l1_loss, INVALID_DISTANCE_SENTINEL


class TestDistanceL1Loss(unittest.TestCase):
    def test_sentinel_excluded(self):
        """Anchors whose target == INVALID_DISTANCE_SENTINEL contribute nothing."""
        B, A = 1, 4
        # anchor 0: valid distance = 1.0, anchor 1: invalid sentinel
        pd   = tf.constant([[1.5], [2.0], [0.0], [0.0]], dtype=tf.float32)
        pd   = tf.reshape(pd, [B, A, 1])
        tgt  = tf.constant(
            [[1.0], [INVALID_DISTANCE_SENTINEL], [1.0], [1.0]], dtype=tf.float32
        )
        tgt  = tf.reshape(tgt, [B, A, 1])
        # Only anchor 0 is foreground and valid
        fg   = tf.constant([[True, True, False, False]])  # [1, 4]
        norm = tf.constant(1.0)

        loss = distance_l1_loss(pd, tgt, fg, norm)
        # Only anchor 0 contributes: |1.5 - 1.0| = 0.5; anchor 1 is excluded
        self.assertAlmostEqual(float(loss), 0.5, places=5)

    def test_all_invalid_returns_zero(self):
        """Loss is exactly 0.0 when all GT distances are the invalid sentinel."""
        B, A = 2, 3
        pd  = tf.ones([B, A, 1])
        tgt = tf.fill([B, A, 1], INVALID_DISTANCE_SENTINEL)
        fg  = tf.ones([B, A], dtype=tf.bool)
        norm = tf.constant(5.0)

        loss = distance_l1_loss(pd, tgt, fg, norm)
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_known_value(self):
        """L1 on a single valid foreground sample is computed correctly."""
        pd   = tf.constant([[[2.3]]])          # [1, 1, 1]
        tgt  = tf.constant([[[1.0]]])
        fg   = tf.constant([[True]])           # [1, 1]
        norm = tf.constant(2.0)                # normalizer

        loss = distance_l1_loss(pd, tgt, fg, norm)
        # L1 = |2.3 - 1.0| = 1.3, normalized by 2.0 → 0.65
        self.assertAlmostEqual(float(loss), 0.65, places=5)


if __name__ == "__main__":
    unittest.main()
