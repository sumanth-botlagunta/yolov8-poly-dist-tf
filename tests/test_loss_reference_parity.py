"""Reference-parity tests for the TAL assignment and box/DFL loss.

These pin two behaviors of the loss against the canonical Ultralytics YOLOv8 recipe.
They are *discriminating* tests — written to FAIL against an implementation that omits
the reference behavior and PASS once it is restored:

    2.1  Soft classification targets are scaled by the GT's localization quality
         (``pos_overlaps`` = per-GT max CIoU, clamped at 0), not normalized to a
         flat max of 1.0.
         Ultralytics: ``target_scores *= align_metric * pos_overlaps / pos_align_metrics``.

    2.2  Box CIoU and DFL losses are weighted per-anchor by ``sum(target_scores, -1)``
         so better-aligned anchors dominate the box gradient.
         Ultralytics: ``(loss * weight).sum() / target_scores_sum``.

The expected values here are derived directly from the reference formulas (hand-derived),
so this file doubles as the reference-parity check without requiring a torch/ultralytics
install. An optional, skipped stub for a direct numerical comparison is at the bottom.
"""

import unittest

import tensorflow as tf

from losses.tal_assigner import TaskAlignedAssigner
from losses.tal_loss import TaskAlignedLossExtended


class TestAssignerPosOverlaps(unittest.TestCase):
    """2.1 — soft target_scores must be scaled by the per-GT max IoU."""

    def test_target_scores_scaled_by_pos_overlaps(self):
        # Single GT box [0,0,40,40] (area 1600). Every prediction is [0,0,40,20]
        # (area 800) → intersection 800, union 1600 → plain IoU 0.5 for all anchors.
        # The overlap metric is CIoU (reference: bbox_iou(..., CIoU=True)):
        #   center penalty  rho2/c2 = ((20-20)^2 + (20-10)^2) / (40^2 + 40^2) = 0.03125
        #   aspect penalty  v = (4/pi^2)(atan(40/40) - atan(40/20))^2 = 0.041926
        #                   alpha_v·v = v^2 / (1 - 0.5 + v) = 0.003243
        #   CIoU = 0.5 - 0.03125 - 0.003243 = 0.465501
        # All four anchor centers sit inside the GT box, so every anchor is a positive
        # with identical alignment. The normalized alignment is therefore 1.0 for all,
        # and the only thing that can pull the soft target below 1.0 is the pos_overlaps
        # (max-CIoU = 0.465501) factor. Reference behavior ⇒ max soft target == 0.4655.
        assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
        B, A, C = 1, 4, 3
        gt_bboxes = tf.constant([[[0.0, 0.0, 40.0, 40.0]]], dtype=tf.float32)
        pd_bboxes = tf.tile(
            tf.constant([[[0.0, 0.0, 40.0, 20.0]]], dtype=tf.float32), [B, A, 1]
        )
        anc_points = tf.constant(
            [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0], [20.0, 20.0]], dtype=tf.float32
        )
        pd_scores = tf.fill([B, A, C], 0.8)
        gt_labels = tf.constant([[1]], dtype=tf.int64)
        mask_gt = tf.constant([[True]])

        _, _, target_scores, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )

        self.assertTrue(tf.reduce_any(fg_mask).numpy(), "expected some foreground anchors")
        max_score = float(tf.reduce_max(target_scores))
        # Reference: scaled by pos_overlaps (= CIoU 0.4655). Buggy variants: 1.0
        # (no pos_overlaps factor) or 0.5 (plain-IoU overlaps instead of CIoU).
        self.assertAlmostEqual(max_score, 0.4655, places=3)

    def test_higher_iou_gt_gets_higher_soft_target(self):
        # Two GTs in two images: image 0's prediction matches its GT better (higher IoU)
        # than image 1's. With pos_overlaps, image 0 must end up with a strictly higher
        # max soft target. Without it, both are normalized to 1.0 and the test fails.
        assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
        C = 2
        gt_bboxes = tf.constant(
            [[[0.0, 0.0, 40.0, 40.0]], [[0.0, 0.0, 40.0, 40.0]]], dtype=tf.float32
        )
        # img0 IoU ~0.875 (box [0,0,40,35]); img1 IoU 0.5 (box [0,0,40,20])
        pd_bboxes = tf.constant(
            [
                [[0.0, 0.0, 40.0, 35.0]] * 3,
                [[0.0, 0.0, 40.0, 20.0]] * 3,
            ],
            dtype=tf.float32,
        )
        anc_points = tf.constant(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 15.0]], dtype=tf.float32
        )
        pd_scores = tf.fill([2, 3, C], 0.8)
        gt_labels = tf.constant([[0], [0]], dtype=tf.int64)
        mask_gt = tf.constant([[True], [True]])

        _, _, target_scores, _, _, _ = assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )
        max0 = float(tf.reduce_max(target_scores[0]))
        max1 = float(tf.reduce_max(target_scores[1]))
        # Reference: max0≈0.875, max1≈0.5. Buggy (no pos_overlaps): both ≈1.0, diff≈0.
        # Require a clear gap (not float noise) and confirm scaling actually applied.
        self.assertGreater(max0 - max1, 0.1)
        self.assertLess(max0, 0.95)


class TestBoxLossAnchorWeighting(unittest.TestCase):
    """2.2 — box CIoU and DFL must be weighted per-anchor by sum(target_scores, -1)."""

    def setUp(self):
        self.loss_fn = TaskAlignedLossExtended(
            num_classes=3, reg_max=16, with_polygons=False, with_distance=False
        )
        B, A, C = 1, 2, 3
        self.fg_mask = tf.constant([[True, True]])
        self.anc_points = tf.constant([[4.0, 4.0], [12.0, 12.0]], dtype=tf.float32)
        self.anc_strides = tf.constant([[8.0], [8.0]], dtype=tf.float32)
        # Targets in xyxy pixels; predictions slightly off so CIoU loss is non-zero.
        self.target_bboxes = tf.constant(
            [[[0.0, 0.0, 8.0, 8.0], [8.0, 8.0, 16.0, 16.0]]], dtype=tf.float32
        )
        self.pd_bboxes = tf.constant(
            [[[0.0, 0.0, 7.0, 7.0], [8.0, 8.0, 15.0, 15.0]]], dtype=tf.float32
        )
        self.pd_box_raw = tf.zeros([B, A, 64], dtype=tf.float32)
        # Per-anchor weight = sum over classes = 0.5 for each anchor.
        self.target_scores = tf.constant(
            [[[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]], dtype=tf.float32
        )
        self.ssum = tf.constant(1.0, dtype=tf.float32)

    def _box_loss(self, target_scores):
        return self.loss_fn._box_loss(
            self.pd_bboxes,
            self.target_bboxes,
            target_scores,
            self.ssum,
            self.fg_mask,
            self.pd_box_raw,
            self.anc_strides,
            self.anc_points,
        )

    def test_box_and_dfl_scale_linearly_with_anchor_weight(self):
        # Holding target_scores_sum fixed, doubling the per-anchor target scores must
        # double both the CIoU and DFL losses (because the numerator is weighted by them).
        # An unweighted implementation ignores target_scores ⇒ losses unchanged ⇒ FAIL.
        ciou_1, dfl_1 = self._box_loss(self.target_scores)
        ciou_2, dfl_2 = self._box_loss(2.0 * self.target_scores)

        self.assertGreater(float(ciou_1), 0.0)
        self.assertGreater(float(dfl_1), 0.0)
        self.assertAlmostEqual(float(ciou_2), 2.0 * float(ciou_1), places=4)
        self.assertAlmostEqual(float(dfl_2), 2.0 * float(dfl_1), places=4)


@unittest.skip(
    "Optional: direct numerical comparison against Ultralytics v8DetectionLoss. "
    "Requires torch + ultralytics; the hand-derived tests above encode the same "
    "reference formulas without the heavy dependency."
)
class TestUltralyticsParity(unittest.TestCase):  # pragma: no cover
    def test_box_cls_dfl_match_reference(self):
        pass


if __name__ == "__main__":
    unittest.main()
