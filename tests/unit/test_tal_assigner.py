"""Tests for TaskAlignedAssigner.

Validates:
    - fg_mask foreground count matches expected top-k coverage.
    - Anchors outside all GT boxes are always background.
    - target_labels match GT class for assigned foreground anchors.
    - Output shapes are correct.
"""

import math

import numpy as np
import tensorflow as tf
import unittest

from losses.tal_assigner import TaskAlignedAssigner, _pairwise_ciou


def _make_assigner() -> TaskAlignedAssigner:
    return TaskAlignedAssigner(topk=2, alpha=0.5, beta=6.0)


class TestTaskAlignedAssigner(unittest.TestCase):
    def test_output_shapes(self):
        """All outputs have the expected shapes."""
        B, A, C, M = 2, 6, 3, 2
        assigner = _make_assigner()
        pd_scores = tf.random.uniform([B, A, C])
        pd_bboxes = tf.tile(
            tf.constant([[[10., 10., 50., 50.]]], dtype=tf.float32),
            [B, A, 1],
        )
        anc_points = tf.constant(
            [[20., 20.], [25., 25.], [30., 30.],
             [60., 60.], [80., 80.], [100., 100.]], dtype=tf.float32
        )
        gt_labels = tf.constant([[0, 1], [1, 0]], dtype=tf.int64)
        # One GT box at pixels [10,10,50,50]; second GT shifted
        gt_bboxes = tf.constant(
            [[[10., 10., 50., 50.], [60., 60., 90., 90.]]] * B,
            dtype=tf.float32,
        )
        mask_gt   = tf.constant([[True, True], [True, False]])
        gt_polys  = tf.zeros([B, M, 72])
        gt_dists  = tf.zeros([B, M])

        (
            target_labels, target_bboxes, target_scores,
            target_polygons, target_dists, fg_mask,
        ) = assigner(
            pd_scores, pd_bboxes, anc_points,
            gt_labels, gt_bboxes, mask_gt,
            gt_polys=gt_polys, gt_dists=gt_dists,
        )

        self.assertEqual(target_labels.shape,    (B, A))
        self.assertEqual(target_bboxes.shape,    (B, A, 4))
        self.assertEqual(target_scores.shape,    (B, A, C))
        self.assertEqual(target_polygons.shape,  (B, A, 72))
        self.assertEqual(target_dists.shape,     (B, A, 1))
        self.assertEqual(fg_mask.shape,          (B, A))

    def test_out_of_box_anchors_are_background(self):
        """Anchors whose center is outside all GT boxes must be background."""
        assigner = _make_assigner()
        # One GT box at pixels [20, 20, 40, 40]
        # Place all anchor centers strictly outside that box
        B, A, C, M = 1, 4, 2, 1
        pd_scores  = tf.ones([B, A, C]) * 0.5
        pd_bboxes  = tf.tile(
            tf.constant([[[25., 25., 35., 35.]]], dtype=tf.float32), [B, A, 1]
        )
        anc_points = tf.constant(
            [[5., 5.], [50., 50.], [60., 10.], [10., 60.]], dtype=tf.float32
        )
        gt_labels  = tf.constant([[0]], dtype=tf.int64)
        gt_bboxes  = tf.constant([[[20., 20., 40., 40.]]], dtype=tf.float32)
        mask_gt    = tf.constant([[True]])
        gt_polys   = tf.zeros([B, M, 72])
        gt_dists   = tf.zeros([B, M])

        _, _, _, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc_points,
            gt_labels, gt_bboxes, mask_gt,
            gt_polys=gt_polys, gt_dists=gt_dists,
        )
        # No anchor center is inside [20,20,40,40] so fg_mask must be all False
        self.assertFalse(tf.reduce_any(fg_mask).numpy())

    def test_foreground_label_matches_gt(self):
        """Every foreground anchor's target_label equals its assigned GT class."""
        assigner = _make_assigner()
        B, A, C, M = 1, 9, 4, 1
        # GT box covering the center of the image; class = 2
        pd_scores = tf.ones([B, A, C]) * 0.5
        # Anchor centers: 3×3 grid at (10,10) to (90,90)
        centers   = [(x, y) for y in [10., 50., 90.] for x in [10., 50., 90.]]
        anc_points = tf.constant(centers, dtype=tf.float32)
        pd_bboxes  = tf.tile(
            tf.constant([[[20., 20., 80., 80.]]], dtype=tf.float32), [B, A, 1]
        )
        gt_labels  = tf.constant([[2]], dtype=tf.int64)
        gt_bboxes  = tf.constant([[[20., 20., 80., 80.]]], dtype=tf.float32)
        mask_gt    = tf.constant([[True]])
        gt_polys   = tf.zeros([B, M, 72])
        gt_dists   = tf.zeros([B, M])

        target_labels, _, _, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc_points,
            gt_labels, gt_bboxes, mask_gt,
            gt_polys=gt_polys, gt_dists=gt_dists,
        )
        fg_labels = tf.boolean_mask(target_labels[0], fg_mask[0])
        # All foreground anchors should be assigned class 2
        self.assertTrue(tf.reduce_all(fg_labels == 2).numpy())

    def test_padded_gt_ignored(self):
        """GTs marked False in mask_gt must not attract any anchor assignments."""
        assigner = _make_assigner()
        B, A, C, M = 1, 6, 2, 2
        pd_scores  = tf.ones([B, A, C]) * 0.5
        # Only the first GT is valid; second GT is padding
        gt_labels  = tf.constant([[0, 1]], dtype=tf.int64)
        gt_bboxes  = tf.constant(
            [[[10., 10., 50., 50.], [10., 10., 50., 50.]]], dtype=tf.float32
        )
        # mask_gt[0, 1] = False → second GT should be ignored
        mask_gt    = tf.constant([[True, False]])
        anc_points = tf.constant(
            [[20., 20.], [25., 25.], [30., 30.],
             [60., 60.], [80., 80.], [100., 100.]], dtype=tf.float32
        )
        pd_bboxes  = tf.tile(
            tf.constant([[[10., 10., 50., 50.]]], dtype=tf.float32), [B, A, 1]
        )
        gt_polys   = tf.zeros([B, M, 72])
        gt_dists   = tf.zeros([B, M])

        target_labels, _, _, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc_points,
            gt_labels, gt_bboxes, mask_gt,
            gt_polys=gt_polys, gt_dists=gt_dists,
        )
        fg_labels = tf.boolean_mask(target_labels[0], fg_mask[0])
        # Class 1 (from padded GT) should never appear
        self.assertFalse(tf.reduce_any(fg_labels == 1).numpy())


class TestCIoUOverlaps(unittest.TestCase):
    """The assigner's overlap metric is Complete IoU clamped at 0 (reference
    recipe: bbox_iou(..., CIoU=True).clamp_(0)), not plain inter/union."""

    def test_pairwise_ciou_matches_loss_ciou(self):
        from losses.tal_assigner import _pairwise_ciou
        from losses.tal_loss import _bbox_iou_loss
        rng = tf.random.Generator.from_seed(3)
        xy1 = rng.uniform([64, 2], 0.0, 300.0)
        wh1 = rng.uniform([64, 2], 1.0, 200.0)
        xy2 = rng.uniform([64, 2], 0.0, 300.0)
        wh2 = rng.uniform([64, 2], 1.0, 200.0)
        b1 = tf.concat([xy1, xy1 + wh1], axis=-1)
        b2 = tf.concat([xy2, xy2 + wh2], axis=-1)
        got = _pairwise_ciou(b1, b2)
        want = 1.0 - _bbox_iou_loss(b1, b2, "ciou")
        self.assertLess(float(tf.reduce_max(tf.abs(got - want))), 1e-5)

    def test_pairwise_ciou_finite_on_zero_boxes(self):
        from losses.tal_assigner import _pairwise_ciou
        pd = tf.constant([[10., 10., 50., 50.]])
        gt = tf.zeros([1, 4])  # padded GT row
        val = _pairwise_ciou(pd, gt)
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(val))))

    def test_duplicate_resolution_prefers_center_aligned_gt(self):
        """Two GTs with IDENTICAL plain IoU against the predicted box; the
        center-aligned one has higher CIoU. Plain-IoU argmax would tie and
        pick index 0 (the offset GT, listed first); CIoU must pick index 1."""
        B, A, C, M = 1, 16, 3, 2
        # All anchors predict the same box [0,0,12,12] (area 144).
        pd_bboxes = tf.tile(tf.constant([[[0., 0., 12., 12.]]]), [B, A, 1])
        pd_scores = tf.fill([B, A, C], 0.5)
        # Anchor grid inside both GTs (spatial candidates for both).
        xs = tf.constant([3., 5., 7., 9.])
        gx, gy = tf.meshgrid(xs, xs)
        anc_points = tf.stack([tf.reshape(gx, [-1]), tf.reshape(gy, [-1])], -1)
        # GT0 offset [2,2,12,12]: inter 100, union 144 -> IoU 100/144, center
        # offset sqrt(2). GT1 centered [1,1,11,11]: inter 100, union 144 ->
        # SAME IoU, zero center offset -> higher CIoU.
        gt_bboxes = tf.constant([[[2., 2., 12., 12.], [1., 1., 11., 11.]]])
        gt_labels = tf.constant([[0, 1]], dtype=tf.int64)
        mask_gt = tf.ones([B, M], dtype=tf.bool)

        assigner = TaskAlignedAssigner(topk=10)
        target_labels, _, _, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc_points,
            gt_labels, gt_bboxes, mask_gt,
            gt_polys=tf.zeros([B, M, 72]), gt_dists=tf.zeros([B, M]),
        )
        fg_labels = tf.boolean_mask(target_labels[0], fg_mask[0])
        self.assertGreater(int(tf.size(fg_labels)), 0)
        # Every contested anchor must resolve to GT1 (class 1, higher CIoU).
        self.assertTrue(bool(tf.reduce_all(fg_labels == 1)),
                        f"expected all class 1, got {fg_labels.numpy()}")


class TestTopkGuard(unittest.TestCase):
    """A GT with no positive-alignment candidate gets ZERO positives.

    Without the guard, the k-th top value for such a GT is exactly 0.0 and the
    >= comparison marks every in-box anchor foreground (thousands for a large
    object), all tie-broken onto GT slot 0. Reference behavior (Ultralytics
    select_topk_candidates) selects nothing for a hopeless GT.
    """

    def _grid_anchors(self):
        # 5x5 grid inside [0, 100]^2
        c = tf.linspace(10.0, 90.0, 5)
        xx, yy = tf.meshgrid(c, c)
        return tf.stack([tf.reshape(xx, [-1]), tf.reshape(yy, [-1])], -1)  # [25, 2]

    def test_hopeless_gt_gets_no_positives(self):
        anc = self._grid_anchors()
        A = 25
        # Zero-area predictions at the anchor points: CIoU clamps to 0 for the
        # box-sized GT, so its alignment underflows to exactly 0 everywhere.
        pd_bboxes = tf.concat([anc, anc], -1)[tf.newaxis]          # [1, A, 4]
        pd_scores = tf.fill([1, A, 3], 0.05)
        gt_bboxes = tf.constant([[[0., 0., 100., 100.]]])          # covers all anchors
        gt_labels = tf.constant([[1]], dtype=tf.int64)
        mask_gt   = tf.constant([[True]])
        assigner  = TaskAlignedAssigner(topk=10)
        _, _, _, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc, gt_labels, gt_bboxes, mask_gt)
        self.assertEqual(int(tf.reduce_sum(tf.cast(fg_mask, tf.int32))), 0)

    def test_partial_candidates_select_only_the_positive_ones(self):
        anc = self._grid_anchors()
        A = 25
        pd_boxes = tf.concat([anc, anc], -1).numpy()               # zero-area
        # Give exactly 3 anchors a decent overlapping prediction (< topk=10).
        for i in (12, 13, 17):
            pd_boxes[i] = [20.0, 20.0, 80.0, 80.0]
        pd_bboxes = tf.constant(pd_boxes[None], tf.float32)
        pd_scores = tf.fill([1, A, 3], 0.05)
        gt_bboxes = tf.constant([[[0., 0., 100., 100.]]])
        gt_labels = tf.constant([[1]], dtype=tf.int64)
        mask_gt   = tf.constant([[True]])
        assigner  = TaskAlignedAssigner(topk=10)
        _, _, _, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc, gt_labels, gt_bboxes, mask_gt)
        fg = fg_mask.numpy()[0]
        self.assertEqual(fg.sum(), 3)
        self.assertTrue(all(fg[i] for i in (12, 13, 17)))


class TestTargetScoreResolvedNormalization(unittest.TestCase):
    """target_scores normalizes each GT's soft label by ITS OWN best REMAINING
    (post duplicate-resolution) align/IoU, not by the raw pre-resolution
    candidate max — see the ``align_norm`` / ``resolved`` block in
    TaskAlignedAssigner.__call__.
    """

    _ALPHA, _BETA, _EPS = 0.5, 6.0, 1e-9

    def _align(self, score, iou):
        """Mirror the assigner's log-space alignment formula exactly."""
        return math.exp(self._ALPHA * math.log(score + self._EPS)) * \
               math.exp(self._BETA * math.log(iou + self._EPS))

    def test_contested_gt_normalizes_by_surviving_max_align(self):
        """Two concentric-square GTs share anchor0 as a spatial candidate.

        anchor0's predicted box exactly matches GT_A (IoU=1.0) and only
        partially overlaps GT_B (IoU=0.25) -- the highest IoU any candidate
        has against GT_B, so duplicate resolution (argmax raw IoU) still
        steals anchor0 for GT_A. GT_B's only surviving candidates are
        anchor1 (IoU=0.16) and anchor2 (IoU=0.09); the top SURVIVOR is
        anchor1, not the stolen anchor0. target_scores for GT_B's survivors
        must be normalized against anchor1's align (post-resolution), not
        anchor0's (higher, pre-resolution) align.
        """
        assigner = TaskAlignedAssigner(topk=10, alpha=self._ALPHA, beta=self._BETA)

        # Concentric squares (same center, same aspect ratio) -> CIoU reduces
        # to plain-IoU (rho2 == 0, aspect term v == 0), so IoU is an exact
        # area-ratio and easy to control by half-width alone.
        gt_a = tf.constant([[40., 40., 60., 60.]])   # half-width 10, center (50,50)
        gt_b = tf.constant([[30., 30., 70., 70.]])   # half-width 20, center (50,50)
        gt_bboxes = tf.stack([gt_a, gt_b], axis=1)   # [1, 2, 4]
        gt_labels = tf.constant([[0, 1]], dtype=tf.int64)
        mask_gt = tf.constant([[True, True]])

        pred0 = tf.constant([40., 40., 60., 60.])    # matches GT_A -> IoU(0,A)=1.0
        pred1 = tf.constant([42., 42., 58., 58.])    # half-width 8
        pred2 = tf.constant([44., 44., 56., 56.])    # half-width 6
        pd_bboxes = tf.stack([pred0, pred1, pred2])[tf.newaxis]  # [1, 3, 4]

        # anchor0 inside both GT_A ([40,60]^2) and GT_B ([30,70]^2); anchor1/2
        # inside GT_B only (x=35, 32 fall outside GT_A's [40,60] x-range).
        anc_points = tf.constant([[50., 50.], [35., 50.], [32., 50.]])

        pd_scores = tf.fill([1, 3, 2], 0.5)  # constant score -> align rank == IoU rank

        target_labels, _, target_scores, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt,
            gt_polys=tf.zeros([1, 2, 72]), gt_dists=tf.zeros([1, 2]),
        )

        self.assertTrue(bool(tf.reduce_all(fg_mask)))
        labels = target_labels.numpy()[0]
        self.assertEqual(labels[0], 0)  # anchor0 -> GT_A (robbed from GT_B)
        self.assertEqual(labels[1], 1)  # anchor1 -> GT_B (top survivor)
        self.assertEqual(labels[2], 1)  # anchor2 -> GT_B (surviving runner-up)

        # Independently reconstruct the IoUs with the assigner's own CIoU
        # primitive (unit-tested elsewhere) to get ground-truth values.
        iou_b1 = float(_pairwise_ciou(pred1[tf.newaxis], gt_b)[0])
        iou_b2 = float(_pairwise_ciou(pred2[tf.newaxis], gt_b)[0])
        iou_b0 = float(_pairwise_ciou(pred0[tf.newaxis], gt_b)[0])
        self.assertGreater(iou_b0, iou_b1)  # confirms anchor0 WAS the best raw candidate
        self.assertGreater(iou_b1, iou_b2)

        align_b1 = self._align(0.5, iou_b1)
        align_b2 = self._align(0.5, iou_b2)

        # Correct (post-resolution): normalize by the max align among the
        # SURVIVING anchors {1, 2} only -> anchor1 is both the align-max and
        # the pos_overlap source, so its own target score reduces to its IoU.
        expected_score_anchor1 = iou_b1
        expected_score_anchor2 = align_b2 * iou_b1 / align_b1

        got = target_scores.numpy()[0]
        self.assertAlmostEqual(got[1, 1], expected_score_anchor1, places=5)
        self.assertAlmostEqual(got[2, 1], expected_score_anchor2, places=5)

        # Discriminating check: the WRONG pre-resolution normalizer (using
        # the stolen anchor0's higher align/IoU as the max) gives a
        # numerically different answer -- confirms this scenario actually
        # exercises the resolved-vs-raw distinction, not a coincidence.
        align_b0 = self._align(0.5, iou_b0)
        wrong_score_anchor1 = align_b1 * iou_b0 / align_b0
        self.assertNotAlmostEqual(got[1, 1], wrong_score_anchor1, places=4)

    def test_uncontested_gt_matches_analytic_align_over_max_form(self):
        """No contest: target_scores == align_norm computed directly from the
        analytic align/IoU formula (byte-close, not just qualitatively right).
        """
        assigner = TaskAlignedAssigner(topk=10, alpha=self._ALPHA, beta=self._BETA)

        gt_box = tf.constant([[30., 30., 70., 70.]])  # half-width 20, center (50,50)
        gt_bboxes = gt_box[:, tf.newaxis, :]           # [1, 1, 4]
        gt_labels = tf.constant([[0]], dtype=tf.int64)
        mask_gt = tf.constant([[True]])

        pred_top = tf.constant([35., 35., 65., 65.])   # half-width 15 -> IoU 0.5625
        pred_low = tf.constant([40., 40., 60., 60.])   # half-width 10 -> IoU 0.25
        pd_bboxes = tf.stack([pred_top, pred_low])[tf.newaxis]  # [1, 2, 4]
        anc_points = tf.constant([[50., 50.], [40., 50.]])       # both inside the box
        pd_scores = tf.fill([1, 2, 1], 0.5)

        _, _, target_scores, _, _, fg_mask = assigner(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt,
            gt_polys=tf.zeros([1, 1, 72]), gt_dists=tf.zeros([1, 1]),
        )
        self.assertTrue(bool(tf.reduce_all(fg_mask)))

        iou_top = float(_pairwise_ciou(pred_top[tf.newaxis], gt_box)[0])
        iou_low = float(_pairwise_ciou(pred_low[tf.newaxis], gt_box)[0])
        align_top = self._align(0.5, iou_top)
        align_low = self._align(0.5, iou_low)
        pos_overlap = iou_top          # max IoU (single GT, no contest)
        align_max = align_top          # max align (single GT, no contest)

        expected_top = align_top * pos_overlap / align_max   # == iou_top
        expected_low = align_low * pos_overlap / align_max

        got = target_scores.numpy()[0]
        np.testing.assert_allclose(got[0, 0], expected_top, atol=1e-6)
        np.testing.assert_allclose(got[1, 0], expected_low, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
