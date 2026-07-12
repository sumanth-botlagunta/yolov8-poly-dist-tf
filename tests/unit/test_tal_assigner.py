"""Tests for TaskAlignedAssigner.

Validates:
    - fg_mask foreground count matches expected top-k coverage.
    - Anchors outside all GT boxes are always background.
    - target_labels match GT class for assigned foreground anchors.
    - Output shapes are correct.
"""

import tensorflow as tf
import unittest

from losses.tal_assigner import TaskAlignedAssigner


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


if __name__ == "__main__":
    unittest.main()
