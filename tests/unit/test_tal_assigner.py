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


if __name__ == "__main__":
    unittest.main()
