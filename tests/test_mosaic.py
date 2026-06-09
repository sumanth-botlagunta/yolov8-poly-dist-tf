"""Tests for Mosaic and MixUp augmentation.

Validates:
    - Output image has the configured output_size.
    - Boxes are clipped to [0, 1] after stitching.
    - mosaic_frequency=0.0 returns first input unchanged.
    - _transform_boxes maps a known box to exact output coordinates (guards the
      new_h/new_w threading fix — the old code re-derived new_h ≈ quad_h - 2*pad_top,
      which drifts by up to 1px for odd letterbox padding).
"""

import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.mosaic import Mosaic, _transform_boxes, _transform_polygons


def _make_batch4(h: int = 20, w: int = 20, n: int = 1) -> dict:
    """Build a batch-of-4 example dict (each field has a leading dim of 4)."""
    box = tf.constant([[0.25, 0.25, 0.75, 0.75]] * n, dtype=tf.float32)  # yxyx norm
    return {
        "image":  tf.fill([4, h, w, 3], tf.constant(100, tf.uint8)),
        "height": tf.constant([h] * 4, tf.int32),
        "width":  tf.constant([w] * 4, tf.int32),
        "groundtruth_boxes":    tf.stack([box] * 4),
        "groundtruth_classes":  tf.zeros([4, n], tf.int64),
        "groundtruth_is_crowd": tf.zeros([4, n], tf.bool),
        "groundtruth_area":     tf.ones([4, n], tf.float32),
        "groundtruth_dontcare": tf.zeros([4, n], tf.int64),
        "groundtruth_dists":    tf.fill([4, n], tf.constant(-1.0)),
        "source_id":            tf.constant(["a", "b", "c", "d"]),
    }


class TestMosaic(unittest.TestCase):
    def test_output_size(self):
        """Mosaic output image matches the configured output_size."""
        mosaic = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=False)
        out = mosaic.mosaic_fn(is_training=True)(_make_batch4())
        self.assertEqual(tuple(out["image"].shape), (1, 32, 32, 3))

    def test_boxes_clipped(self):
        """All boxes lie within [0, 1] after stitching."""
        mosaic = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=False)
        out = mosaic.mosaic_fn(is_training=True)(_make_batch4())
        boxes = out["groundtruth_boxes"].numpy()
        self.assertTrue((boxes >= 0.0).all() and (boxes <= 1.0).all())

    def test_zero_frequency_passthrough(self):
        """mosaic_frequency=0.0 returns the first input image unchanged."""
        batch = _make_batch4()
        mosaic = Mosaic(output_size=[32, 32], mosaic_frequency=0.0, with_polygons=False)
        out = mosaic.mosaic_fn(is_training=True)(batch)
        np.testing.assert_array_equal(
            out["image"][0].numpy(), batch["image"][0].numpy()
        )

    def test_transform_boxes_exact_coordinates(self):
        """A full-image box maps to exact output coords using the true new_h/new_w.

        Content occupies new_h=new_w=50 px, padded by (10,10), no quadrant offset,
        on a 100×100 canvas. A box spanning the whole input [0,0,1,1] must land at
        [pad/H, pad/W, (new+pad)/H, (new+pad)/W] = [0.1, 0.1, 0.6, 0.6].
        """
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)
        out, keep = _transform_boxes(
            boxes,
            scale=tf.constant(0.5),
            pad_top=tf.constant(10), pad_left=tf.constant(10),
            new_h=tf.constant(50), new_w=tf.constant(50),
            offset_y=tf.constant(0), offset_x=tf.constant(0),
            H_out=tf.constant(100), W_out=tf.constant(100),
            area_thresh=0.0,
        )
        self.assertTrue(bool(keep.numpy()[0]))
        np.testing.assert_allclose(
            out.numpy()[0], [0.1, 0.1, 0.6, 0.6], atol=1e-6
        )

    def test_transform_polygons_invalidates_out_of_bounds(self):
        """A source-valid vertex pushed off-canvas must become -1, not clamp to edge.

        Content occupies 50px placed at quadrant offset (80,80) on a 100×100 canvas.
        Vertex (0.1, 0.1) → ((0.1*50+80)/100)=0.85 → in bounds, kept.
        Vertex (0.9, 0.9) → ((0.9*50+80)/100)=1.25 → out of bounds, must be -1.
        Padding vertex (-1,-1) stays -1.
        """
        # one instance, 3 (x,y) pairs flattened to [1, 6]
        polys = tf.constant([[0.1, 0.1, 0.9, 0.9, -1.0, -1.0]], dtype=tf.float32)
        out = _transform_polygons(
            polys,
            pad_top=tf.constant(0), pad_left=tf.constant(0),
            new_h=tf.constant(50), new_w=tf.constant(50),
            offset_y=tf.constant(80), offset_x=tf.constant(80),
            H_out=tf.constant(100), W_out=tf.constant(100),
        ).numpy()[0]
        # in-bounds vertex kept (≈0.85)
        np.testing.assert_allclose(out[0:2], [0.85, 0.85], atol=1e-5)
        # out-of-bounds vertex invalidated, not clamped to 1.0
        self.assertEqual(out[2], -1.0)
        self.assertEqual(out[3], -1.0)
        # padding stays -1
        self.assertEqual(out[4], -1.0)
        self.assertEqual(out[5], -1.0)


if __name__ == "__main__":
    unittest.main()
