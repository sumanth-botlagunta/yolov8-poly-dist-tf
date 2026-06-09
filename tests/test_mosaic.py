"""Tests for the Ultralytics-style Mosaic + random_perspective augmentation.

Validates:
    - Mosaic output image has the configured output_size and boxes stay in [0,1].
    - mosaic_frequency=0.0 with identity affine reproduces the (output-sized) input.
    - _place_in_cell pastes/crops an image into a gray-114 cell at an offset.
    - random_perspective: identity round-trips; boxes clip to edge; polygon vertices
      are clipped to the edge (originally-valid stay in [0,1]; -1 padding stays -1).
"""

import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.mosaic import Mosaic, _place_in_cell
from data_pipeline.augmentations import random_perspective


_MAXV = 8  # 4 (x,y) pairs


def _make_batch4(h: int = 32, w: int = 32, n: int = 1) -> dict:
    """Batch-of-4 example dict (each field has a leading dim of 4)."""
    box = tf.constant([[0.25, 0.25, 0.75, 0.75]] * n, dtype=tf.float32)  # yxyx norm
    poly = tf.constant([[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0]] * n, dtype=tf.float32)
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
        "groundtruth_polygons": tf.stack([poly] * 4),
        "source_id":            tf.constant(["a", "b", "c", "d"]),
    }


def _identity_mosaic(out=32, freq=0.0):
    return Mosaic(
        output_size=[out, out], mosaic_frequency=freq, with_polygons=True,
        aug_scale_min=1.0, aug_scale_max=1.0,
        degrees=0.0, shear=0.0, perspective=0.0, translate=0.0,
        mosaic_center=0.25,
    )


class TestMosaic(unittest.TestCase):
    def test_output_size(self):
        m = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=True)
        out = m.mosaic_fn(is_training=True)(_make_batch4())
        self.assertEqual(tuple(out["image"].shape), (1, 32, 32, 3))

    def test_boxes_in_unit_range(self):
        m = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=True)
        out = m.mosaic_fn(is_training=True)(_make_batch4())
        boxes = out["groundtruth_boxes"].numpy()
        self.assertTrue((boxes >= -1e-4).all() and (boxes <= 1.0 + 1e-4).all())

    def test_identity_single_reproduces_input(self):
        """freq=0 + identity affine → single branch returns the input unchanged."""
        batch = _make_batch4(h=32, w=32)
        out = _identity_mosaic(out=32, freq=0.0).mosaic_fn(is_training=True)(batch)
        np.testing.assert_array_equal(out["image"][0].numpy(), batch["image"][0].numpy())


class TestPlaceInCell(unittest.TestCase):
    def test_paste_with_offset_and_gray_fill(self):
        R = tf.fill([10, 10, 3], tf.constant(200, tf.uint8))
        cell = _place_in_cell(R, tf.constant(20), tf.constant(20),
                              tf.constant(5), tf.constant(5)).numpy()
        self.assertEqual(cell.shape, (20, 20, 3))
        self.assertTrue((cell[5:15, 5:15] == 200).all())   # pasted region
        self.assertTrue((cell[0:5, :] == 114).all())        # gray fill outside

    def test_crop_when_larger_than_cell(self):
        R = tf.fill([30, 30, 3], tf.constant(200, tf.uint8))
        cell = _place_in_cell(R, tf.constant(20), tf.constant(20),
                              tf.constant(-5), tf.constant(-5)).numpy()
        self.assertEqual(cell.shape, (20, 20, 3))
        self.assertTrue((cell == 200).all())   # fully covered by the cropped image


class TestRandomPerspective(unittest.TestCase):
    def setUp(self):
        self.s = 64
        img = np.zeros((self.s, self.s, 3), np.uint8)
        img[8:24, 8:24] = 255
        self.img = tf.constant(img)
        self.boxes = tf.constant([[0.3, 0.3, 0.7, 0.7]], tf.float32)
        self.polys = tf.constant([[0.3, 0.3, 0.7, 0.7, -1.0, -1.0]], tf.float32)

    def test_identity_round_trips(self):
        io, bo, keep, po = random_perspective(
            self.img, self.boxes, self.polys, self.s, self.s,
            degrees=0, translate=0, scale=0, shear=0, perspective=0,
        )
        self.assertTrue(np.array_equal(io.numpy(), self.img.numpy()))
        np.testing.assert_allclose(bo.numpy(), self.boxes.numpy(), atol=1e-3)
        self.assertTrue(bool(keep.numpy()[0]))

    def test_boxes_clipped_to_unit_range(self):
        tf.random.set_seed(0)
        _, bo, keep, _ = random_perspective(
            self.img, self.boxes, self.polys, self.s, self.s, degrees=30,
        )
        b = bo.numpy()
        self.assertTrue((b >= 0.0).all() and (b <= 1.0).all())
        self.assertEqual(keep.numpy().shape, (1,))

    def test_polygon_clip_to_edge_keeps_validity(self):
        """Originally-valid vertices stay in [0,1] (clipped); -1 padding stays -1."""
        tf.random.set_seed(3)
        _, _, _, po = random_perspective(
            self.img, self.boxes, self.polys, self.s, self.s, degrees=25, scale=0.5,
        )
        p = po.numpy()[0]
        valid = p[0:4]     # two transformed (x,y) pairs
        pad = p[4:6]       # the -1 padding pair
        self.assertTrue((valid >= 0.0).all() and (valid <= 1.0).all())
        np.testing.assert_array_equal(pad, [-1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
