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
        """mosaic_frequency=1.0 path produces 4 output images of [4, H, W, 3]."""
        m = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=True)
        out = m.mosaic_fn(is_training=True)(_make_batch4())
        self.assertEqual(tuple(out["image"].shape), (4, 32, 32, 3))

    def test_boxes_in_unit_range(self):
        """mosaic_frequency=1.0 path keeps all (padded + real) boxes within [0,1]."""
        m = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=True)
        out = m.mosaic_fn(is_training=True)(_make_batch4())
        self.assertEqual(out["groundtruth_boxes"].shape[0], 4)
        boxes = out["groundtruth_boxes"].numpy()
        self.assertTrue((boxes >= -1e-4).all() and (boxes <= 1.0 + 1e-4).all())

    def test_identity_single_reproduces_input(self):
        """freq=0 + identity affine → each of the 4 outputs reproduces its input.

        Every decoded image must yield exactly one emitted sample (4-in/4-out),
        and with an identity warp the single branch is a passthrough.
        """
        batch = _make_batch4(h=32, w=32)
        out = _identity_mosaic(out=32, freq=0.0).mosaic_fn(is_training=True)(batch)
        self.assertEqual(tuple(out["image"].shape), (4, 32, 32, 3))
        for i in range(4):
            np.testing.assert_array_equal(
                out["image"][i].numpy(), batch["image"][i].numpy()
            )

    def test_branches_have_identical_structure(self):
        """Both tf.cond branches must emit dicts with identical keys/dtypes/ranks."""
        batch = _make_batch4(h=32, w=32)
        m_mosaic = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=True)
        m_single = Mosaic(output_size=[32, 32], mosaic_frequency=0.0, with_polygons=True)
        out_m = m_mosaic.mosaic_fn(is_training=True)(batch)
        out_s = m_single.mosaic_fn(is_training=True)(batch)

        self.assertEqual(set(out_m.keys()), set(out_s.keys()))
        for k in out_m:
            self.assertEqual(out_m[k].dtype, out_s[k].dtype, f"dtype mismatch {k}")
            self.assertEqual(
                len(out_m[k].shape), len(out_s[k].shape), f"rank mismatch {k}"
            )
            # Leading (sample) dim is always 4.
            self.assertEqual(int(out_m[k].shape[0]), 4)
            self.assertEqual(int(out_s[k].shape[0]), 4)

    def test_padded_rows_are_zero_boxes_and_neg1_polys(self):
        """freq=0 + identity affine: valid (non-zero-box) anns match the 4 inputs;
        padded rows are zero boxes / -1 polygons.

        The incoming batch already has a uniform instance dim (it comes from
        padded_batch upstream); here sample 0 carries a real second box while
        samples 1-3 carry a padded (zero-box / -1-poly) second row. The single
        branch's clip_boxes(min_side=0.005) drops the zero box, so it stays a pad
        row on output. (This pins both the upstream pad-row contract and the
        _stack_results re-pad behaviour.)
        """
        b0 = [[0.25, 0.25, 0.75, 0.75], [0.10, 0.10, 0.40, 0.40]]
        bpad = [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.0, 0.0]]
        p0 = [[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0],
              [0.2, 0.2, 0.35, 0.35, -1.0, -1.0, -1.0, -1.0]]
        ppad = [[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0],
                [-1.0] * 8]
        boxes = tf.constant([b0, bpad, bpad, bpad], tf.float32)      # [4, 2, 4]
        polys = tf.constant([p0, ppad, ppad, ppad], tf.float32)      # [4, 2, 8]
        batch = {
            "image":  tf.fill([4, 32, 32, 3], tf.constant(100, tf.uint8)),
            "height": tf.constant([32] * 4, tf.int32),
            "width":  tf.constant([32] * 4, tf.int32),
            "groundtruth_boxes":    boxes,
            "groundtruth_classes":  tf.zeros([4, 2], tf.int64),
            "groundtruth_is_crowd": tf.zeros([4, 2], tf.bool),
            "groundtruth_area":     tf.ones([4, 2], tf.float32),
            "groundtruth_dontcare": tf.zeros([4, 2], tf.int64),
            "groundtruth_dists":    tf.fill([4, 2], tf.constant(-1.0)),
            "groundtruth_polygons": polys,
            "source_id":            tf.constant(["a", "b", "c", "d"]),
        }

        out = _identity_mosaic(out=32, freq=0.0).mosaic_fn(is_training=True)(batch)
        boxes = out["groundtruth_boxes"].numpy()
        polys = out["groundtruth_polygons"].numpy()
        self.assertEqual(boxes.shape[0], 4)
        self.assertEqual(boxes.shape[1], 2)   # padded up to group-max N=2

        # Sample 0: both boxes valid (non-zero). Samples 1-3: row 1 is a pad row.
        for i in range(4):
            n_valid = 2 if i == 0 else 1
            for j in range(2):
                if j < n_valid:
                    self.assertTrue((boxes[i, j] != 0.0).any(),
                                    f"sample {i} row {j} should be a real box")
                else:
                    np.testing.assert_array_equal(boxes[i, j], [0.0, 0.0, 0.0, 0.0])
                    np.testing.assert_array_equal(polys[i, j], [-1.0] * 8)

    def test_eval_path_four_out(self):
        """is_training=False also emits 4 outputs (consistency)."""
        batch = _make_batch4(h=32, w=32)
        out = _identity_mosaic(out=32, freq=0.0).mosaic_fn(is_training=False)(batch)
        self.assertEqual(tuple(out["image"].shape), (4, 32, 32, 3))


class TestMosaicUnbatchIntegration(unittest.TestCase):
    def test_padded_batch_map_unbatch_yields_four(self):
        """from_tensor_slices(4) → padded_batch(4) → map(mosaic_fn) → unbatch → 4 elems."""
        h = w = 32
        examples = {
            "image":  tf.fill([4, h, w, 3], tf.constant(100, tf.uint8)),
            "height": tf.constant([h] * 4, tf.int32),
            "width":  tf.constant([w] * 4, tf.int32),
            "groundtruth_boxes":    tf.tile([[[0.25, 0.25, 0.75, 0.75]]], [4, 1, 1]),
            "groundtruth_classes":  tf.zeros([4, 1], tf.int64),
            "groundtruth_is_crowd": tf.zeros([4, 1], tf.bool),
            "groundtruth_area":     tf.ones([4, 1], tf.float32),
            "groundtruth_dontcare": tf.zeros([4, 1], tf.int64),
            "groundtruth_dists":    tf.fill([4, 1], tf.constant(-1.0)),
            "groundtruth_polygons": tf.tile(
                [[[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0]]], [4, 1, 1]),
            "source_id":            tf.constant(["a", "b", "c", "d"]),
        }
        m = _identity_mosaic(out=32, freq=0.0)
        ds = (tf.data.Dataset.from_tensor_slices(examples)
              .padded_batch(4, drop_remainder=True)
              .map(m.mosaic_fn(is_training=True))
              .unbatch())

        # Static element spec keeps image [H, W, 3].
        self.assertEqual(ds.element_spec["image"].shape.as_list(), [h, w, 3])

        count = 0
        for el in ds:
            self.assertEqual(tuple(el["image"].shape), (h, w, 3))
            count += 1
        self.assertEqual(count, 4)


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
