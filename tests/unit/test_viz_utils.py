"""Tests for common/viz_utils.render_summary_images (TensorBoard overlays)."""

import unittest

import numpy as np

from common import viz_utils


def _one_pred():
    return {
        "num_detections": 1,
        "bbox": np.array([[0.25, 0.25, 0.75, 0.75]], dtype=np.float32),  # yxyx norm
        "confidence": np.array([0.9], dtype=np.float32),
        "classes": np.array([0], dtype=np.int64),
        "polygons": np.concatenate(
            [
                np.full((1, 24, 1), 0.9),   # conf (already sigmoid-activated, > 0.4 → drawn)
                np.full((1, 24, 1), 0.1),   # radial dist
                np.zeros((1, 24, 1)),       # angle (unused by renderer)
            ],
            axis=-1,
        ).astype(np.float32),
    }


def _one_gt():
    # [M, 72] = [dist, angle_norm, conf] x 24 interleaved (radial about box centre).
    poly = np.zeros((1, 72), dtype=np.float32)
    poly[0, 0::3] = 0.1   # dist per bin
    poly[0, 2::3] = 1.0   # conf per bin (valid → drawn)
    return {
        "n_gt": 1,
        "bbox": np.array([[0.25, 0.25, 0.75, 0.75]], dtype=np.float32),  # yxyx norm
        "classes": np.array([0], dtype=np.int64),
        "polygons": poly,
    }


class TestRenderSummaryImages(unittest.TestCase):
    def setUp(self):
        # Skip cleanly if opencv isn't available (renderer returns None by design).
        self.cv2 = __import__("pytest").importorskip("cv2")

    def test_output_shape_and_dtype(self):
        images = [np.zeros((32, 32, 3), dtype=np.float32)]
        out = viz_utils.render_summary_images(images, [_one_pred()])
        self.assertEqual(out.shape, (1, 32, 32, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_something_is_drawn(self):
        images = [np.zeros((48, 48, 3), dtype=np.float32)]
        out = viz_utils.render_summary_images(images, [_one_pred()])
        # A blank image stays all-zero; drawing a box/polygon must add nonzero pixels.
        self.assertGreater(int(out.sum()), 0)

    def test_no_detections_leaves_image_blank(self):
        images = [np.zeros((16, 16, 3), dtype=np.float32)]
        pred = {
            "num_detections": 0,
            "bbox": np.zeros((1, 4), np.float32),
            "confidence": np.zeros((1,), np.float32),
            "classes": np.zeros((1,), np.int64),
            "polygons": np.zeros((1, 24, 3), np.float32),
        }
        out = viz_utils.render_summary_images(images, [pred])
        self.assertEqual(int(out.sum()), 0)


class TestRenderGtImages(unittest.TestCase):
    def setUp(self):
        self.cv2 = __import__("pytest").importorskip("cv2")

    def test_output_shape_and_dtype(self):
        images = [np.zeros((32, 32, 3), dtype=np.float32)]
        out = viz_utils.render_gt_images(images, [_one_gt()],
                                         class_names={0: "label_0"})
        self.assertEqual(out.shape, (1, 32, 32, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_gt_box_and_polygon_drawn(self):
        images = [np.zeros((48, 48, 3), dtype=np.float32)]
        out = viz_utils.render_gt_images(images, [_one_gt()])
        self.assertGreater(int(out.sum()), 0)

    def test_n_gt_zero_leaves_image_blank(self):
        images = [np.zeros((16, 16, 3), dtype=np.float32)]
        gt = dict(_one_gt(), n_gt=0)
        out = viz_utils.render_gt_images(images, [gt])
        self.assertEqual(int(out.sum()), 0)

    def test_missing_polygons_key_draws_box_only(self):
        # bbox tier: no 'polygons' key → must not crash, still draws the box.
        images = [np.zeros((32, 32, 3), dtype=np.float32)]
        gt = {k: v for k, v in _one_gt().items() if k != "polygons"}
        out = viz_utils.render_gt_images(images, [gt], draw_poly=True)
        self.assertGreater(int(out.sum()), 0)


if __name__ == "__main__":
    unittest.main()
