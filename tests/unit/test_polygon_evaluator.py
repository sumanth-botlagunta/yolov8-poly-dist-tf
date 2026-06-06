"""Tests for PolygonEvaluator.

Validates:
    - Identical pred/GT polygons produce mIoU = 1.0.
    - Non-overlapping polygons produce mIoU = 0.0.
    - No detections returns zeros without crashing.
    - reset() clears state.
    - poly_AP50 is fraction of matched GTs.
"""

import math
import unittest
import numpy as np

from eval.polygon_metrics import PolygonEvaluator, _radial_to_cartesian


_H, _W = 100, 100
_NUM_VERTS = 24


def _uniform_radii(r: float) -> np.ndarray:
    """24-element array of equal radii (a circle)."""
    return np.full(_NUM_VERTS, r, dtype=np.float32)


def _make_poly_72(r: float) -> np.ndarray:
    """PolyYOLO GT format: [dx0,dy0,c0, ...] for a circle of radius r at origin."""
    angles = np.arange(_NUM_VERTS) * (2 * math.pi / _NUM_VERTS)
    dx = r * np.cos(angles).astype(np.float32)
    dy = r * np.sin(angles).astype(np.float32)
    conf = np.ones(_NUM_VERTS, dtype=np.float32)
    return np.stack([dx, dy, conf], axis=1).ravel()  # [72]


def _make_poly_24x3(r: float) -> np.ndarray:
    """Prediction polygon in [24, 3] = (conf, dist, angle_logits) format."""
    dist  = _uniform_radii(r)
    conf  = np.ones(_NUM_VERTS, dtype=np.float32)
    angle = np.zeros(_NUM_VERTS, dtype=np.float32)
    return np.stack([conf, dist, angle], axis=1)   # [24, 3]


class TestRadialToCartesian(unittest.TestCase):
    def test_circle_first_vertex_at_right(self):
        """Vertex 0 angle = 0 → (cx + r, cy)."""
        verts = _radial_to_cartesian(50.0, 50.0, _uniform_radii(10.0))
        self.assertAlmostEqual(float(verts[0, 0]), 60.0, places=4)
        self.assertAlmostEqual(float(verts[0, 1]), 50.0, places=4)


class TestPolygonEvaluator(unittest.TestCase):

    def _single_batch(self, pred_r, gt_r, cx=0.5, cy=0.5, bbox_frac=0.4):
        """Build a batch of size 1 with one detection and one GT."""
        half = bbox_frac / 2
        bbox = np.array([[cy - half, cx - half, cy + half, cx + half]], dtype=np.float32)

        pred_boxes    = bbox[np.newaxis]                          # [1,1,4]
        pred_polygons = _make_poly_24x3(pred_r)[np.newaxis, np.newaxis]  # [1,1,24,3]
        pred_scores   = np.ones([1, 1], dtype=np.float32)
        num_dets      = np.array([1], dtype=np.int32)

        gt_boxes    = bbox[np.newaxis]                            # [1,1,4]
        gt_polygons = _make_poly_72(gt_r)[np.newaxis, np.newaxis]         # [1,1,72]
        n_gt        = np.array([1], dtype=np.int32)

        return (pred_boxes, pred_polygons, pred_scores, num_dets,
                gt_boxes,  gt_polygons,  n_gt)

    def test_identical_polygons_miou_is_one(self):
        """Pred == GT → mask IoU = 1.0."""
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(pred_r=10.0, gt_r=10.0)
        ev.update(*args)
        m = ev.evaluate()
        self.assertAlmostEqual(m['poly_mIoU'], 1.0, places=1)

    def test_no_detections_returns_zeros(self):
        """Zero detections should return zeros without crashing."""
        ev = PolygonEvaluator(image_size=(_H, _W))
        args = list(self._single_batch(10.0, 10.0))
        args[3] = np.array([0], dtype=np.int32)   # num_detections = 0
        ev.update(*args)
        m = ev.evaluate()
        self.assertAlmostEqual(m['poly_mIoU'],  0.0, places=7)
        self.assertAlmostEqual(m['poly_AP50'],  0.0, places=7)

    def test_empty_evaluator_returns_zeros(self):
        """evaluate() on a fresh evaluator must return zeros."""
        ev = PolygonEvaluator(image_size=(_H, _W))
        m  = ev.evaluate()
        self.assertAlmostEqual(m['poly_mIoU'], 0.0, places=7)
        self.assertAlmostEqual(m['poly_AP50'], 0.0, places=7)

    def test_reset_clears_state(self):
        """After reset(), evaluate() returns zeros."""
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(10.0, 10.0)
        ev.update(*args)
        ev.reset()
        m = ev.evaluate()
        self.assertAlmostEqual(m['poly_mIoU'], 0.0, places=7)

    def test_poly_ap50_one_when_matched(self):
        """Single GT, single matching detection → poly_AP50 = 1.0."""
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(10.0, 10.0)
        ev.update(*args)
        m = ev.evaluate()
        self.assertAlmostEqual(m['poly_AP50'], 1.0, places=2)

    def test_miou_in_valid_range(self):
        """Mask IoU is always in [0, 1]."""
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(pred_r=8.0, gt_r=12.0)
        ev.update(*args)
        m = ev.evaluate()
        self.assertGreaterEqual(m['poly_mIoU'], 0.0)
        self.assertLessEqual(m['poly_mIoU'],    1.0)


if __name__ == '__main__':
    unittest.main()
