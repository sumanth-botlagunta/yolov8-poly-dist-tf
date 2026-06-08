"""Tests for PolygonEvaluator.

Validates:
    - Identical pred/GT polygons produce mIoU = 1.0.
    - Partially-overlapping polygons produce 0 < mIoU < 1.
    - No detections returns zeros without crashing.
    - reset() clears state.
    - poly_recall50 is fraction of matched GTs.

Coordinate contract (must match data_pipeline/yolo_parser._preprocess_polygons_v2):
    - GT polygon is [dist, angle_norm, conf] x 24 interleaved (NOT [dx, dy, conf]).
    - Prediction polygon is (conf, dist, angle) per vertex.
    - Radial distances are NORMALIZED [0, ~1.4], scaled to pixels per-axis at
      rasterization. A regression to the old (dx, dy) decode or to pixel-space
      radii makes these tests fail.
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


def _make_gt_poly_72(r: float) -> np.ndarray:
    """GT PolyYOLO format: [dist, angle_norm, conf] x 24 for a circle of radius r.

    r is a NORMALIZED radial distance (the parser emits normalized radii).
    """
    dist = _uniform_radii(r)
    angle_norm = np.zeros(_NUM_VERTS, dtype=np.float32)
    angle_norm[0] = 1.0   # dominant bin one-hot (all radii equal → bin 0)
    conf = np.ones(_NUM_VERTS, dtype=np.float32)
    return np.stack([dist, angle_norm, conf], axis=1).ravel()  # [72]


def _make_pred_poly_24x3(r: float) -> np.ndarray:
    """Prediction polygon in [24, 3] = (conf, dist, angle) format, normalized r."""
    dist  = _uniform_radii(r)
    conf  = np.ones(_NUM_VERTS, dtype=np.float32)
    angle = np.zeros(_NUM_VERTS, dtype=np.float32)
    return np.stack([conf, dist, angle], axis=1)   # [24, 3]


class TestRadialToCartesian(unittest.TestCase):
    def test_circle_first_vertex_at_right(self):
        """Vertex 0 angle = 0 → ((cx_n + r) * W, cy_n * H)."""
        # center (0.5, 0.5) normalized, r = 0.1 normalized, 100x100 px
        verts = _radial_to_cartesian(0.5, 0.5, _uniform_radii(0.1), _W, _H)
        self.assertAlmostEqual(float(verts[0, 0]), 60.0, places=4)
        self.assertAlmostEqual(float(verts[0, 1]), 50.0, places=4)

    def test_non_square_scales_axes_independently(self):
        """A normalized radius scales by W on x and H on y separately."""
        verts = _radial_to_cartesian(0.5, 0.5, _uniform_radii(0.1), 200, 100)
        # vertex 0 (angle 0): x = (0.5 + 0.1)*200 = 120, y = 0.5*100 = 50
        self.assertAlmostEqual(float(verts[0, 0]), 120.0, places=4)
        self.assertAlmostEqual(float(verts[0, 1]),  50.0, places=4)


class TestPolygonEvaluator(unittest.TestCase):

    def _single_batch(self, pred_r, gt_r, cx=0.5, cy=0.5, bbox_frac=0.4):
        """Build a batch of size 1 with one detection and one GT.

        pred_r / gt_r are NORMALIZED radii.
        """
        half = bbox_frac / 2
        bbox = np.array([[cy - half, cx - half, cy + half, cx + half]], dtype=np.float32)

        pred_boxes    = bbox[np.newaxis]                                   # [1,1,4]
        pred_polygons = _make_pred_poly_24x3(pred_r)[np.newaxis, np.newaxis]  # [1,1,24,3]
        pred_scores   = np.ones([1, 1], dtype=np.float32)
        num_dets      = np.array([1], dtype=np.int32)

        gt_boxes    = bbox[np.newaxis]                                     # [1,1,4]
        gt_polygons = _make_gt_poly_72(gt_r)[np.newaxis, np.newaxis]       # [1,1,72]
        n_gt        = np.array([1], dtype=np.int32)

        return (pred_boxes, pred_polygons, pred_scores, num_dets,
                gt_boxes,  gt_polygons,  n_gt)

    def test_identical_polygons_miou_is_one(self):
        """Pred == GT (same normalized radius) → mask IoU = 1.0."""
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(pred_r=0.1, gt_r=0.1)
        ev.update(*args)
        m = ev.evaluate()
        self.assertAlmostEqual(m['poly_mIoU'], 1.0, places=1)

    def test_mismatched_radii_miou_below_one(self):
        """A 2x radius difference must yield mask IoU clearly below 1.

        This is the discriminating assertion: if GT were decoded with the old
        (dx, dy)->sqrt path, or radii were treated as pixels, the GT mask would
        be degenerate and this relationship would not hold.
        """
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(pred_r=0.05, gt_r=0.10)
        ev.update(*args)
        m = ev.evaluate()
        # concentric circles, radius ratio 0.5 → area ratio 0.25 → IoU ~0.25
        self.assertGreater(m['poly_mIoU'], 0.05)
        self.assertLess(m['poly_mIoU'], 0.6)

    def test_no_detections_returns_zeros(self):
        """Zero detections should return zeros without crashing."""
        ev = PolygonEvaluator(image_size=(_H, _W))
        args = list(self._single_batch(0.1, 0.1))
        args[3] = np.array([0], dtype=np.int32)   # num_detections = 0
        ev.update(*args)
        m = ev.evaluate()
        self.assertAlmostEqual(m['poly_mIoU'],     0.0, places=7)
        self.assertAlmostEqual(m['poly_recall50'], 0.0, places=7)

    def test_empty_evaluator_returns_zeros(self):
        """evaluate() on a fresh evaluator must return zeros."""
        ev = PolygonEvaluator(image_size=(_H, _W))
        m  = ev.evaluate()
        self.assertAlmostEqual(m['poly_mIoU'],     0.0, places=7)
        self.assertAlmostEqual(m['poly_recall50'], 0.0, places=7)

    def test_reset_clears_state(self):
        """After reset(), evaluate() returns zeros."""
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(0.1, 0.1)
        ev.update(*args)
        ev.reset()
        m = ev.evaluate()
        self.assertAlmostEqual(m['poly_mIoU'], 0.0, places=7)

    def test_recall50_one_when_matched(self):
        """Single GT, single matching detection → poly_recall50 = 1.0."""
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(0.1, 0.1)
        ev.update(*args)
        m = ev.evaluate()
        self.assertAlmostEqual(m['poly_recall50'], 1.0, places=2)

    def test_miou_in_valid_range(self):
        """Mask IoU is always in [0, 1]."""
        ev   = PolygonEvaluator(image_size=(_H, _W))
        args = self._single_batch(pred_r=0.08, gt_r=0.12)
        ev.update(*args)
        m = ev.evaluate()
        self.assertGreaterEqual(m['poly_mIoU'], 0.0)
        self.assertLessEqual(m['poly_mIoU'],    1.0)


if __name__ == '__main__':
    unittest.main()
