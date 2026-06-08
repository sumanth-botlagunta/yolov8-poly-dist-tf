"""Polygon segmentation evaluator.

Converts PolyYOLO radial-format predictions to Cartesian masks and computes
mask IoU against GT polygon masks.  Matches predictions to GT via bbox IoU > 0.5.

Prediction format (from detection_generator output):
    pred_polygons: [B, max_boxes, 24, 3]  where [..., :] = (conf, dist, angle)
    - dist:  radial distance for each of 24 vertices, in **normalized** image units
    - conf / angle: per-vertex confidence and dominant-angle channel (not used here)

GT format (from yolo_parser._preprocess_polygons_v2):
    gt_polygons: [B, max_gt, 72] = [dist, angle_norm, conf] x 24 interleaved
    - dist (channel 0::3): radial distance per bin, in **normalized** image units
    - angle_norm (1::3): one-hot dominant bin; conf (2::3): per-bin validity

Both prediction and GT radial distances are normalized [0, ~1.4] relative to the
box-center origin (matching the parser).  They are scaled to pixels per-axis at
rasterization time, so non-square inputs stay self-consistent with the parser.

Vertex angles: theta_i = i * 2*pi / 24  (i = 0..23, 0 = right, CCW)

Classes:
    PolygonEvaluator: Accumulates matched polygon pairs, computes mIoU and recall.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_NUM_VERTICES = 24
_ANGLE_STEP   = 2 * math.pi / _NUM_VERTICES   # radians per vertex


def _radial_to_cartesian(
    cx_n: float, cy_n: float,
    radii: np.ndarray,
    w: int, h: int,
) -> np.ndarray:
    """Convert 24 normalized radial distances to Cartesian pixel vertices.

    The origin and radii are in **normalized** image coordinates (matching the
    parser's PolyYOLO encoding).  Vertices are reconstructed in normalized space
    and then scaled to pixels per-axis (``* w`` / ``* h``).

    Args:
        cx_n, cy_n:  Polygon origin in normalized [0, 1] coordinates.
        radii:       Shape [24], normalized radial distance per vertex.
        w, h:        Image width / height in pixels.

    Returns:
        Array of shape [24, 2] in pixel coordinates (x, y).
    """
    angles = np.arange(_NUM_VERTICES, dtype=np.float32) * _ANGLE_STEP
    xs = (cx_n + radii * np.cos(angles)) * w
    ys = (cy_n + radii * np.sin(angles)) * h
    return np.stack([xs, ys], axis=1)   # [24, 2]


def _polygon_to_mask(
    vertices: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """Rasterize a polygon to a binary mask using cv2.

    Args:
        vertices: [N, 2] array of (x, y) pixel coordinates.
        h, w:     Mask height and width.

    Returns:
        Boolean array of shape [h, w].
    """
    try:
        import cv2
        pts = vertices.astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)
    except ImportError:
        # Fallback: axis-aligned bounding box mask (rough approximation)
        mask = np.zeros((h, w), dtype=bool)
        xs, ys = vertices[:, 0].astype(int), vertices[:, 1].astype(int)
        x1, x2 = max(0, xs.min()), min(w - 1, xs.max())
        y1, y2 = max(0, ys.min()), min(h - 1, ys.max())
        mask[y1:y2 + 1, x1:x2 + 1] = True
        return mask


def _bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of yxyx-normalized boxes.

    Returns:
        [len(boxes_a), len(boxes_b)] float32 IoU matrix.
    """
    y1a, x1a, y2a, x2a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    y1b, x1b, y2b, x2b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_y1 = np.maximum(y1a[:, None], y1b[None, :])
    inter_x1 = np.maximum(x1a[:, None], x1b[None, :])
    inter_y2 = np.minimum(y2a[:, None], y2b[None, :])
    inter_x2 = np.minimum(x2a[:, None], x2b[None, :])

    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter   = inter_h * inter_w

    area_a = (y2a - y1a) * (x2a - x1a)
    area_b = (y2b - y1b) * (x2b - x1b)
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class PolygonEvaluator:
    """Accumulates prediction/GT polygon pairs and computes mask IoU metrics.

    Args:
        image_size: (H, W) used for rasterizing polygon masks.
        iou_thresh: Bbox IoU threshold for matching predictions to GT.
    """

    def __init__(self, image_size: Tuple[int, int] = (672, 672), iou_thresh: float = 0.5):
        self._H, self._W = image_size
        self._iou_thresh  = iou_thresh
        self._mask_ious:  List[float] = []
        self._n_matched   = 0
        self._n_gt_total  = 0
        self._n_dt_total  = 0

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(
        self,
        pred_boxes:       np.ndarray,
        pred_polygons:    np.ndarray,
        pred_scores:      np.ndarray,
        num_detections:   np.ndarray,
        gt_boxes:         np.ndarray,
        gt_polygons:      np.ndarray,
        n_gt:             np.ndarray,
    ) -> None:
        """Accumulate one batch.

        Args:
            pred_boxes:     [B, max_det, 4] yxyx-normalized.
            pred_polygons:  [B, max_det, 24, 3] PolyYOLO (conf, dist, angle_logits).
            pred_scores:    [B, max_det] confidence.
            num_detections: [B] valid detection count.
            gt_boxes:       [B, max_gt, 4] yxyx-normalized.
            gt_polygons:    [B, max_gt, 72] PolyYOLO [dist,angle_norm,conf] x 24.
            n_gt:           [B] valid GT count.
        """
        B = int(num_detections.shape[0])
        for i in range(B):
            n_det = int(num_detections[i])
            n_g   = int(n_gt[i])

            self._n_gt_total += n_g
            self._n_dt_total += n_det

            if n_g == 0 or n_det == 0:
                continue

            db = np.asarray(pred_boxes[i, :n_det])   # [n_det, 4]
            gb = np.asarray(gt_boxes[i,  :n_g])       # [n_g, 4]
            iou_mat = _bbox_iou_matrix(db, gb)         # [n_det, n_g]

            matched_dt = set()
            matched_gt = set()

            # Greedy match by descending score
            order = np.asarray(pred_scores[i, :n_det]).argsort()[::-1]
            for di in order:
                best_gt = int(iou_mat[di].argmax())
                if (iou_mat[di, best_gt] >= self._iou_thresh
                        and best_gt not in matched_gt):
                    matched_dt.add(di)
                    matched_gt.add(best_gt)

                    # ---- compute mask IoU for this match ----
                    p_poly = np.asarray(pred_polygons[i, di])   # [24, 3]
                    g_poly = np.asarray(gt_polygons[i, best_gt])  # [72]

                    # Prediction: dist is channel 1 of (conf, dist, angle), normalized.
                    p_dist = np.maximum(p_poly[:, 1], 0.0)   # [24]

                    # GT: dist is channel 0::3 of [dist, angle_norm, conf] x 24, normalized.
                    g_dist = np.maximum(g_poly[0::3], 0.0)   # [24]

                    # Bbox centres as polygon origins (normalized; scaled to px in helper)
                    y1, x1, y2, x2 = db[di]
                    cx_p = (x1 + x2) / 2.0
                    cy_p = (y1 + y2) / 2.0

                    y1g, x1g, y2g, x2g = gb[best_gt]
                    cx_g = (x1g + x2g) / 2.0
                    cy_g = (y1g + y2g) / 2.0

                    p_verts = _radial_to_cartesian(cx_p, cy_p, p_dist, self._W, self._H)
                    g_verts = _radial_to_cartesian(cx_g, cy_g, g_dist, self._W, self._H)

                    p_mask = _polygon_to_mask(p_verts, self._H, self._W)
                    g_mask = _polygon_to_mask(g_verts, self._H, self._W)

                    inter = (p_mask & g_mask).sum()
                    union = (p_mask | g_mask).sum()
                    iou   = inter / union if union > 0 else 0.0
                    self._mask_ious.append(float(iou))
                    self._n_matched += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Compute polygon mIoU and recall@50.

        poly_mIoU:     mean mask IoU over all matched (prediction, GT) pairs.
        poly_recall50: fraction of GT objects matched at bbox IoU >= 0.5. This is
                       recall, NOT average precision — it has no precision term and
                       no score-threshold sweep, so a prediction-flooding model can
                       inflate it. Use poly_mIoU for mask quality.

        Returns:
            Dict with 'poly_mIoU' and 'poly_recall50'.
        """
        if not self._mask_ious:
            return {'poly_mIoU': 0.0, 'poly_recall50': 0.0}

        miou    = float(np.mean(self._mask_ious))
        recall  = self._n_matched / self._n_gt_total if self._n_gt_total > 0 else 0.0
        return {'poly_mIoU': miou, 'poly_recall50': float(recall)}

    def reset(self) -> None:
        """Clear accumulated data."""
        self._mask_ious.clear()
        self._n_matched  = 0
        self._n_gt_total = 0
        self._n_dt_total = 0
