"""Polygon segmentation evaluator.

Converts PolyYOLO radial-format predictions to Cartesian masks and computes
mask IoU against GT polygon masks.  Matches predictions to GT via bbox IoU > 0.5.

Prediction format (from detection_generator output):
    pred_polygons: [B, max_boxes, 24, 3]  where [..., :] = (conf, dist, angle)
    - dist:  radial distance for each of 24 vertices, in **normalized** image units
    - angle: per-bin sub-bin offset in [0, 1) (vertex angle = (i + angle) * step)
    - conf:  per-vertex confidence (sigmoid-activated); bins below ``conf_thresh``
      are EXCLUDED from rasterization, mirroring the decode/viz gate

GT format (from yolo_parser._preprocess_polygons_v2):
    gt_polygons: [B, max_gt, 72] = [dist, angle, conf] x 24 interleaved
    - dist (channel 0::3): radial distance per bin, in **normalized** image units
    - angle (1::3): sub-bin offset in [0, 1); conf (2::3): per-bin validity

Both prediction and GT radial distances are normalized [0, ~1.4] relative to the
box-center origin (matching the parser).  The parser stores each radius as a
single *isotropic* normalized distance (sqrt(dx^2+dy^2)); decomposing it per-axis
(`* w` / `* h`) only reconstructs the original vertex when ``w == h``.  This
evaluator therefore requires square inputs and asserts it at construction time
(see PolygonEvaluator.__init__); non-square support would need the radius stored
in pixel space.

Vertex angles: theta_i = (i + offset_i) * 2*pi / 24  (i = 0..23, 0 = right, CCW)

Classes:
    PolygonEvaluator: Accumulates matched polygon pairs, computes mIoU and recall.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Default radial bin count (angle_step=15 → 24 bins). _radial_to_cartesian
# infers the actual count and angular step from len(radii) so a non-15° config
# reconstructs at the right resolution; this is only the PolygonEvaluator default.
_NUM_VERTICES = 24

# Per-bin confidence gate for PREDICTED polygon vertices. The evaluator and the
# TensorBoard polygon overlay (train/viz_utils.py) MUST share this threshold so
# the visualised contour matches the one scored — viz_utils imports this value
# as its default rather than re-hardcoding 0.4.
DEFAULT_POLY_CONF_THRESH = 0.4


def _radial_to_cartesian(
    cx_n: float, cy_n: float,
    radii: np.ndarray,
    w: int, h: int,
    offsets: np.ndarray = None,
) -> np.ndarray:
    """Convert N normalized radial distances to Cartesian pixel vertices.

    The origin and radii are in **normalized** image coordinates (matching the
    parser's PolyYOLO encoding).  Vertices are reconstructed in normalized space
    and then scaled to pixels per-axis (``* w`` / ``* h``).

    The vertex count N is inferred from ``len(radii)`` and the angular step is
    ``2*pi / N``, so a non-15° config (e.g. angle_step=10 → 36 bins) reconstructs
    at the correct angular resolution instead of assuming 24 bins.

    Args:
        cx_n, cy_n:  Polygon origin in normalized [0, 1] coordinates.
        radii:       Shape [N], normalized radial distance per vertex.
        w, h:        Image width / height in pixels.
        offsets:     Optional shape [N], sub-bin angular offset in [0, 1) per
                     bin (vertex angle = (i + offset) * angle_step). When None,
                     the bin centre angle (i * angle_step) is used.

    Returns:
        Array of shape [N, 2] in pixel coordinates (x, y).
    """
    radii = np.asarray(radii, dtype=np.float32)
    n_verts = radii.shape[0]
    angle_step = 2 * math.pi / n_verts
    idx = np.arange(n_verts, dtype=np.float32)
    if offsets is not None:
        idx = idx + np.asarray(offsets, dtype=np.float32)
    angles = idx * angle_step
    xs = (cx_n + radii * np.cos(angles)) * w
    ys = (cy_n + radii * np.sin(angles)) * h
    return np.stack([xs, ys], axis=1)   # [N, 2]


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
    if vertices.shape[0] < 3:
        # Fewer than 3 vertices cannot enclose area (e.g. a prediction whose
        # conf gate left < 3 bins) — empty mask, not a cv2 error.
        return np.zeros((h, w), dtype=bool)
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


def _eval_gt_mask(n_g, i, gt_is_crowd, gt_is_dontcare) -> np.ndarray:
    """Boolean [n_g] mask of GT to evaluate (drop crowd / dontcare)."""
    keep = np.ones(n_g, dtype=bool)
    if gt_is_crowd is not None:
        keep &= ~np.asarray(gt_is_crowd[i, :n_g], dtype=bool)
    if gt_is_dontcare is not None:
        keep &= ~np.asarray(gt_is_dontcare[i, :n_g], dtype=bool)
    return keep


def _count_eval_gt(n_g, i, gt_is_crowd, gt_is_dontcare) -> int:
    """Count of evaluable (non-crowd/dontcare) GT for image i."""
    if n_g == 0:
        return 0
    return int(_eval_gt_mask(n_g, i, gt_is_crowd, gt_is_dontcare).sum())


class PolygonEvaluator:
    """Accumulates prediction/GT polygon pairs and computes mask IoU metrics.

    Vertex-validity gating: only bins whose conf channel passes the gate are
    rasterized — pred bins need ``conf >= conf_thresh`` (the same 0.4 gate the
    decode/viz path uses), GT bins need ``conf > 0.5`` (the parser writes a
    binary validity). Empty GT bins encode ``dist = 0``; rasterizing them (the
    pre-2026-06-11 behavior) injected a vertex at the box CENTER per empty bin,
    turning any polygon that doesn't occupy all 24 bins into a center-spiked
    star — a 4-vertex GT lost ~90% of its mask area, so poly_mIoU measured
    star-vs-star similarity rather than the decoded polygons that ship.
    Matched pairs whose gated GT has < 3 vertices are counted for recall but
    skipped for mask IoU (no measurable GT mask).

    Args:
        image_size: (H, W) used for rasterizing polygon masks.
        iou_thresh: Bbox IoU threshold for matching predictions to GT.
        conf_thresh: Per-bin confidence gate for PREDICTED vertices; must match
            the decode/viz threshold so the metric scores what is deployed.
    """

    def __init__(self, image_size: Tuple[int, int] = (672, 672), iou_thresh: float = 0.5,
                 conf_thresh: float = DEFAULT_POLY_CONF_THRESH, num_vertices: int = _NUM_VERTICES):
        self._H, self._W = image_size
        # Number of radial bins (= 360 // angle_step). _radial_to_cartesian
        # infers the actual count from len(radii); this is kept for callers that
        # want to validate against the configured value and to document intent.
        self._num_vertices = num_vertices
        # Radial-distance reconstruction is only correct for square inputs (the
        # parser stores a single isotropic radius). Fail loudly rather than
        # silently report distorted mIoU on a non-square config.
        if self._H != self._W:
            raise ValueError(
                "PolygonEvaluator requires square inputs (the radial polygon "
                f"radius is isotropic); got H={self._H}, W={self._W}."
            )
        self._iou_thresh  = iou_thresh
        self._conf_thresh = conf_thresh
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
        gt_is_crowd:      Optional[np.ndarray] = None,
        gt_is_dontcare:   Optional[np.ndarray] = None,
    ) -> None:
        """Accumulate one batch.

        Args:
            pred_boxes:     [B, max_det, 4] yxyx-normalized.
            pred_polygons:  [B, max_det, 24, 3] PolyYOLO (conf, dist, angle) where
                            angle is the sigmoid-activated sub-bin offset in [0, 1),
                            not a raw logit.
            pred_scores:    [B, max_det] confidence.
            num_detections: [B] valid detection count.
            gt_boxes:       [B, max_gt, 4] yxyx-normalized.
            gt_polygons:    [B, max_gt, 72] PolyYOLO [dist, angle, conf] x 24.
            n_gt:           [B] valid GT count.
            gt_is_crowd:    [B, max_gt] bool, optional. Crowd GT are excluded from
                            both the recall denominator and matching (COCO semantics).
            gt_is_dontcare: [B, max_gt] bool, optional. Excluded like crowd GT.
        """
        B = int(num_detections.shape[0])
        for i in range(B):
            n_det = int(num_detections[i])
            n_g   = int(n_gt[i])

            self._n_dt_total += n_det

            if n_g == 0 or n_det == 0:
                # Still count valid (non-crowd/dontcare) GT toward the denominator.
                self._n_gt_total += _count_eval_gt(n_g, i, gt_is_crowd, gt_is_dontcare)
                continue

            # Drop crowd / dontcare GT: they must not inflate the recall denominator
            # nor be matchable (they're un-segmentable / ignore regions).
            keep_gt = _eval_gt_mask(n_g, i, gt_is_crowd, gt_is_dontcare)   # [n_g] bool
            self._n_gt_total += int(keep_gt.sum())

            db = np.asarray(pred_boxes[i, :n_det])              # [n_det, 4]
            gb = np.asarray(gt_boxes[i,  :n_g])[keep_gt]        # [n_keep, 4]
            if gb.shape[0] == 0:
                continue
            gt_keep_idx = np.nonzero(keep_gt)[0]                # map filtered→original
            iou_mat = _bbox_iou_matrix(db, gb)         # [n_det, n_keep]

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
                    self._n_matched += 1

                    # ---- compute mask IoU for this match ----
                    # best_gt indexes the FILTERED GT (iou_mat columns); map back to
                    # the original GT index for the polygon/box lookups in *_polygons.
                    orig_gt = int(gt_keep_idx[best_gt])
                    p_poly = np.asarray(pred_polygons[i, di])        # [24, 3]
                    g_poly = np.asarray(gt_polygons[i, orig_gt])     # [72]

                    # Gate bins on the conf channel. Pred uses the decode/viz
                    # threshold; GT conf is the parser's binary bin validity.
                    # Empty GT bins encode dist=0 — rasterizing them puts a
                    # vertex at the box CENTER per empty bin (center-spiked
                    # star masks; see class docstring).
                    p_keep = np.asarray(p_poly[:, 0]) >= self._conf_thresh  # [24]
                    g_keep = np.asarray(g_poly[2::3]) > 0.5                 # [24]

                    if int(g_keep.sum()) < 3:
                        # Degenerate GT polygon (< 3 occupied bins): the match
                        # counts toward recall but there is no measurable GT
                        # mask for an IoU sample.
                        continue

                    # Prediction: (conf, dist, angle) per vertex. dist normalized;
                    # angle is the sigmoid'd sub-bin offset in [0, 1).
                    p_dist = np.maximum(p_poly[:, 1], 0.0)        # [24]
                    p_off  = np.clip(p_poly[:, 2], 0.0, 1.0)      # [24] sub-bin offset

                    # GT: [dist, angle, conf] x 24. dist at 0::3, sub-bin offset at 1::3.
                    g_dist = np.maximum(g_poly[0::3], 0.0)        # [24]
                    g_off  = np.clip(g_poly[1::3], 0.0, 1.0)      # [24] sub-bin offset

                    # Bbox centres as polygon origins (normalized; scaled to px in helper)
                    y1, x1, y2, x2 = db[di]
                    cx_p = (x1 + x2) / 2.0
                    cy_p = (y1 + y2) / 2.0

                    y1g, x1g, y2g, x2g = gb[best_gt]
                    cx_g = (x1g + x2g) / 2.0
                    cy_g = (y1g + y2g) / 2.0

                    p_verts = _radial_to_cartesian(cx_p, cy_p, p_dist, self._W, self._H, offsets=p_off)[p_keep]
                    g_verts = _radial_to_cartesian(cx_g, cy_g, g_dist, self._W, self._H, offsets=g_off)[g_keep]

                    p_mask = _polygon_to_mask(p_verts, self._H, self._W)
                    g_mask = _polygon_to_mask(g_verts, self._H, self._W)

                    inter = (p_mask & g_mask).sum()
                    union = (p_mask | g_mask).sum()
                    iou   = inter / union if union > 0 else 0.0
                    self._mask_ious.append(float(iou))

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
        # mIoU and recall are decoupled: a matched pair whose gated GT polygon
        # is degenerate (< 3 occupied bins) counts for recall but yields no
        # mask-IoU sample.
        miou    = float(np.mean(self._mask_ious)) if self._mask_ious else 0.0
        recall  = self._n_matched / self._n_gt_total if self._n_gt_total > 0 else 0.0
        return {'poly_mIoU': miou, 'poly_recall50': float(recall)}

    def reset(self) -> None:
        """Clear accumulated data."""
        self._mask_ious.clear()
        self._n_matched  = 0
        self._n_gt_total = 0
        self._n_dt_total = 0
