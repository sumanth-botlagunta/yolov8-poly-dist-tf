"""Visualization utilities for TensorBoard image summaries.

Draws predicted bounding boxes and PolyYOLO polygons onto validation images.
Requires opencv-python; if not installed, image summaries are silently skipped.
"""

import math
import logging

import numpy as np

from eval.polygon_metrics import DEFAULT_POLY_CONF_THRESH

log = logging.getLogger(__name__)

# Distinct BGR colors (OpenCV uses BGR); cycles when class ids exceed the palette.
_PALETTE = [
    (  0, 114, 189), ( 60, 160,  75), (255,  65,  54), (148, 103, 189),
    (140,  86,  75), (227, 119, 194), (127, 127, 127), (188, 189,  34),
    ( 23, 190, 207), ( 31, 119, 180), (255, 127,  14), ( 44, 160,  44),
    (214,  39,  40), (148, 103, 189), (140,  86,  75), (227, 119, 194),
]


def _color(cls_id: int):
    return _PALETTE[int(cls_id) % len(_PALETTE)]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _draw_box(canvas, y1n, x1n, y2n, x2n, color, label: str) -> None:
    """Draws a box and label on a uint8 HWC canvas (in place)."""
    import cv2
    H, W = canvas.shape[:2]
    p1 = (int(x1n * W), int(y1n * H))
    p2 = (int(x2n * W), int(y2n * H))
    cv2.rectangle(canvas, p1, p2, color, thickness=2)
    if label:
        ty = max(p1[1] - 5, 12)
        cv2.putText(canvas, label, (p1[0], ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness=1,
                    lineType=cv2.LINE_AA)


def _draw_polygon(canvas, cxn, cyn, poly_conf, poly_dist, poly_angle=None,
                  n_verts: int = 24,
                  conf_thresh: float = DEFAULT_POLY_CONF_THRESH) -> None:
    """Draws a PolyYOLO radial polygon on a uint8 HWC canvas (in place).

    Args:
        cxn, cyn: Box center in normalized [0, 1] coords.
        poly_conf: [n_verts] sigmoid-activated confidences in [0, 1].
        poly_dist: [n_verts] predicted radial distance (normalized image space).
        poly_angle: Optional [n_verts] sub-bin angular offset in [0, 1); the
            vertex angle becomes (i + offset) * (2*pi/n_verts). When None, the
            bin start angle i * (2*pi/n_verts) is used.
        n_verts: Number of radial vertices (default 24).
        conf_thresh: Per-bin confidence gate; bins below it are skipped. Defaults
            to the value PolygonEvaluator scores with (DEFAULT_POLY_CONF_THRESH)
            so the drawn contour matches the scored one.
    """
    import cv2
    H, W = canvas.shape[:2]
    cx_px = cxn * W
    cy_px = cyn * H

    bin_w = 2.0 * math.pi / n_verts
    pts = []
    for i in range(n_verts):
        conf = float(poly_conf[i])   # already sigmoid-activated by detection_generator
        if conf < conf_thresh:
            continue
        off = float(poly_angle[i]) if poly_angle is not None else 0.0
        angle_rad = (i + off) * bin_w
        d = max(0.0, float(poly_dist[i]))
        # Reference convention: the radial vector is origin - vertex, so the
        # vertex is center minus r*(cos, sin), converted from normalized space
        # to pixels.
        px = int(cx_px - d * math.cos(angle_rad) * W)
        py = int(cy_px - d * math.sin(angle_rad) * H)
        pts.append([px, py])

    if len(pts) >= 3:
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_arr], isClosed=True,
                      color=(0, 220, 100), thickness=2, lineType=cv2.LINE_AA)


def render_summary_images(
    images: list,
    preds_list: list,
    draw_box: bool = True,
    draw_poly: bool = True,
    class_names: list = None,
    conf_thresh: float = DEFAULT_POLY_CONF_THRESH,
) -> "np.ndarray":
    """Draws predictions onto images and returns a uint8 [N, H, W, 3] array.

    Args:
        images: List of float32 [H, W, 3] arrays with pixel values in [0, 1].
        preds_list: List of per-image dicts with numpy arrays:
            bbox           [max_boxes, 4]   yxyx normalized
            classes        [max_boxes]
            confidence     [max_boxes]
            num_detections scalar int
            polygons       [max_boxes, 24, 3] (conf, dist, angle), all
                           sigmoid/softplus activated
        draw_box: Whether to draw bounding boxes.
        draw_poly: Whether to overlay polygon contours.
        class_names: Optional list of class name strings.
        conf_thresh: Per-bin polygon confidence gate; defaults to the value
            PolygonEvaluator scores with so the overlay matches the scored
            contour.

    Returns:
        uint8 numpy array [N, H, W, 3], or None if opencv is unavailable.
    """
    try:
        import cv2  # noqa: F401 - availability check only
    except ImportError:
        log.warning("opencv-python not installed — image summaries skipped.")
        return None

    out = []
    for img, pred in zip(images, preds_list):
        canvas = np.clip(img * 255.0, 0, 255).astype(np.uint8).copy()
        nd = int(pred['num_detections'])
        for i in range(nd):
            box    = pred['bbox'][i]          # [y1, x1, y2, x2]
            score  = float(pred['confidence'][i])
            cls_id = int(pred['classes'][i])
            color  = _color(cls_id)

            if draw_box:
                name  = class_names[cls_id] if class_names and cls_id < len(class_names) else f'c{cls_id}'
                label = f'{name}:{score:.2f}'
                _draw_box(canvas, box[0], box[1], box[2], box[3], color, label)

            if draw_poly and 'polygons' in pred:
                poly = pred['polygons'][i]    # [24, 3]: (conf, dist, angle) activated
                cy_n = (box[0] + box[2]) / 2.0
                cx_n = (box[1] + box[3]) / 2.0
                _draw_polygon(canvas, cx_n, cy_n,
                              poly_conf=poly[:, 0],
                              poly_dist=poly[:, 1],
                              poly_angle=poly[:, 2],
                              conf_thresh=conf_thresh)
        out.append(canvas)

    return np.stack(out, axis=0)   # [N, H, W, 3]


def render_gt_images(
    images: list,
    gts_list: list,
    draw_box: bool = True,
    draw_poly: bool = True,
    class_names=None,
) -> "np.ndarray":
    """Draws ground-truth boxes and PolyYOLO polygons onto images.

    Mirrors render_summary_images but consumes the label dicts emitted by the
    parsers, so it serves both the train-augmentation and validation GT
    summaries. Overlaying the post-augmentation GT makes mosaic/affine/flip
    misalignment visible.

    Args:
        images: List of float32 [H, W, 3] arrays with pixel values in [0, 1].
        gts_list: List of per-image GT dicts with numpy arrays:
            bbox     [M, 4]   yxyx normalized
            classes  [M]
            n_gt     scalar int (valid GT count; rows past it are padding)
            polygons [M, 72]  [dist, angle, conf] x 24 interleaved, radial
                     about the box center (optional; absent or all-zero when
                     with_polygons=False)
        draw_box: Whether to draw GT boxes.
        draw_poly: Whether to overlay GT polygons.
        class_names: Optional id->name map (list or dict) used for the box label.

    Returns:
        uint8 numpy array [N, H, W, 3], or None if opencv is unavailable.
    """
    try:
        import cv2  # noqa: F401 - availability check only
    except Exception as e:  # not only ImportError: missing libGL / shadowed cv2
        log.warning("Ground-truth image summaries skipped — cv2 import failed (%r).", e)
        return None

    out = []
    for img, gt in zip(images, gts_list):
        canvas  = np.clip(img * 255.0, 0, 255).astype(np.uint8).copy()
        ng      = int(gt['n_gt'])
        boxes   = gt['bbox']        # [M, 4] yxyx normalized
        classes = gt['classes']     # [M]
        polys   = gt.get('polygons')  # [M, 72] or None
        for i in range(ng):
            box    = boxes[i]       # [y1, x1, y2, x2]
            cls_id = int(classes[i])
            color  = _color(cls_id)

            if draw_box:
                name = (class_names[cls_id]
                        if class_names is not None and cls_id < len(class_names)
                        else f'c{cls_id}')
                _draw_box(canvas, box[0], box[1], box[2], box[3], color, str(name))

            if draw_poly and polys is not None:
                p    = polys[i]              # [72] = [dist, angle, conf] x 24
                cy_n = (box[0] + box[2]) / 2.0
                cx_n = (box[1] + box[3]) / 2.0
                _draw_polygon(canvas, cx_n, cy_n,
                              poly_conf=p[2::3],   # per-bin validity (0/1)
                              poly_dist=p[0::3],   # per-bin radial distance
                              poly_angle=p[1::3])  # per-bin sub-bin offset [0,1)
        out.append(canvas)

    return np.stack(out, axis=0)   # [N, H, W, 3]
