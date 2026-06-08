"""Visualization utilities for TensorBoard image summaries.

Draws predicted bounding boxes and PolyYOLO polygons onto validation images.
Requires opencv-python; if not installed, image summaries are silently skipped.
"""

import math
import logging

import numpy as np

log = logging.getLogger(__name__)

# Distinct BGR colors (OpenCV uses BGR) for up to 80 classes — cycles if more.
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
    """Draw a box and label on a uint8 HWC canvas (in-place)."""
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


def _draw_polygon(canvas, cxn, cyn, poly_conf, poly_dist, n_verts: int = 24) -> None:
    """Draw a PolyYOLO radial polygon on a uint8 HWC canvas (in-place).

    Args:
        cxn, cyn:  Box center in normalized [0,1] coords.
        poly_conf: [n_verts] sigmoid-activated confidences in [0,1].
        poly_dist: [n_verts] predicted radial distance (normalized image space).
        n_verts:   Number of radial vertices (default 24).
    """
    import cv2
    H, W = canvas.shape[:2]
    cx_px = cxn * W
    cy_px = cyn * H

    pts = []
    for i in range(n_verts):
        conf = float(poly_conf[i])   # already sigmoid-activated by detection_generator
        if conf < 0.4:
            continue
        angle_rad = i * (2.0 * math.pi / n_verts)
        d = max(0.0, float(poly_dist[i]))
        # dx/dy in normalized image space → pixels
        px = int(cx_px + d * math.cos(angle_rad) * W)
        py = int(cy_px + d * math.sin(angle_rad) * H)
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
) -> "np.ndarray":
    """Draw predictions onto images and return a uint8 [N, H, W, 3] array.

    Args:
        images:      List of float32 [H, W, 3] arrays with pixel values in [0, 1].
        preds_list:  List of per-image dicts with numpy arrays:
                         bbox           [max_boxes, 4]   yxyx normalized
                         classes        [max_boxes]
                         confidence     [max_boxes]
                         num_detections scalar int
                         polygons       [max_boxes, 24, 3]  (conf, dist, angle) all sigmoid/softplus activated
        draw_box:    Whether to draw bounding boxes.
        draw_poly:   Whether to overlay polygon contours.
        class_names: Optional list of class name strings.

    Returns:
        uint8 numpy array [N, H, W, 3].
    """
    try:
        import cv2  # noqa: F401 — just to check availability
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
                              poly_dist=poly[:, 1])
        out.append(canvas)

    return np.stack(out, axis=0)   # [N, H, W, 3]
