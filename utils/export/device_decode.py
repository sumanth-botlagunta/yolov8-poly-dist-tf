"""Reconstruct detections from the device-contract SavedModel's flat head outputs.

The single exporter (utils/export/export_saved_model.py) emits the on-device DLC
contract: per-anchor flat tensors instead of post-processed detections. Host tools
that want deploy-style detections from that SavedModel must reproduce the decode the
on-device YoloV8LayerModified applies. This module is that decode, ported from the
device-export validation path, in pure numpy so it imports without TensorFlow.

Device outputs (levels concatenated 3->4->5, batch dim dropped, N anchors):
    box         [N, 4]   DFL-decoded LTRB distances, pre-stride. In [t,l,b,r]
                         (y-first) order when the export used legacy_box_order=True
                         (the export default), else [l,t,r,b].
    cls         [N, C]   raw class logits (pre-sigmoid)
    poly_angle  [N, P]   raw sub-bin offsets (pre-sigmoid)     (optional head)
    poly_dist   [N, P]   raw radial distances (pre-softplus)   (optional head)
    poly_conf   [N, P]   raw per-bin validity (pre-sigmoid)    (optional head)
    dist        [N, 1]   raw log-distance                      (optional head)

reconstruct_detections turns those into the same dict the in-repo deploy path
(models/detection_generator.py::YoloV8Layer) returns, with a leading batch axis of 1:
bbox (yxyx normalized), classes, confidence, num_detections, and — when the polygon
/ distance heads are present — polygons (conf, dist, angle activated) and distance
(metres). All values are numpy arrays; callers wrap them in tf.constant if needed.
"""

import math

import numpy as np

_STRIDES = (8, 16, 32)


def _anchor_grid(h_level, w_level, stride):
    """Anchor centre (cx, cy) pixel coordinates for one FPN level, row-major."""
    ys = (np.arange(h_level) + 0.5) * stride
    xs = (np.arange(w_level) + 0.5) * stride
    gx, gy = np.meshgrid(xs, ys)
    return gx.reshape(-1), gy.reshape(-1)


def _sigmoid(x):
    """Numerically stable elementwise logistic sigmoid (no exp overflow)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _softplus(x):
    """log(1 + exp(x)) without overflow."""
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64))


def _nms(boxes, scores, iou_thresh, max_out, score_thresh):
    """Greedy per-class NMS on yxyx boxes; returns kept indices (score-desc)."""
    idxs = np.where(scores >= score_thresh)[0]
    if idxs.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = idxs[np.argsort(-scores[idxs], kind="stable")]
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(y2 - y1, 0, None) * np.clip(x2 - x1, 0, None)
    keep = []
    while order.size > 0 and len(keep) < max_out:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iy1 = np.maximum(y1[i], y1[rest])
        ix1 = np.maximum(x1[i], x1[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        inter = np.clip(iy2 - iy1, 0, None) * np.clip(ix2 - ix1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def reconstruct_detections(dev_out, model_h, model_w, *, max_boxes=300,
                           score_thresh=0.05, nms_thresh=0.65,
                           min_distance=0.5, max_distance=10.0,
                           legacy_box_order=True):
    """Rebuild the deploy detection dict from one image's device-contract outputs.

    Args:
        dev_out: mapping head name -> array. Must contain 'box' [N,4] and 'cls'
            [N,C]. 'poly_angle'/'poly_dist'/'poly_conf' [N,P] and 'dist' [N,1] are
            decoded when present. Values may be numpy arrays or objects exposing
            ``.numpy()`` (e.g. tf tensors from a SavedModel signature call).
        model_h, model_w: the model input height / width the export was traced at
            (the anchor grid and box normalization use these).
        legacy_box_order: True (the export default) means ``box`` is [t,l,b,r]
            (y-first) and is reordered to [l,t,r,b] before decode; False assumes the
            repo-native [l,t,r,b]. A mismatch transposes every box.

    Returns:
        dict with a leading batch axis of 1: bbox [1,max_boxes,4] yxyx normalized,
        classes [1,max_boxes] int64, confidence [1,max_boxes] float32,
        num_detections [1] int32, and (when the heads are present) polygons
        [1,max_boxes,P,3] (conf,dist,angle activated) and distance [1,max_boxes].
    """
    def _arr(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    box = _arr(dev_out["box"]).astype(np.float64)      # [N,4] pre-stride LTRB
    if legacy_box_order:
        box = box[:, [1, 0, 3, 2]]                     # [t,l,b,r] -> [l,t,r,b]
    cls = _arr(dev_out["cls"]).astype(np.float64)      # [N,C] raw logits
    num_classes = cls.shape[1]

    has_poly = "poly_conf" in dev_out and dev_out["poly_conf"] is not None
    has_dist = "dist" in dev_out and dev_out["dist"] is not None
    if has_poly:
        pa = _arr(dev_out["poly_angle"]).astype(np.float64)   # [N,P] raw
        pd = _arr(dev_out["poly_dist"]).astype(np.float64)
        pc = _arr(dev_out["poly_conf"]).astype(np.float64)
        poly_size = pc.shape[1]
    if has_dist:
        di = _arr(dev_out["dist"]).astype(np.float64).reshape(-1)   # [N] raw log-dist

    # LTRB (pre-stride) -> anchor -> xyxy -> yxyx normalized, per level then concat.
    boxes, off = [], 0
    for stride in _STRIDES:
        h_l, w_l = model_h // stride, model_w // stride
        n = h_l * w_l
        seg = box[off:off + n] * stride
        cx, cy = _anchor_grid(h_l, w_l, stride)
        l, t, r, b = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
        boxes.append(np.stack([(cy - t) / model_h, (cx - l) / model_w,
                               (cy + b) / model_h, (cx + r) / model_w], axis=1))
        off += n
    boxes = np.clip(np.concatenate(boxes, axis=0), 0.0, 1.0)   # [N,4] yxyx norm

    scores = _sigmoid(cls)                                     # [N,C]
    top = scores.argmax(axis=1)
    top_s = scores[np.arange(len(scores)), top]

    # Per-class NMS, tracking the surviving GLOBAL anchor index so the polygon /
    # distance heads can be gathered for exactly the kept boxes.
    sel_box, sel_score, sel_cls, sel_idx = [], [], [], []
    for c in range(num_classes):
        m = np.where(top == c)[0]
        if m.size == 0:
            continue
        keep_local = _nms(boxes[m], top_s[m], nms_thresh, max_boxes, score_thresh)
        if keep_local.size == 0:
            continue
        g = m[keep_local]
        sel_box.append(boxes[g])
        sel_score.append(top_s[g])
        sel_cls.append(np.full(g.shape[0], c, dtype=np.int64))
        sel_idx.append(g)

    if sel_box:
        sb = np.concatenate(sel_box, axis=0)
        ss = np.concatenate(sel_score, axis=0)
        sc = np.concatenate(sel_cls, axis=0)
        sidx = np.concatenate(sel_idx, axis=0)
        order = np.argsort(-ss, kind="stable")[:max_boxes]
        sb, ss, sc, sidx = sb[order], ss[order], sc[order], sidx[order]
    else:
        sb = np.zeros((0, 4), np.float64)
        ss = np.zeros((0,), np.float64)
        sc = np.zeros((0,), np.int64)
        sidx = np.zeros((0,), np.int64)

    k = int(ss.shape[0])
    pad = max_boxes - k
    out = {
        "bbox": np.pad(sb, [[0, pad], [0, 0]]).astype(np.float32)[None],
        "classes": np.pad(sc, [[0, pad]]).astype(np.int64)[None],
        "confidence": np.pad(ss, [[0, pad]]).astype(np.float32)[None],
        "num_detections": np.asarray([k], dtype=np.int32),
    }

    if has_poly:
        if k:
            sel_pa = _sigmoid(pa[sidx])
            sel_pd = _softplus(pd[sidx])
            sel_pc = _sigmoid(pc[sidx])
            poly = np.stack([sel_pc, sel_pd, sel_pa], axis=-1)   # (conf,dist,angle)
        else:
            poly = np.zeros((0, poly_size, 3), np.float64)
        out["polygons"] = np.pad(poly, [[0, pad], [0, 0], [0, 0]]).astype(np.float32)[None]

    if has_dist:
        if k:
            log_di = np.clip(di[sidx], math.log(min_distance), math.log(max_distance))
            dist = np.exp(log_di)
        else:
            dist = np.zeros((0,), np.float64)
        out["distance"] = np.pad(dist, [[0, pad]]).astype(np.float32)[None]

    return out


def is_device_contract(output_keys) -> bool:
    """True if a SavedModel signature's output names are the flat device contract."""
    keys = set(output_keys)
    return "box" in keys and "cls" in keys and "num_detections" not in keys


def is_legacy_contract(output_keys) -> bool:
    """True if the signature's outputs are the (removed) post-processed deploy dict."""
    return "num_detections" in set(output_keys)
