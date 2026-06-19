"""Build a prediction JSON from DLC (or SavedModel) raw outputs — a faithful, INDEPENDENT
reference for the on-device extraction pipeline.

STANDALONE: numpy + stdlib only (pickle/json). No TensorFlow, no repo imports. Run it in
the environment where the net-run results and the transform pkl live.

Why: host eval = 0.68 but device = 0.18, and we've confirmed the input bytes and the model
are fine (SavedModel detects correctly on the device raws). So the gap is in the on-device
EXTRACTION — the decode and/or the un-letterbox transform back to original coordinates.
This script does both the CORRECT way:

  decode (matches the repo detection_generator / YoloV8LayerModified):
    box[N,4] = DFL-decoded LTRB in grid units (pre-stride)
    anchors per level (strides 8/16/32), grid centers + 0.5, levels concatenated 3->4->5
    x1y1 = anchor - lt ; x2y2 = anchor + rb ; * stride  -> pixels in the 672x416 letterbox
    cls -> sigmoid ; top-1 class per anchor ; per-class greedy NMS

  transform (from <...>_transform_info.pkl, the structure written by the raw generator):
    entry = {sub_dir, file_name, info_ratio, info_tblr=(top,bottom,left,right)}
    x_orig = (x_letterbox_px - left) / ratio ; y_orig = (y_letterbox_px - top) / ratio
    so boxes land in ORIGINAL image pixels, comparable to the GT.

H and W are pinned EXPLICITLY (H=672, W=416). A wrong H/W (the classic non-square swap)
would misplace every box — this reference is swap-free on purpose, so if your pipeline
disagrees, the swap/transform is where your bug is.

Read-off:
  this reference scores ~0.68 on the DLC raws  -> the DLC is fine; your extraction
                                                  (decode or transform) is the bug.
  this reference scores ~0.18 too              -> the DLC output itself is wrong (conversion).

Mapping result folders -> transform keys: net-run processes the input_list in order, so
Result_0,1,2,... correspond to fname_idx '000000','000001',... (the raw generator names
files '%06d' per directory). Use --start_index / --index_from_folder if your layout differs.

NOTE on JSON schema: the geometry (boxes in original coords, xywh) is the important part and
is correct. The entry field names (image_id / category_id / bbox / score) follow COCO; if a
downstream consumer uses different names or a category offset, set --category_offset and
--image_id_field to match.

Splits: the list of split ranges to process is defined in the SPLITS constant below (edit it
there). It is printed at startup. Pass --splits to override it for a one-off run.

Usage:
    python tools/device/gen_pred_json_from_dlc.py \
        --raw_root     /path/to/netrun_output \
        --transform_pkl /path/to/cleaner_eval..._672x416_transform_info.pkl \
        --output_json  /tmp/pred_from_dlc.json \
        --input_size 672,416 --num_classes 39 \
        --conf_threshold 0.001 --nms_iou 0.65
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

_STRIDES = [8, 16, 32]
_NODES = ['box', 'cls', 'poly_angle', 'poly_dist', 'poly_conf', 'dist']

# ---------------------------------------------------------------------------
# Splits to process — EDIT THIS LIST.
# ---------------------------------------------------------------------------
# Each entry is an inclusive range "START-END" of zero-padded global frame indices,
# mirroring the raw generator's directory layout: <raw_root>/<split>/Result_<j>/ with
# the global key i = split_start + j. The list is the source of truth; it is printed at
# startup (it can be long). Pass --splits "A-B,C-D" on the command line to override it for
# a single run without editing this file. Leave SPLITS empty (``[]``) to fall back to a
# flat <raw_root>/Result_* layout.
SPLITS = [
    "000000-000999",
    "001000-001999",
]


def _sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    p = x >= 0
    out[p] = 1.0 / (1.0 + np.exp(-x[p]))
    e = np.exp(x[~p])
    out[~p] = e / (1.0 + e)
    return out


def _anchors(H, W):
    """make_anchor_points: levels 8/16/32, centers + 0.5, row-major (ij), concat 3->4->5.
    Returns anchor centers (x, y) in grid units and the per-anchor stride."""
    axy, st = [], []
    for s in _STRIDES:
        h, w = H // s, W // s
        ys = np.arange(h, dtype=np.float32) + 0.5
        xs = np.arange(w, dtype=np.float32) + 0.5
        gy, gx = np.meshgrid(ys, xs, indexing='ij')
        axy.append(np.stack([gx.reshape(-1), gy.reshape(-1)], 1))   # (x, y)
        st.append(np.full((h * w, 1), float(s), np.float32))
    return np.concatenate(axy, 0), np.concatenate(st, 0)


def _find_raw(d, node):
    for name in (f'{node}:0.raw', f'{node}.raw'):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(d, f'{node}*.raw')))
    return g[0] if g else None


def _greedy_nms(boxes, scores, iou_thr):
    """boxes [M,4] x1y1x2y2 pixels; returns kept indices (greedy, single class)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def _decode(box, cls, H, W, conf_thr, nms_iou, num_classes, max_boxes=300):
    """DFL-decoded box[N,4] + raw cls[N,nc] -> detections in 672x416 PIXEL coords."""
    axy, st = _anchors(H, W)
    lt, rb = box[:, :2], box[:, 2:]
    x1y1 = (axy - lt) * st
    x2y2 = (axy + rb) * st
    xyxy = np.concatenate([x1y1, x2y2], 1)            # pixels in the letterbox image
    scores = _sigmoid(cls.astype(np.float64))
    top = scores.argmax(1)
    top_s = scores[np.arange(len(scores)), top]
    sel_b, sel_s, sel_c = [], [], []
    for c in range(num_classes):
        m = (top == c) & (top_s >= conf_thr)
        if not m.any():
            continue
        cb, cs = xyxy[m], top_s[m]
        idx = _greedy_nms(cb, cs, nms_iou)
        sel_b.append(cb[idx]); sel_s.append(cs[idx])
        sel_c.append(np.full(len(idx), c, np.int64))
    if not sel_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
    b = np.concatenate(sel_b); s = np.concatenate(sel_s); cc = np.concatenate(sel_c)
    order = np.argsort(-s)[:max_boxes]
    return b[order], s[order], cc[order]


def _to_original(xyxy_px, entry, H, W):
    """Un-letterbox: 672x416 pixels -> original image pixels, return xywh."""
    ratio = entry['info_ratio']
    ratio = float(ratio[0] if isinstance(ratio, (list, tuple, np.ndarray)) else ratio)
    top, bottom, left, right = entry['info_tblr']
    X1 = (xyxy_px[:, 0] - left) / ratio
    Y1 = (xyxy_px[:, 1] - top) / ratio
    X2 = (xyxy_px[:, 2] - left) / ratio
    Y2 = (xyxy_px[:, 3] - top) / ratio
    # clip to the original image (derivable from the letterbox: resized = target - pads)
    orig_w = (W - left - right) / ratio
    orig_h = (H - top - bottom) / ratio
    X1 = np.clip(X1, 0, orig_w); X2 = np.clip(X2, 0, orig_w)
    Y1 = np.clip(Y1, 0, orig_h); Y2 = np.clip(Y2, 0, orig_h)
    return np.stack([X1, Y1, X2 - X1, Y2 - Y1], 1)     # xywh in original pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_root', required=True,
                    help='base path. With splits: <raw_root>/<split>/Result_<j>/. '
                         'With empty splits: <raw_root>/Result_0,Result_1,...')
    ap.add_argument('--transform_pkl', required=True)
    ap.add_argument('--output_json', required=True)
    ap.add_argument('--splits', default=None,
                    help="OPTIONAL override for the in-file SPLITS constant: a comma list "
                         "like '000000-000999,001000-001999' (i_key=split_start+j). "
                         "Omit to use SPLITS defined at the top of this file; pass an empty "
                         "string ('') to force the flat Result_* layout.")
    ap.add_argument('--input_size', default='672,416', help='H,W')
    ap.add_argument('--num_classes', type=int, default=39)
    ap.add_argument('--conf_threshold', type=float, default=0.001)
    ap.add_argument('--nms_iou', type=float, default=0.65)
    ap.add_argument('--category_offset', type=int, default=0,
                    help='added to the 0-based class index for category_id (0-based confirmed)')
    ap.add_argument('--image_id_field', default='file_name',
                    choices=['file_name', 'file_stem', 'fname_idx'],
                    help="image_id per entry. 01.gen_pred_json.py uses the original "
                         "file_name (with extension).")
    ap.add_argument('--start_index', type=int, default=0,
                    help='flat mode only: fname_idx of the first Result folder')
    a = ap.parse_args()
    H, W = (int(x) for x in a.input_size.split(','))

    # Resolve the splits to process: --splits overrides the in-file SPLITS constant when
    # given (None means "not passed" -> use SPLITS; '' means "force flat Result_* layout").
    if a.splits is None:
        splits = list(SPLITS)
        splits_source = "in-file SPLITS constant"
    else:
        splits = [s for s in a.splits.split(',') if s]
        splits_source = "--splits override"

    # Print the resolved splits up front (the list can be long) so the run is unambiguous.
    if splits:
        print(f"Processing {len(splits)} split range(s) (from {splits_source}):")
        for sp in splits:
            print(f"  {sp}")
    else:
        print(f"No splits ({splits_source}) — using flat <raw_root>/Result_* layout.")

    with open(a.transform_pkl, 'rb') as f:
        tinfo = pickle.load(f)
    print(f"transform entries: {len(tinfo)}   example key: {sorted(tinfo)[0]!r}")
    ex = tinfo[sorted(tinfo)[0]]
    print(f"example entry keys: {list(ex.keys())}")
    print(f"  info_ratio={ex.get('info_ratio')}  info_tblr={ex.get('info_tblr')}")

    # Build the (i_global, result_dir) work list.
    work = []
    if splits:
        for sp in splits:
            st = int(sp.split('-')[0]); ed = int(sp.split('-')[1]) + 1
            for j in range(0, ed - st):
                work.append((st + j, os.path.join(a.raw_root, sp, 'Result_%d' % j)))
    else:
        results = sorted(glob.glob(os.path.join(a.raw_root, 'Result_*')),
                         key=lambda p: int(p.rsplit('_', 1)[-1]))
        if not results and _find_raw(a.raw_root, 'box'):
            results = [a.raw_root]
        if not results:
            raise SystemExit(f"no Result_* folders or box raw under {a.raw_root}")
        for ri, rdir in enumerate(results):
            work.append((a.start_index + (ri if len(results) > 1 else 0), rdir))
    print(f"work list: {len(work)} images")

    preds = []
    N = sum((H // s) * (W // s) for s in _STRIDES)
    missing_tf = miss_raw = done = 0
    for idx, rdir in work:
        key = '%06d' % idx
        entry = tinfo.get(key)
        if entry is None:
            missing_tf += 1
            continue
        pb, pc = _find_raw(rdir, 'box'), _find_raw(rdir, 'cls')
        if not pb or not pc:
            miss_raw += 1
            continue
        box = np.fromfile(pb, np.float32).reshape(N, 4)
        cls = np.fromfile(pc, np.float32).reshape(N, a.num_classes)
        xyxy, score, klass = _decode(box, cls, H, W, a.conf_threshold, a.nms_iou, a.num_classes)
        done += 1
        if len(xyxy) == 0:
            continue
        xywh = _to_original(xyxy, entry, H, W)
        stem = os.path.splitext(entry['file_name'])[0]
        image_id = {'file_name': entry['file_name'], 'file_stem': stem, 'fname_idx': key}[a.image_id_field]
        for j in range(len(xywh)):
            preds.append({
                'image_id': image_id,
                'bbox': [round(float(v), 3) for v in xywh[j]],
                'category_id': int(klass[j]) + a.category_offset,
                'score': round(float(score[j]), 5),
            })

    with open(a.output_json, 'w') as f:
        json.dump(preds, f)
    print(f"\nwrote {len(preds)} detections over {done} images -> {a.output_json}")
    if missing_tf:
        print(f"WARNING: {missing_tf} index(es) had no transform key (check --splits/--start_index).")
    if miss_raw:
        print(f"WARNING: {miss_raw} result folder(s) had no box/cls raw.")
    print("Entry: {image_id: <file_name>, bbox: [x,y,w,h original px], category_id: <0-based>, "
          "score}. Matches 01.gen_pred_json.py's schema.")


if __name__ == '__main__':
    main()
