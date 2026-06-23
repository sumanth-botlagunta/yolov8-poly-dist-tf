"""Aggregate node-by-node raw-output differences across MANY Result folders — the
distribution-level version of ``compare_dlc_raw.py``.

STANDALONE — numpy only, no TensorFlow, no repo imports. Copy this single file anywhere the
two net-run output trees live.

Where ``compare_dlc_raw`` diffs ONE image (one ``Result_N`` pair), this walks every
``Result_*`` folder under two roots (handling the per-split nesting
``<root>/<split>/Result_j``), pairs them by identical relative path, and reports, per head:

  * rel_err   = max|A-B| / max|A|        (median + p95 across images) — range/bit-width loss
  * corr      = Pearson(A, B)            (median) — shape fidelity
  * and the DECODE-AWARE impact, which is what actually moves F1:
      cls:  flip%   = fraction of CANDIDATE anchors (sigmoid(top-1) >= --cand_thresh)
                      whose top-1 CLASS changes A->B  (median across images)
            score|d|= mean |sigmoid(top1_A) - sigmoid(top1_B)| over candidate anchors
      box:  iou_dn  = mean IoU between A-decoded and B-decoded boxes on candidate anchors
                      (1.0 = identical; lower = boxes moved)

Use it to decide WHERE quantization hurts:
  A = CPU (float) DLC raws,  B = quantized DLC raws  -> the head with the worst flip%/iou_dn
  is the one to protect (e.g. keep it int16). Broad rel_err across all heads -> raise
  activation bit-width / grow calibration globally.

Pairing: every ``Result_*`` dir under --a_root is matched to the same relative path under
--b_root (so split layout and Result index must line up — they do when both DLCs ran the
same input_list). Missing/size-mismatched nodes are counted and skipped, not fatal.

Usage:
    python tools/device/compare_dlc_raw_batch.py \
        --a_root /path/cpu_dlc_netrun  --b_root /path/quant_dlc_netrun \
        --input_size 672,416 --num_classes 39 --box_order yfirst \
        --max_images 1000
"""

import argparse
import glob
import os

import numpy as np

_STRIDES = (8, 16, 32)
_HEADS = ['box', 'cls', 'poly_angle', 'poly_dist', 'poly_conf', 'dist']
_CHANS = {'box': 4, 'cls': None, 'poly_angle': 24, 'poly_dist': 24, 'poly_conf': 24, 'dist': 1}


def _find_raw(d, node):
    for name in (f'{node}:0.raw', f'{node}.raw'):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(d, f'{node}*.raw')))
    return g[0] if g else None


def _sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    p = x >= 0
    out[p] = 1.0 / (1.0 + np.exp(-x[p]))
    e = np.exp(x[~p])
    out[~p] = e / (1.0 + e)
    return out


def _anchors(H, W):
    """centers (x, y) in grid units + per-anchor stride, levels 8/16/32 concat 3->4->5."""
    axy, st = [], []
    for s in _STRIDES:
        h, w = H // s, W // s
        ys = np.arange(h, dtype=np.float32) + 0.5
        xs = np.arange(w, dtype=np.float32) + 0.5
        gy, gx = np.meshgrid(ys, xs, indexing='ij')
        axy.append(np.stack([gx.reshape(-1), gy.reshape(-1)], 1))
        st.append(np.full((h * w, 1), float(s), np.float32))
    return np.concatenate(axy, 0), np.concatenate(st, 0)


def _decode_xyxy(box, axy, st, box_order):
    """box[N,4] grid-units -> xyxy pixels. yfirst ([t,l,b,r]) reordered to x-first first."""
    if box_order == 'yfirst':
        box = box[:, [1, 0, 3, 2]]
    lt, rb = box[:, :2], box[:, 2:]
    x1y1 = (axy - lt) * st
    x2y2 = (axy + rb) * st
    return np.concatenate([x1y1, x2y2], 1)


def _iou_rows(a, b):
    """elementwise IoU of two [M,4] xyxy box sets (paired rows)."""
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2]); y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    ab = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return inter / (aa + ab - inter + 1e-9)


def _pairs(a_root, b_root):
    a_dirs = sorted(set(glob.glob(os.path.join(a_root, '**', 'Result_*'), recursive=True))
                    | set(glob.glob(os.path.join(a_root, 'Result_*'))))
    out = []
    for ad in a_dirs:
        if not os.path.isdir(ad):
            continue
        bd = os.path.join(b_root, os.path.relpath(ad, a_root))
        if os.path.isdir(bd):
            out.append((ad, bd))
    return out


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a_root', required=True, help='root A (e.g. CPU/float DLC net-run tree)')
    ap.add_argument('--b_root', required=True, help='root B (e.g. quantized DLC net-run tree)')
    ap.add_argument('--input_size', default='672,416', help='H,W')
    ap.add_argument('--num_classes', type=int, default=39)
    ap.add_argument('--box_order', default='yfirst', choices=['yfirst', 'xfirst'],
                    help="box head order ('yfirst' = legacy/DLC default, reordered before decode)")
    ap.add_argument('--cand_thresh', type=float, default=0.05,
                    help='an anchor is a CANDIDATE if sigmoid(top-1 cls of A) >= this '
                         '(only candidates affect F1; flips/iou are measured on them)')
    ap.add_argument('--max_images', type=int, default=0, help='cap images processed (0 = all)')
    ap.add_argument('--stride', type=int, default=1, help='process every k-th Result folder')
    a = ap.parse_args()
    H, W = (int(x) for x in a.input_size.split(','))
    N = sum((H // s) * (W // s) for s in _STRIDES)
    _CHANS['cls'] = a.num_classes
    axy, st = _anchors(H, W)

    pairs = _pairs(a.a_root, a.b_root)
    if a.stride > 1:
        pairs = pairs[::a.stride]
    if a.max_images:
        pairs = pairs[:a.max_images]
    print(f"A = {a.a_root}\nB = {a.b_root}")
    print(f"matched {len(pairs)} Result-folder pairs   N(anchors)={N}   "
          f"cand_thresh={a.cand_thresh}  box_order={a.box_order}\n")
    if not pairs:
        raise SystemExit("no matching Result_* folders under both roots (check the layout/paths)")

    acc = {n: {'rel': [], 'corr': []} for n in _HEADS}
    acc['cls'].update({'flip': [], 'score': []})
    acc['box'].update({'iou': []})
    used = miss = 0
    for ad, bd in pairs:
        node_arr = {}
        ok = True
        for n in _HEADS:
            pa, pb = _find_raw(ad, n), _find_raw(bd, n)
            if not pa or not pb:
                ok = False; break
            A = np.fromfile(pa, np.float32); B = np.fromfile(pb, np.float32)
            if A.size != B.size or A.size != N * _CHANS[n]:
                ok = False; break
            node_arr[n] = (A, B)
        if not ok:
            miss += 1
            continue
        used += 1

        for n in _HEADS:
            A, B = node_arr[n]
            d = np.abs(A - B)
            acc[n]['rel'].append(float(d.max() / (np.abs(A).max() + 1e-9)))
            if A.std() > 0 and B.std() > 0:
                acc[n]['corr'].append(float(np.corrcoef(A, B)[0, 1]))

        # decode-aware impact on CANDIDATE anchors (the ones that become detections)
        clsA = node_arr['cls'][0].reshape(N, a.num_classes)
        clsB = node_arr['cls'][1].reshape(N, a.num_classes)
        sA = _sigmoid(clsA); topA = sA.argmax(1); topAs = sA[np.arange(N), topA]
        cand = topAs >= a.cand_thresh
        if cand.any():
            sB = _sigmoid(clsB); topB = sB.argmax(1); topBs = sB[np.arange(N), topB]
            acc['cls']['flip'].append(float((topA[cand] != topB[cand]).mean()))
            acc['cls']['score'].append(float(np.abs(topAs[cand] - topBs[cand]).mean()))
            boxA = _decode_xyxy(node_arr['box'][0].reshape(N, 4), axy, st, a.box_order)
            boxB = _decode_xyxy(node_arr['box'][1].reshape(N, 4), axy, st, a.box_order)
            acc['box']['iou'].append(float(_iou_rows(boxA[cand], boxB[cand]).mean()))

    print(f"processed {used} pairs ({miss} skipped: missing/size-mismatch nodes)\n")
    print("=" * 92)
    print(f"{'head':12s}{'rel_err(med)':>13s}{'rel_err(p95)':>13s}{'corr(med)':>11s}"
          f"{'decode-aware impact':>42s}")
    print("-" * 92)
    for n in _HEADS:
        rel = acc[n]['rel']; corr = acc[n]['corr']
        extra = ''
        if n == 'cls' and acc['cls']['flip']:
            extra = (f"flip%(med)={100*np.median(acc['cls']['flip']):6.2f}   "
                     f"score|d|(med)={np.median(acc['cls']['score']):.4f}")
        elif n == 'box' and acc['box']['iou']:
            extra = f"iou_dn(med)={np.median(acc['box']['iou']):.4f}  (1.0=identical)"
        print(f"{n:12s}{np.median(rel):>13.4e}{_pct(rel,95):>13.4e}"
              f"{(np.median(corr) if corr else float('nan')):>11.5f}{extra:>42s}")
    print("=" * 92)
    # verdict: rank heads by decode-aware impact where available, else by rel_err
    print("\nREAD-OFF:")
    if acc['cls']['flip']:
        print(f"  cls top-1 flips on {100*np.median(acc['cls']['flip']):.2f}% of candidate "
              f"anchors (median) -> detections appearing/disappearing or relabeled.")
    if acc['box']['iou']:
        m = np.median(acc['box']['iou'])
        print(f"  box decode IoU A-vs-B on candidate anchors: {m:.4f} (median) "
              f"-> {'boxes stable' if m > 0.95 else 'boxes MOVING under quantization'}.")
    worst = max(_HEADS, key=lambda n: (np.median(acc[n]['rel']) if acc[n]['rel'] else 0))
    print(f"  highest raw rel-error head: '{worst}'. If one head dominates flip%/iou/rel_err,")
    print(f"  keep THAT head higher precision (int16); if all heads have high rel_err, the")
    print(f"  fix is global (more calibration images + --use_enhanced_quantizer / --act_bw 16).")


if __name__ == '__main__':
    main()
