"""Aggregate node-by-node raw-output differences across MANY Result folders — the
distribution-level version of ``compare_dlc_raw.py``, with forward-path TAP support.

STANDALONE — numpy only, no TensorFlow, no repo imports. Copy this single file anywhere the
two net-run output trees live.

Where ``compare_dlc_raw`` diffs ONE image (one ``Result_N`` pair), this walks every
``Result_*`` folder under two roots (handling the per-split nesting
``<root>/<split>/Result_j``), pairs them by identical relative path, and aggregates per node.

NODES compared, in FORWARD order (so the FIRST divergence localizes the fault):
    tap_norm, tap_backbone_3/4/5, tap_neck_3/4/5,   <- forward-path taps (export with
    box, cls, poly_angle, poly_dist, poly_conf, dist   --debug_taps; net-run must emit them)
Taps get rel_err + corr only (they are feature maps); box/cls additionally get the
decode-aware metrics. Taps that are absent in a folder are simply skipped (counted), so this
same tool works whether or not the DLC was exported with --debug_taps.

Per node it reports:
  * rel_err = max|A-B| / max|A|        (median + p95 across images) — range/bit-width loss
  * corr    = Pearson(A, B)            (median) — shape fidelity
  * SIZE-DIFF count                    — a spatial-shape mismatch (e.g. a padding/stem bug)
  * decode-aware impact on CANDIDATE anchors (sigmoid(top-1 cls) >= --cand_thresh):
      cls: flip% (top-1 class changes A->B) + score drift ; box: decoded-IoU A-vs-B
Then it names the FIRST node (forward order) whose median corr < --corr_thresh (or that
SIZE-DIFFs) — everything before it matches, so the break is at/just before that node:
    first divergence at tap_backbone_* -> backbone (stem/convs/padding)
                        tap_neck_*      -> decoder (FPN-PAN)
                        a head only     -> that head's bake (DFL / sigmoid / concat)

Use it for either gap:
  conversion:    A = SavedModel-dumped raws,  B = CPU/float DLC raws
  quantization:  A = CPU/float DLC raws,       B = quantized DLC raws

Usage:
    python tools/device/compare_dlc_raw_batch.py \
        --a_root /path/A_netrun  --b_root /path/B_netrun \
        --input_size 672,416 --num_classes 39 --box_order yfirst \
        --max_images 1000 --corr_thresh 0.99
"""

import argparse
import glob
import os

import numpy as np

_STRIDES = (8, 16, 32)
# Forward path first (taps), then the 6 output heads. The order matters: the first node that
# diverges localizes the fault. Taps appear only when exported with --debug_taps.
_TAPS = ['tap_norm', 'tap_backbone_3', 'tap_backbone_4', 'tap_backbone_5',
         'tap_neck_3', 'tap_neck_4', 'tap_neck_5']
_HEADS = ['box', 'cls', 'poly_angle', 'poly_dist', 'poly_conf', 'dist']
_DEFAULT_NODES = _TAPS + _HEADS


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
    ap.add_argument('--a_root', required=True, help='root A (e.g. SavedModel-dump or CPU DLC)')
    ap.add_argument('--b_root', required=True, help='root B (e.g. CPU DLC or quantized DLC)')
    ap.add_argument('--input_size', default='672,416', help='H,W')
    ap.add_argument('--num_classes', type=int, default=39)
    ap.add_argument('--box_order', default='yfirst', choices=['yfirst', 'xfirst'],
                    help="box head order ('yfirst' = legacy/DLC default, reordered before decode)")
    ap.add_argument('--nodes', default=None,
                    help='comma list of nodes to compare (default: the taps + 6 heads, in '
                         'forward order). Absent nodes are skipped, so this is safe whether or '
                         'not the DLC was exported with --debug_taps.')
    ap.add_argument('--corr_thresh', type=float, default=0.99,
                    help='a node is "diverged" when its MEDIAN corr drops below this; the first '
                         'such node (forward order) is reported as the break point')
    ap.add_argument('--cand_thresh', type=float, default=0.05,
                    help='anchor is a CANDIDATE if sigmoid(top-1 cls of A) >= this '
                         '(only candidates affect F1; flip%%/iou measured on them)')
    ap.add_argument('--max_images', type=int, default=0, help='cap images processed (0 = all)')
    ap.add_argument('--stride', type=int, default=1, help='process every k-th Result folder')
    a = ap.parse_args()
    H, W = (int(x) for x in a.input_size.split(','))
    N = sum((H // s) * (W // s) for s in _STRIDES)
    axy, st = _anchors(H, W)
    nodes = [n for n in (a.nodes.split(',') if a.nodes else _DEFAULT_NODES) if n]

    pairs = _pairs(a.a_root, a.b_root)
    if a.stride > 1:
        pairs = pairs[::a.stride]
    if a.max_images:
        pairs = pairs[:a.max_images]
    print(f"A = {a.a_root}\nB = {a.b_root}")
    print(f"matched {len(pairs)} Result-folder pairs   N(anchors)={N}   "
          f"cand_thresh={a.cand_thresh}  corr_thresh={a.corr_thresh}  box_order={a.box_order}")
    print(f"nodes (forward order): {', '.join(nodes)}\n")
    if not pairs:
        raise SystemExit("no matching Result_* folders under both roots (check the layout/paths)")

    acc = {n: {'rel': [], 'corr': [], 'sizediff': 0, 'missing': 0, 'present': 0} for n in nodes}
    if 'cls' in acc:
        acc['cls'].update({'flip': [], 'score': []})
    if 'box' in acc:
        acc['box'].update({'iou': []})
    used = 0
    for ad, bd in pairs:
        arrs = {}
        touched = False
        for n in nodes:
            pa, pb = _find_raw(ad, n), _find_raw(bd, n)
            if not pa or not pb:
                acc[n]['missing'] += 1
                continue
            A = np.fromfile(pa, np.float32); B = np.fromfile(pb, np.float32)
            if A.size != B.size:
                acc[n]['sizediff'] += 1
                continue
            acc[n]['present'] += 1
            touched = True
            d = np.abs(A - B)
            acc[n]['rel'].append(float(d.max() / (np.abs(A).max() + 1e-9)))
            if A.std() > 0 and B.std() > 0:
                acc[n]['corr'].append(float(np.corrcoef(A, B)[0, 1]))
            arrs[n] = (A, B)
        if touched:
            used += 1

        # decode-aware impact on CANDIDATE anchors (needs cls + box at head sizes)
        if ('cls' in arrs and 'box' in arrs
                and arrs['cls'][0].size == N * a.num_classes and arrs['box'][0].size == N * 4):
            clsA = arrs['cls'][0].reshape(N, a.num_classes)
            clsB = arrs['cls'][1].reshape(N, a.num_classes)
            sA = _sigmoid(clsA); topA = sA.argmax(1); topAs = sA[np.arange(N), topA]
            cand = topAs >= a.cand_thresh
            if cand.any():
                sB = _sigmoid(clsB); topB = sB.argmax(1); topBs = sB[np.arange(N), topB]
                acc['cls']['flip'].append(float((topA[cand] != topB[cand]).mean()))
                acc['cls']['score'].append(float(np.abs(topAs[cand] - topBs[cand]).mean()))
                bA = _decode_xyxy(arrs['box'][0].reshape(N, 4), axy, st, a.box_order)
                bB = _decode_xyxy(arrs['box'][1].reshape(N, 4), axy, st, a.box_order)
                acc['box']['iou'].append(float(_iou_rows(bA[cand], bB[cand]).mean()))

    print(f"processed {used} pairs\n")
    print("=" * 100)
    print(f"{'node':16s}{'n_img':>7s}{'rel_err(med)':>13s}{'rel_err(p95)':>13s}"
          f"{'corr(med)':>11s}{'size!':>6s}{'decode-aware impact':>34s}")
    print("-" * 100)
    for n in nodes:
        e = acc[n]
        if e['present'] == 0:
            tag = 'SIZE-DIFF' if e['sizediff'] else 'absent'
            print(f"{n:16s}{0:>7d}{'-':>13s}{'-':>13s}{'-':>11s}{e['sizediff']:>6d}  {tag}")
            continue
        cmed = np.median(e['corr']) if e['corr'] else float('nan')
        extra = ''
        if n == 'cls' and acc['cls'].get('flip'):
            extra = (f"flip%={100*np.median(acc['cls']['flip']):5.2f} "
                     f"s|d|={np.median(acc['cls']['score']):.3f}")
        elif n == 'box' and acc['box'].get('iou'):
            extra = f"iou_dn={np.median(acc['box']['iou']):.3f}"
        print(f"{n:16s}{e['present']:>7d}{np.median(e['rel']):>13.3e}{_pct(e['rel'],95):>13.3e}"
              f"{cmed:>11.5f}{e['sizediff']:>6d}{extra:>34s}")
    print("=" * 100)

    # First divergence in forward order: a SIZE-DIFF, or median corr < threshold.
    first = None
    for n in nodes:
        e = acc[n]
        if e['sizediff'] and e['present'] == 0:
            first = (n, f"SIZE-DIFF on {e['sizediff']} images"); break
        if e['corr'] and np.median(e['corr']) < a.corr_thresh:
            first = (n, f"median corr {np.median(e['corr']):.4f} < {a.corr_thresh}"); break
    print("\nREAD-OFF:")
    if first:
        print(f"  FIRST DIVERGENCE: '{first[0]}'  ({first[1]})")
        print("    -> nodes before it match; the fault is at/just before this node.")
        if first[0].startswith('tap_backbone'):
            print("    -> backbone (stem / convs / padding).")
        elif first[0].startswith('tap_neck'):
            print("    -> decoder (FPN-PAN).")
        elif first[0] == 'tap_norm':
            print("    -> input normalization / the baked /255 (or the input bytes).")
        elif first[0] in _HEADS:
            print(f"    -> the '{first[0]}' head's bake only (everything upstream is faithful).")
    else:
        print(f"  No node's median corr fell below {a.corr_thresh}: the forward path is faithful;")
        print("    any F1 gap is decode/threshold sensitivity, not a graph divergence.")
    if acc.get('cls', {}).get('flip'):
        print(f"  cls top-1 flips on {100*np.median(acc['cls']['flip']):.2f}% of candidate "
              f"anchors (median).")
    if acc.get('box', {}).get('iou'):
        m = np.median(acc['box']['iou'])
        print(f"  box decode IoU A-vs-B on candidate anchors: {m:.4f} (median) "
              f"-> {'stable' if m > 0.95 else 'boxes MOVING'}.")
    if not any(acc[n]['present'] for n in _TAPS if n in acc):
        print("  (no tap_* nodes found — export with --debug_taps and emit them in net-run to "
              "localize WHICH stage diverges, not just which head.)")


if __name__ == '__main__':
    main()
