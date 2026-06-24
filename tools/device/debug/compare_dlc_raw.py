"""Compare two SNPE/SavedModel raw-output folders node-by-node, in ONE clean table.

STANDALONE — numpy only, no TensorFlow, no repo imports. Copy this single file to any
machine that has the two Result folders.

Reads each ``<node>:0.raw`` (float32) from an "A" folder and a "B" folder and prints, per
node: element count (A and B), max|diff|, mean|diff|, Pearson correlation, and a verdict
(MATCH / DIFF / SIZE-DIFF / MISSING). Then it reports the FIRST node that diverges — that
is where the two graphs disagree — and decodes the box for a quick geometric sanity check.

Typical use (expected-from-SavedModel  vs  actual-from-DLC):
    # 1) in the TF env, dump expected outputs from the SavedModel on ONE raw image:
    python tools/device/dump_savedmodel_raw.py --saved_model <sm> --raw_image <img.raw> \
        --out_dir /tmp/expected
    # 2) here, diff against the DLC net-run result for the SAME image:
    python tools/device/compare_dlc_raw.py \
        --a /tmp/expected/Result_0 \
        --b <dlc_netrun>/Result_0 \
        --input_size 672,416

A-vs-B read-off:
    all nodes MATCH (corr~1)         -> conversion is FAITHFUL; the F1 gap is the
                                        on-device decode harness or the input bytes.
    a SIZE-DIFF on a tap_backbone_*  -> the spatial size changed (e.g. width not padded);
                                        a padding/stem bug.
    first DIFF at tap_backbone_3     -> backbone (padding/convs). at tap_neck_* -> decoder.
                                        at a head only -> that head's bake.
"""

import argparse
import glob
import os

import numpy as np

# Canonical node order = forward path (taps) then the 6 output heads. Only those present
# in BOTH folders are compared; missing ones are reported, not fatal.
_DEFAULT_NODES = [
    'tap_norm',
    'tap_backbone_3', 'tap_backbone_4', 'tap_backbone_5',
    'tap_neck_3', 'tap_neck_4', 'tap_neck_5',
    'box', 'cls', 'poly_angle', 'poly_dist', 'poly_conf', 'dist',
]


def _find_raw(d, node):
    for name in (f'{node}:0.raw', f'{node}.raw'):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(d, f'{node}*.raw')))
    return g[0] if g else None


def _stats(a):
    return (float(a.min()), float(a.max()), float(a.mean()), float(a.std()))


def _anchors_strides(H, W):
    pts, strd = [], []
    for s in (8, 16, 32):
        h, w = H // s, W // s
        sy = np.arange(h, dtype=np.float32) + 0.5
        sx = np.arange(w, dtype=np.float32) + 0.5
        gy, gx = np.meshgrid(sy, sx, indexing='ij')
        pts.append(np.stack([gy.reshape(-1), gx.reshape(-1)], 1))
        strd.append(np.full((h * w, 1), float(s), np.float32))
    return np.concatenate(pts, 0), np.concatenate(strd, 0)


def _decode_box(box, H, W, box_order='yfirst'):
    """box [N,4] (grid units) -> normalized yxyx (YoloV8LayerModified convention).

    box_order: 'yfirst' ([t,l,b,r] — the legacy/DLC export default, --legacy_box_order) is
    reordered to x-first [l,t,r,b] before decode; 'xfirst' assumes [l,t,r,b]."""
    if box_order == 'yfirst':
        box = box[:, [1, 0, 3, 2]]
    ap, st = _anchors_strides(H, W)
    lt, rb = box[:, :2], box[:, 2:]
    axy = ap[:, ::-1]
    x1y1 = axy - lt
    x2y2 = axy + rb
    yxyx = np.stack([x1y1[:, 1], x1y1[:, 0], x2y2[:, 1], x2y2[:, 0]], 1) * st
    yxyx[:, 0::2] /= H
    yxyx[:, 1::2] /= W
    return yxyx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='folder A (e.g. expected from SavedModel)')
    ap.add_argument('--b', required=True, help='folder B (e.g. DLC net-run result)')
    ap.add_argument('--input_size', default='672,416', help='H,W')
    ap.add_argument('--nodes', default=','.join(_DEFAULT_NODES),
                    help='comma list of node names to compare')
    ap.add_argument('--samples', type=int, default=6, help='sample values to print per node')
    ap.add_argument('--box_order', default='yfirst', choices=['yfirst', 'xfirst'],
                    help="box head order of both raw sets ('yfirst' = legacy/DLC default, "
                         "reordered to x-first before the decoded-box printout; 'xfirst' for "
                         "--legacy_box_order=False). Affects only the decoded-box display, "
                         "not the per-node raw diff.")
    a = ap.parse_args()
    H, W = (int(x) for x in a.input_size.split(','))
    N = sum((H // s) * (W // s) for s in (8, 16, 32))

    print("=" * 100)
    print(" RAW OUTPUT COMPARISON   A vs B")
    print(f"   A = {a.a}")
    print(f"   B = {a.b}")
    print(f"   H x W = {H} x {W}     N(anchors) = {N}")
    print("=" * 100)

    # ---- per-node comparison table ----
    print(f"{'node':16s}{'A.count':>11s}{'B.count':>11s}{'max|diff|':>13s}"
          f"{'mean|diff|':>13s}{'corr':>8s}  verdict")
    print("-" * 100)
    first_diff = None
    box_pair = {}
    for node in a.nodes.split(','):
        pa, pb = _find_raw(a.a, node), _find_raw(a.b, node)
        if not pa or not pb:
            miss = ('A' if not pa else '') + ('B' if not pb else '')
            print(f"{node:16s}{'-':>11s}{'-':>11s}{'-':>13s}{'-':>13s}{'-':>8s}  MISSING-in-{miss}")
            continue
        A = np.fromfile(pa, np.float32)
        B = np.fromfile(pb, np.float32)
        if A.size != B.size:
            print(f"{node:16s}{A.size:>11d}{B.size:>11d}{'-':>13s}{'-':>13s}{'-':>8s}  *** SIZE-DIFF ***")
            if first_diff is None:
                first_diff = (node, 'SIZE-DIFF')
            continue
        d = np.abs(A - B)
        corr = float(np.corrcoef(A, B)[0, 1]) if A.std() > 0 and B.std() > 0 else float('nan')
        match = d.max() <= 1e-3 + 1e-3 * float(np.abs(B).max())
        verdict = "MATCH" if match else ("DIFF <--" if (np.isnan(corr) or corr < 0.999) else "near")
        print(f"{node:16s}{A.size:>11d}{B.size:>11d}{d.max():>13.4e}{d.mean():>13.4e}{corr:>8.4f}  {verdict}")
        if not match and first_diff is None:
            first_diff = (node, f"corr={corr:.4f}")
        if node == 'box' and A.size == N * 4:
            box_pair = {'A': A.reshape(N, 4), 'B': B.reshape(N, 4)}

    print("-" * 100)
    if first_diff:
        print(f" FIRST DIVERGENCE: '{first_diff[0]}'  ({first_diff[1]})")
        print("   -> everything before this matches; the break is at/just before this node.")
    else:
        print(" ALL COMPARED NODES MATCH -> conversion is faithful; look at the on-device")
        print("    decode harness or the input bytes, not the DLC.")
    print("=" * 100)

    # ---- per-node detail (stats + sample values), so you can read them back ----
    print("\nPER-NODE DETAIL (A then B):")
    for node in a.nodes.split(','):
        pa, pb = _find_raw(a.a, node), _find_raw(a.b, node)
        if not pa or not pb:
            continue
        A = np.fromfile(pa, np.float32)
        B = np.fromfile(pb, np.float32)
        print(f"\n  [{node}]  A.count={A.size}  B.count={B.size}")
        amn, amx, amu, asd = _stats(A)
        print(f"     A  min={amn:>10.4f} max={amx:>10.4f} mean={amu:>10.4f} std={asd:>10.4f}")
        if A.size == B.size:
            bmn, bmx, bmu, bsd = _stats(B)
            print(f"     B  min={bmn:>10.4f} max={bmx:>10.4f} mean={bmu:>10.4f} std={bsd:>10.4f}")
            k = min(a.samples, A.size)
            print(f"     A[:{k}] = {np.array2string(A[:k], precision=4, suppress_small=True)}")
            print(f"     B[:{k}] = {np.array2string(B[:k], precision=4, suppress_small=True)}")

    # ---- decoded box geometry ----
    if box_pair:
        print("\nDECODED BOX (ltrb grid -> normalized yxyx):")
        dl = _decode_box(box_pair['A'], H, W, box_order=a.box_order)
        dn = _decode_box(box_pair['B'], H, W, box_order=a.box_order)
        dd = np.abs(dl - dn)
        print(f"   max|diff|={dd.max():.4e}  mean|diff|={dd.mean():.4e}")
        for r in (0, 1, 2):
            print(f"   anchor{r}  A {np.round(dl[r], 4)}   B {np.round(dn[r], 4)}")


if __name__ == '__main__':
    main()
