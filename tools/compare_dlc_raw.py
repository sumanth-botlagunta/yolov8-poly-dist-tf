"""Compare two DLC net-run raw outputs (legacy vs new) node-by-node — all at once.

Reads each `<node>:0.raw` from a legacy Result folder and a new Result folder, and for
every head prints size / max|diff| / mean|diff| / correlation / sample rows, then decodes
the box (ltrb -> normalized yxyx, the YoloV8LayerModified convention: anchor-lt / anchor+rb,
* stride, / image size) and compares the decoded boxes. The head with a large max|diff| /
low correlation is where the two DLCs disagree.

Usage:
    python tools/compare_dlc_raw.py \
        --legacy /path/to/legacy_dsp_result/Result_0 \
        --new    /path/to/new_dsp_result/Result_0 \
        --input_size 672,416            # N (5733) inferred from this
"""

import argparse
import glob
import os

import numpy as np


def _find_raw(d, node):
    for name in (f'{node}:0.raw', f'{node}.raw'):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    g = sorted(glob.glob(os.path.join(d, f'{node}*.raw')))
    return g[0] if g else None


def _anchors_strides(H, W):
    """make_anchor_points convention: levels 8/16/32, +0.5, row-major, anchor=(y,x)."""
    pts, strd = [], []
    for s in (8, 16, 32):
        h, w = H // s, W // s
        sy = np.arange(h, dtype=np.float32) + 0.5
        sx = np.arange(w, dtype=np.float32) + 0.5
        gy, gx = np.meshgrid(sy, sx, indexing='ij')           # [h, w]
        pts.append(np.stack([gy.reshape(-1), gx.reshape(-1)], 1))   # (y, x)
        strd.append(np.full((h * w, 1), float(s), np.float32))
    return np.concatenate(pts, 0), np.concatenate(strd, 0)


def _decode_box(box, H, W):
    """box [N,4]=ltrb (grid units) -> normalized yxyx, as YoloV8LayerModified decodes."""
    ap, st = _anchors_strides(H, W)            # ap=(y,x)
    lt, rb = box[:, :2], box[:, 2:]
    axy = ap[:, ::-1]                          # (x, y)
    x1y1 = axy - lt                            # (x1, y1)
    x2y2 = axy + rb                            # (x2, y2)
    yxyx = np.stack([x1y1[:, 1], x1y1[:, 0], x2y2[:, 1], x2y2[:, 0]], 1) * st  # pixels
    yxyx[:, 0::2] /= H
    yxyx[:, 1::2] /= W
    return yxyx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy', required=True, help='legacy Result_N folder')
    ap.add_argument('--new', required=True, help='new Result_N folder')
    ap.add_argument('--input_size', default='672,416', help='H,W (N inferred)')
    ap.add_argument('--nodes', default='box,cls,poly_angle,poly_dist,poly_conf,dist')
    a = ap.parse_args()
    H, W = (int(x) for x in a.input_size.split(','))
    N = sum((H // s) * (W // s) for s in (8, 16, 32))
    print(f"legacy = {a.legacy}")
    print(f"new    = {a.new}")
    print(f"size   = {H}x{W}   N(anchors) = {N}\n")

    print("================= raw per-node comparison =================")
    box_pair = {}
    for node in a.nodes.split(','):
        pa, pb = _find_raw(a.legacy, node), _find_raw(a.new, node)
        if not pa or not pb:
            print(f"{node:11s} MISSING  (legacy={'ok' if pa else 'NO'}, new={'ok' if pb else 'NO'})")
            continue
        A = np.fromfile(pa, np.float32)
        B = np.fromfile(pb, np.float32)
        if A.size != B.size:
            print(f"{node:11s} *** SIZE DIFF ***  legacy={A.size}  new={B.size}  -> different output shape")
            continue
        C = A.size // N if A.size % N == 0 else 0
        d = np.abs(A - B)
        corr = float(np.corrcoef(A, B)[0, 1]) if A.std() > 0 and B.std() > 0 else float('nan')
        print(f"{node:11s} floats={A.size}  C={C}  max|diff|={d.max():.4e}  mean|diff|={d.mean():.4e}  corr={corr:.4f}")
        if C:
            A2, B2 = A.reshape(N, C), B.reshape(N, C)
            if node == 'box':
                box_pair = {'legacy': A2, 'new': B2}
            for r in (0, 1, 2):
                la = np.array2string(A2[r], precision=3, max_line_width=240, suppress_small=True)
                nb = np.array2string(B2[r], precision=3, max_line_width=240, suppress_small=True)
                print(f"    row{r}  legacy {la}")
                print(f"          new    {nb}")

    if box_pair:
        print("\n=========== decoded box (ltrb -> normalized yxyx) ===========")
        dl = _decode_box(box_pair['legacy'], H, W)
        dn = _decode_box(box_pair['new'], H, W)
        dd = np.abs(dl - dn)
        print(f"decoded-box  max|diff|={dd.max():.4e}  mean|diff|={dd.mean():.4e}")
        for r in (0, 1, 2):
            print(f"    anchor{r}  legacy {np.round(dl[r], 4)}   new {np.round(dn[r], 4)}")

    print("\nRead-off: the head with a large max|diff| and low correlation is where the two")
    print("DLCs disagree. box matches but cls differs -> class head. Everything matches ->")
    print("the gap is in the eval/extraction harness or the input, not the DLC.")


if __name__ == '__main__':
    main()
