"""Recursively compare ALL per-layer .raw outputs of two `snpe-net-run --debug` trees and
rank every layer by divergence — to localize WHICH layer a conversion/quantization fault
first appears in.

STANDALONE — numpy + stdlib only, no TensorFlow, no repo imports. Copy this single file to
the machine that holds the two net-run output trees.

Context
-------
`snpe-net-run --debug` dumps EVERY intermediate layer's output (not just --output_node) for
each input, as float32 .raw files, in subfolders mirroring the layer-name hierarchy (deep).
Run it twice (e.g. A = float/CPU DLC, B = quantized DLC), each with --debug; both trees have
IDENTICAL structure and filenames, so we match every file by its path relative to its Result
folder — no need to know the layer names in advance.

What it does
------------
1. Finds the Result_* folders under --a (recursively; handles <root>/<split>/Result_i), pairs
   each with the same relative path under --b. (If there are no Result_* folders, --a and --b
   are treated as a single pair and compared directly.)
2. Within each Result pair, recursively walks every *.raw, keying each by its path RELATIVE to
   the Result folder = the layer identity (stable across images).
3. Per layer, accumulates across the sampled images: rel_err = max|A-B|/max|A|, Pearson corr,
   max|diff|, and SIZE-DIFF / missing counts.
4. Prints the layers ranked WORST-FIRST (lowest median corr) so the top of the list is where
   the graphs disagree most, and writes a full per-layer CSV (--csv) for offline analysis.

Read-off
--------
  * One early layer flips from corr~1.0 to low, and everything downstream is also bad
        -> the fault originates THERE (first divergence). Read its name for the stage
           (stem/backbone/neck/a specific head op).
  * Divergence grows gradually across many layers, none catastrophic
        -> accumulated quantization error -> global fix (more calibration / --act_bw 16).
  * A SIZE-DIFF on any layer -> a spatial/shape mismatch (padding/reshape) — a hard bug.

Usage
-----
    python tools/device/compare_dlc_debug.py \
        --a /path/cpu_dlc_debug_out  --b /path/quant_dlc_debug_out \
        --max_results 25 --top 50 --csv /tmp/layer_diff.csv
    # only Result_0..Result_200 (even if more exist):  --result_range 0-200
    # widen once localized:  --max_results 0 (all)   --sort name
"""

import argparse
import csv as csvmod
import glob
import os

import numpy as np


def _result_index(path):
    """Integer N parsed from a 'Result_<N>' folder name (-1 if it doesn't parse)."""
    try:
        return int(os.path.basename(path).rsplit('_', 1)[-1])
    except ValueError:
        return -1


def _result_dirs(root):
    """Result_* dirs anywhere under root (flat or split-nested), sorted NUMERICALLY by index
    (so Result_2 precedes Result_10 and a 0-200 range is intuitive). Empty -> single-pair."""
    found = set(glob.glob(os.path.join(root, '**', 'Result_*'), recursive=True))
    found |= set(glob.glob(os.path.join(root, 'Result_*')))
    dirs = [d for d in found if os.path.isdir(d)]
    return sorted(dirs, key=lambda d: (os.path.dirname(d), _result_index(d)))


def _raws(root):
    """Every *.raw under root, as paths relative to root (the layer key)."""
    files = glob.glob(os.path.join(root, '**', '*.raw'), recursive=True)
    return sorted(os.path.relpath(f, root) for f in files)


def _metrics(A, B):
    """(rel_err, corr, max_abs_diff) for two equal-size float32 vectors; corr may be nan."""
    d = np.abs(A - B)
    rel = float(d.max() / (np.abs(A).max() + 1e-12))
    corr = float('nan')
    if A.std() > 0 and B.std() > 0:
        corr = float(np.corrcoef(A, B)[0, 1])
    return rel, corr, float(d.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', required=True, help='root A (e.g. float/CPU DLC --debug output)')
    ap.add_argument('--b', required=True, help='root B (e.g. quantized DLC --debug output)')
    ap.add_argument('--result_range', default=None,
                    help='only Result folders whose index N (Result_<N>) is in this INCLUSIVE '
                         'range, e.g. "0-200" — even if more folders exist. Applied before '
                         '--stride / --max_results.')
    ap.add_argument('--max_results', type=int, default=-1,
                    help='sample at most this many Result folders (0 = all). Default: 25 when no '
                         '--result_range, or ALL-in-range when --result_range is given. A '
                         'diverging layer diverges on every image, so a small sample localizes '
                         'it fast.')
    ap.add_argument('--stride', type=int, default=1, help='use every k-th Result folder')
    ap.add_argument('--corr_thresh', type=float, default=0.99,
                    help='a layer counts as "diverged" when its median corr is below this')
    ap.add_argument('--sort', default='divergence', choices=['divergence', 'name'],
                    help='rank by worst-divergence (default) or by layer path')
    ap.add_argument('--top', type=int, default=50, help='how many layers to print (0 = all)')
    ap.add_argument('--csv', default=None, help='write the full per-layer table here')
    a = ap.parse_args()

    # default cap: 25 normally, but ALL-in-range when an explicit --result_range is given
    max_res = a.max_results if a.max_results >= 0 else (0 if a.result_range else 25)
    rdirs = _result_dirs(a.a)
    if rdirs:
        if a.result_range:
            lo, hi = (int(x) for x in a.result_range.split('-'))
            rdirs = [d for d in rdirs if lo <= _result_index(d) <= hi]
        if a.stride > 1:
            rdirs = rdirs[::a.stride]
        if max_res:
            rdirs = rdirs[:max_res]
        pairs = [(rd, os.path.join(a.b, os.path.relpath(rd, a.a))) for rd in rdirs]
        pairs = [(x, y) for x, y in pairs if os.path.isdir(y)]
        mode = f"{len(pairs)} Result-folder pairs (sampled)"
    else:
        pairs = [(a.a, a.b)]
        mode = "single folder pair (no Result_* found)"
    print(f"A = {a.a}\nB = {a.b}\nmode: {mode}")
    if not pairs:
        raise SystemExit("no matching Result_* folders under both roots (check paths/layout)")

    # layer key -> accumulators
    acc = {}
    n_files_seen = 0
    for i, (ad, bd) in enumerate(pairs):
        for key in _raws(ad):
            bfile = os.path.join(bd, key)
            e = acc.setdefault(key, {'rel': [], 'corr': [], 'mx': [],
                                     'size': 0, 'miss': 0, 'n': 0})
            if not os.path.exists(bfile):
                e['miss'] += 1
                continue
            A = np.fromfile(os.path.join(ad, key), np.float32)
            B = np.fromfile(bfile, np.float32)
            if A.size != B.size or A.size == 0:
                e['size'] += 1
                continue
            rel, corr, mx = _metrics(A, B)
            e['rel'].append(rel); e['mx'].append(mx); e['n'] += 1
            if corr == corr:  # not nan
                e['corr'].append(corr)
            n_files_seen += 1
        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            print(f"  ...processed {i + 1}/{len(pairs)} result folders", flush=True)
    print(f"compared {n_files_seen} layer-files across {len(pairs)} image(s); "
          f"{len(acc)} distinct layers\n")

    # summarize per layer
    rows = []
    for key, e in acc.items():
        if e['n'] == 0:
            rows.append({'layer': key, 'n': 0, 'rel_med': float('nan'), 'rel_p95': float('nan'),
                         'corr_med': float('nan'), 'corr_min': float('nan'),
                         'maxdiff_med': float('nan'), 'size': e['size'], 'miss': e['miss']})
            continue
        rows.append({
            'layer': key, 'n': e['n'],
            'rel_med': float(np.median(e['rel'])),
            'rel_p95': float(np.percentile(e['rel'], 95)),
            'corr_med': float(np.median(e['corr'])) if e['corr'] else float('nan'),
            'corr_min': float(np.min(e['corr'])) if e['corr'] else float('nan'),
            'maxdiff_med': float(np.median(e['mx'])),
            'size': e['size'], 'miss': e['miss'],
        })

    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csvmod.DictWriter(f, fieldnames=['layer', 'n', 'rel_med', 'rel_p95',
                                                 'corr_med', 'corr_min', 'maxdiff_med',
                                                 'size', 'miss'])
            w.writeheader()
            for r in sorted(rows, key=lambda r: r['layer']):
                w.writerow(r)
        print(f"wrote full per-layer table ({len(rows)} layers) -> {a.csv}\n")

    # rank: worst divergence first (lowest corr_med; nan corr -> use rel_med as tiebreak)
    def _badness(r):
        c = r['corr_med']
        c = 2.0 if (c != c) else c        # push nan-corr (constant tensors) to the bottom
        return (c, -(r['rel_med'] if r['rel_med'] == r['rel_med'] else 0))
    ranked = sorted(rows, key=_badness) if a.sort == 'divergence' \
        else sorted(rows, key=lambda r: r['layer'])

    diverged = [r for r in rows if r['corr_med'] == r['corr_med'] and r['corr_med'] < a.corr_thresh]
    sizebad = [r for r in rows if r['size'] and r['n'] == 0]
    print("=" * 108)
    print(f"{'corr_med':>10s}{'corr_min':>10s}{'rel_med':>11s}{'rel_p95':>11s}"
          f"{'maxdiff':>11s}{'n':>5s}  layer")
    print("-" * 108)
    show = ranked if a.top == 0 else ranked[:a.top]
    for r in show:
        cm = r['corr_med']; cn = r['corr_min']
        flag = '  <-- SIZE-DIFF' if (r['size'] and r['n'] == 0) else ''
        print(f"{cm:>10.5f}{cn:>10.5f}{r['rel_med']:>11.3e}{r['rel_p95']:>11.3e}"
              f"{r['maxdiff_med']:>11.3e}{r['n']:>5d}  {r['layer']}{flag}")
    print("=" * 108)

    print("\nREAD-OFF:")
    print(f"  layers compared: {len(rows)}   diverged (median corr < {a.corr_thresh}): "
          f"{len(diverged)}   size-mismatched: {len(sizebad)}")
    if sizebad:
        print(f"  SIZE-DIFF (structural — fix first): {sizebad[0]['layer']}"
              + (f"  (+{len(sizebad)-1} more)" if len(sizebad) > 1 else ""))
    if ranked and ranked[0]['corr_med'] == ranked[0]['corr_med'] \
            and ranked[0]['corr_med'] < a.corr_thresh:
        w = ranked[0]
        print(f"  WORST layer: {w['layer']}")
        print(f"     corr_med={w['corr_med']:.4f}  rel_med={w['rel_med']:.3e}")
        print("  -> read the worst layers' names: if they cluster in one stage (a backbone "
              "block, a neck level, one head op) the fault is THERE; if divergence is mild and "
              "spread over many layers it is accumulated quantization (global fix).")
    else:
        print("  no layer's median corr fell below the threshold — the per-layer graph is "
              "faithful; the gap is decode/threshold sensitivity, not a layer fault.")


if __name__ == '__main__':
    main()
