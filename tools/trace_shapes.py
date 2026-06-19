"""Compare two model architectures by variable position — no name matching.

Variables from each source are sorted by their normalized name and then
compared index-by-index. If two models have the same architecture (even with
completely different naming), every positional shape pair should be MATCH.

Sources can be a YAML config (builds the model) or a checkpoint path.

Usage
-----
# Default positional compare (sorted by normalized name):
python tools/trace_shapes.py \\
    --src1 configs/experiments/yolo/yolov8_poly_dist.yaml \\
    --src2 initial_checkpoint_folder/ckpt-920304

# Sort both sides by shape before comparing (removes ordering ambiguity):
#   If architectures are identical, every row will be MATCH.
#   Any SHAPE MISMATCH here means a true architectural difference.
python tools/trace_shapes.py \\
    --src1 ... --src2 ... --by-shape

# Shape histogram + module counts only (no per-row table):
python tools/trace_shapes.py \\
    --src1 ... --src2 ... --stats-only

# Filter to backbone only, hide matching rows:
python tools/trace_shapes.py \\
    --src1 ... --src2 ... --filter backbone --only-mismatch

# Save full report:
python tools/trace_shapes.py \\
    --src1 ... --src2 ... --no-colour > report.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure the repo root is importable so ``tools.shared`` resolves when this file is
# run directly as a script (python tools/trace_shapes.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.shared._table import Table, coloured


# ---------------------------------------------------------------------------
# Name normalization (same rules as compare_checkpoints)
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = ["yolo_model/", "model/", "yolov8/", "yolo_v8/"]

_SUBS = [
    ("/Conv2D/kernel",          "/conv/kernel"),
    ("/Conv2D/bias",            "/conv/bias"),
    ("/BatchNorm/",             "/bn/"),
    ("/batch_normalization/",   "/bn/"),
    ("/BatchNormalization/",    "/bn/"),
    ("/bn/moving_average",      "/bn/moving_mean"),
    ("/bn/Momentum",            "/bn/moving_variance"),
]


def _strip_colon_zero(name: str) -> str:
    """Strip the Keras ``:0`` suffix only (not ``str.rstrip(":0")``, which would
    mangle names ending in ``0`` such as ``conv2d_10:0`` -> ``conv2d_1``)."""
    return name[:-2] if name.endswith(":0") else name


def _normalize(name: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = _strip_colon_zero(name)
    for old, new in _SUBS:
        name = name.replace(old, new)
    return name


def _top_module(name: str) -> str:
    """Return the first path segment of the normalized name (e.g. 'backbone')."""
    norm = _normalize(name)
    return norm.split("/")[0]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _is_config(path: str) -> bool:
    return path.endswith(".yaml") or path.endswith(".yml")


def load_source(path: str) -> Tuple[str, List[Tuple[str, tuple]], List[str]]:
    """Load a source and return (label, model_items, scalar_meta_names).

    model_items  — [(name, shape), ...] with shape != (), sorted by normalized name.
    scalar_meta_names — names of shape-() variables (global_step, optimizer scalars).
    """
    if _is_config(path):
        return _load_config(path)
    else:
        return _load_ckpt(path)


def _load_config(config_path: str) -> Tuple[str, List[Tuple[str, tuple]], List[str]]:
    repo_root = str(Path(config_path).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import tensorflow as tf
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8

    cfg   = load_config(config_path)
    model = build_yolov8(cfg.task.model)
    model.deploy = False
    model.build_and_init(cfg.task.model.input_size)

    label = Path(config_path).stem
    items = [(_strip_colon_zero(v.name), tuple(v.shape)) for v in model.variables]
    return label, items, []   # configs have no scalar metadata


def _load_ckpt(ckpt_path: str) -> Tuple[str, List[Tuple[str, tuple]], List[str]]:
    import tensorflow as tf
    reader    = tf.train.load_checkpoint(ckpt_path)
    shape_map = reader.get_variable_to_shape_map()
    skip      = {".OPTIMIZER_SLOT/", "_CHECKPOINTABLE_OBJECT_GRAPH", "save_counter"}

    items: List[Tuple[str, tuple]] = []
    meta:  List[str] = []
    for name, shape in shape_map.items():
        if any(s in name for s in skip):
            continue
        shape_t = tuple(shape)
        if len(shape_t) == 0:
            meta.append(name)          # scalar training metadata — excluded from comparison
        else:
            items.append((name, shape_t))

    items.sort(key=lambda t: _normalize(t[0]))
    meta.sort()
    label = Path(ckpt_path).name
    return label, items, meta


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align(
    items1: List[Tuple[str, tuple]],
    items2: List[Tuple[str, tuple]],
    name_filter: str,
    only_mismatch: bool,
) -> List[dict]:
    """Zip both lists by index position and build comparison rows."""
    from itertools import zip_longest

    rows = []
    sentinel = ("—", ())
    idx = 0

    for (n1, s1), (n2, s2) in zip_longest(items1, items2, fillvalue=sentinel):
        if name_filter and name_filter not in n1 and name_filter not in n2:
            idx += 1
            continue

        if s1 and s2:
            status = "MATCH" if s1 == s2 else "SHAPE MISMATCH"
        elif s1:
            status = "EXTRA"
        else:
            status = "MISSING"

        if only_mismatch and status == "MATCH":
            idx += 1
            continue

        rows.append(dict(idx=idx, n1=n1, s1=s1, n2=n2, s2=s2, status=status))
        idx += 1

    return rows


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_scalar_meta(
    meta1: List[str],
    meta2: List[str],
    label1: str,
    label2: str,
    use_colour: bool,
) -> None:
    """Print scalar () metadata variables found in either source."""
    all_names = sorted(set(meta1) | set(meta2))
    if not all_names:
        return
    set1, set2 = set(meta1), set(meta2)
    w = max((len(n) for n in all_names), default=20)
    w = max(w, 20)
    print(f"\n{'Scalar metadata variables (shape=(), excluded from comparison)':}")
    print(f"  {'Name':<{w}}  {label1:>10}  {label2:>10}")
    print(f"  {'-'*w}  {'-'*10}  {'-'*10}")
    for name in all_names:
        in1 = "✓" if name in set1 else "—"
        in2 = "✓" if name in set2 else "—"
        print(f"  {name:<{w}}  {in1:>10}  {in2:>10}")


def print_module_counts(
    items1: List[Tuple[str, tuple]],
    items2: List[Tuple[str, tuple]],
    label1: str,
    label2: str,
    use_colour: bool,
) -> None:
    """Print variable counts broken down by top-level module (backbone/decoder/head/…)."""
    c1: Counter = Counter(_top_module(n) for n, _ in items1)
    c2: Counter = Counter(_top_module(n) for n, _ in items2)
    all_modules = sorted(set(c1) | set(c2))

    w = max((len(m) for m in all_modules), default=10)
    w = max(w, len(label1), len(label2), 10)

    print(f"\n{'Module counts by top-level prefix':}")
    print(f"  {'Module':<{w}}  {label1:>10}  {label2:>10}  {'Status':>8}")
    print(f"  {'-'*w}  {'-'*10}  {'-'*10}  {'-'*8}")
    for mod in all_modules:
        n1, n2 = c1.get(mod, 0), c2.get(mod, 0)
        ok = n1 == n2
        status = "OK" if ok else "DIFF"
        colour = None if ok else "red"
        line = f"  {mod:<{w}}  {n1:>10}  {n2:>10}  {status:>8}"
        print(coloured(line, colour, use_colour))
    total1, total2 = len(items1), len(items2)
    ok = total1 == total2
    colour = None if ok else "red"
    print(f"  {'-'*w}  {'-'*10}  {'-'*10}  {'-'*8}")
    line = f"  {'TOTAL':<{w}}  {total1:>10}  {total2:>10}  {'OK' if ok else 'DIFF':>8}"
    print(coloured(line, colour, use_colour))


def print_shape_histogram(
    items1: List[Tuple[str, tuple]],
    items2: List[Tuple[str, tuple]],
    label1: str,
    label2: str,
    use_colour: bool,
) -> None:
    """Print a side-by-side count of every unique shape in both sources.

    Identical counts → architectures have the same multiset of shapes.
    Any DIFF row → one source has a shape the other does not, or in different quantity.
    """
    hist1: Counter = Counter(str(s) for _, s in items1)
    hist2: Counter = Counter(str(s) for _, s in items2)
    all_shapes = sorted(set(hist1) | set(hist2))

    w = max((len(s) for s in all_shapes), default=20)
    w = max(w, 20)

    print(f"\n{'Shape histogram (unique shapes × count)':}")
    print(f"  {'Shape':<{w}}  {label1:>10}  {label2:>10}  {'Status':>8}")
    print(f"  {'-'*w}  {'-'*10}  {'-'*10}  {'-'*8}")
    diff_count = 0
    for shape in all_shapes:
        n1, n2 = hist1.get(shape, 0), hist2.get(shape, 0)
        ok = n1 == n2
        if not ok:
            diff_count += 1
        status = "OK" if ok else "DIFF"
        colour = None if ok else "red"
        line = f"  {shape:<{w}}  {n1:>10}  {n2:>10}  {status:>8}"
        print(coloured(line, colour, use_colour))

    print()
    if diff_count == 0:
        msg = "Shape histograms are IDENTICAL — any row mismatches above are purely ordering artifacts."
        print(coloured(f"  ✓ {msg}", "green", use_colour))
    else:
        msg = f"{diff_count} shape(s) differ in count — true architectural difference exists."
        print(coloured(f"  ✗ {msg}", "red", use_colour))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Positional shape comparison with shape-histogram diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--src1", required=True,
                    help="First source: YAML config path or checkpoint path prefix")
    ap.add_argument("--src2", required=True,
                    help="Second source: YAML config path or checkpoint path prefix")
    ap.add_argument("--filter", default="",
                    help="Only show rows where either name contains this substring")
    ap.add_argument("--only-mismatch", action="store_true",
                    help="Hide MATCH rows, show only mismatches and extras")
    ap.add_argument("--by-shape", action="store_true",
                    help="Sort both sources by shape before comparing (removes ordering ambiguity). "
                         "MATCH everywhere = architectures are identical.")
    ap.add_argument("--stats-only", action="store_true",
                    help="Skip the per-row table; only print module counts and shape histogram.")
    ap.add_argument("--no-colour", action="store_true",
                    help="Disable ANSI colours (useful for file output)")
    args = ap.parse_args()

    use_colour = not args.no_colour

    print(f"Loading src1: {args.src1}")
    label1, items1, meta1 = load_source(args.src1)
    print(f"  {len(items1)} model variables  ({len(meta1)} scalar metadata excluded)"
          f"  (source type: {'config' if _is_config(args.src1) else 'checkpoint'})")

    print(f"Loading src2: {args.src2}")
    label2, items2, meta2 = load_source(args.src2)
    print(f"  {len(items2)} model variables  ({len(meta2)} scalar metadata excluded)"
          f"  (source type: {'config' if _is_config(args.src2) else 'checkpoint'})\n")

    # ---- always print diagnostics first ----
    print_module_counts(items1, items2, label1, label2, use_colour)
    print_shape_histogram(items1, items2, label1, label2, use_colour)

    if args.stats_only:
        print_scalar_meta(meta1, meta2, label1, label2, use_colour)
        return

    if len(items1) != len(items2):
        print(f"\nWARNING: variable counts differ ({len(items1)} vs {len(items2)}) — "
              f"extra rows will be marked EXTRA / MISSING")

    # ---- optional: sort by shape to remove ordering bias ----
    if args.by_shape:
        print(f"\n[--by-shape] Sorting both sources by (shape, normalized_name) before comparison.")
        items1 = sorted(items1, key=lambda t: (str(t[1]), _normalize(t[0])))
        items2 = sorted(items2, key=lambda t: (str(t[1]), _normalize(t[0])))

    rows = align(items1, items2, args.filter, args.only_mismatch)

    if not rows:
        print("\nNo rows to display (all matched or all filtered out).")
        return

    print()
    tbl = Table(label1, label2, show_index=True, use_colour=use_colour)
    tbl.header()

    counts: Dict[str, int] = {}
    for r in rows:
        tbl.row(r["n1"], r["s1"], r["n2"], r["s2"], r["status"], idx=r["idx"])
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    tbl.footer()

    total = max(len(items1), len(items2))
    tbl.summary(counts, total)

    print_scalar_meta(meta1, meta2, label1, label2, use_colour)


if __name__ == "__main__":
    main()
