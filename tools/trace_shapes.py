"""Compare two model architectures by variable position — no name matching.

Variables from each source are sorted by their normalized name and then
compared index-by-index. If two models have the same architecture (even with
completely different naming), every positional shape pair should be MATCH.

Sources can be a YAML config (builds the model) or a checkpoint path.

Usage
-----
# Two configs — build both, compare variable order:
python tools/trace_shapes.py \\
    --src1 configs/experiments/yolo/yolov8_poly_dist.yaml \\
    --src2 configs/other_model.yaml

# Config vs checkpoint (e.g. init ckpt):
python tools/trace_shapes.py \\
    --src1 configs/experiments/yolo/yolov8_poly_dist.yaml \\
    --src2 initial_checkpoint_folder/ckpt-920304

# Two checkpoints:
python tools/trace_shapes.py \\
    --src1 ckpt-920304 --src2 ckpt-other

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from _table import Table


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


def _normalize(name: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.rstrip(":0")
    for old, new in _SUBS:
        name = name.replace(old, new)
    return name


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _is_config(path: str) -> bool:
    return path.endswith(".yaml") or path.endswith(".yml")


def load_source(path: str) -> Tuple[str, List[Tuple[str, tuple]]]:
    """Load a source and return (label, [(name, shape), ...]) sorted by normalized name."""
    if _is_config(path):
        return _load_config(path)
    else:
        return _load_ckpt(path)


def _load_config(config_path: str) -> Tuple[str, List[Tuple[str, tuple]]]:
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

    # model.variables preserves build order; use that order (most meaningful)
    label = Path(config_path).stem
    items = [(v.name.rstrip(":0"), tuple(v.shape)) for v in model.variables]
    return label, items


def _load_ckpt(ckpt_path: str) -> Tuple[str, List[Tuple[str, tuple]]]:
    import tensorflow as tf
    reader    = tf.train.load_checkpoint(ckpt_path)
    shape_map = reader.get_variable_to_shape_map()
    skip      = {".OPTIMIZER_SLOT/", "_CHECKPOINTABLE_OBJECT_GRAPH", "save_counter"}
    items     = [
        (name, tuple(shape))
        for name, shape in shape_map.items()
        if not any(s in name for s in skip)
    ]
    # Sort checkpoints by normalized name for a deterministic order
    items.sort(key=lambda t: _normalize(t[0]))
    label = Path(ckpt_path).name
    return label, items


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
            status = "EXTRA"    # in src1 only
        else:
            status = "MISSING"  # in src2 only

        if only_mismatch and status == "MATCH":
            idx += 1
            continue

        rows.append(dict(idx=idx, n1=n1, s1=s1, n2=n2, s2=s2, status=status))
        idx += 1

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Positional (index-by-index) shape comparison of two model sources.",
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
    ap.add_argument("--no-colour", action="store_true",
                    help="Disable ANSI colours (useful for file output)")
    args = ap.parse_args()

    print(f"Loading src1: {args.src1}")
    label1, items1 = load_source(args.src1)
    print(f"  {len(items1)} variables  (source type: {'config' if _is_config(args.src1) else 'checkpoint'})")

    print(f"Loading src2: {args.src2}")
    label2, items2 = load_source(args.src2)
    print(f"  {len(items2)} variables  (source type: {'config' if _is_config(args.src2) else 'checkpoint'})\n")

    if len(items1) != len(items2):
        print(f"WARNING: variable counts differ ({len(items1)} vs {len(items2)}) — "
              f"extra rows will be marked EXTRA / MISSING\n")

    rows = align(items1, items2, args.filter, args.only_mismatch)

    if not rows:
        print("No rows to display (all matched or all filtered out).")
        return

    tbl = Table(label1, label2, show_index=True, use_colour=not args.no_colour)
    tbl.header()

    counts: Dict[str, int] = {}
    for r in rows:
        tbl.row(r["n1"], r["s1"], r["n2"], r["s2"], r["status"], idx=r["idx"])
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    tbl.footer()

    total = len(items1) if len(items1) >= len(items2) else len(items2)
    tbl.summary(counts, total)


if __name__ == "__main__":
    main()
