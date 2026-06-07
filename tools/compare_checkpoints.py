"""Compare two TF checkpoints (or checkpoint vs current model) variable by variable.

Prints a side-by-side table with shape comparison and MATCH / MISMATCH / UNMATCHED status.

Usage examples
--------------
# Two checkpoints against each other:
python tools/compare_checkpoints.py \
    --ckpt1 initial_checkpoint_folder/ckpt-920304 \
    --ckpt2 path/to/new/ckpt-1000

# Checkpoint vs current model built from config:
python tools/compare_checkpoints.py \
    --ckpt1 initial_checkpoint_folder/ckpt-920304 \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml

# Filter to backbone + decoder only, hide unmatched head vars:
python tools/compare_checkpoints.py \
    --ckpt1 A --ckpt2 B --modules backbone decoder

# Grep for a specific layer name substring:
python tools/compare_checkpoints.py \
    --ckpt1 A --ckpt2 B --grep cls_pred
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Name normalization (shared with checkpoint_migration.py)
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

def load_ckpt_vars(path: str) -> Dict[str, Tuple[tuple, str]]:
    """Return {name: (shape, dtype)} from a checkpoint, skipping optimizer slots."""
    import tensorflow as tf
    reader = tf.train.load_checkpoint(path)
    shape_map = reader.get_variable_to_shape_map()
    dtype_map  = reader.get_variable_to_dtype_map()
    skip = {".OPTIMIZER_SLOT/", "_CHECKPOINTABLE_OBJECT_GRAPH", "save_counter"}
    return {
        name: (tuple(shape), dtype_map[name].name)
        for name, shape in shape_map.items()
        if not any(s in name for s in skip)
    }


def load_model_vars(config_path: str) -> Dict[str, Tuple[tuple, str]]:
    """Build the model from a YAML config and return its variable shapes."""
    repo_root = str(Path(config_path).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import tensorflow as tf
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8

    cfg = load_config(config_path)
    model = build_yolov8(cfg.task.model)
    model.deploy = False
    model.build_and_init(cfg.task.model.input_size)

    return {
        v.name.rstrip(":0"): (tuple(v.shape), v.dtype.name)
        for v in model.variables
    }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def build_rows(
    vars1: Dict[str, Tuple[tuple, str]],
    vars2: Dict[str, Tuple[tuple, str]],
    modules: Optional[List[str]],
) -> List[dict]:
    """Match each var in vars1 to the best var in vars2 by normalized name."""

    def _keep(name: str) -> bool:
        return not modules or any(m in name for m in modules)

    vars1_f = {k: v for k, v in vars1.items() if _keep(k)}

    # normalized → full name for ckpt2
    norm2: Dict[str, str] = {}
    for k in vars2:
        nk = _normalize(k)
        # keep shortest name if collision (prefer unqualified path)
        if nk not in norm2 or len(k) < len(norm2[nk]):
            norm2[nk] = k

    rows: List[dict] = []
    matched2: set = set()

    for name1, (shape1, _) in sorted(vars1_f.items()):
        norm1 = _normalize(name1)

        if norm1 in norm2:
            name2 = norm2[norm1]
            shape2 = vars2[name2][0]
            status = "MATCH" if shape1 == shape2 else "SHAPE MISMATCH"
            rows.append(dict(name1=name1, shape1=shape1,
                             name2=name2, shape2=shape2, status=status))
            matched2.add(name2)
        else:
            candidates = list(norm2.keys())
            best = difflib.get_close_matches(norm1, candidates, n=1, cutoff=0.55)
            if best:
                name2 = norm2[best[0]]
                shape2 = vars2[name2][0]
                status = "MATCH~" if shape1 == shape2 else "MISMATCH~"
                rows.append(dict(name1=name1, shape1=shape1,
                                 name2=f"≈{name2}", shape2=shape2, status=status))
                matched2.add(name2)
            else:
                rows.append(dict(name1=name1, shape1=shape1,
                                 name2="—", shape2=(), status="UNMATCHED 1"))

    # vars2 entries with no counterpart in vars1
    for name2, (shape2, _) in sorted(vars2.items()):
        if name2 not in matched2 and _keep(name2):
            rows.append(dict(name1="—", shape1=(),
                             name2=name2, shape2=shape2, status="ONLY IN 2"))

    return rows


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_STATUS_COLOUR = {
    "MATCH":          "\033[92m",  # green
    "MATCH~":         "\033[96m",  # cyan (fuzzy)
    "SHAPE MISMATCH": "\033[91m",  # red
    "MISMATCH~":      "\033[93m",  # yellow
    "UNMATCHED 1":    "\033[91m",  # red
    "ONLY IN 2":      "\033[93m",  # yellow
}
_RESET = "\033[0m"


def print_table(
    rows: List[dict],
    label1: str,
    label2: str,
    use_colour: bool = True,
    grep: str = "",
) -> None:
    if grep:
        rows = [r for r in rows if grep in r["name1"] or grep in r["name2"]]
        if not rows:
            print(f"No rows matching --grep '{grep}'")
            return

    C1 = min(max((len(r["name1"]) for r in rows), default=10), 70)
    C1 = max(C1, len(label1), 30)
    C2 = min(max((len(r["name2"]) for r in rows), default=10), 70)
    C2 = max(C2, len(label2), 30)
    CS = 20
    ST = 15

    sep = f"+{'-'*(C1+2)}+{'-'*(CS+2)}+{'-'*(C2+2)}+{'-'*(CS+2)}+{'-'*(ST+2)}+"

    def hdr(n1, s1, n2, s2, st):
        return (f"| {n1:<{C1}} | {s1:<{CS}} | {n2:<{C2}} | {s2:<{CS}} | {st:<{ST}} |")

    print(sep)
    print(hdr(label1, "Shape 1", label2, "Shape 2", "Status"))
    print(sep)

    counts = {}
    for r in rows:
        st   = r["status"]
        s1   = str(r["shape1"]) if r["shape1"] else ""
        s2   = str(r["shape2"]) if r["shape2"] else ""
        n1   = r["name1"]
        n2   = r["name2"]
        line = hdr(n1, s1, n2, s2, st)
        if use_colour and st in _STATUS_COLOUR:
            line = _STATUS_COLOUR[st] + line + _RESET
        print(line)
        counts[st] = counts.get(st, 0) + 1

    print(sep)
    print()
    print("Summary:")
    total = len(rows)
    for k, v in sorted(counts.items()):
        print(f"  {k:<20}: {v}")
    print(f"  {'TOTAL':<20}: {total}")
    exact = counts.get("MATCH", 0) + counts.get("MATCH~", 0)
    pct = 100 * exact / total if total else 0
    print(f"\n  Matched (exact+fuzzy) : {exact} / {total}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of two TF checkpoint architectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ckpt1", required=True,
                        help="First checkpoint path prefix")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt2",
                     help="Second checkpoint path prefix")
    grp.add_argument("--config",
                     help="YAML config — builds current model and compares against --ckpt1")
    parser.add_argument("--modules", nargs="*",
                        help="Filter to variables containing these substrings (e.g. backbone decoder)")
    parser.add_argument("--grep", default="",
                        help="Only show rows where either name contains this string")
    parser.add_argument("--no-colour", action="store_true",
                        help="Disable ANSI colour codes (useful for piping to a file)")
    args = parser.parse_args()

    print(f"Loading ckpt1: {args.ckpt1}")
    vars1 = load_ckpt_vars(args.ckpt1)
    print(f"  → {len(vars1)} variables")

    if args.ckpt2:
        print(f"Loading ckpt2: {args.ckpt2}")
        vars2 = load_ckpt_vars(args.ckpt2)
        label2 = Path(args.ckpt2).name
    else:
        print(f"Building model from config: {args.config}")
        vars2 = load_model_vars(args.config)
        label2 = "current model"
    print(f"  → {len(vars2)} variables\n")

    label1 = Path(args.ckpt1).name
    rows = build_rows(vars1, vars2, modules=args.modules or None)

    print_table(
        rows, label1, label2,
        use_colour=not args.no_colour,
        grep=args.grep,
    )


if __name__ == "__main__":
    main()
