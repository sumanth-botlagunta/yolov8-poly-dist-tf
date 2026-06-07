"""Compare two TF checkpoints (or checkpoint vs current model) by variable name.

Variables from source 1 are matched to source 2 by normalized name (exact first,
then fuzzy). Output is a terminal-fitted table — names are truncated to fit;
use --no-colour and redirect to a file for full names.

Usage
-----
# Checkpoint vs checkpoint:
python tools/compare_checkpoints.py \\
    --ckpt1 initial_checkpoint_folder/ckpt-920304 \\
    --ckpt2 path/to/other/ckpt-1000

# Checkpoint vs current model built from YAML:
python tools/compare_checkpoints.py \\
    --ckpt1 initial_checkpoint_folder/ckpt-920304 \\
    --config configs/experiments/yolo/yolov8_poly_dist.yaml

# Filter to backbone + decoder only:
python tools/compare_checkpoints.py \\
    --ckpt1 A --config C.yaml --modules backbone decoder

# Grep for a layer name substring:
python tools/compare_checkpoints.py \\
    --ckpt1 A --config C.yaml --grep cls_pred

# Save full-width report to file:
python tools/compare_checkpoints.py \\
    --ckpt1 A --config C.yaml --no-colour > report.txt
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from _table import Table


# ---------------------------------------------------------------------------
# Name normalization
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

def load_ckpt(path: str) -> Tuple[Dict[str, Tuple[tuple, str]], List[str]]:
    """Return ({name: (shape, dtype)}, scalar_meta_names) from a checkpoint.

    Scalar () variables (global_step, optimizer scalars) are separated into
    scalar_meta_names and excluded from the main comparison dict.
    """
    import tensorflow as tf
    reader    = tf.train.load_checkpoint(path)
    shape_map = reader.get_variable_to_shape_map()
    dtype_map = reader.get_variable_to_dtype_map()
    skip = {".OPTIMIZER_SLOT/", "_CHECKPOINTABLE_OBJECT_GRAPH", "save_counter"}

    result: Dict[str, Tuple[tuple, str]] = {}
    meta:   List[str] = []
    for name, shape in shape_map.items():
        if any(s in name for s in skip):
            continue
        shape_t = tuple(shape)
        if len(shape_t) == 0:
            meta.append(name)
        else:
            result[name] = (shape_t, dtype_map[name].name)
    return result, sorted(meta)


def load_model(config_path: str) -> Dict[str, Tuple[tuple, str]]:
    """Build model from YAML config, return {var_name: (shape, dtype)}."""
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

    return {
        v.name.rstrip(":0"): (tuple(v.shape), v.dtype.name)
        for v in model.variables
    }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_vars(
    vars1: Dict[str, Tuple[tuple, str]],
    vars2: Dict[str, Tuple[tuple, str]],
    modules: Optional[List[str]],
) -> List[dict]:
    """Match vars1 → vars2 by normalized name (exact then fuzzy)."""

    def _keep(n: str) -> bool:
        return not modules or any(m in n for m in modules)

    filtered1 = {k: v for k, v in vars1.items() if _keep(k)}

    # normalized → full-name lookup for vars2
    norm2: Dict[str, str] = {}
    for k in vars2:
        nk = _normalize(k)
        if nk not in norm2 or len(k) < len(norm2[nk]):
            norm2[nk] = k

    rows: List[dict] = []
    matched2: set = set()

    for name1, (shape1, _) in sorted(filtered1.items()):
        norm1 = _normalize(name1)

        if norm1 in norm2:
            name2  = norm2[norm1]
            shape2 = vars2[name2][0]
            status = "MATCH" if shape1 == shape2 else "SHAPE MISMATCH"
            rows.append(dict(n1=name1, s1=shape1, n2=name2, s2=shape2, status=status))
            matched2.add(name2)
        else:
            best = difflib.get_close_matches(norm1, list(norm2), n=1, cutoff=0.55)
            if best:
                name2  = norm2[best[0]]
                shape2 = vars2[name2][0]
                status = "MATCH~" if shape1 == shape2 else "MISMATCH~"
                rows.append(dict(n1=name1, s1=shape1, n2=f"≈{name2}", s2=shape2, status=status))
                matched2.add(name2)
            else:
                rows.append(dict(n1=name1, s1=shape1, n2="—", s2=(), status="UNMATCHED"))

    for name2, (shape2, _) in sorted(vars2.items()):
        if name2 not in matched2 and _keep(name2):
            rows.append(dict(n1="—", s1=(), n2=name2, s2=shape2, status="ONLY IN 2"))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Side-by-side name-matched comparison of two TF checkpoint architectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ckpt1", required=True, help="First checkpoint path prefix")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt2",   help="Second checkpoint path prefix")
    grp.add_argument("--config",  help="YAML config — builds current model for comparison")
    ap.add_argument("--modules",  nargs="*",
                    help="Only show variables containing these substrings (e.g. backbone decoder)")
    ap.add_argument("--grep",     default="",
                    help="Only show rows where either name contains this string")
    ap.add_argument("--no-colour", action="store_true",
                    help="Disable ANSI colours (useful for file output)")
    args = ap.parse_args()

    print(f"Loading:  {args.ckpt1}")
    vars1, meta1 = load_ckpt(args.ckpt1)
    print(f"  {len(vars1)} variables  ({len(meta1)} scalar metadata excluded)")

    meta2: List[str] = []
    if args.ckpt2:
        print(f"Loading:  {args.ckpt2}")
        vars2, meta2 = load_ckpt(args.ckpt2)
        label2 = Path(args.ckpt2).name
    else:
        print(f"Building model from: {args.config}")
        vars2  = load_model(args.config)
        label2 = "current model"
    print(f"  {len(vars2)} variables  ({len(meta2)} scalar metadata excluded)\n")

    label1 = Path(args.ckpt1).name
    rows   = match_vars(vars1, vars2, args.modules)

    if args.grep:
        rows = [r for r in rows if args.grep in r["n1"] or args.grep in r["n2"]]
        if not rows:
            print(f"No rows match --grep '{args.grep}'")
            return

    tbl = Table(label1, label2, use_colour=not args.no_colour)
    tbl.header()

    counts: Dict[str, int] = {}
    for r in rows:
        tbl.row(r["n1"], r["s1"], r["n2"], r["s2"], r["status"])
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    tbl.footer()
    tbl.summary(counts, len(rows))

    # ---- scalar metadata section ----
    all_meta = sorted(set(meta1) | set(meta2))
    if all_meta:
        set1, set2 = set(meta1), set(meta2)
        print(f"\nScalar metadata variables (shape=(), excluded from comparison above):")
        w = max((len(n) for n in all_meta), default=20)
        print(f"  {'Name':<{w}}  {label1:>14}  {label2:>14}")
        print(f"  {'-'*w}  {'-'*14}  {'-'*14}")
        for name in all_meta:
            in1 = "✓" if name in set1 else "—"
            in2 = "✓" if name in set2 else "—"
            print(f"  {name:<{w}}  {in1:>14}  {in2:>14}")


if __name__ == "__main__":
    main()
