"""Migrate an old TF checkpoint to the new model variable names.

Usage:
    # Step 1 — inspect what variables are in the old checkpoint:
    python tools/checkpoint_migration.py list \
        --ckpt initial_checkpoint_folder/ckpt-920304

    # Step 2 — dry-run the mapping (shows which vars matched/missed):
    python tools/checkpoint_migration.py map \
        --ckpt initial_checkpoint_folder/ckpt-920304 \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml

    # Step 3 — migrate and save:
    python tools/checkpoint_migration.py migrate \
        --ckpt initial_checkpoint_folder/ckpt-920304 \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --output /tmp/migrated_ckpt/ckpt \
        --modules backbone decoder
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Lazy TF import so `python tools/checkpoint_migration.py list --help` is fast.

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_checkpoint_variables(ckpt_path: str) -> Dict[str, Tuple]:
    """Return {name: shape} for every variable stored in *ckpt_path*."""
    import tensorflow as tf

    reader = tf.train.load_checkpoint(ckpt_path)
    shape_map = reader.get_variable_to_shape_map()
    dtype_map = reader.get_variable_to_dtype_map()
    return {
        name: (tuple(shape), dtype_map[name].name)
        for name, shape in shape_map.items()
    }


def build_name_mapping(
    old_vars: Dict[str, Tuple],
    new_vars: Dict[str, Tuple],
    modules: List[str] = ("backbone", "decoder"),
) -> Dict[str, str]:
    """Attempt to map old checkpoint variable names to new model variable names.

    Strategy
    --------
    1. Filter old vars to those belonging to *modules* (substring match).
    2. For each old name, strip common framework prefixes and compare the
       remaining suffix against every new name using SequenceMatcher.
    3. Accept a match when similarity > 0.6 AND shapes agree.
    4. Log unmatched old names as warnings.

    Returns
    -------
    {old_name: new_name}  for successfully matched variables only.
    """
    module_filter = tuple(modules)

    # Only consider old vars that belong to the requested modules.
    old_filtered = {
        name: info
        for name, info in old_vars.items()
        if any(m in name for m in module_filter)
    }

    # Strip common prefix tokens that differ between old and new naming.
    _STRIP_PREFIXES = [
        "yolo_model/",
        "model/",
        "yolov8/",
        ".OPTIMIZER_SLOT/",
    ]

    # Substitution rules: normalize legacy TF1/Keras layer name patterns to
    # match the new codebase's explicit sub-layer naming (conv/bn/act).
    _SUBS = [
        # Conv2D layer named inline (TF1) → sub-layer named 'conv'
        ("/Conv2D/kernel",    "/conv/kernel"),
        ("/Conv2D/bias",      "/conv/bias"),
        # BatchNorm variants → 'bn'
        ("/BatchNorm/",             "/bn/"),
        ("/batch_normalization/",   "/bn/"),
        ("/BatchNormalization/",    "/bn/"),
        # BN parameter names differ across TF versions
        ("/bn/moving_average",      "/bn/moving_mean"),
        ("/bn/Momentum",            "/bn/moving_variance"),
    ]

    def _normalize(name: str) -> str:
        for prefix in _STRIP_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
        # Remove trailing ":0" if present (TF1 style)
        name = name.rstrip(":0")
        for old_pat, new_pat in _SUBS:
            name = name.replace(old_pat, new_pat)
        return name

    # Build normalized → full-name lookup for new vars.
    new_stripped: Dict[str, str] = {}
    for new_name in new_vars:
        stripped = _normalize(new_name)
        new_stripped[stripped] = new_name

    mapping: Dict[str, str] = {}
    unmatched: List[str] = []

    for old_name, (old_shape, _) in old_filtered.items():
        old_stripped = _normalize(old_name)

        # Exact match on stripped name.
        if old_stripped in new_stripped:
            new_name = new_stripped[old_stripped]
            new_shape = new_vars[new_name][0]
            if old_shape == new_shape:
                mapping[old_name] = new_name
            else:
                log.warning(
                    "Shape mismatch — old=%s %s  new=%s %s (skipped)",
                    old_name, old_shape, new_name, new_shape,
                )
            continue

        # Fuzzy match by sequence similarity.
        new_candidates = list(new_stripped.keys())
        matches = difflib.get_close_matches(
            old_stripped, new_candidates, n=3, cutoff=0.6
        )
        matched = False
        for candidate in matches:
            new_name = new_stripped[candidate]
            new_shape = new_vars[new_name][0]
            if old_shape == new_shape:
                mapping[old_name] = new_name
                log.debug("Fuzzy match: %s  →  %s", old_name, new_name)
                matched = True
                break
        if not matched:
            unmatched.append(old_name)

    if unmatched:
        log.warning(
            "%d old variables could not be mapped (head vars are expected "
            "to be unmatched):", len(unmatched)
        )
        for name in unmatched[:20]:
            log.warning("  UNMATCHED: %s", name)
        if len(unmatched) > 20:
            log.warning("  ... and %d more", len(unmatched) - 20)

    log.info(
        "Mapping summary: %d matched, %d unmatched out of %d old vars "
        "(filtered to modules: %s)",
        len(mapping), len(unmatched), len(old_filtered), list(modules),
    )
    return mapping


def migrate_checkpoint(
    old_ckpt_path: str,
    new_model,                  # tf.keras.Model
    output_ckpt_path: str,
    modules: List[str] = ("backbone", "decoder"),
) -> Dict[str, int]:
    """Load *old_ckpt_path*, assign matching weights to *new_model*, save.

    Parameters
    ----------
    old_ckpt_path : str
        Path prefix to the old checkpoint (e.g. ``ckpt-920304``).
    new_model : tf.keras.Model
        Already-built new model.  Must have been called at least once so
        all variables exist.
    output_ckpt_path : str
        Path prefix for the output checkpoint (e.g. ``/tmp/migrated/ckpt``).
    modules : list of str
        Only load variables belonging to these module names.

    Returns
    -------
    dict with keys ``loaded``, ``skipped``, ``not_found``.
    """
    import tensorflow as tf

    old_vars = list_checkpoint_variables(old_ckpt_path)

    # Build new var dict from model: {name_without_:0: variable}
    new_var_dict: Dict[str, tf.Variable] = {
        v.name.rstrip(":0"): v for v in new_model.variables
    }
    new_var_info: Dict[str, Tuple] = {
        name: (tuple(v.shape), str(v.dtype).replace("tf.", ""))
        for name, v in new_var_dict.items()
    }

    mapping = build_name_mapping(old_vars, new_var_info, modules=list(modules))

    reader = tf.train.load_checkpoint(old_ckpt_path)
    stats = {"loaded": 0, "skipped": 0, "not_found": 0}

    for old_name, new_name in mapping.items():
        old_shape, old_dtype = old_vars[old_name]
        try:
            tensor = reader.get_tensor(old_name)
        except Exception as e:
            log.warning("Could not read %s: %s", old_name, e)
            stats["not_found"] += 1
            continue

        if new_name not in new_var_dict:
            log.warning("New var not found in model: %s", new_name)
            stats["not_found"] += 1
            continue

        var = new_var_dict[new_name]
        if var.shape != tensor.shape:
            log.warning(
                "Shape mismatch at assign: %s %s vs %s — skipping",
                new_name, var.shape, tensor.shape,
            )
            stats["skipped"] += 1
            continue

        var.assign(tensor)
        stats["loaded"] += 1

    log.info(
        "Assignment complete: loaded=%d  skipped=%d  not_found=%d",
        stats["loaded"], stats["skipped"], stats["not_found"],
    )

    # Save the migrated weights as a new checkpoint.
    output_path = Path(output_ckpt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=new_model)
    ckpt.write(output_ckpt_path)
    log.info("Migrated checkpoint saved to: %s", output_ckpt_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list(args: argparse.Namespace) -> None:
    variables = list_checkpoint_variables(args.ckpt)
    print(f"\nCheckpoint: {args.ckpt}")
    print(f"Total variables: {len(variables)}\n")
    header = f"{'Name':<80} {'Shape':<20} {'Dtype'}"
    print(header)
    print("-" * len(header))
    for name, (shape, dtype) in sorted(variables.items()):
        print(f"{name:<80} {str(shape):<20} {dtype}")


def _cmd_map(args: argparse.Namespace) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.yaml_loader import load_config
    from models.yolo_v8 import YoloV8
    import tensorflow as tf

    cfg = load_config(args.config)
    model = YoloV8(cfg.task.model)
    # Build model by calling it once with dummy input.
    dummy = tf.zeros([1] + cfg.task.model.input_size)
    model(dummy, training=False)

    old_vars = list_checkpoint_variables(args.ckpt)
    new_var_info = {
        v.name.rstrip(":0"): (tuple(v.shape), v.dtype.name)
        for v in model.variables
    }
    modules = args.modules or ["backbone", "decoder"]
    mapping = build_name_mapping(old_vars, new_var_info, modules=modules)

    print(f"\nMapping ({len(mapping)} matches):")
    for old, new in sorted(mapping.items()):
        print(f"  {old}  →  {new}")


def _cmd_migrate(args: argparse.Namespace) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.yaml_loader import load_config
    from models.yolo_v8 import YoloV8
    import tensorflow as tf

    cfg = load_config(args.config)
    model = YoloV8(cfg.task.model)
    dummy = tf.zeros([1] + cfg.task.model.input_size)
    model(dummy, training=False)

    modules = args.modules or ["backbone", "decoder"]
    stats = migrate_checkpoint(
        old_ckpt_path=args.ckpt,
        new_model=model,
        output_ckpt_path=args.output,
        modules=modules,
    )
    print(f"\nDone: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8 checkpoint migration tool"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list subcommand
    p_list = sub.add_parser("list", help="List all variables in an old checkpoint")
    p_list.add_argument("--ckpt", required=True, help="Path prefix to old checkpoint")

    # map subcommand
    p_map = sub.add_parser("map", help="Dry-run: show the old→new variable mapping")
    p_map.add_argument("--ckpt", required=True)
    p_map.add_argument("--config", required=True, help="Experiment YAML config path")
    p_map.add_argument(
        "--modules", nargs="+", default=["backbone", "decoder"],
        help="Module names to include (default: backbone decoder)"
    )

    # migrate subcommand
    p_mig = sub.add_parser("migrate", help="Migrate old checkpoint to new model")
    p_mig.add_argument("--ckpt", required=True)
    p_mig.add_argument("--config", required=True)
    p_mig.add_argument("--output", required=True, help="Output checkpoint path prefix")
    p_mig.add_argument(
        "--modules", nargs="+", default=["backbone", "decoder"],
    )

    args = parser.parse_args()
    {"list": _cmd_list, "map": _cmd_map, "migrate": _cmd_migrate}[args.command](args)


if __name__ == "__main__":
    main()
