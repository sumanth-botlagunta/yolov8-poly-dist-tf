"""Migrate an old TF checkpoint into the new model — by STRUCTURE, not by name.

Why structural matching
-----------------------
Keras 3 gives leaf variables auto-generated names (``conv2d_59/kernel``) or bare
role names (``kernel``/``gamma``/``beta``), so variable names carry no module
path and do not survive a codebase rewrite. Name matching is therefore fragile.

Instead the architecture is recovered from structure and the two sides aligned
position-by-position, gated by leaf *role* + *shape*:

    * Old checkpoint → DFS over the stored ``_CHECKPOINTABLE_OBJECT_GRAPH`` proto
      (TF2 object-based checkpoints record the trackable tree and its ordering,
      independent of leaf layer names).
    * New model      → ordered ``module.variables`` per weight module (Keras
      tracks list-stored sublayers like the C2f blocks here, which the model's
      own object-graph checkpoint does not).

Each variable carries a standardized role (kernel/bias/gamma/beta/moving_mean/
moving_variance), so same-shape variables (e.g. BN gamma vs beta) are never
confused. Old variables with no structural match are reported as possible
architecture changes — never silently copied. This is the default ``structural``
strategy; ``--strategy name`` keeps the legacy fuzzy name matcher as a fallback.

Usage:
    # Step 1 — inspect variables in the old checkpoint:
    python tools/checkpoint_migration.py list \
        --ckpt initial_checkpoint_folder/ckpt-920304

    # Step 2 — dry-run the structural mapping (per-module match counts):
    python tools/checkpoint_migration.py map \
        --ckpt initial_checkpoint_folder/ckpt-920304 \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml

    # Step 2b — write the full old→new variable mapping to JSON for auditing:
    python tools/checkpoint_migration.py mapping \
        --ckpt initial_checkpoint_folder/ckpt-920304 \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --output /tmp/var_mapping.json

    # Step 3 — migrate and save (modules auto-detected from class count):
    python tools/checkpoint_migration.py migrate \
        --ckpt initial_checkpoint_folder/ckpt-920304 \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --output /tmp/migrated_ckpt/ckpt --mapping-json /tmp/var_mapping.json

    # ...or force the modules explicitly (skips auto class detection):
    python tools/checkpoint_migration.py migrate \
        --ckpt ... --config ... --output ... --modules backbone decoder

Module auto-selection
---------------------
The new model is built by *input-graph tracking* (a dummy forward pass
materialises every variable). The classification head width (``num_classes``)
is read from the ``cls_pred`` conv shape in both the old checkpoint and the
freshly built model:

    * class counts MATCH    → migrate backbone + decoder + head (full transfer)
    * class counts DIFFER   → migrate backbone + decoder only  (head re-trained)

Passing ``--modules`` explicitly overrides this and disables auto-detection.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
from collections import Counter
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


def detect_num_classes(var_info: Dict[str, Tuple]) -> Optional[int]:
    """Infer the classification head width (``num_classes``) from variable shapes.

    The cls prediction conv is named ``cls_pred_{level}`` (see ``models/head.py``);
    its kernel is ``[1, 1, in_ch, num_classes]`` and its bias is ``[num_classes]``.
    We scan for those, falling back to any variable whose name contains ``cls``.

    Parameters
    ----------
    var_info : dict
        ``{name: (shape_tuple, dtype)}`` for either an old checkpoint or a
        freshly-built model.

    Returns
    -------
    int or None
        The most common class count found across cls-head variables, or ``None``
        if no classification head variable could be identified.
    """
    def _scan(predicate) -> List[int]:
        found: List[int] = []
        for name, info in var_info.items():
            shape = info[0]
            if not predicate(name.lower()):
                continue
            if len(shape) == 4:        # conv kernel [kh, kw, in, num_classes]
                found.append(int(shape[-1]))
            elif len(shape) == 1:      # conv bias [num_classes]
                found.append(int(shape[0]))
        return found

    # Prefer the precise cls_pred match, then loosen to any 'cls' tensor.
    candidates = _scan(lambda n: "cls_pred" in n)
    if not candidates:
        candidates = _scan(lambda n: "cls" in n and "pred" in n)
    if not candidates:
        candidates = _scan(lambda n: "cls" in n)

    if not candidates:
        return None
    return Counter(candidates).most_common(1)[0][0]


def resolve_modules(
    old_var_info: Dict[str, Tuple],
    new_var_info: Dict[str, Tuple],
    requested: Optional[List[str]],
) -> Tuple[List[str], str]:
    """Decide which modules to migrate.

    If *requested* is provided, it is honoured verbatim (auto-detection off).
    Otherwise the classification head widths of the old checkpoint and the new
    model are compared:

        * equal class counts → ``[backbone, decoder, head]`` (full transfer)
        * differing / unknown → ``[backbone, decoder]`` (head re-initialised)

    Returns
    -------
    (modules, reason)
        ``modules`` is the resolved module list; ``reason`` is a human-readable
        explanation for logging.
    """
    if requested is not None:
        return list(requested), f"user-specified modules: {list(requested)}"

    base = ["backbone", "decoder"]
    old_nc = detect_num_classes(old_var_info)
    new_nc = detect_num_classes(new_var_info)

    if old_nc is not None and new_nc is not None and old_nc == new_nc:
        return (
            base + ["head"],
            f"class counts match (num_classes={new_nc}) — migrating head too",
        )

    if old_nc is None or new_nc is None:
        reason = (
            f"could not determine class counts (old={old_nc}, new={new_nc}) — "
            "migrating backbone + decoder only"
        )
    else:
        reason = (
            f"class counts differ (old={old_nc}, new={new_nc}) — "
            "migrating backbone + decoder only; head will be re-trained"
        )
    return base, reason


# ---------------------------------------------------------------------------
# Structural (name-independent) matching
# ---------------------------------------------------------------------------
#
# Keras 3 gives leaf variables auto-generated names ("conv2d_59/kernel") or bare
# role names ("kernel"/"gamma"/"beta"), so *no* name carries the module path.
# Matching by name is therefore unreliable across a codebase rewrite.
#
# Instead we recover the architecture from STRUCTURE:
#   * Old checkpoint  → DFS over the stored _CHECKPOINTABLE_OBJECT_GRAPH proto
#                       (TF2 object-based checkpoints record the trackable tree
#                       and its ordering, independent of leaf layer names).
#   * New model       → DFS-equivalent ordered list from ``module.variables``
#                       (Keras tracks list-stored sublayers here, which the
#                       checkpoint object graph does NOT — so we must use this
#                       for the new side).
#
# Both traversals visit variables in the same architectural order; each leaf
# carries a standardized *role* (kernel/bias/gamma/beta/moving_mean/
# moving_variance). We align the two ordered streams role+shape gated, which
# disambiguates same-shape variables (e.g. BN gamma vs beta) that pure shape
# matching would confuse. Divergences are reported, never silently copied.

# Modules that bear weights and partition the model cleanly.
_WEIGHT_MODULES = ("backbone", "decoder", "head")


def _normalize_role(name: str) -> str:
    """Reduce a variable/attribute name to its standardized leaf role.

    Examples: ``_kernel`` → ``kernel``, ``gamma`` → ``gamma``, ``kernel:0`` →
    ``kernel``. The object graph stores Keras conv weights as ``_kernel`` while
    the live Variable is named ``kernel``; normalizing makes the two comparable.
    """
    name = name.rstrip(":0")
    return name.lstrip("_")


def flatten_checkpoint_structure(ckpt_path: str) -> List[dict]:
    """DFS the old checkpoint's trackable object graph into ordered records.

    Returns a list (in structural traversal order) of::

        {"path": (local_name, ...), "role": str, "shape": tuple,
         "key": checkpoint_key, "module": str}

    Variables are tagged with ``module`` = the first path segment that is a known
    weight module (backbone/decoder/head), else ``"other"``. Optimizer slots and
    bookkeeping scalars are skipped.
    """
    import tensorflow as tf
    from tensorflow.core.protobuf import (  # type: ignore
        trackable_object_graph_pb2 as tog,
    )

    reader = tf.train.load_checkpoint(ckpt_path)
    shape_map = reader.get_variable_to_shape_map()

    graph_bytes = reader.get_tensor("_CHECKPOINTABLE_OBJECT_GRAPH")
    object_graph = tog.TrackableObjectGraph()
    object_graph.ParseFromString(graph_bytes)
    nodes = object_graph.nodes

    def _module_of(path: Tuple[str, ...]) -> str:
        for seg in path:
            if seg in _WEIGHT_MODULES:
                return seg
        return "other"

    records: List[dict] = []
    visited = {0}

    def _walk(node_id: int, edge: str, path: Tuple[str, ...]) -> None:
        node = nodes[node_id]
        var_key = None
        for attr in node.attributes:
            if attr.name == "VARIABLE_VALUE":
                var_key = attr.checkpoint_key
                break
        if var_key is not None and "OPTIMIZER_SLOT" not in var_key:
            records.append({
                "path": path,
                "role": _normalize_role(edge),
                "shape": tuple(shape_map.get(var_key, ())),
                "key": var_key,
                "module": _module_of(path),
            })
        for child in node.children:
            if child.node_id in visited:
                continue
            visited.add(child.node_id)
            _walk(child.node_id, child.local_name, path + (child.local_name,))

    for child in nodes[0].children:
        if child.node_id in visited:
            continue
        visited.add(child.node_id)
        _walk(child.node_id, child.local_name, (child.local_name,))

    return records


def flatten_model_structure(new_model) -> List[dict]:
    """DFS-equivalent ordered records for the live new model.

    Uses ``module.variables`` per weight module, which (unlike the model's own
    object-graph checkpoint) includes variables held in Python-list sublayers
    (e.g. the C2f blocks). Validated to be in the same architectural order as the
    checkpoint object-graph DFS.

    Returns records of::

        {"path": (module,), "role": str, "shape": tuple,
         "var": tf.Variable, "module": str}
    """
    records: List[dict] = []
    for module_name in _WEIGHT_MODULES:
        module = getattr(new_model, module_name, None)
        if module is None:
            continue
        for var in module.variables:
            records.append({
                "path": (module_name,),
                "role": _normalize_role(var.name),
                "shape": tuple(var.shape),
                "var": var,
                "module": module_name,
            })
    return records


def align_structures(
    old_records: List[dict],
    new_records: List[dict],
    modules: List[str],
) -> Tuple[List[dict], Dict[str, dict]]:
    """Align old→new records per module by in-order (role, shape) matching.

    For each requested module the old records are walked in order; for every old
    record we advance through the new records until a not-yet-consumed variable
    with the same ``role`` and ``shape`` is found, and pair them. This tolerates
    the new side having *extra* variables (partial transfer — e.g. a randomly
    initialised head) while still catching real architectural divergence: an old
    variable with no forward-matching new variable is reported, not copied.

    Returns
    -------
    pairs : list of dict
        ``{"key": old_checkpoint_key, "var": new_tf_Variable, "role": str,
           "shape": tuple, "module": str, "new_path": (module,)}`` for every
        aligned variable.
    diagnostics : dict
        Per-module ``{matched, old_count, new_count, unmatched_old}`` plus a
        top-level ``"_status"`` summarising whether the transfer is clean.
    """
    def _by_module(records: List[dict]) -> Dict[str, List[dict]]:
        grouped: Dict[str, List[dict]] = {}
        for rec in records:
            grouped.setdefault(rec["module"], []).append(rec)
        return grouped

    old_by_mod = _by_module(old_records)
    new_by_mod = _by_module(new_records)

    pairs: List[dict] = []
    diagnostics: Dict[str, dict] = {}
    clean = True

    for module in modules:
        olds = old_by_mod.get(module, [])
        news = new_by_mod.get(module, [])
        new_ptr = 0
        matched = 0
        unmatched_old: List[dict] = []

        for old in olds:
            target = None
            scan = new_ptr
            while scan < len(news):
                cand = news[scan]
                if cand["role"] == old["role"] and cand["shape"] == old["shape"]:
                    target = cand
                    new_ptr = scan + 1
                    break
                scan += 1
            if target is None:
                unmatched_old.append(old)
                clean = False
                continue
            pairs.append({
                "key": old["key"],
                "var": target["var"],
                "role": old["role"],
                "shape": old["shape"],
                "module": module,
                "new_path": target["path"],
            })
            matched += 1

        diagnostics[module] = {
            "matched": matched,
            "old_count": len(olds),
            "new_count": len(news),
            "unmatched_old": [
                {"key": r["key"], "role": r["role"], "shape": r["shape"]}
                for r in unmatched_old
            ],
        }
        if unmatched_old:
            log.warning(
                "[%s] %d old variable(s) had no structural match in the new model "
                "(possible architecture change) — not copied",
                module, len(unmatched_old),
            )

    diagnostics["_status"] = {
        "clean": clean,
        "total_pairs": len(pairs),
    }
    return pairs, diagnostics


def build_structural_mapping(
    old_ckpt_path: str,
    new_model,
    modules: List[str],
) -> Tuple[List[dict], Dict[str, dict]]:
    """Convenience wrapper: flatten both sides and align them.

    Returns ``(pairs, diagnostics)`` from :func:`align_structures`.
    """
    old_records = flatten_checkpoint_structure(old_ckpt_path)
    new_records = flatten_model_structure(new_model)
    return align_structures(old_records, new_records, modules)


def _write_mapping_json(
    pairs: List[dict],
    diagnostics: Dict[str, dict],
    json_path: str,
) -> None:
    """Serialise the old→new structural mapping (and diagnostics) to JSON."""
    import json

    payload = {
        "pairs": [
            {
                "old_key": p["key"],
                "new_path": "/".join(p["new_path"] + (f"#{i}",)),
                "role": p["role"],
                "shape": list(p["shape"]),
                "module": p["module"],
            }
            for i, p in enumerate(pairs)
        ],
        "diagnostics": diagnostics,
    }
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    log.info("Variable mapping written to: %s", json_path)


def migrate_checkpoint(
    old_ckpt_path: str,
    new_model,                  # tf.keras.Model
    output_ckpt_path: str,
    modules: Optional[List[str]] = None,
    strategy: str = "structural",
    mapping_json: Optional[str] = None,
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
    modules : list of str, optional
        Only load variables belonging to these module names. When ``None``
        (default) the modules are auto-selected by comparing the old and new
        classification head widths (see :func:`resolve_modules`).
    strategy : {"structural", "name"}
        ``"structural"`` (default) matches variables by DFS structure + role +
        shape — name independent and the recommended path. ``"name"`` falls back
        to the legacy fuzzy name matcher (:func:`build_name_mapping`).
    mapping_json : str, optional
        If set (structural strategy only), write the resolved old→new variable
        mapping and per-module diagnostics to this JSON path for auditing/reuse.

    Returns
    -------
    dict with keys ``loaded``, ``skipped``, ``not_found``.
    """
    import tensorflow as tf

    old_vars = list_checkpoint_variables(old_ckpt_path)
    new_var_dict: Dict[str, tf.Variable] = {
        v.name.rstrip(":0"): v for v in new_model.variables
    }
    new_var_info: Dict[str, Tuple] = {
        name: (tuple(v.shape), str(v.dtype).replace("tf.", ""))
        for name, v in new_var_dict.items()
    }

    resolved_modules, reason = resolve_modules(old_vars, new_var_info, modules)
    log.info("Module selection: %s", reason)
    log.info("Migration strategy: %s", strategy)

    reader = tf.train.load_checkpoint(old_ckpt_path)
    stats = {"loaded": 0, "skipped": 0, "not_found": 0}

    if strategy == "structural":
        pairs, diagnostics = build_structural_mapping(
            old_ckpt_path, new_model, resolved_modules
        )
        for module in resolved_modules:
            d = diagnostics.get(module, {})
            log.info(
                "[%s] structural match: %d/%d old vars aligned (new has %d)",
                module, d.get("matched", 0), d.get("old_count", 0),
                d.get("new_count", 0),
            )
        if not diagnostics["_status"]["clean"]:
            log.warning(
                "Structural alignment is NOT clean — some old variables had no "
                "match (see per-module diagnostics). Aligned variables are still "
                "copied; review before training."
            )

        if mapping_json:
            _write_mapping_json(pairs, diagnostics, mapping_json)

        for pair in pairs:
            var = pair["var"]
            try:
                tensor = reader.get_tensor(pair["key"])
            except Exception as e:
                log.warning("Could not read %s: %s", pair["key"], e)
                stats["not_found"] += 1
                continue
            if tuple(var.shape) != tuple(tensor.shape):
                # Should not happen (shape is part of the match key) but guard anyway.
                log.warning(
                    "Shape mismatch at assign: %s %s vs %s — skipping",
                    pair["new_path"], var.shape, tensor.shape,
                )
                stats["skipped"] += 1
                continue
            var.assign(tensor)
            stats["loaded"] += 1

    else:  # legacy name-based fallback
        mapping = build_name_mapping(old_vars, new_var_info, modules=resolved_modules)
        for old_name, new_name in mapping.items():
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


def _build_model_from_config(config_path: str):
    """Load YAML config and return a built+materialised YoloV8 (+ model_cfg)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8

    cfg = load_config(config_path)
    model_cfg = cfg.task.model
    # Assemble the model and materialise every variable via input-graph tracking
    # (a dummy forward pass). YoloV8 takes sub-modules, not a config — it must be
    # built through the factory.
    model = build_yolov8(model_cfg)
    model.build_and_init(model_cfg.input_size)
    return model, model_cfg


def _cmd_map(args: argparse.Namespace) -> None:
    model, _ = _build_model_from_config(args.config)

    old_vars = list_checkpoint_variables(args.ckpt)
    new_var_info = {
        v.name.rstrip(":0"): (tuple(v.shape), v.dtype.name)
        for v in model.variables
    }
    modules, reason = resolve_modules(old_vars, new_var_info, args.modules)
    print(f"\nModule selection: {reason}")
    print(f"Strategy: {args.strategy}")

    if args.strategy == "structural":
        pairs, diagnostics = build_structural_mapping(args.ckpt, model, modules)
        for module in modules:
            d = diagnostics.get(module, {})
            print(f"  [{module}] aligned {d.get('matched', 0)}/{d.get('old_count', 0)}"
                  f" old vars (new has {d.get('new_count', 0)})")
            for um in d.get("unmatched_old", [])[:10]:
                print(f"      UNMATCHED old: {um['role']} {um['shape']}  {um['key']}")
        print(f"\nStructural mapping ({len(pairs)} pairs, "
              f"clean={diagnostics['_status']['clean']}):")
        for p in pairs[:40]:
            print(f"  {p['module']:<9} {p['role']:<16} {str(p['shape']):<22} ← {p['key']}")
        if len(pairs) > 40:
            print(f"  ... and {len(pairs) - 40} more")
    else:
        mapping = build_name_mapping(old_vars, new_var_info, modules=modules)
        print(f"\nName mapping ({len(mapping)} matches):")
        for old, new in sorted(mapping.items()):
            print(f"  {old}  →  {new}")


def _cmd_mapping(args: argparse.Namespace) -> None:
    """Emit the structural old→new variable mapping as JSON (no weight copy)."""
    model, _ = _build_model_from_config(args.config)

    old_vars = list_checkpoint_variables(args.ckpt)
    new_var_info = {
        v.name.rstrip(":0"): (tuple(v.shape), v.dtype.name)
        for v in model.variables
    }
    modules, reason = resolve_modules(old_vars, new_var_info, args.modules)
    print(f"Module selection: {reason}")

    pairs, diagnostics = build_structural_mapping(args.ckpt, model, modules)
    _write_mapping_json(pairs, diagnostics, args.output)
    print(f"Wrote {len(pairs)} pairs (clean={diagnostics['_status']['clean']}) "
          f"to {args.output}")


def _cmd_migrate(args: argparse.Namespace) -> None:
    model, _ = _build_model_from_config(args.config)

    # args.modules is None unless the user forced it → migrate_checkpoint
    # auto-selects modules from the class count when None.
    stats = migrate_checkpoint(
        old_ckpt_path=args.ckpt,
        new_model=model,
        output_ckpt_path=args.output,
        modules=args.modules,
        strategy=args.strategy,
        mapping_json=args.mapping_json,
    )
    print(f"\nDone: {stats}")


_MODULES_HELP = (
    "Module names to include. Default: auto — include the head only when the "
    "old/new class counts match, else backbone + decoder."
)


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
    p_map.add_argument("--modules", nargs="+", default=None, help=_MODULES_HELP)
    p_map.add_argument(
        "--strategy", choices=["structural", "name"], default="structural",
        help="structural (default, name-independent) or name (legacy fuzzy)."
    )

    # mapping subcommand — emit JSON mapping for auditing/reuse
    p_jsn = sub.add_parser(
        "mapping", help="Write the structural old→new variable mapping as JSON"
    )
    p_jsn.add_argument("--ckpt", required=True)
    p_jsn.add_argument("--config", required=True, help="Experiment YAML config path")
    p_jsn.add_argument("--output", required=True, help="Output JSON path")
    p_jsn.add_argument("--modules", nargs="+", default=None, help=_MODULES_HELP)

    # migrate subcommand
    p_mig = sub.add_parser("migrate", help="Migrate old checkpoint to new model")
    p_mig.add_argument("--ckpt", required=True)
    p_mig.add_argument("--config", required=True)
    p_mig.add_argument("--output", required=True, help="Output checkpoint path prefix")
    p_mig.add_argument("--modules", nargs="+", default=None, help=_MODULES_HELP)
    p_mig.add_argument(
        "--strategy", choices=["structural", "name"], default="structural",
        help="structural (default, name-independent) or name (legacy fuzzy)."
    )
    p_mig.add_argument(
        "--mapping-json", default=None,
        help="Optional path to also write the resolved variable mapping as JSON."
    )

    args = parser.parse_args()
    {
        "list": _cmd_list,
        "map": _cmd_map,
        "mapping": _cmd_mapping,
        "migrate": _cmd_migrate,
    }[args.command](args)


if __name__ == "__main__":
    main()
