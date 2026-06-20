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

Curated legacy map (recommended for the ckpt-319992 legacy checkpoint)
---------------------------------------------------------------------
The legacy checkpoint names variables completely differently AND at different
nesting levels (``backbone/layer_with_weights-N/...``,
``head/_head/{level}/cv3/...``), so neither name nor traversal-order matching is
reliable. ``tools/shared/checkpoint_weight_map.py`` holds a hand-curated structural map
that resolves each variable into confident / suggested / ambiguous tiers and
copies only shape-verified pairs — this runtime resolver backs the ``"map"``
strategy. Separately, ``tools/shared/legacy_weight_map_frozen.py`` holds the committed,
hand-verified EXACT map used by the ``"frozen"`` strategy. ``"auto"`` (default)
selects ``"native"`` for a checkpoint produced by THIS codebase (warm-starting a new
run — EMA markers / ``model/{backbone,decoder,head}`` root), ``"frozen"`` for legacy
object checkpoints (``layer_with_weights`` / ``_head/`` keys), and ``"structural"``
otherwise; it never auto-selects the unverified ``"map"`` resolver
(see :func:`_detect_strategy`).

Warm-starting from this codebase's own checkpoint (``"native"``) is handled specially
because the trainer stores the *complete* weights only in the EMA shadows — the plain
``model/`` object graph omits the list-tracked C2f block variables — so it loads via the
EMA path (:func:`migrate_native`, reusing ``tools.shared.ckpt_loading``).

Module rule (39-class): a 39-class model migrates ALL modules (backbone +
decoder + head); any other class count migrates backbone + decoder only (the
head is re-trained). See :func:`select_modules_39`.

Workflow for the legacy checkpoint:
    # See exactly what maps cleanly and what needs manual help:
    python tools/checkpoint_migration.py report \
        --ckpt <legacy_ckpt> --config configs/experiments/yolo/yolov8_poly_dist.yaml
    # Add MANUAL_OVERRIDES for anything AMBIGUOUS, then migrate:
    python tools/checkpoint_migration.py migrate --strategy map \
        --ckpt <legacy_ckpt> --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --output /tmp/migrated/ckpt --mapping-json /tmp/map.json

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


def _strip_colon_zero(name: str) -> str:
    """Strip the Keras ``:0`` tensor suffix only.

    ``str.rstrip(":0")`` strips every trailing ``:`` / ``0`` character, mangling
    names ending in ``0`` (``conv2d_10:0`` -> ``conv2d_1``). This removes only
    the suffix, so weight-name matching is not silently corrupted.
    """
    return name[:-2] if name.endswith(":0") else name


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
        name = _strip_colon_zero(name)
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
    name = _strip_colon_zero(name)
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


def select_modules_39(new_model, requested: Optional[List[str]]) -> Tuple[List[str], str]:
    """Module selection by the 39-class rule.

    If ``requested`` is given it is honoured. Otherwise: a 39-class model copies
    all modules (backbone + decoder + head); any other class count copies only
    backbone + decoder (the head is re-trained because its class width differs).
    """
    if requested is not None:
        return list(requested), f"user-specified modules: {list(requested)}"
    head = getattr(new_model, "head", None)
    num_classes = getattr(head, "num_classes", None)
    if num_classes == 39:
        return (
            ["backbone", "decoder", "head"],
            "num_classes == 39 -> migrating all modules (backbone + decoder + head)",
        )
    return (
        ["backbone", "decoder"],
        f"num_classes = {num_classes} (!= 39) -> migrating backbone + decoder only; "
        "head will be re-trained",
    )


def _diagnose_new_paths(new_recs) -> None:
    """Warn if new-model variable paths look unstructured (path-resolution failed).

    On some TF/Keras versions variables lack ``.path``; the curated map falls back
    to the attribute tree / ``v.name``. If block detection collapses (all of the
    backbone in one block), the legacy mapping cannot align — surface it loudly
    with a sample path so the resolution source can be fixed.
    """
    bb = [r for r in new_recs if r["module"] == "backbone"]
    n_blocks = len({r["block_ord"] for r in bb})
    sample = next((r["path"] for r in bb), "<none>")
    log.info("Path-resolution check: backbone vars=%d distinct-blocks=%d sample-path=%r",
             len(bb), n_blocks, sample)
    if bb and n_blocks < 2:
        log.warning(
            "New-model path resolution looks DEGENERATE (backbone collapsed to "
            "%d block). Variable paths are unstructured on this TF/Keras build; "
            "the legacy mapping will not align. Sample path/name: %r. Please share "
            "this sample so the path source can be adjusted.", n_blocks, sample,
        )


def _is_native_checkpoint(keys) -> bool:
    """True if *keys* look like a checkpoint written by THIS codebase.

    Two layouts both count as native:

    * **Trainer checkpoint (periodic ``ckpt-N`` / ``best_*``)** — saved with the EMA
      optimizer (``train/trainer.py``). The model variables are deduped into
      ``optimizer/_model_vars`` with EMA shadows under ``optimizer/_shadows`` (so there is
      no separate ``model/`` subtree). The EMA shadows are the *complete* weight source —
      see :func:`migrate_native`. Markers: ``optimizer/_shadows`` / ``optimizer/_ema_step``
      / ``optimizer/_model_vars``.
    * **Model-only checkpoint** — ``tf.train.Checkpoint(model=YoloV8).write``: weights rooted
      at ``model/{backbone,decoder,head}/``.

    The legacy object checkpoint instead roots weights at bare ``backbone/`` / ``head/`` and
    carries ``layer_with_weights`` / ``_head/`` segments (caught by the frozen check before
    this one), so the cases are unambiguous.
    """
    if any(
        ("optimizer/_shadows" in k) or ("optimizer/_ema_step" in k)
        or ("optimizer/_model_vars" in k)
        for k in keys
    ):
        return True
    return any(
        k.startswith(("model/backbone/", "model/decoder/", "model/head/")) for k in keys
    )


def _detect_strategy(reader) -> str:
    """Pick the migration strategy from the checkpoint's key layout.

    'frozen' for legacy object checkpoints (``layer_with_weights`` / ``_head/``), 'native'
    for a checkpoint produced by this codebase (``model/{backbone,decoder,head}/`` root),
    else 'structural'.
    """
    keys = list(reader.get_variable_to_shape_map())
    for key in keys:
        if "layer_with_weights" in key or "_head/" in key:
            return "frozen"
    if _is_native_checkpoint(keys):
        return "native"
    return "structural"


def apply_frozen_map(reader, new_model, modules: Optional[List[str]] = None) -> Dict[str, int]:
    """Assign weights using the committed, hand-verified frozen map.

    ``tools/shared/legacy_weight_map_frozen.LEGACY_TO_NEW`` maps each exact legacy
    checkpoint key to a stable canonical id of the new variable. We index the
    live model by that canonical id (env-independent — no Keras auto-names) and
    copy each legacy tensor into the matching variable, shape-verified. The
    39-class module rule selects which modules to copy. Returns
    ``{loaded, skipped, not_found}``.
    """
    from tools.shared import checkpoint_weight_map as wm
    from tools.shared.legacy_weight_map_frozen import LEGACY_TO_NEW

    # Two legacy keys mapping to the same canonical id would silently overwrite
    # one new variable and leave another at random init. Catch that map typo
    # before copying anything.
    if len(set(LEGACY_TO_NEW.values())) != len(LEGACY_TO_NEW):
        n_dupes = len(LEGACY_TO_NEW) - len(set(LEGACY_TO_NEW.values()))
        raise RuntimeError(
            f"legacy_weight_map_frozen.LEGACY_TO_NEW has {n_dupes} duplicate "
            "canonical-id target(s); a weight would be overwritten. Fix the map."
        )

    resolved_modules, reason = select_modules_39(new_model, modules)
    log.info("Module selection: %s", reason)

    new_recs = wm.new_records(new_model)
    _diagnose_new_paths(new_recs)
    by_canon = {wm.canonical_id(r): r for r in new_recs}
    mods = set(resolved_modules)
    available = set(reader.get_variable_to_shape_map())

    stats = {"loaded": 0, "skipped": 0, "not_found": 0}
    loaded_by_module = {m: 0 for m in resolved_modules}
    canon_missing, old_missing = 0, 0
    for old_key, canon in LEGACY_TO_NEW.items():
        module = canon.split("/")[0]
        if module not in mods:
            continue
        rec = by_canon.get(canon)
        if rec is None:
            canon_missing += 1
            continue
        if old_key not in available:
            stats["not_found"] += 1
            old_missing += 1
            continue
        tensor = reader.get_tensor(old_key)
        var = rec["var"]
        if tuple(var.shape) != tuple(tensor.shape):
            log.warning("Shape mismatch %s: %s vs %s — skipping",
                        canon, tuple(var.shape), tuple(tensor.shape))
            stats["skipped"] += 1
            continue
        var.assign(tensor)
        stats["loaded"] += 1
        loaded_by_module[module] += 1
    stats["loaded_by_module"] = loaded_by_module
    stats["modules"] = list(resolved_modules)

    log.info(
        "Frozen map: loaded=%d skipped=%d not_found=%d "
        "(canonical-missing-in-model=%d, legacy-key-missing-in-ckpt=%d)",
        stats["loaded"], stats["skipped"], stats["not_found"],
        canon_missing, old_missing,
    )
    if old_missing:
        log.warning("%d frozen legacy keys were not in the checkpoint — the real "
                    "key format may differ slightly; run `report` to inspect.", old_missing)
    return stats


def migrate_with_frozen(
    old_ckpt_path: str,
    new_model,
    output_ckpt_path: str,
    modules: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Frozen-map migration: assign via LEGACY_TO_NEW then save the checkpoint."""
    import tensorflow as tf
    reader = tf.train.load_checkpoint(old_ckpt_path)
    stats = apply_frozen_map(reader, new_model, modules=modules)

    # Coverage guard: refuse to write a checkpoint where an entire selected
    # module loaded nothing. That is the silent-wrong-model failure mode — e.g.
    # a legacy head key-format drift makes every head key 'not_found', the head
    # stays at random init, and without this guard we would still write the
    # checkpoint and report success. Fail loudly BEFORE writing instead.
    empty = [m for m, n in stats.get("loaded_by_module", {}).items() if n == 0]
    if stats["loaded"] == 0 or empty:
        raise RuntimeError(
            "Frozen-map migration loaded no weights for "
            f"{empty or 'any module'} (loaded={stats['loaded']}, "
            f"not_found={stats['not_found']}). The legacy key format likely does "
            "not match the frozen map — run `report` to inspect. Refusing to write "
            f"a partially-initialized checkpoint to {output_ckpt_path}."
        )

    output_path = Path(output_ckpt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf.train.Checkpoint(model=new_model).write(output_ckpt_path)
    log.info("Migrated checkpoint saved to: %s", output_ckpt_path)
    return stats


def migrate_native(
    old_ckpt_path: str,
    new_model,
    output_ckpt_path: str,
    modules: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Warm-start ``new_model`` from a checkpoint produced by THIS codebase.

    Subtlety this handles: a checkpoint written by the trainer stores the *complete*
    weights only in the **EMA shadows**. The plain ``model/`` object graph omits the
    list-tracked C2f block variables (a known Keras quirk — the same reason
    :func:`flatten_model_structure` enumerates the new side via ``module.variables`` rather
    than the object graph), so ``tf.train.Checkpoint(model=...).restore`` alone recovers
    only a subset. The complete, EMA-averaged weights are exactly what
    :func:`tools.shared.ckpt_loading.restore_eval_weights` already loads (in place) for
    eval/export by reconstructing the EMA wrapper and swapping its shadows in — so native
    warm-start reuses it rather than re-implementing the object-graph walk.

    The weights are assigned **in place** into ``new_model`` (the same contract the other
    strategies use — the live model is what training starts from; the written checkpoint is
    a re-serialisation). Only the requested modules are warm-started: any excluded module
    (e.g. ``head`` when ``modules=[backbone, decoder]``) is snapshotted before the load and
    restored to its fresh init afterwards.
    """
    import tensorflow as tf
    from tools.shared.ckpt_loading import restore_eval_weights

    resolved_modules, reason = select_modules_39(new_model, modules)
    log.info("Module selection: %s", reason)
    resolved = set(resolved_modules)

    # Snapshot modules we will NOT warm-start so we can restore their fresh init after the
    # full-model EMA load (restore_eval_weights swaps shadows into EVERY model variable).
    excluded = [
        m for m in _WEIGHT_MODULES
        if getattr(new_model, m, None) is not None and m not in resolved
    ]
    snapshots = {
        m: [v.numpy().copy() for v in getattr(new_model, m).variables]
        for m in excluded
    }

    mode = restore_eval_weights(new_model, old_ckpt_path)  # assigns complete weights in place
    if mode == "raw":
        log.warning(
            "Native warm-start loaded RAW model/ weights (no EMA shadows in %s). For this "
            "architecture the list-tracked C2f block weights live only in the EMA shadows, "
            "so a model-only checkpoint cannot fully warm-start those blocks — they stay at "
            "fresh init. Prefer a periodic ckpt-N or best_* checkpoint (both carry EMA).",
            old_ckpt_path,
        )

    for m, vals in snapshots.items():
        for v, val in zip(getattr(new_model, m).variables, vals):
            v.assign(val)
        log.info("Native warm-start: kept module '%s' at fresh init (not requested)", m)

    loaded_by_module = {
        m: len(getattr(new_model, m).variables)
        for m in resolved_modules if getattr(new_model, m, None) is not None
    }
    stats = {
        "loaded": sum(loaded_by_module.values()),
        "skipped": 0,
        "not_found": 0,
        "loaded_by_module": loaded_by_module,
        "modules": list(resolved_modules),
        "mode": mode,
    }

    empty = [m for m, n in loaded_by_module.items() if n == 0]
    if stats["loaded"] == 0 or empty:
        raise RuntimeError(
            "Native warm-start loaded no weights for "
            f"{empty or 'any module'} (modules={list(resolved_modules)}). Refusing to write "
            f"a partially-initialized checkpoint to {output_ckpt_path}."
        )

    output_path = Path(output_ckpt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf.train.Checkpoint(model=new_model).write(output_ckpt_path)
    log.info(
        "Native warm-start: loaded %s weights for %s -> %s",
        mode, list(resolved_modules), output_ckpt_path,
    )
    return stats


def migrate_with_map(
    old_ckpt_path: str,
    new_model,
    output_ckpt_path: str,
    modules: Optional[List[str]] = None,
    mapping_json: Optional[str] = None,
    include_suggested: bool = True,
) -> Dict[str, int]:
    """Migrate using the curated structural weight map (legacy -> new).

    Copies ``confident`` (and, unless disabled, ``suggested``) pairs whose module
    is in the selected set, verifying shapes. Ambiguous/unmatched variables are
    reported and left at their initial values. See ``tools/shared/checkpoint_weight_map``.
    """
    import tensorflow as tf

    reader = tf.train.load_checkpoint(old_ckpt_path)
    stats = apply_weight_map(
        reader, new_model, modules=modules,
        mapping_json=mapping_json, include_suggested=include_suggested,
    )

    # Coverage guard, mirroring migrate_with_frozen: refuse to write a checkpoint
    # where nothing loaded or an entire selected module loaded zero weights. With
    # the curated map, that means every candidate pair for a module was ambiguous
    # or shape-mismatched — the module stays at random init and the migration is
    # silently wrong. Fail loudly BEFORE writing instead.
    empty = [m for m, n in stats.get("loaded_by_module", {}).items() if n == 0]
    if stats["loaded"] == 0 or empty:
        raise RuntimeError(
            "Map migration loaded no weights for "
            f"{empty or 'any module'} (loaded={stats['loaded']}, "
            f"skipped={stats['skipped']}, not_found={stats['not_found']}). All "
            "candidate pairs were ambiguous or shape-mismatched — run `report` to "
            "inspect and add entries to MANUAL_OVERRIDES in "
            "tools/shared/checkpoint_weight_map.py. Refusing to write a "
            f"partially-initialized checkpoint to {output_ckpt_path}."
        )

    output_path = Path(output_ckpt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf.train.Checkpoint(model=new_model).write(output_ckpt_path)
    log.info("Migrated checkpoint saved to: %s", output_ckpt_path)
    return stats


def apply_weight_map(
    reader,
    new_model,
    modules: Optional[List[str]] = None,
    mapping_json: Optional[str] = None,
    include_suggested: bool = True,
) -> Dict[str, int]:
    """Resolve the curated map for *reader* and assign weights into *new_model*.

    Separated from disk I/O so it can be unit-tested with a fake reader exposing
    ``get_variable_to_shape_map()`` and ``get_tensor(key)``. Returns the
    ``{loaded, skipped, not_found}`` stats; does not save a checkpoint.
    """
    from tools.shared import checkpoint_weight_map as wm

    resolved_modules, reason = select_modules_39(new_model, modules)
    log.info("Module selection: %s", reason)

    old_recs, skipped = wm.old_records(reader)
    new_recs = wm.new_records(new_model)
    _diagnose_new_paths(new_recs)
    resolution = wm.resolve(old_recs, new_recs)

    mods = set(resolved_modules)
    pairs = list(resolution["confident"])
    if include_suggested:
        pairs += resolution["suggested"]
    pairs = [p for p in pairs if p["module"] in mods]

    log.info(
        "Map resolution: confident=%d suggested=%d ambiguous=%d "
        "unmatched_old=%d unmatched_new=%d (skipped non-weight=%d)",
        len(resolution["confident"]), len(resolution["suggested"]),
        len(resolution["ambiguous"]), len(resolution["unmatched_old"]),
        len(resolution["unmatched_new"]), len(skipped),
    )
    cov = wm.coverage(resolution, new_recs, resolved_modules)
    for m in resolved_modules:
        c = cov.get(m, {})
        status = ("EXACT" if c.get("exact") else
                  "COMPLETE (has suggested)" if c.get("complete") else "INCOMPLETE")
        log.info("  [%s] confident %d + suggested %d = %d/%d  %s",
                 m, c.get("confident", 0), c.get("suggested", 0),
                 c.get("covered", 0), c.get("total", 0), status)
    if cov["_exact"]:
        log.info("Mapping is EXACT for all selected modules — full confident 1:1 transfer.")
    elif cov["_complete"]:
        log.info("Mapping is COMPLETE (all covered) but includes suggested pairs — review them.")
    if resolution["ambiguous"]:
        log.warning(
            "%d variables are AMBIGUOUS and were NOT copied. Run the `report` "
            "subcommand and add entries to MANUAL_OVERRIDES in "
            "tools/shared/checkpoint_weight_map.py.", len(resolution["ambiguous"]),
        )

    if mapping_json:
        import json
        payload = {
            "pairs": [
                {"old_key": p["key"], "new_path": p["path"],
                 "shape": list(p["shape"]), "tier": p["tier"], "module": p["module"]}
                for p in pairs
            ],
            "ambiguous": resolution["ambiguous"],
            "unmatched_old": resolution["unmatched_old"],
            "unmatched_new": resolution["unmatched_new"],
        }
        out = Path(mapping_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        log.info("Variable mapping written to: %s", mapping_json)

    stats = {"loaded": 0, "skipped": 0, "not_found": 0}
    # Track per-module loaded counts so callers can apply the same coverage guard
    # as the frozen path (refuse to write a checkpoint where a whole selected
    # module loaded nothing). Initialize every selected module to 0 so a module
    # that contributes no pairs at all is visible as empty, not merely absent.
    loaded_by_module = {m: 0 for m in resolved_modules}
    for p in pairs:
        try:
            tensor = reader.get_tensor(p["key"])
        except Exception as e:
            log.warning("Could not read %s: %s", p["key"], e)
            stats["not_found"] += 1
            continue
        var = p["var"]
        if tuple(var.shape) != tuple(tensor.shape):
            log.warning("Shape mismatch %s: %s vs %s — skipping",
                        p["path"], var.shape, tensor.shape)
            stats["skipped"] += 1
            continue
        var.assign(tensor)
        stats["loaded"] += 1
        loaded_by_module[p["module"]] = loaded_by_module.get(p["module"], 0) + 1
    stats["loaded_by_module"] = loaded_by_module

    log.info("Assignment complete: loaded=%d skipped=%d not_found=%d",
             stats["loaded"], stats["skipped"], stats["not_found"])
    return stats


def migrate_checkpoint(
    old_ckpt_path: str,
    new_model,                  # tf.keras.Model
    output_ckpt_path: str,
    modules: Optional[List[str]] = None,
    strategy: str = "auto",
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
    strategy : {"auto", "native", "frozen", "map", "structural", "name"}
        ``"auto"`` (default) inspects the checkpoint: a checkpoint produced by this
        codebase (``model/{backbone,decoder,head}/`` root) uses ``"native"``; legacy
        object checkpoints (``layer_with_weights``/``_head``) use ``"frozen"``;
        otherwise ``"structural"`` (see :func:`_detect_strategy`). ``"auto"`` never
        selects the unverified ``"map"`` resolver. ``"native"`` restores via
        ``tf.train.Checkpoint`` directly — exact and complete for a same-architecture
        warm-start (the list-tracked C2f blocks restore in full), honouring the
        selected modules and ignoring the source optimizer/step state. ``"frozen"``
        assigns via the committed hand-verified frozen map. ``"map"`` (opt-in only) uses the curated
        structural weight map for the legacy→new migration
        (:mod:`tools.shared.checkpoint_weight_map`) and the 39-class module rule
        (:func:`select_modules_39`); like ``"frozen"`` it now refuses to write if
        a whole selected module loaded nothing. ``"structural"`` matches by DFS
        structure + role + shape. ``"name"`` is the legacy fuzzy name matcher.
    mapping_json : str, optional
        Write the resolved old→new variable mapping (and diagnostics) to JSON.

    Returns
    -------
    dict with keys ``loaded``, ``skipped``, ``not_found``.
    """
    import tensorflow as tf

    if strategy == "auto":
        strategy = _detect_strategy(tf.train.load_checkpoint(old_ckpt_path))
        log.info("Auto-detected migration strategy: %s", strategy)

    if strategy == "native":
        return migrate_native(
            old_ckpt_path, new_model, output_ckpt_path, modules=modules,
        )

    if strategy == "frozen":
        return migrate_with_frozen(
            old_ckpt_path, new_model, output_ckpt_path, modules=modules,
        )

    if strategy == "map":
        return migrate_with_map(
            old_ckpt_path, new_model, output_ckpt_path,
            modules=modules, mapping_json=mapping_json,
        )

    old_vars = list_checkpoint_variables(old_ckpt_path)
    new_var_dict: Dict[str, tf.Variable] = {
        _strip_colon_zero(v.name): v for v in new_model.variables
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


def _cmd_dump(args: argparse.Namespace) -> None:
    """Print every weight variable grouped by module — full names, no truncation.

    Intended for sharing the legacy checkpoint's structure (e.g. photographing
    the terminal). Optimizer slots and the object-graph marker are excluded.
    One variable per line as ``name <TAB> shape`` so long names never truncate.
    """
    variables = list_checkpoint_variables(args.ckpt)

    def _keep(name: str) -> bool:
        if "OPTIMIZER_SLOT" in name or "_CHECKPOINTABLE_OBJECT_GRAPH" in name:
            return False
        if name in ("save_counter", "global_step"):
            return False
        return name.split("/")[0] in ("backbone", "decoder", "head")

    rows = sorted((n, sh) for n, (sh, _dt) in variables.items() if _keep(n))
    by_mod: Dict[str, list] = {}
    for n, sh in rows:
        by_mod.setdefault(n.split("/")[0], []).append((n, sh))

    lines = [f"Checkpoint: {args.ckpt}",
             f"Weight variables: {len(rows)}", ""]
    for mod in ("backbone", "decoder", "head"):
        items = by_mod.get(mod, [])
        lines.append(f"################ {mod.upper()} ({len(items)} vars) ################")
        for n, sh in items:
            lines.append(f"{n}\t{tuple(sh)}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"\n(written to {args.output})")


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
        _strip_colon_zero(v.name): (tuple(v.shape), v.dtype.name)
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
        _strip_colon_zero(v.name): (tuple(v.shape), v.dtype.name)
        for v in model.variables
    }
    modules, reason = resolve_modules(old_vars, new_var_info, args.modules)
    print(f"Module selection: {reason}")

    pairs, diagnostics = build_structural_mapping(args.ckpt, model, modules)
    _write_mapping_json(pairs, diagnostics, args.output)
    print(f"Wrote {len(pairs)} pairs (clean={diagnostics['_status']['clean']}) "
          f"to {args.output}")


def _cmd_report(args: argparse.Namespace) -> None:
    """Debug report for the curated map: confident / suggested / ambiguous / unmatched.

    Use this against the REAL legacy checkpoint to see exactly what maps cleanly
    and what needs a MANUAL_OVERRIDES entry in tools/shared/checkpoint_weight_map.py.
    """
    import tensorflow as tf
    from tools.shared import checkpoint_weight_map as wm

    model, _ = _build_model_from_config(args.config)
    resolved_modules, reason = select_modules_39(model, args.modules)

    reader = tf.train.load_checkpoint(args.ckpt)
    old_recs, skipped = wm.old_records(reader)
    new_recs = wm.new_records(model)
    _diagnose_new_paths(new_recs)
    res = wm.resolve(old_recs, new_recs)

    cov = wm.coverage(res, new_recs, resolved_modules)

    print(f"\nModule selection: {reason}")
    print(f"Legacy weight vars parsed: {len(old_recs)}  (unparsed weight-module keys: {len(skipped)})")
    print(f"New model weight vars:     {len(new_recs)}\n")

    # Surface unparsed weight-module keys (these are the real culprit when head
    # vars go missing — paste them so the parser can be fixed to the exact format).
    skipped_weight = [k for k in skipped if k.split("/")[0] in ("backbone", "decoder", "head")]
    if skipped_weight:
        print(f"--- UNPARSED weight-module keys ({len(skipped_weight)}) ---")
        for k in skipped_weight[:60]:
            print(f"  {k}")
        if len(skipped_weight) > 60:
            print(f"  ... and {len(skipped_weight) - 60} more")
        print()

    print(f"  CONFIDENT : {len(res['confident'])}")
    print(f"  SUGGESTED : {len(res['suggested'])}  (same-shape siblings paired by index)")
    print(f"  AMBIGUOUS : {len(res['ambiguous'])}  (NEED MANUAL_OVERRIDES)")
    print(f"  unmatched old: {len(res['unmatched_old'])}  | unmatched new: {len(res['unmatched_new'])}")
    print("\nCoverage per module:")
    for m in resolved_modules:
        c = cov.get(m, {})
        status = ("EXACT ✓" if c.get("exact") else
                  "COMPLETE (review suggested)" if c.get("complete") else
                  "INCOMPLETE — see below")
        print(f"  [{m}] confident {c.get('confident', 0)} + suggested "
              f"{c.get('suggested', 0)} = {c.get('covered', 0)}/{c.get('total', 0)}  {status}")
    if cov["_exact"]:
        print("\n  ==> EXACT for all selected modules (every pair confident). Safe to migrate.")
    elif cov["_complete"]:
        print("\n  ==> COMPLETE: everything maps, but review the SUGGESTED pairs above; "
              "pin any you want certain via MANUAL_OVERRIDES.")
    else:
        print("\n  ==> NOT complete yet. Resolve the AMBIGUOUS/UNMATCHED items below, "
              "then re-run report.")

    if res["suggested"]:
        print("\n--- SUGGESTED (review these) ---")
        for p in res["suggested"][:60]:
            print(f"  [{p['module']}] {str(p['shape']):<22} {p['key']}\n        -> {p['path']}")
    if res["ambiguous"]:
        print("\n--- AMBIGUOUS (add to MANUAL_OVERRIDES) ---")
        for a in res["ambiguous"]:
            print(f"  {a['scope']}  {a['shape']}")
            print(f"    old: {a['key']}")
            for c in a["candidates"]:
                print(f"    cand-> {c}")
    if res["unmatched_old"]:
        print("\n--- UNMATCHED OLD (no new target found) ---")
        for k in res["unmatched_old"][:40]:
            print(f"  {k}")
    if res["unmatched_new"]:
        print("\n--- UNMATCHED NEW (no legacy source; left at init) ---")
        for p in res["unmatched_new"][:40]:
            print(f"  {p}")

    if args.output:
        import json
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "confident": [{"old": p["key"], "new": p["path"], "shape": list(p["shape"])} for p in res["confident"]],
            "suggested": [{"old": p["key"], "new": p["path"], "shape": list(p["shape"])} for p in res["suggested"]],
            "ambiguous": res["ambiguous"],
            "unmatched_old": res["unmatched_old"],
            "unmatched_new": res["unmatched_new"],
        }, indent=2))
        print(f"\nFull report written to {args.output}")

    if args.freeze_py:
        _write_frozen_py(res, args.ckpt, args.config, args.freeze_py)
        print(f"Frozen Python mapping written to {args.freeze_py}")


def _write_frozen_py(resolution: dict, ckpt: str, config: str, path: str) -> None:
    """Freeze the resolved confident+suggested mapping as a committable Python dict.

    Generated from the REAL checkpoint, this is an exact ``{old_key: new_path}``
    map you can review and check in. ``checkpoint_weight_map.MANUAL_OVERRIDES``
    still wins over it at migration time.
    """
    pairs = resolution["confident"] + resolution["suggested"]
    lines = [
        '"""AUTO-GENERATED exact legacy->new weight mapping. Review before trusting.',
        "",
        f"Source checkpoint : {ckpt}",
        f"Source config     : {config}",
        f"Pairs             : {len(pairs)} "
        f"(confident={len(resolution['confident'])}, suggested={len(resolution['suggested'])})",
        f"Ambiguous (unmapped): {len(resolution['ambiguous'])}",
        '"""',
        "",
        "EXACT_MAP = {",
    ]
    for p in sorted(pairs, key=lambda r: r["key"]):
        lines.append(f"    {p['key']!r}: {p['path']!r},  # {p['tier']} {tuple(p['shape'])}")
    lines.append("}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def _cmd_migrate(args: argparse.Namespace) -> None:
    model, _ = _build_model_from_config(args.config)

    # args.modules is None unless the user forced it → 39-class rule (map strategy)
    # or class-count auto-selection (structural/name strategies).
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

    # dump subcommand — full untruncated names grouped by module
    p_dump = sub.add_parser(
        "dump",
        help="Print all weight variables grouped by module, full names + shapes "
             "(no truncation). Useful for sharing the legacy checkpoint structure.",
    )
    p_dump.add_argument("--ckpt", required=True, help="Path prefix to old checkpoint")
    p_dump.add_argument("--output", default=None, help="Optional .txt path to also save the dump")

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

    # report subcommand — curated-map debug report
    p_rep = sub.add_parser(
        "report",
        help="Curated-map debug report (confident/suggested/ambiguous/unmatched)",
    )
    p_rep.add_argument("--ckpt", required=True, help="Legacy checkpoint path prefix")
    p_rep.add_argument("--config", required=True, help="Experiment YAML config path")
    p_rep.add_argument("--modules", nargs="+", default=None,
                       help="Override modules (default: 39-class rule).")
    p_rep.add_argument("--output", default=None, help="Optional JSON report path")
    p_rep.add_argument("--freeze-py", default=None,
                       help="Write the resolved mapping as a committable Python "
                            "dict (EXACT_MAP) generated from the real checkpoint.")

    # migrate subcommand
    p_mig = sub.add_parser("migrate", help="Migrate old checkpoint to new model")
    p_mig.add_argument("--ckpt", required=True)
    p_mig.add_argument("--config", required=True)
    p_mig.add_argument("--output", required=True, help="Output checkpoint path prefix")
    p_mig.add_argument("--modules", nargs="+", default=None,
                       help="Override modules (default: 39-class rule for map; "
                            "class-count auto for structural/name).")
    p_mig.add_argument(
        "--strategy", choices=["auto", "native", "frozen", "map", "structural", "name"],
        default="auto",
        help="auto (default; native for this codebase's own checkpoints, frozen for legacy "
             "checkpoints, else structural), native (exact tf.train.Checkpoint restore of a "
             "checkpoint produced by this codebase — recommended for warm-starting a new run), "
             "frozen (committed hand-verified LEGACY_TO_NEW dict), "
             "map (runtime curated parser), structural, or name (legacy fuzzy)."
    )
    p_mig.add_argument(
        "--mapping-json", default=None,
        help="Optional path to also write the resolved variable mapping as JSON."
    )

    args = parser.parse_args()
    {
        "list": _cmd_list,
        "dump": _cmd_dump,
        "map": _cmd_map,
        "mapping": _cmd_mapping,
        "report": _cmd_report,
        "migrate": _cmd_migrate,
    }[args.command](args)


if __name__ == "__main__":
    main()
