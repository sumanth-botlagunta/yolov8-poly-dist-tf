"""Hand-curated structural weight map: legacy object-checkpoint -> new model.

The legacy checkpoint (e.g. ``ckpt-319992``) and the new model implement the
*same* architecture but with completely different variable naming, and at
different nesting levels. Name matching cannot work; ordering matching cannot
work (legacy enumerates BN-before-conv and alphabetically, the new model uses
Keras creation order). So this module maps by **structure + role + shape**, and
splits every variable into one of three confidence tiers:

    confident  — exactly one (role, shape) candidate in the matching scope, OR a
                 fully-determined head semantic. These are safe to copy.
    suggested  — more than one same-shape sibling in the scope; paired by index
                 order. Very likely correct, but review before trusting.
    ambiguous  — could not be resolved automatically. Listed for you to fill in
                 ``MANUAL_OVERRIDES`` below.

Head (fully determined)
-----------------------
The new head exposes named attributes per level (see ``models/head.py``):
``cv2feat_s1/s2`` (box+poly stem), ``box_pred``, ``cls_s1/s2``, ``cls_pred``,
``pa_pred/pd_pred/pc_pred``, ``dist_s0``, ``dist_pred``. The legacy head keys map
onto these one-to-one (see ``_OLD_HEAD_SEMANTIC``), so the head is always
confident.

Backbone / decoder
------------------
Both enumerate 10 / 6 top blocks in creation order, so the legacy
``layer_with_weights-{N}`` aligns with the N-th new block. Within a block we
match by (role, shape). Shape-distinct conv units (e.g. C2f ``cv1`` 64->64 vs
``cv2`` 96->64) are confident; identical-shape siblings inside a bottleneck
(``cv1``/``cv2`` both 3x3xCxC) are *suggested* by index, or ambiguous if the
index cannot be read.

No offline reference needed
---------------------------
This module derives everything at runtime from (a) the real legacy checkpoint
via ``tf.train.load_checkpoint`` and (b) the live new model. It does NOT read any
spreadsheet or reference doc — point it at the real checkpoint (local or pulled
from the cloud) and it produces an exact, verified mapping.

How to finish the mapping (with the real 39-class checkpoint)
-------------------------------------------------------------
1. Run ``python tools/checkpoint_migration.py report --ckpt <legacy> --config
   <yaml>`` to print confident / suggested / ambiguous / unmatched and a per-
   module EXACT vs COMPLETE status. With the real checkpoint, shape-distinct and
   index-resolvable variables are CONFIDENT; only genuinely undecidable cases
   remain.
2. ``--freeze-py <path>`` writes the resolved mapping as a committable
   ``EXACT_MAP`` dict generated from that checkpoint, for review/version control.
3. For anything ambiguous (or a suggested pair you want pinned), add
   ``"<old_checkpoint_key>": "<new_variable_path>"`` to ``MANUAL_OVERRIDES``
   (overrides always win). Re-run ``report`` until EXACT, then ``migrate``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

ROLES = ("kernel", "bias", "gamma", "beta", "moving_mean", "moving_variance")
WEIGHT_MODULES = ("backbone", "decoder", "head")


def _strip_colon_zero(name: str) -> str:
    """Strip the Keras ``:0`` tensor suffix only.

    ``str.rstrip(":0")`` strips every trailing ``:`` / ``0`` character, which
    mangles legitimate names ending in ``0`` (``conv2d_10:0`` -> ``conv2d_1``,
    ``bn_0:0`` -> ``bn_``). This removes the suffix and nothing else.
    """
    return name[:-2] if name.endswith(":0") else name

# ---------------------------------------------------------------------------
# Manual overrides — fill these for anything `report` lists as ambiguous/wrong.
# Key   = exact legacy checkpoint key (incl. /.ATTRIBUTES/VARIABLE_VALUE).
# Value = new variable path (the v.path shown by `report`, e.g.
#         "backbone/stem_c2f/bn0/cv1/conv2d_4/kernel").
# ---------------------------------------------------------------------------
MANUAL_OVERRIDES: Dict[str, str] = {
    # "backbone/layer_with_weights-2/model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE":
    #     "backbone/stem_c2f/bn0/cv1/conv2d_4/kernel",
}

# Legacy head sub-path -> (semantic_name) used by the new head attributes.
# Order of the regexes matters (longest / most specific first).
# Separators between the group name and its layer index vary across checkpoints
# (``cv2feat_layer_with_weights-0`` vs ``cv2feat/layer_with_weights-0`` vs
# ``cv2feat/0``), so the patterns accept ``_``, ``/`` or ``layer_with_weights-``.
def _sub(group: str, idx: int) -> str:
    return rf"{group}[_/](?:layer_with_weights[-_])?0*{idx}(?:/|$)"


_OLD_HEAD_SEMANTIC: List[Tuple[str, str]] = [
    (_sub("cv2feat", 0), "cv2feat_s1"),
    (_sub("cv2feat", 1), "cv2feat_s2"),
    (_sub("cv3", 0),     "cls_s1"),
    (_sub("cv3", 1),     "cls_s2"),
    (_sub("cv3", 2),     "cls_pred"),
    (_sub("cv4", 0),     "dist_s0"),
    (_sub("cv4", 1),     "dist_pred"),
    (r"poly_angle",      "pa_pred"),
    (r"poly_dist",       "pd_pred"),
    (r"poly_conf",       "pc_pred"),
    (r"(^|/)box(/|$)",   "box_pred"),
]

# New head semantics that exist per level (subset present depends on config).
_NEW_HEAD_SEMANTICS = (
    "cv2feat_s1", "cv2feat_s2", "box_pred",
    "cls_s1", "cls_s2", "cls_pred",
    "pa_pred", "pd_pred", "pc_pred",
    "dist_s0", "dist_pred",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_attr_suffix(key: str) -> str:
    """Drop the object-checkpoint ``/.ATTRIBUTES/VARIABLE_VALUE`` tail."""
    return re.split(r"/\.ATTRIBUTES", key, maxsplit=1)[0]


def _checkpoint_order(reader) -> Dict[str, int]:
    """Return ``{checkpoint_key: tracking_order_index}`` from the object graph.

    TF2 object checkpoints store the trackable tree in the order layers/variables
    were created. A DFS over it gives each variable a deterministic *architecture
    order* index — the real construction order, not the alphabetical order of the
    flat variable map. This is what lets us pair same-shape siblings correctly.
    """
    order: Dict[str, int] = {}
    try:
        from tensorflow.core.protobuf import (  # type: ignore
            trackable_object_graph_pb2 as tog,
        )
        og = tog.TrackableObjectGraph()
        og.ParseFromString(reader.get_tensor("_CHECKPOINTABLE_OBJECT_GRAPH"))
        nodes = og.nodes
        visited = {0}
        counter = [0]

        def walk(node_id: int) -> None:
            node = nodes[node_id]
            for attr in node.attributes:
                if attr.name == "VARIABLE_VALUE":
                    order[attr.checkpoint_key] = counter[0]
                    counter[0] += 1
            for child in node.children:
                if child.node_id in visited:
                    continue
                visited.add(child.node_id)
                walk(child.node_id)

        for child in nodes[0].children:
            if child.node_id not in visited:
                visited.add(child.node_id)
                walk(child.node_id)
    except Exception:  # pragma: no cover - defensive across TF versions
        pass
    return order


def role_of(name_or_tail: str) -> Optional[str]:
    """Map a leaf name/tail to a standardized role (handles truncation).

    Robust to full scoped names: takes the last ``/`` segment first, so
    ``backbone/stem_conv1/conv2d/kernel:0`` -> ``kernel``.
    """
    n = _strip_colon_zero(name_or_tail.split("/")[-1]).lstrip("_")
    # exact first
    if n in ROLES:
        return n
    # truncated tails like 'moving_varia...' / 'moving_mean/...'. Use an 8-char
    # prefix so 'moving_mean' ('moving_m') and 'moving_variance' ('moving_v')
    # stay distinguishable — a 6-char prefix ('moving') collides and silently
    # mis-roles variance as mean, swapping BN running stats.
    for r in ROLES:
        if n.startswith(r[:8]):
            return r
    return None


def _variable_path_map(module, module_name: str) -> Dict[int, str]:
    """Build {id(var): 'module/attr/.../leaf'} from the module attribute tree.

    Works without ``KerasVariable.path`` (which only exists on Keras 3): walks
    the tf.Module attribute hierarchy and labels each variable by its attribute
    path. Used as a fallback when ``v.path`` is unavailable (older Keras / raw
    ResourceVariables).
    """
    out: Dict[int, str] = {}
    target_ids = {id(v) for v in module.variables}
    try:
        for path, val in module._flatten(
            predicate=lambda v: id(v) in target_ids, with_path=True
        ):
            out[id(val)] = "/".join([module_name] + [str(p) for p in path])
    except Exception:  # pragma: no cover - defensive across TF versions
        pass
    return out


def _resolve_path(v, module_name: str, path_map: Dict[int, str]) -> str:
    """Best available structural path for a variable, across Keras versions."""
    p = getattr(v, "path", None)
    if isinstance(p, str) and p:
        return p
    p = path_map.get(id(v))
    if p:
        return p
    # last resort: the (possibly scoped) variable name
    return _strip_colon_zero(getattr(v, "name", ""))


def _new_block_name(path: str, module: str) -> str:
    """First path segment after the module (the top block, e.g. 'stem_c2f')."""
    p = re.sub(r"^yolo_v8/", "", path)
    segs = p.split("/")
    # find the module segment, return the segment right after it
    if module in segs:
        i = segs.index(module)
        if i + 1 < len(segs):
            return segs[i + 1]
    return segs[1] if len(segs) > 1 else segs[0]


_LEAF_LAYER = re.compile(r"^(conv2d|conv|batch_normalization|bn)(_\d+)?$")


def _new_subblock(path: str, module: str) -> str:
    """Conv-unit sub-block of a new backbone/decoder variable.

    Returns '' for a plain ConvBnAct block, 'cv1'/'cv2' for C2f/SPPF outer convs,
    and 'bn{i}/cv{k}' for bottleneck convs — i.e. the path between the top block
    and the leaf conv/bn layer. Works for both ``v.path`` (``.../cv1/conv2d/...``)
    and the attribute-tree fallback (``.../cv1/conv/...``).
    """
    p = re.sub(r"^yolo_v8/", "", path)
    segs = p.split("/")
    if module in segs:
        segs = segs[segs.index(module) + 1:]
    if len(segs) < 2:
        return ""
    rest = segs[1:]  # drop the top block name
    for i, s in enumerate(rest):
        if _LEAF_LAYER.match(s):
            return "/".join(rest[:i])
    return "/".join(rest[:-1])  # fallback: drop the role


def _legacy_subblock(struct: str) -> Optional[str]:
    """Translate a legacy backbone/decoder key to the NEW sub-block vocabulary.

    The legacy C2f conv names are architecturally inverted; the correspondence is
    fixed by data-flow role (see tools/legacy_checkpoint_structure.md):
        _route/_conv2 -> cv1,  _connect/_conv1 -> cv2,
        _model_to_wrap/{i}/_conv{k} -> bn{i}/cv{k},
        _conv1 -> cv1, _conv2 -> cv2  (SPPF),  plain block -> ''.
    Returns None if the sub-path is unrecognised (so it is reported, not guessed).
    """
    m = re.search(r"layer_with_weights-\d+/(.*)$", struct)
    if not m:
        return None
    rest = m.group(1)
    # strip the trailing conv/<role> or bn/<role> leaf
    rest = re.sub(r"/?(conv|bn)/[^/]+$", "", rest)
    if rest == "":
        return ""                       # plain ConvBnAct block
    if rest.startswith("_route"):
        return "cv1"
    if rest.startswith("_connect"):
        return "cv2"
    mm = re.match(r"_model_to_wrap/(\d+)/_conv(\d+)$", rest)
    if mm:
        return f"bn{mm.group(1)}/cv{mm.group(2)}"
    if rest == "_conv1":
        return "cv1"                    # SPPF input conv
    if rest == "_conv2":
        return "cv2"                    # SPPF output conv
    return None


# ---------------------------------------------------------------------------
# New-model records (authoritative) — by OBJECT structure, not variable paths.
# ---------------------------------------------------------------------------
#
# Backbone/decoder blocks are stable Python attributes (same code on every TF /
# Keras build), and each _ConvBnAct exposes .conv/.bn, each C2f exposes
# .cv1/.cv2/.bottlenecks (named bn{i} with .cv1/.cv2), each SPPF .cv1/.cv2. We
# walk those objects directly so the (block_ord, subblock, role) identity does
# NOT depend on v.path (which is absent on non-Keras-3 builds). This is what made
# the frozen-map canonical ids match everywhere.

_BACKBONE_BLOCKS = [
    "stem_conv1", "stem_conv2", "stem_c2f", "down1", "c2f_p3",
    "down2", "c2f_p4", "down3", "c2f_p5_pre", "sppf",
]
_DECODER_BLOCKS = [
    "fpn_c2f_p4", "fpn_c2f_p3", "pan_down_p3",
    "pan_c2f_p4", "pan_down_p4", "pan_c2f_p5",
]


def _objrec(module: str, block_ord: int, subblock: str, role: str, var) -> dict:
    return {
        "module": module, "block_ord": block_ord, "subblock": subblock,
        "role": role, "shape": tuple(var.shape), "var": var,
        "path": f"{module}/blk{block_ord}/{subblock or '-'}/{role}",
    }


def _convbnact_records(layer, module: str, block_ord: int, subblock: str) -> List[dict]:
    """Records for a _ConvBnAct (.conv + .bn) or a bare Conv2D (.kernel/.bias)."""
    recs: List[dict] = []
    conv = getattr(layer, "conv", layer)  # _ConvBnAct.conv, or the layer itself
    kernel = getattr(conv, "kernel", None)
    if kernel is not None:
        recs.append(_objrec(module, block_ord, subblock, "kernel", kernel))
        bias = getattr(conv, "bias", None)
        if bias is not None:
            recs.append(_objrec(module, block_ord, subblock, "bias", bias))
    bn = getattr(layer, "bn", None)
    if bn is not None:
        for role in ("gamma", "beta", "moving_mean", "moving_variance"):
            v = getattr(bn, role, None)
            if v is not None:
                recs.append(_objrec(module, block_ord, subblock, role, v))
    return recs


def _block_records(block, module: str, block_ord: int) -> List[dict]:
    """Records for one top-level block: C2f / SPPF / plain ConvBnAct."""
    recs: List[dict] = []
    if hasattr(block, "bottlenecks"):                 # C2f
        recs += _convbnact_records(block.cv1, module, block_ord, "cv1")
        recs += _convbnact_records(block.cv2, module, block_ord, "cv2")
        for i, bott in enumerate(block.bottlenecks):
            recs += _convbnact_records(bott.cv1, module, block_ord, f"bn{i}/cv1")
            recs += _convbnact_records(bott.cv2, module, block_ord, f"bn{i}/cv2")
    elif hasattr(block, "pool"):                      # SPPF
        recs += _convbnact_records(block.cv1, module, block_ord, "cv1")
        recs += _convbnact_records(block.cv2, module, block_ord, "cv2")
    else:                                             # plain _ConvBnAct
        recs += _convbnact_records(block, module, block_ord, "")
    return recs


def _path_based_bd_records(module, module_name: str) -> List[dict]:
    """Fallback: derive backbone/decoder records from variable paths.

    Only used if the expected block attributes are missing (unknown variant).
    """
    recs: List[dict] = []
    path_map = _variable_path_map(module, module_name)
    order: Dict[str, int] = {}
    for v in module.variables:
        block = _new_block_name(_resolve_path(v, module_name, path_map), module_name)
        if block not in order:
            order[block] = len(order)
    for v in module.variables:
        path = _resolve_path(v, module_name, path_map)
        block = _new_block_name(path, module_name)
        recs.append({
            "module": module_name, "block_ord": order[block],
            "subblock": _new_subblock(path, module_name),
            "role": role_of(getattr(v, "name", "") or path),
            "shape": tuple(v.shape), "var": v, "path": path,
        })
    return recs


def new_records(model) -> List[dict]:
    """Flatten the live new model into structural records.

    backbone/decoder records carry ``block_ord`` + ``subblock`` (from the object
    tree); head records carry ``level`` + ``semantic`` (from named attributes).
    Every record has role, shape, the live ``var``, and a readable ``path``.
    """
    recs: List[dict] = []

    for module_name, block_names in (("backbone", _BACKBONE_BLOCKS),
                                     ("decoder", _DECODER_BLOCKS)):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        if all(hasattr(module, n) for n in block_names):
            for block_ord, name in enumerate(block_names):
                recs += _block_records(getattr(module, name), module_name, block_ord)
        else:  # unknown architecture variant — fall back to path parsing
            recs += _path_based_bd_records(module, module_name)

    head = getattr(model, "head", None)
    if head is not None:
        head_path_map = _variable_path_map(head, "head")
        for level in getattr(head, "_levels", []):
            for sem in _NEW_HEAD_SEMANTICS:
                layer = getattr(head, f"{sem}_{level}", None)
                if layer is None:
                    continue
                for v in layer.variables:
                    path = _resolve_path(v, "head", head_path_map)
                    # prefer the semantic+level identity for the head path so it is
                    # stable and readable even without v.path
                    recs.append({
                        "module": "head",
                        "level": str(level),
                        "semantic": sem,
                        "role": role_of(getattr(v, "name", "") or path),
                        "shape": tuple(v.shape),
                        "var": v,
                        "path": path or f"head/{sem}_{level}/{role_of(getattr(v, 'name', '')) or '?'}",
                    })
    return recs


# ---------------------------------------------------------------------------
# Legacy-checkpoint records
# ---------------------------------------------------------------------------

def _parse_old_head(struct: str) -> Optional[Tuple[str, str, str]]:
    """Parse a legacy head key (without suffix) -> (level, semantic, role).

    Handles both observed forms: ``head/_head/3/...`` (with the ``_head``
    container) and ``head/_3/...``. The level is the first standalone digit
    segment (``3``/``_3``/``4``/``_5``); the sub-path after it selects the
    semantic.
    """
    segs = struct.split("/")
    level = None
    lvl_i = None
    for i, s in enumerate(segs[1:], start=1):
        if re.fullmatch(r"_?\d+", s):
            lvl_i = i
            level = re.search(r"\d", s).group(0)
            break
    if lvl_i is None:
        return None
    role = role_of(segs[-1])
    if role is None:
        return None
    sub = "/".join(segs[lvl_i + 1:])
    for pat, sem in _OLD_HEAD_SEMANTIC:
        if re.search(pat, sub):
            return level, sem, role
    return None


def old_records(reader) -> Tuple[List[dict], List[str]]:
    """Read a legacy checkpoint -> (records, skipped_keys).

    ``reader`` is a ``tf.train.load_checkpoint`` reader. Records carry the same
    fields as the matching new records plus ``key`` (the exact checkpoint key).
    """
    shape_map = reader.get_variable_to_shape_map()
    order_map = _checkpoint_order(reader)
    recs: List[dict] = []
    skipped: List[str] = []

    for key, shape in shape_map.items():
        if "OPTIMIZER_SLOT" in key or "_CHECKPOINTABLE_OBJECT_GRAPH" in key:
            continue
        module = key.split("/")[0]
        if module not in WEIGHT_MODULES:
            skipped.append(key)
            continue
        struct = strip_attr_suffix(key)
        shape_t = tuple(shape)
        # architecture/tracking order from the object graph (fallback: large)
        order = order_map.get(key, 1_000_000 + len(recs))

        if module == "head":
            parsed = _parse_old_head(struct)
            if parsed is None:
                skipped.append(key)
                continue
            level, sem, role = parsed
            recs.append({
                "module": "head", "level": level, "semantic": sem,
                "role": role, "shape": shape_t, "key": key, "order": order,
            })
            continue

        # backbone / decoder
        m = re.search(r"layer_with_weights-(\d+)", struct)
        if m is None:
            skipped.append(key)
            continue
        block_ord = int(m.group(1))
        role = role_of(struct.split("/")[-1])
        subblock = _legacy_subblock(struct)
        if role is None or subblock is None:
            skipped.append(key)
            continue
        recs.append({
            "module": module, "block_ord": block_ord, "subblock": subblock,
            "role": role, "shape": shape_t, "key": key, "order": order,
        })
    return recs, skipped


# ---------------------------------------------------------------------------
# Stable canonical id (env-independent — no Keras auto-numbers)
# ---------------------------------------------------------------------------

def canonical_id(rec: dict) -> str:
    """A stable identity for a variable, identical on the old and new side.

    Encodes the architecture position, NOT the Keras auto-name, so it is the
    same across TF/Keras versions and is what the frozen map keys the new side
    on. Head: ``head/L{level}/{semantic}/{role}``; backbone/decoder:
    ``{module}/blk{block_ord}/{subblock or '-'}/{role}``.
    """
    if rec["module"] == "head":
        return f"head/L{rec['level']}/{rec['semantic']}/{rec['role']}"
    sb = rec.get("subblock") or "-"
    return f"{rec['module']}/blk{rec['block_ord']}/{sb}/{rec['role']}"


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve(old_recs: List[dict], new_recs: List[dict]) -> dict:
    """Match legacy -> new. Returns a dict with confident/suggested/ambiguous.

    confident / suggested : list of {"key", "path", "var", "shape", "tier"}
    ambiguous             : list of {"key", "shape", "scope", "candidates"}
    unmatched_old         : list of legacy keys with no new counterpart
    unmatched_new         : list of new paths with no legacy source
    """
    confident: List[dict] = []
    suggested: List[dict] = []
    ambiguous: List[dict] = []

    used_new = set()  # id(var)

    # ---- HEAD: by (level, semantic, role) ----
    new_head = {}
    for r in new_recs:
        if r["module"] == "head":
            new_head.setdefault((r["level"], r["semantic"], r["role"]), []).append(r)
    for o in [r for r in old_recs if r["module"] == "head"]:
        key = (o["level"], o["semantic"], o["role"])
        cands = new_head.get(key, [])
        if len(cands) == 1 and cands[0]["shape"] == o["shape"]:
            n = cands[0]
            confident.append(_pair(o, n, "head"))
            used_new.add(id(n["var"]))
        else:
            ambiguous.append({
                "key": o["key"], "shape": o["shape"],
                "scope": f"head L{o['level']} {o['semantic']} {o['role']}",
                "candidates": [c["path"] for c in cands],
            })

    # ---- BACKBONE / DECODER: by (module, block_ord, role, shape) ----
    for module in ("backbone", "decoder"):
        # Exact architectural key: (block ordinal, conv-unit sub-block, role).
        # The sub-block resolves same-shape siblings by their position in the
        # architecture (cv1 vs cv2, bn0/cv1 vs bn0/cv2), NOT by shape — see
        # tools/legacy_checkpoint_structure.md.
        new_by_sig: Dict[tuple, dict] = {}
        for r in new_recs:
            if r["module"] == module:
                new_by_sig[(r["block_ord"], r["subblock"], r["role"])] = r

        for o in (r for r in old_recs if r["module"] == module):
            sig = (o["block_ord"], o["subblock"], o["role"])
            n = new_by_sig.get(sig)
            if n is not None and n["shape"] == o["shape"]:
                confident.append(_pair(o, n, "arch"))
                used_new.add(id(n["var"]))
            else:
                ambiguous.append({
                    "key": o["key"], "shape": o["shape"],
                    "scope": f"{module} block{o['block_ord']} {o['subblock']} {o['role']}",
                    "candidates": [n["path"]] if n is not None else [],
                })

    # ---- apply MANUAL_OVERRIDES (win over everything) ----
    new_by_path = {r["path"]: r for r in new_recs}
    if MANUAL_OVERRIDES:
        # drop any auto pair whose old key is overridden
        ov_keys = set(MANUAL_OVERRIDES)
        confident = [p for p in confident if p["key"] not in ov_keys]
        suggested = [p for p in suggested if p["key"] not in ov_keys]
        ambiguous = [a for a in ambiguous if a["key"] not in ov_keys]
        old_by_key = {o.get("key"): o for o in old_recs}
        for old_key, new_path in MANUAL_OVERRIDES.items():
            n = new_by_path.get(new_path)
            o = old_by_key.get(old_key)
            if n is None or o is None:
                continue
            confident.append(_pair(o, n, "manual"))
            used_new.add(id(n["var"]))

    matched_old = {p["key"] for p in confident + suggested}
    unmatched_old = [a["key"] for a in ambiguous] + [
        o.get("key") for o in old_recs
        if o.get("key") not in matched_old and o.get("key") not in {a["key"] for a in ambiguous}
    ]
    unmatched_new = [r["path"] for r in new_recs if id(r["var"]) not in used_new]

    return {
        "confident": confident,
        "suggested": suggested,
        "ambiguous": ambiguous,
        "unmatched_old": sorted(set(k for k in unmatched_old if k)),
        "unmatched_new": sorted(unmatched_new),
    }


def _pair(o: dict, n: dict, tier: str) -> dict:
    return {
        "key": o["key"], "path": n["path"], "var": n["var"],
        "shape": n["shape"], "tier": tier, "module": n["module"],
    }


def coverage(resolution: dict, new_recs: List[dict], modules) -> dict:
    """Assess how completely the selected modules are covered by the mapping.

    A module is COMPLETE when every new variable is matched (confident or
    suggested). It is EXACT (the strong guarantee) only when every match is
    CONFIDENT — no suggested, no ambiguous. Returns
    ``{module: {confident, suggested, covered, total, exact, complete}, "_exact",
    "_complete"}``.
    """
    mods = set(modules)
    conf_paths = {p["path"] for p in resolution["confident"]}
    sugg_paths = {p["path"] for p in resolution["suggested"]}

    out: dict = {}
    all_exact = True
    all_complete = True
    for m in mods:
        new_paths = [r["path"] for r in new_recs if r["module"] == m]
        total = len(new_paths)
        confident = sum(1 for p in new_paths if p in conf_paths)
        suggested = sum(1 for p in new_paths if p in sugg_paths)
        covered = confident + suggested
        complete = covered == total and not any(
            a["scope"].startswith(m) for a in resolution["ambiguous"]
        )
        exact = complete and suggested == 0
        out[m] = {
            "confident": confident, "suggested": suggested,
            "covered": covered, "total": total,
            "exact": exact, "complete": complete,
        }
        all_exact = all_exact and exact
        all_complete = all_complete and complete
    out["_exact"] = all_exact
    out["_complete"] = all_complete
    return out
