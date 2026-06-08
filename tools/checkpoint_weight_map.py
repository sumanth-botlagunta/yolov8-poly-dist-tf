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
_OLD_HEAD_SEMANTIC: List[Tuple[str, str]] = [
    (r"cv2feat_layer_with_weights-0", "cv2feat_s1"),
    (r"cv2feat_layer_with_weights-1", "cv2feat_s2"),
    (r"cv3/layer_with_weights-0",     "cls_s1"),
    (r"cv3/layer_with_weights-1",     "cls_s2"),
    (r"cv3/layer_with_weights-2",     "cls_pred"),
    (r"cv4/layer_with_weights-0",     "dist_s0"),
    (r"cv4/layer_with_weights-1",     "dist_pred"),
    (r"poly_angle",                   "pa_pred"),
    (r"poly_dist",                    "pd_pred"),
    (r"poly_conf",                    "pc_pred"),
    (r"(^|/)box(/|$)",                "box_pred"),
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


def role_of(name_or_tail: str) -> Optional[str]:
    """Map a leaf name/tail to a standardized role (handles truncation).

    Robust to full scoped names: takes the last ``/`` segment first, so
    ``backbone/stem_conv1/conv2d/kernel:0`` -> ``kernel``.
    """
    n = name_or_tail.split("/")[-1].rstrip(":0").lstrip("_")
    # exact first
    if n in ROLES:
        return n
    # truncated tails like 'moving_varia...' / 'moving_mean/...'
    for r in ROLES:
        if n.startswith(r[:6]):
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
    return getattr(v, "name", "").rstrip(":0")


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


# ---------------------------------------------------------------------------
# New-model records (authoritative)
# ---------------------------------------------------------------------------

def new_records(model) -> List[dict]:
    """Flatten the live new model into structural records.

    backbone/decoder records carry ``block_ord``; head records carry
    ``level`` + ``semantic``. Every record has role, shape, the live ``var`` and
    its ``path``.
    """
    recs: List[dict] = []

    for module_name in ("backbone", "decoder"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        path_map = _variable_path_map(module, module_name)
        order: Dict[str, int] = {}
        for v in module.variables:
            path = _resolve_path(v, module_name, path_map)
            block = _new_block_name(path, module_name)
            if block not in order:
                order[block] = len(order)
        for v in module.variables:
            path = _resolve_path(v, module_name, path_map)
            block = _new_block_name(path, module_name)
            recs.append({
                "module": module_name,
                "block_ord": order[block],
                "role": role_of(getattr(v, "name", "") or path),
                "shape": tuple(v.shape),
                "var": v,
                "path": path,
            })

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

        if module == "head":
            parsed = _parse_old_head(struct)
            if parsed is None:
                skipped.append(key)
                continue
            level, sem, role = parsed
            recs.append({
                "module": "head", "level": level, "semantic": sem,
                "role": role, "shape": shape_t, "key": key,
            })
            continue

        # backbone / decoder
        m = re.search(r"layer_with_weights-(\d+)", struct)
        if m is None:
            skipped.append(key)
            continue
        block_ord = int(m.group(1))
        role = role_of(struct.split("/")[-1])
        if role is None:
            skipped.append(key)
            continue
        # index hint for same-shape siblings: bottleneck + conv-unit numbers
        recs.append({
            "module": module, "block_ord": block_ord,
            "role": role, "shape": shape_t, "key": key,
            "idx_hint": _old_index_hint(struct),
        })
    return recs, skipped


def _old_index_hint(struct: str) -> Tuple[int, ...]:
    """Best-effort conv-unit index for disambiguating same-shape siblings.

    Combines the bottleneck index (model_to_wrap/<i>/) and the conv index
    (_conv<k> / cv<k>) so two identical-shape convs in a block can be ordered.
    """
    hint: List[int] = []
    mt = re.search(r"model_to_wrap/(\d+)/", struct)
    hint.append(int(mt.group(1)) if mt else -1)
    cv = re.search(r"_conv(\d+)|/cv(\d+)/|connect_conv(\d+)|route_conv(\d+)", struct)
    if cv:
        hint.append(next(int(g) for g in cv.groups() if g is not None))
    else:
        hint.append(-1)
    return tuple(hint)


def _new_index_hint(path: str) -> Tuple[int, ...]:
    p = re.sub(r"^yolo_v8/", "", path)
    hint: List[int] = []
    bn = re.search(r"/bn(\d+)/", p)
    hint.append(int(bn.group(1)) if bn else -1)
    cv = re.search(r"/cv(\d+)/", p)
    hint.append(int(cv.group(1)) if cv else -1)
    return tuple(hint)


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
        new_by_sig: Dict[tuple, List[dict]] = {}
        for r in new_recs:
            if r["module"] == module:
                new_by_sig.setdefault((r["block_ord"], r["role"], r["shape"]), []).append(r)
        old_by_sig: Dict[tuple, List[dict]] = {}
        for r in old_recs:
            if r["module"] == module:
                old_by_sig.setdefault((r["block_ord"], r["role"], r["shape"]), []).append(r)

        for sig, olds in old_by_sig.items():
            news = new_by_sig.get(sig, [])
            if len(olds) == 1 and len(news) == 1:
                confident.append(_pair(olds[0], news[0], "shape-unique"))
                used_new.add(id(news[0]["var"]))
            elif len(olds) == len(news) and len(olds) > 1:
                # same-shape siblings (e.g. C2f bottleneck cv1/cv2). If both sides
                # carry fully-parsed, distinct index hints that form a bijection,
                # the pairing is deterministic -> CONFIDENT. Otherwise fall back to
                # order-based SUGGESTED for the user to review.
                o_idx = {tuple(o.get("idx_hint", ())): o for o in olds}
                n_idx = {_new_index_hint(n["path"]): n for n in news}
                fully_indexed = (
                    len(o_idx) == len(olds) and len(n_idx) == len(news)
                    and set(o_idx) == set(n_idx)
                    and all(-1 not in k for k in o_idx)
                )
                if fully_indexed:
                    for k, o in o_idx.items():
                        n = n_idx[k]
                        confident.append(_pair(o, n, "index-exact"))
                        used_new.add(id(n["var"]))
                else:
                    o_sorted = sorted(olds, key=lambda r: r.get("idx_hint", ()))
                    n_sorted = sorted(news, key=lambda r: _new_index_hint(r["path"]))
                    for o, n in zip(o_sorted, n_sorted):
                        suggested.append(_pair(o, n, "index"))
                        used_new.add(id(n["var"]))
            else:
                for o in olds:
                    ambiguous.append({
                        "key": o["key"], "shape": o["shape"],
                        "scope": f"{module} block{sig[0]} {sig[1]} {sig[2]}",
                        "candidates": [n["path"] for n in news],
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
