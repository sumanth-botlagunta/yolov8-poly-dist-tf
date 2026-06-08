"""Tests for the curated legacy->new structural weight map.

The legacy checkpoint uses a different naming scheme at different nesting levels
(``backbone/layer_with_weights-N/...``, ``head/_head/{level}/cv3/...``) than the
new model. These tests verify that:

  * select_modules_39 applies the 39-class rule,
  * the head maps fully and confidently,
  * the backbone/decoder resolver recovers a synthesized legacy enumeration,
  * apply_weight_map copies values into the correct new variables via a fake
    checkpoint reader (no real legacy checkpoint required).
"""

import numpy as np
import tensorflow as tf

from configs.model_config import ModelConfig
from models.yolo_v8 import build_yolov8
from tools import checkpoint_weight_map as wm
from tools.checkpoint_migration import (
    apply_frozen_map,
    apply_weight_map,
    select_modules_39,
)


_H = _W = 128


def _build(num_classes=39, polys=True, dist=True):
    cfg = ModelConfig(
        input_size=[_H, _W, 3], num_classes=num_classes,
        with_polygons=polys, with_distance=dist, deploy=False,
    )
    model = build_yolov8(cfg)
    model.build_and_init(cfg.input_size)
    return model


class _FakeReader:
    """Minimal stand-in for tf.train.load_checkpoint's reader."""

    def __init__(self, shapes, tensors):
        self._shapes = shapes
        self._tensors = tensors

    def get_variable_to_shape_map(self):
        return self._shapes

    def get_tensor(self, key):
        return self._tensors[key]


# ---------------------------------------------------------------------------
# Module selection (39-class rule)
# ---------------------------------------------------------------------------

def test_39_class_rule_includes_head():
    model = _build(num_classes=39)
    mods, _ = select_modules_39(model, None)
    assert mods == ["backbone", "decoder", "head"]


def test_non_39_class_rule_excludes_head():
    model = _build(num_classes=10)
    mods, _ = select_modules_39(model, None)
    assert mods == ["backbone", "decoder"]


def test_explicit_modules_override_rule():
    model = _build(num_classes=39)
    mods, _ = select_modules_39(model, ["backbone"])
    assert mods == ["backbone"]


# ---------------------------------------------------------------------------
# Head is fully + confidently mapped
# ---------------------------------------------------------------------------

def test_head_maps_fully_confident():
    """Every new head var has a confident legacy source (synthesized)."""
    model = _build(num_classes=39)
    new_recs = wm.new_records(model)
    head_new = [r for r in new_recs if r["module"] == "head"]

    # Build a synthetic legacy head enumeration with the legacy naming scheme.
    sem_to_old = {
        "cv2feat_s1": "cv2feat_layer_with_weights-0",
        "cv2feat_s2": "cv2feat_layer_with_weights-1",
        "cls_s1": "cv3/layer_with_weights-0",
        "cls_s2": "cv3/layer_with_weights-1",
        "cls_pred": "cv3/layer_with_weights-2",
        "dist_s0": "cv4/layer_with_weights-0",
        "dist_pred": "cv4/layer_with_weights-1",
        "pa_pred": "poly_angle", "pd_pred": "poly_dist", "pc_pred": "poly_conf",
        "box_pred": "box",
    }
    old_recs = []
    for r in head_new:
        sub = sem_to_old[r["semantic"]]
        # conv vs bn role placement mirrors the legacy keys
        if r["role"] in ("kernel", "bias"):
            tail = f"{sub}/conv/{r['role']}" if "pred" not in r["semantic"] or r["semantic"] in ("box_pred", "pa_pred", "pd_pred", "pc_pred", "dist_pred") else f"{sub}/{r['role']}"
        else:
            tail = f"{sub}/bn/{r['role']}"
        key = f"head/_head/{r['level']}/{tail}/.ATTRIBUTES/VARIABLE_VALUE"
        old_recs.append({
            "module": "head", "level": r["level"], "semantic": r["semantic"],
            "role": r["role"], "shape": r["shape"], "key": key,
        })

    res = wm.resolve(old_recs, new_recs)
    head_conf = [p for p in res["confident"] if p["module"] == "head"]
    assert len(head_conf) == len(head_new), (
        f"head confident {len(head_conf)} != {len(head_new)}; "
        f"ambiguous={len(res['ambiguous'])}"
    )


# ---------------------------------------------------------------------------
# Backbone/decoder resolver recovers a synthesized legacy enumeration
# ---------------------------------------------------------------------------

def test_backbone_decoder_resolver_recovers_truth():
    """A synthetic legacy enumeration carrying the architectural sub-block maps
    1:1 to the new vars (confident, no ambiguous)."""
    model = _build(num_classes=39)
    new_recs = wm.new_records(model)
    bd = [r for r in new_recs if r["module"] in ("backbone", "decoder")]

    fake_old, truth = [], {}
    for i, r in enumerate(bd):
        conv_or_bn = "conv" if r["role"] in ("kernel", "bias") else "bn"
        key = (f"{r['module']}/layer_with_weights-{r['block_ord']}/x{i}/"
               f"{conv_or_bn}/{r['role']}/.ATTRIBUTES/VARIABLE_VALUE")
        fake_old.append({
            "module": r["module"], "block_ord": r["block_ord"],
            "subblock": r["subblock"], "role": r["role"],
            "shape": r["shape"], "key": key,
        })
        truth[key] = r["path"]

    res = wm.resolve(fake_old, new_recs)
    assert len(res["ambiguous"]) == 0
    assert len(res["confident"]) == len(fake_old)
    assert all(truth[p["key"]] == p["path"] for p in res["confident"])


# ---------------------------------------------------------------------------
# legacy C2f sub-block translation (architecture, not shape)
# ---------------------------------------------------------------------------

def test_legacy_c2f_subblock_translation():
    """The inverted legacy C2f names map to the right new conv units."""
    f = wm._legacy_subblock
    assert f("backbone/layer_with_weights-2/_route/_conv2/conv/kernel") == "cv1"
    assert f("backbone/layer_with_weights-2/_connect/_conv1/conv/kernel") == "cv2"
    assert f("backbone/layer_with_weights-2/_model_to_wrap/0/_conv1/conv/kernel") == "bn0/cv1"
    assert f("backbone/layer_with_weights-4/_model_to_wrap/1/_conv2/bn/gamma") == "bn1/cv2"
    assert f("backbone/layer_with_weights-9/_conv1/conv/kernel") == "cv1"   # SPPF
    assert f("backbone/layer_with_weights-9/_conv2/bn/beta") == "cv2"
    assert f("backbone/layer_with_weights-0/conv/kernel") == ""              # plain
    assert f("backbone/layer_with_weights-0/bn/gamma") == ""


def test_new_subblock_extraction():
    """New paths parse to the matching sub-block vocabulary."""
    g = wm._new_subblock
    assert g("backbone/stem_conv1/conv2d/kernel", "backbone") == ""
    assert g("backbone/stem_c2f/cv1/conv2d/kernel", "backbone") == "cv1"
    assert g("backbone/stem_c2f/cv2/conv2d/kernel", "backbone") == "cv2"
    assert g("backbone/stem_c2f/bn0/cv1/conv2d/kernel", "backbone") == "bn0/cv1"
    assert g("backbone/stem_c2f/bn0/cv2/batch_normalization/gamma", "backbone") == "bn0/cv2"
    # attribute-tree fallback form (no .path)
    assert g("backbone/stem_c2f/bn0/cv1/conv/_kernel", "backbone") == "bn0/cv1"


def test_apply_weight_map_transfers_values():
    model = _build(num_classes=39)
    new_recs = wm.new_records(model)

    def find(pred):
        return next(r for r in new_recs if pred(r))

    cases = [
        ("head/_head/3/cv3/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE",
         find(lambda r: r["module"] == "head" and r.get("semantic") == "cls_pred"
              and r["level"] == "3" and r["role"] == "kernel")),
        ("head/_head/3/cv2feat_layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE",
         find(lambda r: r["module"] == "head" and r.get("semantic") == "cv2feat_s1"
              and r["level"] == "3" and r["role"] == "gamma")),
        ("backbone/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE",
         find(lambda r: r["module"] == "backbone" and r["block_ord"] == 0
              and r["role"] == "kernel")),
    ]

    shapes, tensors = {}, {}
    for i, (key, r) in enumerate(cases):
        val = np.full(r["shape"], 0.5 + 0.1 * i, dtype=np.float32)
        shapes[key] = list(r["shape"])
        tensors[key] = val
        r["var"].assign(tf.zeros_like(r["var"]))

    stats = apply_weight_map(_FakeReader(shapes, tensors), model, modules=None)
    assert stats["loaded"] == len(cases)
    assert stats["skipped"] == 0 and stats["not_found"] == 0
    for i, (key, r) in enumerate(cases):
        assert np.allclose(r["var"].numpy(), 0.5 + 0.1 * i), f"value not transferred: {key}"


# ---------------------------------------------------------------------------
# Full legacy-checkpoint simulation: exact 1:1 for ALL 3 modules
# ---------------------------------------------------------------------------

def _legacy_key_for(rec, bord_name):
    """Reconstruct the exact legacy checkpoint key for a new variable record.

    Mirrors tools/legacy_checkpoint_structure.md so the whole mapping can be
    validated against realistic legacy names without the real checkpoint.
    """
    import re

    def leaf(role):
        return f"conv/{role}" if role in ("kernel", "bias") else f"bn/{role}"

    if rec["module"] == "head":
        sem = {
            "cv2feat_s1": "cv2feat/layer_with_weights-0",
            "cv2feat_s2": "cv2feat/layer_with_weights-1",
            "box_pred": "box",
            "cls_s1": "cv3/layer_with_weights-0",
            "cls_s2": "cv3/layer_with_weights-1",
            "cls_pred": "cv3/layer_with_weights-2",
            "dist_s0": "cv4/layer_with_weights-0",
            "dist_pred": "cv4/layer_with_weights-1",
            "pa_pred": "poly_angle", "pd_pred": "poly_dist", "pc_pred": "poly_conf",
        }[rec["semantic"]]
        tail = (leaf(rec["role"]) if rec["semantic"] in
                ("cv2feat_s1", "cv2feat_s2", "cls_s1", "cls_s2", "dist_s0")
                else f"conv/{rec['role']}")
        return f"head/_head/{rec['level']}/{sem}/{tail}/.ATTRIBUTES/VARIABLE_VALUE"

    mod, bo, sb, role = rec["module"], rec["block_ord"], rec["subblock"], rec["role"]
    base = f"{mod}/layer_with_weights-{bo}"
    bn = bord_name[(mod, bo)]
    if sb == "":
        sub = ""
    elif "sppf" in bn:
        sub = "/_conv1" if sb == "cv1" else "/_conv2"
    elif sb == "cv1":
        sub = "/_route/_conv2"
    elif sb == "cv2":
        sub = "/_connect/_conv1"
    else:
        m = re.match(r"bn(\d+)/cv(\d+)", sb)
        sub = f"/_model_to_wrap/{m.group(1)}/_conv{m.group(2)}"
    return f"{base}{sub}/{leaf(role)}/.ATTRIBUTES/VARIABLE_VALUE"


def test_full_legacy_simulation_exact_all_modules():
    """Generate exact legacy names for all 336 vars and assert EXACT 1:1 + values."""
    import re
    model = _build(num_classes=39)
    new_recs = wm.new_records(model)

    bord_name = {}
    for r in new_recs:
        if r["module"] in ("backbone", "decoder"):
            p = re.sub(r"^yolo_v8/", "", r["path"]).split("/")
            name = p[p.index(r["module"]) + 1] if r["module"] in p else p[1]
            bord_name[(r["module"], r["block_ord"])] = name

    shapes, tensors, keymap = {}, {}, {}
    for i, r in enumerate(new_recs):
        key = _legacy_key_for(r, bord_name)
        shapes[key] = list(r["shape"])
        tensors[key] = np.full(r["shape"], (i % 97) * 0.01 + 0.001, dtype=np.float32)
        keymap[key] = r
        r["var"].assign(tf.zeros_like(r["var"]))

    reader = _FakeReader(shapes, tensors)
    stats = apply_weight_map(reader, model, modules=None)
    assert stats == {"loaded": len(new_recs), "skipped": 0, "not_found": 0}

    old_recs, skipped = wm.old_records(reader)
    res = wm.resolve(old_recs, new_recs)
    assert len(skipped) == 0
    assert len(res["ambiguous"]) == 0
    assert len(res["suggested"]) == 0
    cov = wm.coverage(res, new_recs, ["backbone", "decoder", "head"])
    assert cov["_exact"] is True
    for m in ("backbone", "decoder", "head"):
        assert cov[m]["confident"] == cov[m]["total"]

    # every variable received its exact stamped value
    for key, r in keymap.items():
        assert np.allclose(r["var"].numpy(), tensors[key]), f"value mismatch: {key}"


# ---------------------------------------------------------------------------
# Frozen committed map (tools/legacy_weight_map_frozen.py)
# ---------------------------------------------------------------------------

def test_frozen_map_covers_model_one_to_one():
    """The committed LEGACY_TO_NEW dict maps to every model variable, 1:1."""
    from tools.legacy_weight_map_frozen import LEGACY_TO_NEW

    model = _build(num_classes=39)
    new_recs = wm.new_records(model)
    canon_in_model = {wm.canonical_id(r) for r in new_recs}

    assert len(LEGACY_TO_NEW) == len(new_recs) == 336
    # every frozen target exists in the model, and the mapping is injective
    targets = list(LEGACY_TO_NEW.values())
    assert len(set(targets)) == len(targets), "frozen map has duplicate targets"
    assert set(targets) == canon_in_model, "frozen targets != model variables"
    # legacy keys are unique
    assert len(set(LEGACY_TO_NEW)) == len(LEGACY_TO_NEW)


def test_frozen_map_transfers_values_end_to_end():
    """apply_frozen_map copies stamped legacy tensors into the right variables."""
    from tools.legacy_weight_map_frozen import LEGACY_TO_NEW

    model = _build(num_classes=39)
    new_recs = wm.new_records(model)
    by_canon = {wm.canonical_id(r): r for r in new_recs}

    shapes, tensors = {}, {}
    for i, (old_key, canon) in enumerate(LEGACY_TO_NEW.items()):
        r = by_canon[canon]
        shapes[old_key] = list(r["shape"])
        tensors[old_key] = np.full(r["shape"], (i % 89) * 0.01 + 0.002, dtype=np.float32)
        r["var"].assign(tf.zeros_like(r["var"]))

    stats = apply_frozen_map(_FakeReader(shapes, tensors), model, modules=None)
    assert stats == {"loaded": 336, "skipped": 0, "not_found": 0}
    for i, (old_key, canon) in enumerate(LEGACY_TO_NEW.items()):
        r = by_canon[canon]
        assert np.allclose(r["var"].numpy(), tensors[old_key]), f"value mismatch: {canon}"
