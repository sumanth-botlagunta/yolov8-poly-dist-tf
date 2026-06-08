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
from tools.checkpoint_migration import apply_weight_map, select_modules_39


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
    model = _build(num_classes=39)
    new_recs = wm.new_records(model)
    bd = [r for r in new_recs if r["module"] in ("backbone", "decoder")]

    fake_old, truth = [], {}
    for i, r in enumerate(bd):
        conv_or_bn = "conv" if r["role"] in ("kernel", "bias") else "bn"
        key = (f"{r['module']}/layer_with_weights-{r['block_ord']}/synth{i}/"
               f"{conv_or_bn}/{r['role']}/.ATTRIBUTES/VARIABLE_VALUE")
        fake_old.append({
            "module": r["module"], "block_ord": r["block_ord"], "role": r["role"],
            "shape": r["shape"], "key": key, "idx_hint": wm._new_index_hint(r["path"]),
        })
        truth[key] = r["path"]

    res = wm.resolve(fake_old, new_recs)
    pairs = res["confident"] + res["suggested"]
    assert len(res["ambiguous"]) == 0
    assert all(truth[p["key"]] == p["path"] for p in pairs)
    assert len(pairs) == len(fake_old)


# ---------------------------------------------------------------------------
# apply_weight_map copies values into the correct new variables
# ---------------------------------------------------------------------------

def test_index_siblings_promote_to_confident_and_exact():
    """Same-shape C2f siblings with clean bottleneck indices are CONFIDENT/EXACT."""
    model = _build(num_classes=39)
    new_recs = wm.new_records(model)
    bd = [r for r in new_recs if r["module"] in ("backbone", "decoder")]

    fake_old = []
    for i, r in enumerate(bd):
        nh = wm._new_index_hint(r["path"])  # (bn_i, cv_k)
        cob = "conv" if r["role"] in ("kernel", "bias") else "bn"
        if nh[0] >= 0:                       # bottleneck conv -> legacy model_to_wrap
            sub = f"model_to_wrap/{nh[0]}/_conv{nh[1]}"
        elif nh[1] >= 0:                     # outer cv (shape-unique anyway)
            sub = f"_connect_conv{nh[1]}"
        else:
            sub = f"x{i}"
        key = (f"{r['module']}/layer_with_weights-{r['block_ord']}/{sub}/"
               f"{cob}/{r['role']}/.ATTRIBUTES/VARIABLE_VALUE")
        fake_old.append({
            "module": r["module"], "block_ord": r["block_ord"], "role": r["role"],
            "shape": r["shape"], "key": key,
            "idx_hint": wm._old_index_hint(wm.strip_attr_suffix(key)),
        })

    res = wm.resolve(fake_old, new_recs)
    assert len(res["ambiguous"]) == 0
    # every bottleneck sibling pairing must be confident (index-exact), not suggested
    sib_paths = {r["path"] for r in bd if wm._new_index_hint(r["path"])[0] >= 0}
    conf_paths = {p["path"] for p in res["confident"]}
    assert sib_paths <= conf_paths, "bottleneck siblings should be confident via index"

    cov = wm.coverage(res, new_recs, ["backbone", "decoder"])
    assert cov["backbone"]["covered"] == cov["backbone"]["total"]
    assert cov["decoder"]["covered"] == cov["decoder"]["total"]
    assert cov["_complete"] is True


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
