"""Tests for tools/device/export_device_dlc.py — the on-device DLC export.

The device DLC contract (from the on-device SNPE tooling, see docs/device_export.md):
    input  node  input_image  float32 [1, 672, 416, 3]  pixels in [0,255]
    output nodes box/cls/poly_angle/poly_dist/poly_conf/dist — RAW logits, levels
                 concatenated 3→4→5 channels-last → [1, N, C], N = 5733 @ 672×416.

These tests build a random-init model (no checkpoint), export, and assert the
contract holds — including the SNPE-critical requirement that the GraphDef carries
TOP-LEVEL ops literally named input_image + each head (so `--out_node box`
resolves to `box:0` and the device dumps `box:0.raw`).
"""

import numpy as np
import pytest
import tensorflow as tf

from configs.model_config import ModelConfig
from models.yolo_v8 import build_yolov8
from tools.device import export_device_dlc as ed

H, W = 672, 416
N_ANCHORS = sum((H // s) * (W // s) for s in (8, 16, 32))   # 5733


@pytest.fixture(scope="module")
def full_model():
    tf.keras.mixed_precision.set_global_policy("float32")
    cfg = ModelConfig()                       # all 6 heads
    cfg.input_size = [H, W, 3]
    model = build_yolov8(cfg)
    model.deploy = False
    if getattr(model, "decoder", None) is not None:
        model.decoder.static_resize = True   # mirror the exporter (fixed-size, SNPE-clean)
    model.build_and_init([H, W, 3])
    return model, cfg


def _head_chan(cfg):
    hc = [("box", 64), ("cls", cfg.num_classes)]
    if cfg.with_polygons:
        hc += [("poly_angle", cfg.output_poly_size),
               ("poly_dist", cfg.output_poly_size),
               ("poly_conf", cfg.output_poly_size)]
    if cfg.with_distance:
        hc += [("dist", 1)]
    return hc


def _frozen_conv_bn(use_bias):
    """Tiny conv-BN(-relu) graph frozen to constants, with non-trivial BN params."""
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    m = tf.keras.Sequential([
        tf.keras.Input((16, 16, 3)),
        tf.keras.layers.Conv2D(8, 3, padding="same", use_bias=use_bias),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=use_bias),
        tf.keras.layers.BatchNormalization(),
    ])
    m(tf.zeros((1, 16, 16, 3)))
    rng = np.random.default_rng(0)
    for l in m.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.set_weights([rng.uniform(0.5, 2.0, l.gamma.shape).astype("f"),
                           rng.uniform(-1, 1, l.beta.shape).astype("f"),
                           rng.uniform(-1, 1, l.moving_mean.shape).astype("f"),
                           rng.uniform(0.2, 2.0, l.moving_variance.shape).astype("f")])

    @tf.function
    def f(x):
        return m(x, training=False)
    cf = f.get_concrete_function(tf.TensorSpec([1, 16, 16, 3], tf.float32))
    return convert_variables_to_constants_v2(cf).graph.as_graph_def()


def _run_graph(gd, in_name, out_name, x):
    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(gd, name="")
        with tf.compat.v1.Session(graph=g) as s:
            return s.run(g.get_tensor_by_name(out_name + ":0"),
                         {g.get_tensor_by_name(in_name + ":0"): x})


@pytest.mark.parametrize("use_bias", [False, True])
def test_batch_norm_folded_into_conv(use_bias):
    """_fold_batch_norms removes every FusedBatchNorm* AND is numerically identical.

    A standalone FusedBatchNormV3 left in the exported graph quantizes badly — its
    per-channel scale gets forced into one per-tensor int8 activation encoding (the
    "merge 1 encoding ... found 0" converter warning), crushing narrow-range channels.
    The exporter folds BN into the preceding conv's per-channel weights instead. This
    pins both invariants the fix relies on: no BN op survives, and the float output is
    unchanged — so quantization sees a clean Conv2D+BiasAdd it can quantize per-channel.
    """
    gd = _frozen_conv_bn(use_bias)
    in_name = [n.name for n in gd.node if n.op == "Placeholder"][0]
    out_name = [n.name for n in gd.node if n.op in ("Relu", "FusedBatchNormV3")][-1]
    assert sum(1 for n in gd.node if n.op in ed._BN_OPS) > 0          # precondition: BN present
    x = np.random.default_rng(1).standard_normal((1, 16, 16, 3)).astype("f")
    before = _run_graph(gd, in_name, out_name, x)

    gd2, folded, skipped = ed._fold_batch_norms(gd)
    out2 = out_name if out_name in {n.name for n in gd2.node} else out_name + "/fold_ba"
    assert sum(1 for n in gd2.node if n.op in ed._BN_OPS) == 0        # no BN survives
    assert (folded, skipped) == (2, 0)
    after = _run_graph(gd2, in_name, out2, x)
    assert np.abs(before - after).max() < 1e-4                        # numerically identical


def test_concat_levels_shape_and_order(full_model):
    """_concat_levels flattens row-major and concatenates levels 3→4→5."""
    model, cfg = full_model
    raw = model(tf.zeros([1, H, W, 3]), training=False)
    box = ed._concat_levels(raw["box"], 64)
    assert tuple(box.shape) == (1, N_ANCHORS, 64)

    # First level (stride 8) occupies the first 84*52 anchors, row-major.
    lvl3 = tf.reshape(raw["box"]["3"], [1, -1, 64])
    np.testing.assert_array_equal(box[:, : 84 * 52].numpy(), lvl3.numpy())


@pytest.fixture(scope="module")
def exported(full_model, tmp_path_factory):
    model, cfg = full_model
    out_dir = str(tmp_path_factory.mktemp("dev_export"))
    head_chan = _head_chan(cfg)
    serving_fn = ed.build_serving_fn(model, H, W, head_chan, normalize=True)
    ed._save_named_savedmodel(serving_fn, [n for n, _ in head_chan], out_dir)
    return out_dir, model, head_chan


def test_top_level_op_names_for_snpe(exported):
    """SNPE --out_node needs literal top-level ops input_image + each head."""
    out_dir, _, head_chan = exported
    from tensorflow.python.saved_model import loader_impl
    sm = loader_impl.parse_saved_model(out_dir)
    op_names = {n.name for n in sm.meta_graphs[0].graph_def.node}
    for t in ["input_image"] + [n for n, _ in head_chan]:
        assert t in op_names, f"top-level op '{t}' missing — SNPE --out_node would fail"


def test_signature_shapes(exported):
    """Device-DLC node layout: box DFL-decoded to [N,4], others raw [N,C], no batch."""
    out_dir, _, head_chan = exported
    fn = tf.saved_model.load(out_dir).signatures["serving_default"]
    img255 = np.random.RandomState(0).uniform(0, 255, [1, H, W, 3]).astype(np.float32)
    out = fn(input_image=tf.constant(img255))
    assert set(out.keys()) == {n for n, _ in head_chan}
    for name, c in head_chan:
        oc = 4 if name == "box" else c
        assert tuple(out[name].shape) == (N_ANCHORS, oc)


def test_normalization_baked_in(exported):
    """Raw heads: device([0,255]) == concat(raw-model(img/255)), batch dropped."""
    out_dir, model, head_chan = exported
    fn = tf.saved_model.load(out_dir).signatures["serving_default"]
    img255 = np.random.RandomState(1).uniform(0, 255, [1, H, W, 3]).astype(np.float32)
    out = fn(input_image=tf.constant(img255))
    raw = model(tf.constant(img255) / 255.0, training=False)
    for name, c in head_chan:
        if name == "box":
            continue
        man = ed._concat_levels(raw[name], c)[0].numpy()   # [N, c]
        np.testing.assert_allclose(out[name].numpy(), man, rtol=1e-4, atol=1e-2)


def test_force_float32_policy_passes_when_clean():
    """The float32 guard is a no-op under a float32 global policy."""
    tf.keras.mixed_precision.set_global_policy("float32")
    ed._force_float32_policy()
    assert tf.keras.mixed_precision.global_policy().compute_dtype == "float32"


def test_force_float32_policy_raises_on_leaked_bf16(monkeypatch):
    """If a mixed_bfloat16 policy cannot be cleared, fail loudly at the source.

    This is the root cause of the cryptic `--verify` cls tolerance failure: a leaked
    bf16 policy makes the stems compute bf16 while the float32-pinned heads still
    report float32 dtype, so the export silently diverges from the float32 SavedModel.
    """
    monkeypatch.setattr(tf.keras.mixed_precision, "set_global_policy", lambda *a, **k: None)
    monkeypatch.setattr(tf.keras.mixed_precision, "global_policy",
                        lambda: tf.keras.mixed_precision.Policy("mixed_bfloat16"))
    with pytest.raises(RuntimeError, match="not 'float32'"):
        ed._force_float32_policy()


def test_assert_close_tolerates_benign_accumulation():
    """Benign float32 graph accumulation (tiny relative error) must PASS even when a
    large fraction of elements exceed a strict rtol=1e-5 band."""
    rng = np.random.RandomState(0)
    ref = rng.uniform(-15, 15, size=[1, 100, 39]).astype(np.float32)
    got = ref + rng.normal(0, 7e-4 * 15, size=ref.shape).astype(np.float32)  # ~7e-4 rel
    # Most elements fall outside rtol=1e-5, but the relative magnitude is benign.
    assert np.mean(~np.isclose(got, ref, rtol=1e-5, atol=1e-4)) > 0.5
    ed._assert_close("cls", got, ref)                   # must not raise


def test_assert_close_flags_real_divergence():
    """An O(1) relative error (wrong layout / dropped weights / bf16 stems) must fail
    loudly, naming the real causes."""
    ref = np.ones([1, 100, 39], np.float32)
    got = np.zeros_like(ref)                            # 100% relative error
    with pytest.raises(AssertionError, match="REAL fault"):
        ed._assert_close("cls", got, ref)


def test_graph_is_snpe_compatible(exported):
    """The exported GraphDef must not contain ops the Qualcomm SNPE tensorflow-to-dlc
    converter rejects: StridedSlice with ellipsis_mask/new_axis_mask (from `x[..., c]`
    style indexing), nor a dynamic Shape→Pack reshape subgraph (the device input is
    fixed 1xHxWx3, so reshapes must be static). Regression guard for the C2f channel
    split and the static `_concat_levels` reshape."""
    out_dir, _, _ = exported
    from tensorflow.python.saved_model import loader_impl
    gd = loader_impl.parse_saved_model(out_dir).meta_graphs[0].graph_def

    ops = [n.op for n in gd.node]
    # SNPE's StridedSliceLayerBuilder rejects the strided slices this model used to
    # emit (C2f channel split `y[..., :c]`; FPN dynamic-resize `tf.shape(ref)[1:3]`).
    # The C2f split is now tf.split (Split op) and the resize size is static, so the
    # exported graph must contain NO StridedSlice at all.
    ss = [n.name for n in gd.node if n.op == "StridedSlice"]
    assert not ss, f"StridedSlice present — SNPE StridedSliceLayerBuilder will fail: {ss}"
    # And no dynamic Shape/Pack subgraph (the device input is fixed 1xHxWx3).
    assert "Pack" not in ops, "dynamic Pack reshape subgraph present — SNPE wants static shapes"
    assert "Shape" not in ops, "dynamic Shape op present — reshape/resize size should be static"


def test_box_dfl_decode_matches_reference(exported):
    """The baked box DFL decode (reshape→softmax→Σ·bins) must equal the in-repo
    detection_generator._decode_dfl, per level then concat 3→4→5 — proving the baked
    decode reproduces the in-repo box pipeline and the concat layout is correct.
    Pre-stride.

    The default export uses legacy_box_order=True: it emits [top,left,bottom,right]
    (y-first) to match the on-device box_ops.dist2bbox(ver=1) + (y,x) anchors, so the
    repo-native [left,top,right,bottom] reference is reordered [1,0,3,2] before comparing."""
    out_dir, model, _ = exported
    fn = tf.saved_model.load(out_dir).signatures["serving_default"]
    img255 = np.random.RandomState(2).uniform(0, 255, [1, H, W, 3]).astype(np.float32)
    out = fn(input_image=tf.constant(img255))
    raw = model(tf.constant(img255) / 255.0, training=False)

    parts = []
    for lvl in ["3", "4", "5"]:
        ltrb = model.detection_generator._decode_dfl(tf.cast(raw["box"][lvl], tf.float32))
        parts.append(tf.reshape(ltrb, [1, -1, 4]))
    box_ref = tf.concat(parts, axis=1)[0].numpy()[:, [1, 0, 3, 2]]   # [l,t,r,b] -> [t,l,b,r]

    assert tuple(out["box"].shape) == (N_ANCHORS, 4)
    np.testing.assert_allclose(out["box"].numpy(), box_ref, rtol=1e-3, atol=1e-2)
