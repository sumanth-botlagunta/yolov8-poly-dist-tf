"""Tests for tools/export_device_dlc.py — the legacy-DLC drop-in device export.

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
from tools import export_device_dlc as ed

H, W = 672, 416
N_ANCHORS = sum((H // s) * (W // s) for s in (8, 16, 32))   # 5733


@pytest.fixture(scope="module")
def full_model():
    tf.keras.mixed_precision.set_global_policy("float32")
    cfg = ModelConfig()                       # all 6 heads
    cfg.input_size = [H, W, 3]
    model = build_yolov8(cfg)
    model.deploy = False
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
    out_dir, _, head_chan = exported
    fn = tf.saved_model.load(out_dir).signatures["serving_default"]
    img255 = np.random.RandomState(0).uniform(0, 255, [1, H, W, 3]).astype(np.float32)
    out = fn(input_image=tf.constant(img255))
    assert set(out.keys()) == {n for n, _ in head_chan}
    for name, c in head_chan:
        assert tuple(out[name].shape) == (1, N_ANCHORS, c)


def test_normalization_baked_in(exported):
    """device([0,255]) == raw-model(img/255), concatenated."""
    out_dir, model, head_chan = exported
    fn = tf.saved_model.load(out_dir).signatures["serving_default"]
    img255 = np.random.RandomState(1).uniform(0, 255, [1, H, W, 3]).astype(np.float32)
    out = fn(input_image=tf.constant(img255))
    raw = model(tf.constant(img255) / 255.0, training=False)
    for name, c in head_chan:
        man = ed._concat_levels(raw[name], c).numpy()
        np.testing.assert_allclose(out[name].numpy(), man, rtol=1e-5, atol=1e-4)


def test_decode_equivalence(exported):
    """Splitting the concatenated nodes back to per-level and decoding reproduces
    the deploy path — proving the concat layout is the lossless one the on-device
    YoloV8LayerModified expects."""
    out_dir, model, head_chan = exported
    fn = tf.saved_model.load(out_dir).signatures["serving_default"]
    img255 = np.random.RandomState(2).uniform(0, 255, [1, H, W, 3]).astype(np.float32)
    out = fn(input_image=tf.constant(img255))
    raw = model(tf.constant(img255) / 255.0, training=False)

    counts = [(H // s) * (W // s) for s in (8, 16, 32)]
    hw = [(H // s, W // s) for s in (8, 16, 32)]

    def _split(name, c):
        flat = out[name].numpy()[0]
        per, off = {}, 0
        for lvl, n, (lh, lw) in zip(["3", "4", "5"], counts, hw):
            per[lvl] = tf.constant(flat[off:off + n].reshape(1, lh, lw, c))
            off += n
        return per

    rebuilt = {name: _split(name, c) for name, c in head_chan}
    from_raw = model.detection_generator(rebuilt)
    from_deploy = model.detection_generator(raw)
    np.testing.assert_allclose(from_raw["bbox"].numpy(),
                               from_deploy["bbox"].numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(from_raw["num_detections"].numpy(),
                                  from_deploy["num_detections"].numpy())
