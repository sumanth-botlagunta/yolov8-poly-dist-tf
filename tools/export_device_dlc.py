"""Export a trained checkpoint to a SavedModel laid out as a DROP-IN replacement
for the legacy on-device Qualcomm SNPE DLC.

Unlike ``tools/export_saved_model.py`` (which bakes NMS into the graph and emits
the post-processed deploy dict for [0, 1]-normalized input), this tool reproduces
the *legacy device contract* so the existing SNPE conversion / quantization /
net-run / result-extraction pipeline keeps working **unchanged** — the new ``.dlc``
simply replaces the old one.

Legacy contract (reverse-engineered from the on-device tooling — see the
``snpe-tensorflow-to-dlc`` command and the result-extraction script in
``prompts/dlc_conversion.txt`` / ``docs/device_export.md``):

    Input  node:  ``input_image``   float32  [1, 672, 416, 3]   pixels in [0, 255]
    Output nodes (RAW head logits, one flat tensor per head, levels concatenated
                  3→4→5, channels-last, NO activations / NO DFL decode / NO NMS —
                  the on-device ``YoloV8LayerModified`` does all of that):

        box         float32 [1, N, 4*reg_max]  (= [1, 5733, 64])  raw DFL logits
        cls         float32 [1, N, num_classes] (= [1, 5733, 39])  raw class logits
        poly_angle  float32 [1, N, poly_size]   (= [1, 5733, 24])  raw (pre-sigmoid)
        poly_dist   float32 [1, N, poly_size]   (= [1, 5733, 24])  raw (pre-softplus)
        poly_conf   float32 [1, N, poly_size]   (= [1, 5733, 24])  raw (pre-sigmoid)
        dist        float32 [1, N, 1]           (= [1, 5733,  1])  raw log-distance

    N = total anchors over the 3 FPN levels for the given input size
        (672×416 → 84·52 + 42·26 + 21·13 = 5733).

Two device-specific transforms vs. the [0,1] export contract:
  1. ``input_image`` carries raw [0, 255] pixels (the on-device raw-image generator
     sets ``IMAGE_NROM_FLAG=False``), so this graph divides by 255 internally to
     feed the model the [0, 1] tensors it was trained on (train.task.normalize_images).
  2. The forward pass runs in float32 (NOT the training mixed_bfloat16 policy) so the
     exported GraphDef is a clean float32 graph for the SNPE converter / quantizer.

The per-head concatenation here mirrors ``models/detection_generator.py`` exactly
(reshape each level [B,H,W,C]→[B,H*W,C] row-major, concat levels 3→4→5 on the anchor
axis). ``--verify`` proves this is lossless by reconstructing the per-level dict from
the concatenated nodes and re-running the in-repo decoder (the faithful port of the
on-device ``YoloV8LayerModified``) — its detections must match the deploy path.

Usage:
    python tools/export_device_dlc.py \
        --config     configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/ckpt-or-epoch \
        --output_dir /path/to/saved_model \
        --input_size 672,416 \
        --verify

Then, exactly as before (drop-in — only the SavedModel path changes):
    ./snpe-tensorflow-to-dlc --input_network /path/to/saved_model \
        --output_path model_pre.dlc --input_dim input_image 1,672,416,3 \
        --out_node cls --out_node box --out_node poly_angle \
        --out_node poly_conf --out_node poly_dist --out_node dist
"""

import logging
import os

from absl import app, flags
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',     None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('checkpoint', None, 'Checkpoint path prefix.',          required=True)
    flags.DEFINE_string('output_dir', None, 'Directory to write the SavedModel.', required=True)
    flags.DEFINE_string('input_size', '672,416',
                        'Device input H,W (comma-separated). Matches the legacy DLC '
                        '(--input_dim input_image 1,H,W,3).')
    flags.DEFINE_bool  ('normalize', True,
                        'Bake /255 into the graph so the device can feed raw [0,255] '
                        'pixels (IMAGE_NROM_FLAG=False). Set False only if the device '
                        'is changed to feed [0,1].')
    flags.DEFINE_bool  ('verify', False,
                        'After export, load the SavedModel back and assert node names, '
                        'shapes, /255 equivalence, and decode equivalence vs the deploy path.')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)

# Legacy output-node order is irrelevant (the extractor reads by name), but we
# keep the canonical head order for readable logs / signature.
_LEVELS = ["3", "4", "5"]


def _concat_levels(per_level: dict, channels: int) -> tf.Tensor:
    """Flatten + concat one head across FPN levels → [B, N, channels].

    Mirrors models/detection_generator.py: each level [B,H,W,C] is reshaped to
    [B, H*W, C] (row-major over H then W) and the levels are concatenated 3→4→5
    on the anchor axis. Channels-last, raw (no activation).
    """
    parts = []
    for lvl in _LEVELS:
        x = tf.cast(per_level[lvl], tf.float32)
        b = tf.shape(x)[0]
        parts.append(tf.reshape(x, [b, -1, channels]))
    return tf.concat(parts, axis=1)


def main(_):
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8
    from tools.ckpt_loading import restore_eval_weights

    h_str, w_str = FLAGS.input_size.split(',')
    H, W = int(h_str), int(w_str)

    # Force a clean float32 graph for the SNPE converter. The training
    # mixed_bfloat16 policy (heads pinned float32) is for throughput only; float32
    # is numerically a superset, restores from the same checkpoint, and avoids
    # bf16 ops the SNPE TF converter would choke on. (Do NOT call
    # tools.runtime_setup.apply_eval_precision_policy here — that re-enables bf16.)
    tf.keras.mixed_precision.set_global_policy('float32')

    config    = load_config(FLAGS.config)
    model_cfg = config.task.model

    # Build at the device input size. The model is fully convolutional, so a
    # 672×672-trained checkpoint restores and runs at 672×416 unchanged (same as
    # the legacy export, which also ran 672×416).
    model_cfg.input_size = [H, W, 3]
    poly_size  = model_cfg.output_poly_size
    n_classes  = model_cfg.num_classes
    reg_max    = 16
    with_poly  = model_cfg.with_polygons
    with_dist  = model_cfg.with_distance

    model = build_yolov8(model_cfg)
    model.deploy = False                 # raw head dict, NOT the NMS/deploy path
    model.build_and_init([H, W, 3])

    kind = restore_eval_weights(model, FLAGS.checkpoint)
    log.info("Checkpoint restored (%s weights): %s", kind, FLAGS.checkpoint)

    do_norm = FLAGS.normalize

    # Head order → channel count. tf.identity(name=...) tags each output so the op
    # survives freezing as ``StatefulPartitionedCall/<name>`` and can be promoted to
    # a clean top-level node below.
    head_chan = [('box', 4 * reg_max), ('cls', n_classes)]
    if with_poly:
        head_chan += [('poly_angle', poly_size), ('poly_dist', poly_size), ('poly_conf', poly_size)]
    if with_dist:
        head_chan += [('dist', 1)]
    head_names = [n for n, _ in head_chan]
    serving_fn = build_serving_fn(model, H, W, head_chan, do_norm)

    # The on-device SNPE pipeline resolves ``--out_node box`` to tensor ``box:0`` and
    # dumps ``box:0.raw``, so the GraphDef must contain TOP-LEVEL ops literally named
    # box/cls/poly_*/dist (and input_image). A plain tf.saved_model.save buries them
    # in a StatefulPartitionedCall and renames the outputs to Identity:0.. — so we
    # freeze (inline + variables→constants), promote each tagged op to a clean
    # top-level Identity, and re-emit a v1 SavedModel mirroring the legacy graph.
    _save_named_savedmodel(serving_fn, head_names, FLAGS.output_dir)
    log.info("Device SavedModel written to %s", FLAGS.output_dir)

    # Log the concrete output shapes (the SNPE/.raw element counts).
    n_anchors = sum((H // s) * (W // s) for s in (8, 16, 32))
    log.info("Input: input_image [1, %d, %d, 3] float32, pixels in %s",
             H, W, "[0,255] (/255 baked in)" if do_norm else "[0,1]")
    log.info("Anchors N = %d  (levels: %s)", n_anchors,
             " + ".join(f"{(H//s)}x{(W//s)}" for s in (8, 16, 32)))
    for name, c in head_chan:
        log.info("  out_node %-11s [1, %d, %d]  (%d floats)", name, n_anchors, c, n_anchors * c)

    if FLAGS.verify:
        _verify(FLAGS.output_dir, model, H, W, n_anchors, head_chan, do_norm)


def build_serving_fn(model, H, W, head_chan, normalize):
    """Build the device serving tf.function.

    Bakes /255 (when ``normalize``), runs the raw (deploy=False) model, and emits
    one ``tf.identity``-tagged tensor per head — levels concatenated 3→4→5,
    channels-last, raw logits — named exactly box/cls/poly_*/dist.
    """
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, H, W, 3], dtype=tf.float32, name='input_image')
    ])
    def serving_fn(input_image):
        images = input_image / 255.0 if normalize else input_image
        raw = model(images, training=False)
        return {n: tf.identity(_concat_levels(raw[n], c), name=n) for n, c in head_chan}

    return serving_fn


def _save_named_savedmodel(serving_fn, head_names, output_dir):
    """Freeze ``serving_fn``, promote each tagged head op to a clean top-level node
    named exactly box/cls/..., and write a v1 SavedModel (SNPE-ready graph)."""
    import shutil
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    cf     = serving_fn.get_concrete_function()
    frozen = convert_variables_to_constants_v2(cf)
    gd     = frozen.graph.as_graph_def()

    op_names = {n.name for n in gd.node}
    for name in head_names:
        # The tagged tf.identity survives freezing either as a clean top-level op
        # 'name' (concrete-function freeze) or scoped 'scope/name' (SavedModel
        # freeze). If it is already top-level, keep it; otherwise promote it.
        if name in op_names:
            continue
        cands = [n.name for n in gd.node if n.op == 'Identity' and n.name.split('/')[-1] == name]
        src = next((c for c in cands if c.endswith('/' + name)), cands[0] if cands else None)
        if src is None:
            raise RuntimeError(f"could not locate frozen output op for '{name}'")
        node = gd.node.add()
        node.op = 'Identity'
        node.name = name
        node.input.append(src)
        node.attr['T'].type = tf.float32.as_datatype_enum

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(gd, name='')
        with tf.compat.v1.Session(graph=g) as sess:
            inp  = g.get_tensor_by_name('input_image:0')
            outs = {n: g.get_tensor_by_name(n + ':0') for n in head_names}
            tf.compat.v1.saved_model.simple_save(
                sess, output_dir, inputs={'input_image': inp}, outputs=outs)


def _verify(saved_model_dir, model, H, W, n_anchors, head_chan, do_norm):
    """Assert the exported graph matches the legacy device contract."""
    import numpy as np
    from tensorflow.python.saved_model import loader_impl

    log.info("---- verification ----")
    head_names = [n for n, _ in head_chan]

    # 0) SNPE-critical: the GraphDef must contain TOP-LEVEL ops literally named
    #    input_image + each head, so `--out_node box` resolves to `box:0` and the
    #    extractor finds `box:0.raw`.
    sm = loader_impl.parse_saved_model(saved_model_dir)
    op_names = {n.name for n in sm.meta_graphs[0].graph_def.node}
    missing = [t for t in (['input_image'] + head_names) if t not in op_names]
    assert not missing, f"top-level op(s) absent from GraphDef (SNPE --out_node would fail): {missing}"
    log.info("[ok] top-level graph ops present for SNPE: %s", ['input_image'] + head_names)

    loaded = tf.saved_model.load(saved_model_dir)
    fn = loaded.signatures['serving_default']

    # Deterministic synthetic image in [0,255].
    rng = np.random.RandomState(0)
    img255 = rng.uniform(0, 255, size=[1, H, W, 3]).astype(np.float32)
    out = fn(input_image=tf.constant(img255))

    # 1) signature node names present
    got = set(out.keys())
    assert set(head_names) <= got, f"missing output nodes: {set(head_names) - got} (got {got})"
    log.info("[ok] signature output nodes present: %s", sorted(got))

    # 2) shapes / element counts
    for name, c in head_chan:
        shp = tuple(out[name].shape)
        assert shp == (1, n_anchors, c), f"{name}: expected (1,{n_anchors},{c}), got {shp}"
    log.info("[ok] all node shapes == [1, %d, C]", n_anchors)

    # 3) /255 equivalence: device([0,255]) == raw-model(img/255) concatenated.
    raw = model(tf.constant(img255) / 255.0 if do_norm else tf.constant(img255),
                training=False)
    n_classes = dict(head_chan)['cls']
    man = _concat_levels(raw['cls'], n_classes).numpy()
    np.testing.assert_allclose(out['cls'].numpy(), man, rtol=1e-5, atol=1e-4)
    log.info("[ok] /255 + concat reproduces the raw model exactly")

    # 4) decode equivalence: split the concatenated nodes back into a per-level
    #    dict and run the in-repo YoloV8Layer (the port of the on-device
    #    YoloV8LayerModified). Its detections must match the deploy path that
    #    consumes the native per-level dict — proving the concatenation is the
    #    correct, lossless layout the device decoder expects.
    if model.detection_generator is not None:
        counts = [(H // s) * (W // s) for s in (8, 16, 32)]
        hw     = [(H // s, W // s) for s in (8, 16, 32)]

        def _split(name, c):
            flat = out[name].numpy()[0]               # [N, c]
            per = {}
            off = 0
            for lvl, n, (lh, lw) in zip(_LEVELS, counts, hw):
                per[lvl] = tf.constant(flat[off:off + n].reshape(1, lh, lw, c))
                off += n
            return per

        rebuilt = {name: _split(name, c) for name, c in head_chan}
        from_raw    = model.detection_generator(rebuilt)
        from_deploy = model.detection_generator(raw)
        np.testing.assert_allclose(from_raw['bbox'].numpy(),
                                   from_deploy['bbox'].numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(from_raw['num_detections'].numpy(),
                                      from_deploy['num_detections'].numpy())
        log.info("[ok] decode(raw nodes) == decode(native per-level) — drop-in layout confirmed")

    log.info("---- verification PASSED ----")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
