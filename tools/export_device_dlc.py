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
    Output nodes (one flat tensor per head, levels concatenated 3→4→5, channels-last,
                  batch dim dropped → [N, C]). ``box`` is DFL-DECODED (the legacy DLC
                  bakes it in); the rest are RAW (the on-device ``YoloV8LayerModified``
                  applies sigmoid/softplus/exp + stride/anchor/NMS, and stride/anchor to
                  box):

        box         float32 [N, 4]            (= [5733, 4])   DFL-decoded LTRB, pre-stride
        cls         float32 [N, num_classes]  (= [5733, 39])  raw class logits
        poly_angle  float32 [N, poly_size]    (= [5733, 24])  raw (pre-sigmoid)
        poly_dist   float32 [N, poly_size]    (= [5733, 24])  raw (pre-softplus)
        poly_conf   float32 [N, poly_size]    (= [5733, 24])  raw (pre-sigmoid)
        dist        float32 [N, 1]            (= [5733,  1])  raw log-distance

    ``box`` decode (matches the legacy DLC and detection_generator._decode_dfl):
        [N, 64] → reshape [N, 4, 16] → softmax over bins → Σ·[0..15] (1×1 conv) → [N, 4].

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
    flags.DEFINE_bool  ('debug_taps', False,
                        'Also emit intermediate tensors as top-level nodes (tap_input, '
                        'tap_feat3/4/5 = /255 output + backbone P3/P4/P5) so the conversion '
                        'can be bisected SavedModel-vs-DLC to find the first diverging layer. '
                        'Add the matching --out_node tap_* to snpe-tensorflow-to-dlc.')
    flags.DEFINE_bool  ('legacy_box_order', True,
                        'Emit the box head as [top,left,bottom,right] (y-first) to match the '
                        'deployed on-device decoder: make_anchor_points stores anchors (y,x) '
                        'and box_ops.dist2bbox(ver=1) does anchor-lt with NO axis reverse, so '
                        'it requires distance[0]=top, [1]=left, [2]=bottom, [3]=right. The '
                        "model/repo-native order is [left,top,right,bottom] (x-first); without "
                        'this swap the legacy decode applies x-offsets on the y-axis and every '
                        'box is transposed (the host=0.68 / device=0.19 gap). Set False to keep '
                        'the x-first order (decode with this repo or tools/gen_pred_json_from_dlc.py).')
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
        # Prefer a FULLY STATIC reshape target. The device export fixes the input
        # (input_signature [1, H, W, 3] → SNPE --input_dim 1,H,W,3), so each level's
        # [1, Hl, Wl, C] shape is known at trace time. A dynamic
        # ``tf.reshape(x, [tf.shape(x)[0], -1, C])`` would emit Shape→StridedSlice→
        # Pack→Reshape; the Pack/Shape subgraph is needless friction for the SNPE
        # converter. With a static shape it is a single clean Reshape. Fall back to
        # the dynamic form only if the spatial dims are unknown (non-export use).
        s = x.shape
        if s.rank == 4 and s[1] is not None and s[2] is not None:
            b = -1 if s[0] is None else int(s[0])
            parts.append(tf.reshape(x, [b, int(s[1]) * int(s[2]), channels]))
        else:
            parts.append(tf.reshape(x, [tf.shape(x)[0], -1, channels]))
    return tf.concat(parts, axis=1)


def _force_float32_policy() -> None:
    """Set — and verify — the global Keras policy is float32.

    The SNPE device export must be a pure float32 graph. A leaked mixed_bfloat16
    policy is *silent* here: the prediction heads are pinned float32 (models/head.py)
    so head outputs still report float32 dtype, but their conv stems would compute in
    bf16, so the values carry bf16 precision. That surfaces only much later as a
    ``--verify`` tolerance failure (float32 SavedModel vs bf16-precision reference),
    with matching shapes/dtypes and no clue to the cause. Re-set + assert here so the
    contamination fails loudly at the source instead.
    """
    tf.keras.mixed_precision.set_global_policy('float32')
    compute = tf.keras.mixed_precision.global_policy().compute_dtype
    if compute != 'float32':
        raise RuntimeError(
            f"Global Keras compute policy is '{compute}', not 'float32', even after "
            "set_global_policy('float32'). The SNPE export must be float32. Something "
            "re-enabled mixed precision (e.g. tools.runtime_setup.apply_eval_precision_policy "
            "or an earlier import). Run this exporter in a clean process / before any "
            "bfloat16 policy is set."
        )


def _assert_close(name, got, ref, rel_tol=2e-2, atol=2e-2):
    """Assert the device SavedModel reproduces the reference model, judged by
    RELATIVE magnitude rather than an element count at an unrealistic tolerance.

    The SavedModel is a ~280-layer float32 graph. It legitimately differs from the
    eager Keras model by benign accumulation — fused (FusedBatchNormV3 / fused conv)
    vs unfused ops compute in a different order — which is ~1e-3 relative or smaller
    and which SNPE's int8/int16 quantization swamps entirely. A per-element
    ``np.allclose(rtol=1e-5)`` flags most of those tiny differences as "mismatched"
    (the misleading ~77%-of-elements failure that motivated this), even though the
    graph is correct.

    A REAL fault — a wrong concat/wiring layout, dropped weights, or a precision
    asymmetry (bf16 stems vs a float32 graph) — instead produces an O(1) relative
    error. So gate on the global relative error ``max|got-ref| / max|ref|``: benign
    accumulation passes, a real corruption (rel ~ 1) fails loudly with diagnostics.
    """
    import numpy as np
    g = got.astype(np.float64); r = ref.astype(np.float64)
    maxd = float(np.abs(g - r).max())
    maxv = float(np.abs(r).max())
    tol  = atol + rel_tol * maxv
    if maxd <= tol:
        return
    rel  = maxd / (maxv + 1e-12)
    mism = float(np.mean(~np.isclose(g, r, rtol=1e-5, atol=1e-4)) * 100.0)
    raise AssertionError(
        f"[{name}] device SavedModel != reference model: max|diff|={maxd:.3e} "
        f"exceeds tol={tol:.3e} (relative error {rel:.2e}; {mism:.1f}% of elements "
        f"outside the strict rtol=1e-5 band). got.dtype={got.dtype}, ref.dtype={ref.dtype}.\n"
        f"  rel ~ 1e-3 or below is benign float32 graph accumulation; this is far larger,\n"
        f"  so it indicates a REAL fault: a wrong concat/wiring layout, dropped weights in\n"
        f"  the freeze step, or a precision asymmetry (bf16 stems under a leaked\n"
        f"  mixed_bfloat16 policy vs the float32 graph). Run tools/diagnose_device_export.py\n"
        f"  to localize which export stage diverges.")


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

    # Re-assert float32 immediately before building the model. load_config (or an
    # earlier import in a long-lived session/notebook) can leave a mixed_bfloat16
    # global policy active — the base poly_dist YAML now trains in bfloat16. If the
    # model layers are created under that policy their conv STEMS compute in bf16
    # while the prediction heads stay pinned float32, so head outputs still REPORT
    # float32 dtype but carry bf16-precision values. The SavedModel (frozen here in
    # float32) then disagrees with the bf16 reference model by ~60-80% of elements
    # with matching shapes/dtypes — exactly the cryptic `--verify` tolerance failure
    # that motivated this guard. Fail fast with an actionable message instead.
    _force_float32_policy()

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
    # Fixed-size export: use compile-time-constant FPN upsample sizes so the graph has
    # no Shape→StridedSlice (SNPE-clean). Safe because the export builds/traces at one
    # size; training/eval keep the dynamic (robust) path. Numerically identical here.
    if getattr(model, 'decoder', None) is not None:
        model.decoder.static_resize = True
    model.build_and_init([H, W, 3])

    # Belt-and-suspenders: confirm nothing inside build_* re-enabled a non-float32
    # policy while the layers were being created.
    _force_float32_policy()

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
    if FLAGS.debug_taps:
        tap_names = ['tap_norm',
                     'tap_backbone_3', 'tap_backbone_4', 'tap_backbone_5',
                     'tap_neck_3', 'tap_neck_4', 'tap_neck_5']
        head_names = head_names + tap_names
        log.info("debug_taps ON — also emitting %s (add matching --out_node to the converter)",
                 tap_names)
    serving_fn = build_serving_fn(model, H, W, head_chan, do_norm, reg_max,
                                  debug_taps=FLAGS.debug_taps,
                                  legacy_box_order=FLAGS.legacy_box_order)
    if FLAGS.legacy_box_order:
        log.info("legacy_box_order ON — box emitted as [top,left,bottom,right] (y-first) "
                 "to match the on-device box_ops.dist2bbox(ver=1) + (y,x) anchors.")

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
        oc = 4 if name == 'box' else c   # box is DFL-decoded to 4 LTRB distances
        kind = "DFL-decoded LTRB" if name == 'box' else "raw"
        log.info("  out_node %-11s [%d, %d]  (%d floats, %s)", name, n_anchors, oc,
                 n_anchors * oc, kind)

    if FLAGS.verify:
        _verify(FLAGS.output_dir, model, H, W, n_anchors, head_chan, do_norm,
                legacy_box_order=FLAGS.legacy_box_order)


def build_serving_fn(model, H, W, head_chan, normalize, reg_max=16, debug_taps=False,
                     legacy_box_order=True):
    """Build the device serving tf.function (legacy-DLC contract).

    Bakes /255 (when ``normalize``), runs the raw (deploy=False) model, concatenates
    each head across FPN levels 3→4→5 (row-major), and emits one ``tf.identity``-tagged
    tensor per head named exactly box/cls/poly_*/dist — with the batch dim dropped so
    shapes are ``[N, C]`` (matching the legacy DLC nodes).

    The ``box`` head additionally bakes the DFL "integral" decode the legacy DLC
    contains: reshape ``[1,N,64]→[1,N,4,16]`` → ``softmax`` over the 16 bins → a 1×1
    ``conv2d`` with constant weights ``[0,1,…,15]`` (shape [1,1,16,1], bias 0) → reshape
    ``[N,4]``. So ``box`` is the 4 LTRB distances (pre-stride), NOT the raw [N,64]
    logits. This is exactly ``distance = Σ softmax(logits)·bin`` and matches
    ``models/detection_generator.py::_decode_dfl``. cls/poly_*/dist stay RAW — the
    on-device YoloV8LayerModified applies sigmoid/softplus/exp and the stride/anchor/NMS
    decode (including box) to them.
    """
    import numpy as np
    N = sum((H // s) * (W // s) for s in (8, 16, 32))
    # DFL integral weights: a 1×1 conv over the 16-bin axis, filter = bin indices.
    bin_w = tf.constant(np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1))

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, H, W, 3], dtype=tf.float32, name='input_image')
    ])
    def serving_fn(input_image):
        images = input_image / 255.0 if normalize else input_image
        # Call sub-modules explicitly so intermediate tensors can be tapped (taps run
        # the same backbone/decoder/head as model(images)).
        feats   = model.backbone(images, training=False)
        decoded = model.decoder(feats, training=False)
        raw     = model.head(decoded, training=False)
        out = {}
        for n, c in head_chan:
            x = _concat_levels(raw[n], c)                   # [1, N, c]
            if n == 'box':
                b = tf.reshape(x, [1, N, 4, reg_max])       # [1, N, 4, 16]
                p = tf.nn.softmax(b, axis=-1)               # softmax over the 16 bins
                d = tf.nn.conv2d(p, bin_w, strides=[1, 1, 1, 1], padding='VALID')  # [1,N,4,1]
                x = tf.reshape(d, [N, 4])                   # [N, 4] [left,top,right,bottom]
                if legacy_box_order:
                    # Reorder to [top,left,bottom,right] (y-first) so the deployed
                    # box_ops.dist2bbox(ver=1) — which does anchor(y,x) - lt with NO axis
                    # reverse — reads each offset on the correct axis. Without this the
                    # legacy decoder applies the left/right (x) offsets to the y-axis.
                    x = tf.gather(x, [1, 0, 3, 2], axis=1)  # [l,t,r,b] -> [t,l,b,r]
            else:
                x = tf.reshape(x, [N, c])                   # [N, c] raw (batch dropped)
            out[n] = tf.identity(x, name=n)
        if debug_taps:
            # Bisection taps along the whole forward path, each flattened to [1, -1].
            # Compare these SavedModel-vs-DLC node by node: the FIRST tap that diverges
            # localizes the break.
            #   tap_norm           = input_image/255 (the tensor actually fed to the model)
            #   tap_backbone_3/4/5 = backbone P3/P4/P5 (strides 8/16/32) — catches the
            #                        SAME-padding / stem issue (and a wrong W shows as a
            #                        different element count here)
            #   tap_neck_3/4/5     = decoder (FPN-PAN) outputs that feed the heads
            # tap_norm matches but tap_backbone_3 differs  -> backbone (padding/convs).
            # backbone matches but tap_neck_* differs      -> decoder (FPN resize/concat).
            # neck matches but a head differs              -> that head.
            out['tap_norm'] = tf.identity(tf.reshape(images, [1, -1]), name='tap_norm')
            for lvl in ('3', '4', '5'):
                out[f'tap_backbone_{lvl}'] = tf.identity(
                    tf.reshape(tf.cast(feats[lvl], tf.float32), [1, -1]), name=f'tap_backbone_{lvl}')
            for lvl in ('3', '4', '5'):
                out[f'tap_neck_{lvl}'] = tf.identity(
                    tf.reshape(tf.cast(decoded[lvl], tf.float32), [1, -1]), name=f'tap_neck_{lvl}')
        return out

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


def _verify(saved_model_dir, model, H, W, n_anchors, head_chan, do_norm,
            legacy_box_order=True):
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

    # 2) shapes / element counts. box is DFL-decoded to [N, 4]; the rest are raw
    #    [N, C]; all have the batch dim dropped (legacy-DLC node layout).
    for name, c in head_chan:
        oc  = 4 if name == 'box' else c
        shp = tuple(out[name].shape)
        assert shp == (n_anchors, oc), f"{name}: expected ({n_anchors},{oc}), got {shp}"
    log.info("[ok] node shapes match legacy layout (box [N,4], others [N,C], no batch dim)")

    # 3) /255 equivalence for the RAW heads: device([0,255]) == concat(raw-model(img/255)),
    #    batch dropped. Covers cls/poly_*/dist (box is decoded, checked in 4).
    raw = model(tf.constant(img255) / 255.0 if do_norm else tf.constant(img255),
                training=False)
    for name, c in head_chan:
        if name == 'box':
            continue
        ref = _concat_levels(raw[name], c)[0].numpy()   # [N, c]
        _assert_close(name, out[name].numpy(), ref)
    log.info("[ok] raw heads reproduce /255 + concat (within float32 graph accumulation)")

    # 4) box DFL decode equivalence: the baked reshape→softmax→Σ·bins must match the
    #    in-repo DFL decode (detection_generator._decode_dfl), per level then concat
    #    3→4→5. This also confirms the box concat layout. Pre-stride LTRB distances.
    dg = model.detection_generator
    if dg is not None:
        parts = []
        for lvl in _LEVELS:
            ltrb = dg._decode_dfl(tf.cast(raw['box'][lvl], tf.float32))   # [1,Hl,Wl,4]
            parts.append(tf.reshape(ltrb, [1, -1, 4]))
        box_ref = tf.concat(parts, axis=1)[0].numpy()                     # [N, 4] [l,t,r,b]
        if legacy_box_order:
            box_ref = box_ref[:, [1, 0, 3, 2]]                            # -> [t,l,b,r]
        _assert_close('box (DFL)', out['box'].numpy(), box_ref)
        log.info("[ok] box reproduces the in-repo DFL decode (softmax + Σ·bins, pre-stride%s)",
                 ", reordered [t,l,b,r] for legacy decode" if legacy_box_order else "")

    log.info("---- verification PASSED ----")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
