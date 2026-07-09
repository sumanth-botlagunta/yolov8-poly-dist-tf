"""Export a trained checkpoint to a SavedModel matching the deployed SNPE DLC.

Drop-in replacement for the deployed on-device Qualcomm SNPE DLC: the exported
SavedModel feeds the existing snpe-tensorflow-to-dlc conversion / quantization /
net-run / extraction pipeline unchanged. Unlike export_saved_model.py, it emits
per-head raw tensors on the device contract (no in-graph NMS) and bakes /255 so
the device can feed raw [0, 255] pixels; the forward pass runs in float32.

Device contract (input node, then one flat [N, C] tensor per head with the FPN
levels concatenated 3→4→5, channels-last, batch dim dropped):

    node         shape                dtype    status
    input_image  [1, 672, 416, 3]     float32  pixels in [0, 255] (/255 baked in)
    box          [N, 4]   (5733, 4)   float32  DFL-decoded LTRB, pre-stride
    cls          [N, 39]  (5733, 39)  float32  raw class logits
    poly_angle   [N, 24]  (5733, 24)  float32  raw (pre-sigmoid)
    poly_dist    [N, 24]  (5733, 24)  float32  raw (pre-softplus)
    poly_conf    [N, 24]  (5733, 24)  float32  raw (pre-sigmoid)
    dist         [N, 1]   (5733, 1)   float32  raw log-distance

N = total anchors over the 3 FPN levels (672×416 → 84·52 + 42·26 + 21·13 = 5733).
Only box is decoded in-graph (the deployed DLC bakes it in); every other head is
raw, with the on-device YoloV8LayerModified applying sigmoid/softplus/exp and the
stride/anchor/NMS decode.

Usage:
    python utils/export/export_device_savedmodel.py \
        --config     configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/ckpt-or-epoch \
        --output_dir /path/to/saved_model \
        --input_size 672,416

    snpe-tensorflow-to-dlc --input_network /path/to/saved_model \
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
                        'Device input H,W (comma-separated). Matches the deployed DLC '
                        '(--input_dim input_image 1,H,W,3).')
    flags.DEFINE_bool  ('normalize', True,
                        'Bake /255 into the graph so the device can feed raw [0,255] '
                        'pixels (IMAGE_NROM_FLAG=False). Set False only if the device '
                        'is changed to feed [0,1].')
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
                        'this swap the on-device decode applies x-offsets on the y-axis and every '
                        'box is transposed (the host=0.68 / device=0.19 gap). Set False to keep '
                        'the x-first order (decode with this repo).')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)

# Output-node order is irrelevant to the extractor (it reads by name), but we
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
    """Set and assert the global Keras policy is float32.

    The SNPE export must be a pure float32 graph. A leaked mixed_bfloat16 policy is
    silent: the heads are pinned float32 (models/head.py) so head outputs still
    report float32 dtype, but their conv stems would compute in bf16 and carry bf16
    precision, surfacing only later as a ``--verify`` tolerance failure. Asserting
    here fails at the source.
    """
    tf.keras.mixed_precision.set_global_policy('float32')
    compute = tf.keras.mixed_precision.global_policy().compute_dtype
    if compute != 'float32':
        raise RuntimeError(
            f"Global Keras compute policy is '{compute}', not 'float32', even after "
            "set_global_policy('float32'). The SNPE export must be float32. Something "
            "re-enabled mixed precision (e.g. common.runtime_setup.apply_eval_precision_policy "
            "or an earlier import). Run this exporter in a clean process / before any "
            "bfloat16 policy is set."
        )


def main(_):
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8
    from common.ckpt_loading import restore_eval_weights

    h_str, w_str = FLAGS.input_size.split(',')
    H, W = int(h_str), int(w_str)

    # Force a clean float32 graph for the SNPE converter. The training
    # mixed_bfloat16 policy (heads pinned float32) is for throughput only; float32
    # restores from the same checkpoint and avoids bf16 ops the SNPE converter
    # rejects. Do not call common.runtime_setup.apply_eval_precision_policy here —
    # it re-enables bf16.
    tf.keras.mixed_precision.set_global_policy('float32')

    config    = load_config(FLAGS.config)
    model_cfg = config.task.model

    # Re-assert float32 immediately before building the model. load_config (or an
    # earlier import) can leave a mixed_bfloat16 policy active, in which case the
    # conv stems build in bf16 while the heads stay pinned float32 — the frozen
    # float32 SavedModel would then silently disagree with the bf16 reference.
    _force_float32_policy()

    # Build at the device input size. The model is fully convolutional, so a
    # 672×672-trained checkpoint restores and runs at 672×416 unchanged (the
    # input size the deployed DLC runs at).
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

    # Confirm nothing inside build_* re-enabled a non-float32 policy while the
    # layers were being created.
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
    # top-level Identity, and re-emit a v1 SavedModel with those top-level nodes.
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


def build_serving_fn(model, H, W, head_chan, normalize, reg_max=16, debug_taps=False,
                     legacy_box_order=True):
    """Build the device serving tf.function (deployed-DLC contract).

    Bakes /255 (when ``normalize``), runs the raw (deploy=False) model, concatenates
    each head across FPN levels 3→4→5 (row-major), and emits one ``tf.identity``-tagged
    tensor per head named exactly box/cls/poly_*/dist — with the batch dim dropped so
    shapes are ``[N, C]`` (matching the deployed DLC nodes).

    The ``box`` head additionally bakes the DFL "integral" decode the deployed DLC
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
                    # on-device decoder applies the left/right (x) offsets to the y-axis.
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


# Inference batch-norm op names — none of these may survive into a graph that is about to
# be quantized; each is folded into its preceding Conv2D by _fold_batch_norms below.
_BN_OPS = ('FusedBatchNormV3', 'FusedBatchNormV2', 'FusedBatchNorm')


def _fold_batch_norms(gd):
    """Fold inference BatchNorm into the preceding Conv2D, in place on a FROZEN GraphDef.

    Why: a standalone FusedBatchNormV3 left in the graph makes snpe-tensorflow-to-dlc warn
    ``can only merge 1 encoding for src op: .../FusedBatchNormV3 .../Conv2D, but found 0`` and
    keeps a separate BN layer. In float that runs exactly; once QUANTIZED, the BN's per-channel
    scale (gamma/sqrt(var+eps)) is forced into one per-tensor int8 activation encoding, crushing
    narrow-range channels and cascading downstream. Folding the scale into the conv's per-channel
    weights lets SNPE quantize it per-channel correctly — no standalone BN, no bad encoding.

    Math (BN inference): y = (x - mean) * gamma/sqrt(var+eps) + beta. With ``s = gamma/sqrt(var+eps)``
    and a preceding ``conv(x) [+ bias]``: fold ``W' = W * s`` (per output channel) and
    ``bias' = (bias - mean) * s + beta``, then replace BN with a BiasAdd. Numerically identical
    to the original (verified to ~1e-6 on synthetic conv-BN and conv-bias-BN graphs).

    Safe by construction: only folds when the conv output feeds ONLY this BN (else scaling the
    shared conv weights would corrupt the other consumers) and all params are constants (true
    after convert_variables_to_constants_v2). Anything else is skipped, not corrupted. Returns
    (gd, folded_count, skipped_count).
    """
    import numpy as np
    from tensorflow.python.framework import tensor_util

    nodes = {n.name: n for n in gd.node}

    def base(s):
        return s.split(':')[0].lstrip('^')

    consumers = {}
    for n in gd.node:
        for inp in n.input:
            consumers.setdefault(base(inp), []).append(n.name)

    def deref(name):
        n = nodes.get(base(name)); seen = set()
        while n is not None and n.op == 'Identity' and n.name not in seen:
            seen.add(n.name); n = nodes.get(base(n.input[0]))
        return n

    def const_of(name):
        n = deref(name)
        if n is None or n.op != 'Const':
            return None, None
        return tensor_util.MakeNdarray(n.attr['value'].tensor), n

    remove = set(); folded = skipped = 0
    for bn in [n for n in gd.node if n.op in _BN_OPS]:
        prod = deref(bn.input[0]); conv = None; bias_node = None
        if prod is not None and prod.op in ('Conv2D', 'DepthwiseConv2dNative'):
            conv = prod
        elif prod is not None and prod.op in ('BiasAdd', 'AddV2', 'Add'):
            inner = deref(prod.input[0])
            if inner is not None and inner.op in ('Conv2D', 'DepthwiseConv2dNative'):
                conv = inner; bias_node = prod
        if conv is None:
            skipped += 1; continue
        allowed = {bn.name} | ({bias_node.name} if bias_node else set())
        if any(c not in allowed for c in consumers.get(conv.name, [])):
            skipped += 1; continue              # conv feeds something else — folding would corrupt it
        if bias_node and any(c != bn.name for c in consumers.get(bias_node.name, [])):
            skipped += 1; continue
        W, Wn = const_of(conv.input[1]); g, _ = const_of(bn.input[1]); b, _ = const_of(bn.input[2])
        m, _ = const_of(bn.input[3]); v, _ = const_of(bn.input[4])
        if any(x is None for x in (W, g, b, m, v)):
            skipped += 1; continue
        eps = bn.attr['epsilon'].f if bn.attr['epsilon'].f > 0 else 1e-3
        s = (g / np.sqrt(v + eps)).astype(np.float32)
        b0 = np.zeros_like(s)
        if bias_node is not None:
            bv, _ = const_of(bias_node.input[1])
            if bv is not None:
                b0 = bv.astype(np.float32)
        if conv.op == 'Conv2D':
            newW = (W * s.reshape(1, 1, 1, -1)).astype(np.float32)
        else:                                    # DepthwiseConv2dNative [kh,kw,cin,mult]
            kh, kw, cin, mult = W.shape
            newW = (W * s.reshape(1, 1, cin, mult)).astype(np.float32)
        newb = ((b0 - m) * s + b).astype(np.float32)
        Wn.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(newW))
        bc = gd.node.add(); bc.op = 'Const'; bc.name = bn.name + '/fold_bias'
        bc.attr['dtype'].type = tf.float32.as_datatype_enum
        bc.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(newb))
        ba = gd.node.add(); ba.op = 'BiasAdd'; ba.name = bn.name + '/fold_ba'
        ba.input.extend([conv.name, bc.name]); ba.attr['T'].type = tf.float32.as_datatype_enum
        if 'data_format' in conv.attr:
            ba.attr['data_format'].CopyFrom(conv.attr['data_format'])
        for n in gd.node:                        # rewire BN's (output-0) consumers to the BiasAdd
            for i, inp in enumerate(n.input):
                if base(inp) == bn.name and inp.split(':')[-1] in ('0', bn.name):
                    n.input[i] = ba.name
        remove.add(bn.name); folded += 1

    keep = [n for n in gd.node if n.name not in remove]
    del gd.node[:]; gd.node.extend(keep)
    return gd, folded, skipped


def _save_named_savedmodel(serving_fn, head_names, output_dir):
    """Freeze ``serving_fn``, promote each tagged head op to a clean top-level node
    named exactly box/cls/..., and write a v1 SavedModel (SNPE-ready graph)."""
    import shutil
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    cf     = serving_fn.get_concrete_function()
    frozen = convert_variables_to_constants_v2(cf)
    gd     = frozen.graph.as_graph_def()

    # Fold inference BatchNorm into the preceding conv so the DLC has NO standalone
    # FusedBatchNormV3 (which quantizes badly — see _fold_batch_norms). Numerically identical.
    gd, folded, skipped = _fold_batch_norms(gd)
    log.info("BatchNorm fold: folded %d BN into conv, skipped %d", folded, skipped)
    remaining = [n.name for n in gd.node if n.op in _BN_OPS]
    if remaining:
        log.warning("BN fold: %d FusedBatchNorm* op(s) did NOT fold (will quantize poorly): %s",
                    len(remaining), remaining[:5])

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
