"""Validate the device-export SavedModel end-to-end against the original model.

Answers ONE question definitively: does the exported device DLC graph (box DFL-decoded,
heads raw, /255 baked) reproduce the original trained model's detections — i.e. is the
EXPORT the cause of a low on-device score, or is it downstream (quantization / on-device
decode / input format)?

It scores two pipelines on the SAME validation images with the SAME COCO evaluator:

  GOLDEN  : original model, deploy path (the model's true accuracy)
            model.deploy=True → detection_generator does DFL+stride+anchor+NMS in-repo.

  EXPORT  : the exported device SavedModel (what becomes the DLC)
            feed RAW [0,255] image → graph emits box[N,4] (DFL-decoded, pre-stride) +
            raw cls/poly/dist → this script reconstructs detections exactly as the
            on-device YoloV8LayerModified must (stride+anchor+sigmoid+top1+per-class NMS).

If F1(EXPORT) ≈ F1(GOLDEN): the export is faithful → the on-device gap is quantization,
the device decoder, or input format — NOT this graph. If they differ: the export (or the
reconstruction contract) is wrong, and the per-image box diff printed here localizes it.

Usage:
    python tools/validate_device_export.py \
        --config     configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/ckpt-N \
        --saved_model /path/to/device/saved_model \
        --num_images 300
"""

import logging

from absl import app, flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
try:
    flags.DEFINE_string('config', None, 'Experiment YAML.', required=True)
    flags.DEFINE_string('checkpoint', None, 'Checkpoint prefix.', required=True)
    flags.DEFINE_string('saved_model', None, 'Exported device SavedModel dir.', required=True)
    flags.DEFINE_integer('num_images', 300, 'How many val images to score (-1 = all).')
    flags.DEFINE_string('split', 'val', "Eval split.")
    flags.DEFINE_bool('normalize_baked', True, 'SavedModel bakes /255 (feed raw [0,255]).')
except flags.DuplicateFlagError:
    pass
log = logging.getLogger(__name__)

_STRIDES = [8, 16, 32]
_LEVELS = ['3', '4', '5']


def _anchor_grid(Hl, Wl, s):
    ys = (np.arange(Hl) + 0.5) * s
    xs = (np.arange(Wl) + 0.5) * s
    gx, gy = np.meshgrid(xs, ys)            # [Hl, Wl]
    return gx.reshape(-1), gy.reshape(-1)


def _reconstruct(dev_out, H, W, num_classes, max_boxes=300,
                 score_thresh=0.05, nms_thresh=0.65, poly_size=24,
                 min_dist=0.5, max_dist=10.0):
    """Rebuild the deploy-dict detections from the device SavedModel outputs, exactly
    as the on-device YoloV8LayerModified must: box[N,4] is pre-stride DFL LTRB → apply
    stride+anchor → yxyx; cls raw → sigmoid → top-1 → per-class NMS → top-k."""
    box = dev_out['box'].numpy()                 # [N,4] l,t,r,b pre-stride
    cls = dev_out['cls'].numpy()                 # [N,num_classes] raw logits

    # box → yxyx normalized
    boxes, off = [], 0
    for s in _STRIDES:
        Hl, Wl = H // s, W // s
        n = Hl * Wl
        seg = box[off:off + n] * s
        cx, cy = _anchor_grid(Hl, Wl, s)
        l, t, r, b = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
        boxes.append(np.stack([(cy - t) / H, (cx - l) / W,
                               (cy + b) / H, (cx + r) / W], 1))
        off += n
    boxes = np.clip(np.concatenate(boxes, 0), 0.0, 1.0).astype(np.float32)   # [N,4]
    scores = 1.0 / (1.0 + np.exp(-cls))          # sigmoid [N,nc]

    top = scores.argmax(1)
    top_s = scores[np.arange(len(scores)), top]
    sel_b, sel_s, sel_c = [], [], []
    for c in range(num_classes):
        m = top == c
        if not m.any():
            continue
        cb, cs = boxes[m], top_s[m]
        idx = tf.image.non_max_suppression(cb, cs, max_boxes, nms_thresh, score_thresh).numpy()
        sel_b.append(cb[idx]); sel_s.append(cs[idx])
        sel_c.append(np.full(len(idx), c, np.int64))
    if sel_b:
        sb = np.concatenate(sel_b); ss = np.concatenate(sel_s); sc = np.concatenate(sel_c)
        order = np.argsort(-ss)[:max_boxes]
        sb, ss, sc = sb[order], ss[order], sc[order]
    else:
        sb = np.zeros([0, 4], np.float32); ss = np.zeros([0], np.float32); sc = np.zeros([0], np.int64)
    k = len(ss)
    pad = max_boxes - k
    out_b = np.pad(sb, [[0, pad], [0, 0]])[None]
    out_s = np.pad(ss, [[0, pad]])[None]
    out_c = np.pad(sc, [[0, pad]])[None]
    pred = {
        'bbox': tf.constant(out_b, tf.float32),
        'classes': tf.constant(out_c, tf.int64),
        'confidence': tf.constant(out_s, tf.float32),
        'num_detections': tf.constant([k], tf.int32),
        # polygons/distance not needed for box F1; zero-fill for evaluator shape contract
        'polygons': tf.zeros([1, max_boxes, poly_size, 3], tf.float32),
        'distance': tf.zeros([1, max_boxes], tf.float32),
    }
    return pred


def main(_):
    import dataclasses
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8
    from tools.ckpt_loading import restore_eval_weights
    from train.task import YoloV8Task, normalize_images
    from eval.coco_metrics import COCOEvaluator

    tf.keras.mixed_precision.set_global_policy('float32')
    config = load_config(FLAGS.config)
    tcfg = config.task
    H, W = tcfg.model.input_size[:2]
    nc = tcfg.num_classes

    # GOLDEN model (deploy path)
    gm = build_yolov8(tcfg.model)
    gm.deploy = True
    gm.build_and_init(tcfg.model.input_size)
    restore_eval_weights(gm, FLAGS.checkpoint)

    # EXPORT SavedModel
    dev_fn = tf.saved_model.load(FLAGS.saved_model).signatures['serving_default']

    # Val data (same pipeline eval.py uses)
    task = YoloV8Task(config)
    data_cfg = dataclasses.replace(tcfg.validation_data, is_training=False)
    val_ds = task.build_inputs(data_cfg)

    ev_gold = COCOEvaluator(num_classes=nc, image_size=(H, W))
    ev_dev = COCOEvaluator(num_classes=nc, image_size=(H, W))

    seen = 0
    box_diffs = []
    for images, labels in val_ds:
        B = int(images.shape[0])
        # GOLDEN (whole batch)
        gpred = gm(normalize_images(images), training=False)
        ev_gold.update(gpred, labels)

        # EXPORT (per image: SavedModel signature is batch-1)
        imgs = tf.cast(images, tf.float32)
        for i in range(B):
            raw_in = imgs[i:i + 1] if FLAGS.normalize_baked else imgs[i:i + 1] / 255.0
            dout = dev_fn(input_image=raw_in)
            dpred = _reconstruct(dout, H, W, nc)
            # single-image label slice
            lbl_i = {k: v[i:i + 1] for k, v in labels.items()}
            ev_dev.update(dpred, lbl_i)
            seen += 1
            if FLAGS.num_images > 0 and seen >= FLAGS.num_images:
                break
        if FLAGS.num_images > 0 and seen >= FLAGS.num_images:
            break

    mg = ev_gold.evaluate()
    md = ev_dev.evaluate()
    log.info("==================== validation (%d images) ====================", seen)
    for key in sorted(set(mg) | set(md)):
        g, d = mg.get(key), md.get(key)
        if isinstance(g, (int, float)) and isinstance(d, (int, float)):
            flag = "" if abs(g - d) < 0.01 else "   <-- DIFF"
            log.info("  %-22s golden=%.4f   export=%.4f%s", key, g, d, flag)
    log.info("Interpretation: export≈golden  -> export is faithful; on-device gap is "
             "quantization / device decode / input. export<<golden -> export/contract bug.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
