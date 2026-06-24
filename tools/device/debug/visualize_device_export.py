"""Visualize the device-export model's detections on random val images (at 672×416).

Eyeball check that the DLC graph is producing sensible boxes/polygons: it builds the
exact export graph in-process at the device size (box DFL-decoded LTRB, raw heads, /255
baked — byte-identical to the written SavedModel, proven by export --verify), runs it on
N random validation images, reconstructs detections exactly as the on-device decoder must
(LTRB→stride→anchor→xyxy, sigmoid cls, top-1, per-class NMS; polygons sigmoid/softplus),
and writes one annotated PNG per image (predictions in class colors; GT boxes in white).

Runs the ACTUAL exported SavedModel (the exact artifact converted to the DLC), at its
own native input size (read from the signature, e.g. 672×416), so the picture reflects
exactly what the DLC consumes — no in-process rebuild.

Usage:
    python -m tools.visualize_device_export \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --saved_model /path/to/device/saved_model \
        --num_images 20 --output_dir /tmp/dlc_viz
"""

import logging
import os
import sys

# repo root is four levels up: tools/device/debug/<file> -> tools/device/debug -> tools/device -> tools -> repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from absl import app, flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
try:
    flags.DEFINE_string('config', None, 'Experiment YAML (for the val data + class names).', required=True)
    flags.DEFINE_string('saved_model', None, 'The exported device SavedModel — the exact '
                        'artifact that gets converted to the DLC.', required=True)
    flags.DEFINE_integer('num_images', 20, 'How many random val images to render.')
    flags.DEFINE_string('output_dir', '/tmp/dlc_viz', 'Where to write annotated PNGs.')
    flags.DEFINE_float('score_thresh', 0.25, 'Only draw detections with score >= this.')
    flags.DEFINE_bool('draw_gt', True, 'Overlay ground-truth boxes (white).')
    flags.DEFINE_bool('draw_poly', True, 'Overlay predicted polygon contours.')
    flags.DEFINE_integer('seed', 0, 'Shuffle seed for image selection.')
    flags.DEFINE_string('box_order', 'yfirst', "box head order of the SavedModel: 'yfirst' "
                        "(legacy/DLC default, --legacy_box_order; reordered to x-first before "
                        "decode) or 'xfirst' (--legacy_box_order=False).")
except flags.DuplicateFlagError:
    pass
log = logging.getLogger(__name__)

_STRIDES = [8, 16, 32]


def _sigmoid(x):
    out = np.empty_like(x, dtype=np.float32)
    p = x >= 0
    out[p] = 1.0 / (1.0 + np.exp(-x[p]))
    e = np.exp(x[~p])
    out[~p] = e / (1.0 + e)
    return out


def _softplus(x):
    return np.logaddexp(0.0, x).astype(np.float32)   # stable log(1+exp(x))


def _anchor_grid(Hl, Wl, s):
    ys = (np.arange(Hl) + 0.5) * s
    xs = (np.arange(Wl) + 0.5) * s
    gx, gy = np.meshgrid(xs, ys)
    return gx.reshape(-1), gy.reshape(-1)


def _reconstruct_full(dev_out, H, W, nc, poly_size, max_boxes=300,
                      score_thresh=0.05, nms_thresh=0.65, box_order='yfirst'):
    """Full deploy-dict (boxes + activated polygons) from device outputs, carrying the
    polygon heads through the same top-1 / per-class NMS the on-device decoder uses.

    box_order: 'yfirst' ([t,l,b,r] — the legacy/DLC export default) is reordered to x-first
    before decode; 'xfirst' assumes [l,t,r,b] (a --legacy_box_order=False export)."""
    box = dev_out['box'].numpy()
    if box_order == 'yfirst':
        box = box[:, [1, 0, 3, 2]]               # [t,l,b,r] -> [l,t,r,b]
    cls = dev_out['cls'].numpy()
    pa = dev_out['poly_angle'].numpy() if 'poly_angle' in dev_out else None
    pd = dev_out['poly_dist'].numpy() if 'poly_dist' in dev_out else None
    pc = dev_out['poly_conf'].numpy() if 'poly_conf' in dev_out else None
    has_poly = pa is not None

    boxes, off = [], 0
    for s in _STRIDES:
        Hl, Wl = H // s, W // s
        n = Hl * Wl
        seg = box[off:off + n] * s
        cx, cy = _anchor_grid(Hl, Wl, s)
        l, t, r, b = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
        boxes.append(np.stack([(cy - t) / H, (cx - l) / W, (cy + b) / H, (cx + r) / W], 1))
        off += n
    boxes = np.clip(np.concatenate(boxes, 0), 0.0, 1.0).astype(np.float32)
    scores = _sigmoid(cls)
    top = scores.argmax(1)
    top_s = scores[np.arange(len(scores)), top]

    sel_global, sel_cls = [], []
    for c in range(nc):
        m = np.where(top == c)[0]
        if m.size == 0:
            continue
        idx = tf.image.non_max_suppression(boxes[m], top_s[m], max_boxes,
                                           nms_thresh, score_thresh).numpy()
        sel_global.append(m[idx])
        sel_cls.append(np.full(len(idx), c, np.int64))
    if sel_global:
        gi = np.concatenate(sel_global)
        gc = np.concatenate(sel_cls)
        order = np.argsort(-top_s[gi])[:max_boxes]
        gi, gc = gi[order], gc[order]
    else:
        gi = np.zeros([0], int); gc = np.zeros([0], np.int64)
    k = len(gi)

    polys = np.zeros([k, poly_size, 3], np.float32)
    if has_poly and k:
        polys[:, :, 0] = _sigmoid(pc[gi])     # conf
        polys[:, :, 1] = _softplus(pd[gi])    # dist
        polys[:, :, 2] = _sigmoid(pa[gi])     # angle
    return {
        'bbox': boxes[gi],
        'classes': gc,
        'confidence': top_s[gi],
        'num_detections': k,
        'polygons': polys,
    }


def main(_):
    import dataclasses
    from configs.yaml_loader import load_config
    from train.task import YoloV8Task
    from train.viz_utils import render_summary_images, _draw_box
    try:
        from configs.class_map import DETECTION_CLASSES
        class_names = [DETECTION_CLASSES[i] for i in range(len(DETECTION_CLASSES))]
    except Exception:
        class_names = None

    tf.keras.mixed_precision.set_global_policy('float32')
    config = load_config(FLAGS.config)
    tcfg = config.task
    nc = tcfg.num_classes
    poly_size = tcfg.model.output_poly_size

    # Load the ACTUAL exported SavedModel — the exact graph that becomes the DLC — and
    # take its native input size from the signature, so the val images are letterboxed to
    # exactly what the SavedModel/DLC consumes (no rebuild, no size mismatch).
    loaded = tf.saved_model.load(FLAGS.saved_model)
    serving = loaded.signatures['serving_default']
    in_shape = serving.inputs[0].shape          # [1, H, W, 3]
    H, W = int(in_shape[1]), int(in_shape[2])
    tcfg.model.input_size = [H, W, 3]
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    log.info("Loaded SavedModel %s — native input %dx%d (the DLC's size). Rendering %d images → %s",
             FLAGS.saved_model, H, W, FLAGS.num_images, FLAGS.output_dir)

    task = YoloV8Task(config)
    data_cfg = dataclasses.replace(tcfg.validation_data, is_training=False)
    val_ds = task.build_inputs(data_cfg).shuffle(256, seed=FLAGS.seed)

    import cv2
    written = 0
    for images, labels in val_ds:
        B = int(images.shape[0])
        imgs = tf.cast(images, tf.float32)
        for i in range(B):
            out = serving(input_image=imgs[i:i + 1])     # the real SavedModel, raw [0,255] in
            pred = _reconstruct_full(out, H, W, nc, poly_size, box_order=FLAGS.box_order)
            # display threshold (NMS already ran at 0.05; filter for a clean overlay)
            keep = pred['confidence'] >= FLAGS.score_thresh
            pred = {'bbox': pred['bbox'][keep], 'classes': pred['classes'][keep],
                    'confidence': pred['confidence'][keep],
                    'num_detections': int(keep.sum()), 'polygons': pred['polygons'][keep]}

            img01 = (imgs[i].numpy() / 255.0)
            canvas = render_summary_images([img01], [pred], draw_box=True,
                                           draw_poly=FLAGS.draw_poly and tcfg.with_polygons,
                                           class_names=class_names)[0]
            if FLAGS.draw_gt:
                ng = int(labels['n_gt'][i])
                gb = labels['bbox'].numpy()[i, :ng]
                for j in range(ng):
                    _draw_box(canvas, gb[j, 0], gb[j, 1], gb[j, 2], gb[j, 3],
                              (255, 255, 255), 'gt')
            path = os.path.join(FLAGS.output_dir, f'val_{written:03d}.png')
            cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            written += 1
            if written >= FLAGS.num_images:
                break
        if written >= FLAGS.num_images:
            break
    log.info("Wrote %d annotated images to %s (predictions=class colors, GT=white).",
             written, FLAGS.output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
