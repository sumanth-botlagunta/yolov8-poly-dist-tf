"""Validate the device-export SavedModel end-to-end: precision / recall / F1.

You already know the model is good (host eval F1 ≈ 0.71) but the on-device DLC scores
~0.16. This isolates WHERE that drop happens by scoring two pipelines on the SAME
validation images with the SAME metrics:

  GOLDEN  : original model, deploy path (model.deploy=True → in-repo DFL+stride+anchor+NMS)
            — should reproduce your ~0.71.

  EXPORT  : the exported device SavedModel (what becomes the DLC). Feed RAW [0,255] →
            graph emits box[N,4] (DFL-decoded LTRB, pre-stride) + raw cls/poly/dist →
            this script reconstructs detections exactly as the on-device
            YoloV8LayerModified must (LTRB→stride→anchor→xyxy; sigmoid; top-1; per-class NMS).

Reports, for each pipeline:
  - F1score50 / mAP / mAP50 / AR100 from the repo COCOEvaluator (directly comparable to
    your host eval), and
  - direct peak-F1 with its precision, recall, and TP/FP/FN (transparent, IoU≥0.5).

Read-off:
  EXPORT ≈ GOLDEN (≈0.71)  → the exported graph is FAITHFUL; the on-device 0.16 is
                             quantization / on-device decode convention / input format.
  EXPORT << GOLDEN         → the export (or this reconstruction contract) is wrong; the
                             per-pipeline numbers + box format tell you which.

Usage:
    python -m tools.validate_device_export \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/ckpt-N \
        --saved_model /path/to/device/saved_model \
        --num_images 500
"""

import logging
import os
import sys

# Repo root before this script's dir, so `import eval` finds the eval/ package and not
# tools/eval.py (which shadows it when run as `python tools/device/validate_device_export.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from absl import app, flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
try:
    flags.DEFINE_string('config', None, 'Experiment YAML.', required=True)
    flags.DEFINE_string('checkpoint', None, 'Checkpoint prefix (for the GOLDEN reference).', required=True)
    flags.DEFINE_string('saved_model', None, 'The exported device SavedModel — the exact '
                        'artifact that becomes the DLC. Scored at its own native size.',
                        required=True)
    flags.DEFINE_integer('num_images', 500, 'How many val images to score (-1 = all).')
    flags.DEFINE_float('iou_thr', 0.5, 'IoU threshold for the direct precision/recall/F1.')
    flags.DEFINE_bool('normalize_baked', True, 'SavedModel bakes /255 (feed raw [0,255]).')
    flags.DEFINE_string('box_order', 'yfirst', "box head order of the device SavedModel: "
                        "'yfirst' (deployed-DLC default, --legacy_box_order; reordered to "
                        "x-first to compare to the golden) or 'xfirst' (--legacy_box_order=False).")
except flags.DuplicateFlagError:
    pass
log = logging.getLogger(__name__)

_STRIDES = [8, 16, 32]


def _anchor_grid(Hl, Wl, s):
    ys = (np.arange(Hl) + 0.5) * s
    xs = (np.arange(Wl) + 0.5) * s
    gx, gy = np.meshgrid(xs, ys)
    return gx.reshape(-1), gy.reshape(-1)


def _reconstruct(dev_out, H, W, num_classes, max_boxes=300,
                 score_thresh=0.05, nms_thresh=0.65, poly_size=24, box_order='yfirst'):
    """Rebuild deploy-dict detections from device outputs, as the on-device decoder must:
    box[N,4] LTRB pre-stride → stride+anchor → yxyx; cls raw → sigmoid → top-1 → NMS.

    box_order: 'yfirst' ([t,l,b,r] — the deployed-DLC export default, --legacy_box_order) is
    reordered to x-first before decode; 'xfirst' assumes [l,t,r,b] (a --legacy_box_order=False
    export). A mismatch transposes every box."""
    box = dev_out['box'].numpy()                 # pre-stride; [t,l,b,r] if yfirst else [l,t,r,b]
    if box_order == 'yfirst':
        box = box[:, [1, 0, 3, 2]]               # -> [l,t,r,b]
    cls = dev_out['cls'].numpy()                 # [N,num_classes] raw logits
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
    boxes = np.clip(np.concatenate(boxes, 0), 0.0, 1.0).astype(np.float32)
    # Numerically-stable sigmoid. A naive 1/(1+exp(-x)) overflows exp() for very
    # negative logits; np.where still evaluates both arms so it overflows too. Apply
    # exp only to each safe subset (exact, no overflow warning).
    scores = np.empty_like(cls, dtype=np.float32)
    pos = cls >= 0
    scores[pos] = 1.0 / (1.0 + np.exp(-cls[pos]))
    e = np.exp(cls[~pos])
    scores[~pos] = e / (1.0 + e)
    top = scores.argmax(1)
    top_s = scores[np.arange(len(scores)), top]
    sel_b, sel_s, sel_c = [], [], []
    for c in range(num_classes):
        m = top == c
        if not m.any():
            continue
        cb, cs = boxes[m], top_s[m]
        idx = tf.image.non_max_suppression(cb, cs, max_boxes, nms_thresh, score_thresh).numpy()
        sel_b.append(cb[idx]); sel_s.append(cs[idx]); sel_c.append(np.full(len(idx), c, np.int64))
    if sel_b:
        sb = np.concatenate(sel_b); ss = np.concatenate(sel_s); sc = np.concatenate(sel_c)
        order = np.argsort(-ss)[:max_boxes]
        sb, ss, sc = sb[order], ss[order], sc[order]
    else:
        sb = np.zeros([0, 4], np.float32); ss = np.zeros([0], np.float32); sc = np.zeros([0], np.int64)
    k = len(ss); pad = max_boxes - k
    return {
        'bbox': tf.constant(np.pad(sb, [[0, pad], [0, 0]])[None], tf.float32),
        'classes': tf.constant(np.pad(sc, [[0, pad]])[None], tf.int64),
        'confidence': tf.constant(np.pad(ss, [[0, pad]])[None], tf.float32),
        'num_detections': tf.constant([k], tf.int32),
        'polygons': tf.zeros([1, max_boxes, poly_size, 3], tf.float32),
        'distance': tf.zeros([1, max_boxes], tf.float32),
    }


def _collect(pred, labels, i):
    """Pull per-image (pred boxes/scores/classes, gt boxes/classes) for direct metrics."""
    nd = int(pred['num_detections'][i])
    pb = pred['bbox'].numpy()[i, :nd]
    ps = pred['confidence'].numpy()[i, :nd]
    pc = pred['classes'].numpy()[i, :nd]
    ng = int(labels['n_gt'][i])
    gb = labels['bbox'].numpy()[i, :ng]
    gc = labels['classes'].numpy()[i, :ng]
    return pb, ps, pc, gb, gc


def _direct_prf(records, iou_thr):
    """Peak-F1 (and its precision/recall, TP/FP/FN) over a confidence sweep, greedy
    same-class IoU matching at iou_thr. Transparent, pycocotools-free."""
    from eval.polygon_metrics import _bbox_iou_matrix
    scores, is_tp = [], []
    total_gt = 0
    for pb, ps, pc, gb, gc in records:
        total_gt += len(gb)
        if len(pb) == 0:
            continue
        order = np.argsort(-ps)
        pb, ps, pc = pb[order], ps[order], pc[order]
        matched = np.zeros(len(gb), bool)
        iou = _bbox_iou_matrix(pb, gb) if len(gb) else None
        for k in range(len(pb)):
            scores.append(float(ps[k]))
            tp = 0
            if len(gb):
                cand = [j for j in range(len(gb))
                        if not matched[j] and gc[j] == pc[k] and iou[k, j] >= iou_thr]
                if cand:
                    j = max(cand, key=lambda j: iou[k, j])
                    matched[j] = True
                    tp = 1
            is_tp.append(tp)
    if not scores:
        return dict(peak_f1=0.0, precision=0.0, recall=0.0, thresh=0.0,
                    tp=0, fp=0, fn=total_gt, total_gt=total_gt)
    scores = np.array(scores); is_tp = np.array(is_tp)
    order = np.argsort(-scores)
    tp_cum = np.cumsum(is_tp[order])
    fp_cum = np.cumsum(1 - is_tp[order])
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    rec = tp_cum / max(total_gt, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(np.argmax(f1))
    return dict(peak_f1=float(f1[bi]), precision=float(prec[bi]), recall=float(rec[bi]),
                thresh=float(scores[order][bi]), tp=int(tp_cum[bi]),
                fp=int(fp_cum[bi]), fn=total_gt - int(tp_cum[bi]), total_gt=total_gt)


def main(_):
    import dataclasses
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8
    from tools.shared.ckpt_loading import restore_eval_weights
    from train.task import YoloV8Task, normalize_images
    from eval.coco_metrics import COCOEvaluator

    tf.keras.mixed_precision.set_global_policy('float32')
    config = load_config(FLAGS.config)
    tcfg = config.task
    nc = tcfg.num_classes

    # Score the ACTUAL exported SavedModel (the exact artifact that becomes the DLC),
    # at its own native input size from the signature (e.g. 672x416). Overriding
    # model.input_size makes the eval parser letterbox images to exactly that size, and
    # builds golden + anchors at it — so the comparison reflects the real DLC at the real
    # device size, no rebuild, no size mismatch.
    loaded = tf.saved_model.load(FLAGS.saved_model)
    dev_serving = loaded.signatures['serving_default']
    in_shape = dev_serving.inputs[0].shape          # [1, H, W, 3]
    H, W = int(in_shape[1]), int(in_shape[2])
    tcfg.model.input_size = [H, W, 3]
    log.info("Scoring SavedModel %s — native %dx%d (the DLC's size)", FLAGS.saved_model, H, W)

    gm = build_yolov8(tcfg.model)
    gm.deploy = True
    gm.build_and_init(tcfg.model.input_size)
    restore_eval_weights(gm, FLAGS.checkpoint)

    task = YoloV8Task(config)
    data_cfg = dataclasses.replace(tcfg.validation_data, is_training=False)
    val_ds = task.build_inputs(data_cfg)

    ev_g = COCOEvaluator(num_classes=nc, image_size=(H, W))
    ev_d = COCOEvaluator(num_classes=nc, image_size=(H, W))
    rec_g, rec_d = [], []

    seen = 0
    for images, labels in val_ds:
        B = int(images.shape[0])
        gpred = gm(normalize_images(images), training=False)
        ev_g.update(gpred, labels)

        imgs = tf.cast(images, tf.float32)
        for i in range(B):
            raw_in = imgs[i:i + 1] if FLAGS.normalize_baked else imgs[i:i + 1] / 255.0
            dpred = _reconstruct(dev_serving(input_image=raw_in), H, W, nc,
                                 box_order=FLAGS.box_order)
            lbl_i = {k: v[i:i + 1] for k, v in labels.items()}
            ev_d.update(dpred, lbl_i)
            rec_g.append(_collect(gpred, labels, i))
            rec_d.append(_collect(dpred, lbl_i, 0))
            seen += 1
            if FLAGS.num_images > 0 and seen >= FLAGS.num_images:
                break
        if FLAGS.num_images > 0 and seen >= FLAGS.num_images:
            break

    mg, md = ev_g.evaluate(), ev_d.evaluate()
    pg = _direct_prf(rec_g, FLAGS.iou_thr)
    pd = _direct_prf(rec_d, FLAGS.iou_thr)

    log.info("================ validation on %d images ================", seen)
    log.info("COCO evaluator (same as host eval):")
    log.info("  %-12s GOLDEN=%.4f   EXPORT=%.4f", "F1score50", mg.get('F1score50', 0), md.get('F1score50', 0))
    log.info("  %-12s GOLDEN=%.4f   EXPORT=%.4f", "mAP50", mg.get('mAP50', 0), md.get('mAP50', 0))
    log.info("  %-12s GOLDEN=%.4f   EXPORT=%.4f", "mAP", mg.get('mAP', 0), md.get('mAP', 0))
    log.info("  %-12s GOLDEN=%.4f   EXPORT=%.4f", "AR100", mg.get('AR100', 0), md.get('AR100', 0))
    log.info("Direct peak-F1 @ IoU%.2f (precision / recall / F1, TP/FP/FN):", FLAGS.iou_thr)
    for tag, p in (("GOLDEN", pg), ("EXPORT", pd)):
        log.info("  %-6s  P=%.4f  R=%.4f  F1=%.4f  (TP=%d FP=%d FN=%d, GT=%d, thr=%.3f)",
                 tag, p['precision'], p['recall'], p['peak_f1'],
                 p['tp'], p['fp'], p['fn'], p['total_gt'], p['thresh'])
    gap = (md.get('F1score50', 0) - mg.get('F1score50', 0))
    log.info("--------------------------------------------------------")
    if abs(gap) < 0.02:
        log.info("VERDICT: EXPORT ≈ GOLDEN → exported graph is FAITHFUL. The on-device "
                 "drop is quantization / on-device decode convention / input format, NOT "
                 "the export.")
    else:
        log.info("VERDICT: EXPORT differs from GOLDEN by %.3f F1 → export or reconstruction "
                 "contract issue; inspect box format (LTRB pre-stride) and concat layout.", gap)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
