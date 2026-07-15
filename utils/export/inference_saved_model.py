"""Run a trained model (checkpoint or exported SavedModel) over images or a TFDS split.

Point it at a checkpoint (+config) or an exported SavedModel, plus either a
file/folder of images (--images, searched recursively) or a TFDS split
(--tfds_split, read via the config's validation_data). In BOTH modes the
predictions JSON uses image_id = file_name = the image basename with its
extension (folder mode: the file's basename; TFDS mode: the records'
image/filename), matching the GT annotations directly. Bboxes and scores are
written at full float precision. Per image it produces:

  * visual — an annotated image (boxes + polygons + class/score, plus distance when
    the model has a distance head), drawn on the model-input geometry or back on the
    original full-resolution image.
  * predictions — a COCO-style JSON of all detections (bbox + score + class +
    distance) in the chosen coordinate space.

The exported SavedModel is the device-contract artifact (utils/export/
export_saved_model.py): its signature has flat per-anchor outputs (box/cls/poly_*/
dist) and takes ``input_image`` float32 pixels in [0, 255]. This tool reconstructs
deploy-style detections from those flat heads (utils/export/device_decode.py —
LTRB->anchor->NMS, sigmoid/softplus activations), so boxes, classes, scores,
polygons, and distance are all available from the SavedModel path. An older
post-processed SavedModel (with ``num_detections`` in its signature) is still
consumed on its original path.

--draw_on selects the output coordinate space (JSON bbox and drawn overlays):
  * model    — the exported input size (e.g. 672 or 672x416), from the signature/config.
  * original — mapped back to source-image pixels via the inverse letterbox. Default.

Usage:
    python utils/export/inference_saved_model.py --saved_model /export/saved_model \
        --images /path/to/images_dir --output_dir /tmp/infer_out \
        --emit both --draw_on original

    python utils/export/inference_saved_model.py \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /run/ckpt-100000 --images /path/to/imgs --emit visual --draw_on model

    # Predictions JSON over the TFDS test split (no image dump needed):
    python utils/export/inference_saved_model.py --saved_model /export/saved_model \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --tfds_split test --emit json --score 0.05 --output_dir /tmp/dev_preds
"""

import glob
import json
import logging
import math
import os
import time

from absl import app, flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config', None, 'Experiment YAML (required with --checkpoint).')
    flags.DEFINE_string('checkpoint', None, 'Checkpoint path prefix.')
    flags.DEFINE_string('saved_model', None, 'Exported SavedModel dir (alternative to --checkpoint).')
    flags.DEFINE_string('images', None, 'Image file or directory of images (searched '
                        'recursively). Alternative to --tfds_split.')
    flags.DEFINE_string('tfds_split', None, "TFDS split to run over (e.g. 'test'), read via "
                        "--config's validation_data (tfds_name / tfds_data_dir). Alternative "
                        "to --images. image_id = file_name = the image basename with extension "
                        "(from the records' image/filename), matching the GT annotations.")
    flags.DEFINE_string('output_dir', '/tmp/infer_out', 'Where to write outputs.')
    flags.DEFINE_enum('emit', 'both', ['visual', 'json', 'both'], 'What to produce.')
    flags.DEFINE_enum('draw_on', 'original', ['original', 'model'],
                      'Output coordinate space: original image pixels or model input size.')
    flags.DEFINE_float('score', 0.25, 'Min detection confidence to keep/draw.')
    flags.DEFINE_integer('input_size', 0, 'Override square input size; 0 = read from config/SavedModel.')
    flags.DEFINE_enum('device_box_order', 'yfirst', ['yfirst', 'xfirst'],
                      "Box-channel order of the device SavedModel's box head: 'yfirst' "
                      "([t,l,b,r], the export's legacy_box_order=True default) or 'xfirst' "
                      "([l,t,r,b], a --legacy_box_order=False export).")
    flags.DEFINE_bool('no_poly', False, 'Disable polygon overlay.')
    flags.DEFINE_string('predictions_json', None, 'JSON path (default: <output_dir>/predictions.json).')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)

_IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
_N_VERTS = 24
try:
    from eval.polygon_metrics import DEFAULT_POLY_CONF_THRESH as _POLY_CONF
except Exception:
    _POLY_CONF = 0.4


def _list_images(path: str):
    """List image files under ``path`` recursively (any nesting depth).

    A set dedupes the lower/upper-case extension patterns, which both match the
    same file on case-insensitive filesystems.
    """
    if os.path.isfile(path):
        return [path]
    files = set()
    for ext in _IMG_EXTS:
        for pat in (ext, ext.upper()):
            files.update(glob.glob(os.path.join(path, '**', pat), recursive=True))
    return sorted(files)


def _letterbox(img_hw3_uint8, out_h, out_w, pad=114):
    """Aspect-preserving resize to (out_h, out_w) with gray padding.

    Returns (canvas01 float32 [out_h,out_w,3] in [0,1], r, top, left) where r/top/left
    are the scale and pad offsets needed to invert the letterbox.
    """
    import cv2
    h, w = img_hw3_uint8.shape[:2]
    r = min(out_h / h, out_w / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_hw3_uint8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), pad, np.uint8)
    top, left = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas.astype(np.float32) / 255.0, r, top, left


def _inv_point(xn, yn, out_w, out_h, r, top, left):
    """model-input-normalized (xn,yn) -> original-image pixel (x,y)."""
    return (xn * out_w - left) / r, (yn * out_h - top) / r


def _per_image_preds(predictions, i):
    pred = {
        'bbox':           predictions['bbox'][i].numpy(),
        'classes':        predictions['classes'][i].numpy(),
        'confidence':     predictions['confidence'][i].numpy(),
        'num_detections': int(predictions['num_detections'][i]),
    }
    if 'polygons' in predictions:
        pred['polygons'] = predictions['polygons'][i].numpy()
    return pred


def _poly_vertices_norm(poly_24x3, cxn, cyn, conf_thresh):
    """Decode a radial polygon [24,(conf,dist,angle)] to model-normalized (x,y) vertices."""
    bin_w = 2.0 * math.pi / _N_VERTS
    pts = []
    for i in range(_N_VERTS):
        conf = float(poly_24x3[i, 0])
        if conf < conf_thresh:
            continue
        d = max(0.0, float(poly_24x3[i, 1]))
        off = float(poly_24x3[i, 2])
        ang = (i + off) * bin_w
        pts.append((cxn + d * math.cos(ang), cyn + d * math.sin(ang)))
    return pts


def _load_checkpoint_model(config, ckpt_path):
    from models.yolo_v8 import build_yolov8
    from common.ckpt_loading import restore_eval_weights
    from common.runtime_setup import apply_eval_precision_policy

    apply_eval_precision_policy(config)
    model = build_yolov8(config.task.model)
    model.deploy = True
    model.build_and_init(config.task.model.input_size)
    kind = restore_eval_weights(model, ckpt_path)
    log.info("Restored %s weights from %s", kind, ckpt_path)
    return model


def main(_):
    # Validate the flag combination FIRST — model/SavedModel loading takes tens of
    # seconds, and a bad source flag should not cost that wait.
    if FLAGS.tfds_split and FLAGS.images:
        raise SystemExit("Provide --images OR --tfds_split, not both.")
    if not FLAGS.tfds_split and not FLAGS.images:
        raise SystemExit("Provide --images or --tfds_split.")
    if FLAGS.tfds_split and not FLAGS.config:
        raise SystemExit("--tfds_split needs --config (for tfds_name / tfds_data_dir).")

    tf.config.run_functions_eagerly(False)
    import cv2
    from common.viz_utils import render_summary_images
    from common.progress import Progress

    class_names = None
    try:
        from configs.class_map import DETECTION_CLASSES
        class_names = [str(DETECTION_CLASSES[i]) for i in sorted(DETECTION_CLASSES)]
    except Exception:
        pass

    def _name(c):
        return class_names[int(c)] if class_names and int(c) < len(class_names) else str(int(c))

    # --- load model (checkpoint or SavedModel); establish run() + model (H, W) ---
    if FLAGS.saved_model:
        from utils.export.device_decode import (
            reconstruct_detections, is_device_contract, is_legacy_contract)
        t0 = time.time()
        log.info("Loading SavedModel %s (TF deserializes the full frozen graph — "
                 "this is the slow startup step)...", FLAGS.saved_model)
        loaded = tf.saved_model.load(FLAGS.saved_model)
        log.info("SavedModel loaded in %.1fs", time.time() - t0)
        infer = loaded.signatures['serving_default']
        out_keys = list(infer.structured_outputs.keys())
        sig_in = infer.inputs[0].shape
        mh = FLAGS.input_size or int(sig_in[1])
        mw = FLAGS.input_size or int(sig_in[2])

        if is_device_contract(out_keys):
            legacy_order = (FLAGS.device_box_order == 'yfirst')
            log.info("Device-contract SavedModel (box_order=%s) — reconstructing detections "
                     "from flat heads.", FLAGS.device_box_order)

            def run(batch):
                dev = infer(input_image=tf.constant((batch * 255.0).astype(np.float32)))
                npd = reconstruct_detections(dict(dev), mh, mw, legacy_box_order=legacy_order)
                return {k: tf.constant(v) for k, v in npd.items()}
        elif is_legacy_contract(out_keys):
            log.info("Post-processed SavedModel (num_detections signature) — using its "
                     "detections directly.")

            def run(batch):
                return infer(tf.constant(batch))
        else:
            raise SystemExit(
                f"SavedModel signature outputs {sorted(out_keys)} match neither the device "
                "contract (flat 'box'+'cls' heads) nor a post-processed deploy dict "
                "(with 'num_detections'). Re-export with utils/export/export_saved_model.py.")
    else:
        if not FLAGS.config or not FLAGS.checkpoint:
            raise SystemExit("Provide --saved_model, or both --config and --checkpoint.")
        from configs.yaml_loader import load_config
        from train.task import normalize_images
        config = load_config(FLAGS.config)
        mh = mw = FLAGS.input_size or int(config.task.model.input_size[0])
        t0 = time.time()
        model = _load_checkpoint_model(config, FLAGS.checkpoint)
        log.info("Model built + checkpoint restored in %.1fs", time.time() - t0)

        def run(batch):
            return model(normalize_images(tf.constant((batch * 255.0).astype(np.uint8))),
                         training=False)

    # --- image source: a folder tree (--images) or a TFDS split (--tfds_split) ---
    # Both yield (img_id, name, rgb_uint8). TFDS ids come from image/id, so the
    # predictions JSON is directly scoreable against the GT annotations.
    if FLAGS.tfds_split:
        from configs.yaml_loader import load_config
        import tensorflow_datasets as tfds
        val_cfg = load_config(FLAGS.config).task.validation_data
        t0 = time.time()
        ds, ds_info = tfds.load(val_cfg.tfds_name, split=FLAGS.tfds_split,
                                data_dir=val_cfg.tfds_data_dir, with_info=True)
        total = ds_info.splits[FLAGS.tfds_split].num_examples
        log.info("TFDS %s split '%s' opened in %.1fs — %d examples",
                 val_cfg.tfds_name, FLAGS.tfds_split, time.time() - t0, total)

        def _source():
            for ex in ds:
                # image_id convention: the image BASENAME with extension (string),
                # matching the GT annotations. image/filename is the source of
                # truth; fall back to the numeric image/id only if absent.
                if 'image/filename' in ex:
                    name = os.path.basename(ex['image/filename'].numpy().decode('utf-8'))
                else:
                    name = str(int(ex['image/id'].numpy()))
                yield name, name, ex['image'].numpy()
        src_name = f"{val_cfg.tfds_name}[{FLAGS.tfds_split}]"
    else:
        files = _list_images(FLAGS.images)
        if not files:
            raise SystemExit(f"No images found at {FLAGS.images}")
        total = len(files)

        def _source():
            for f in files:
                bgr = cv2.imread(f)
                if bgr is None:
                    log.warning("Could not read %s — skipping", f)
                    continue
                # image_id convention: the image BASENAME with extension (string).
                # Subfolder paths are dropped — GT annotations key by basename.
                name = os.path.basename(f)
                yield name, name, bgr[..., ::-1]
        src_name = FLAGS.images

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    want_vis = FLAGS.emit in ('visual', 'both')
    want_json = FLAGS.emit in ('json', 'both')
    log.info("Running on %d image(s) from %s | model=%dx%d | emit=%s | draw_on=%s -> %s",
             total, src_name, mh, mw, FLAGS.emit, FLAGS.draw_on, FLAGS.output_dir)

    coco_preds = []          # COCO-style detection list (for the JSON)
    n_images = 0
    pbar = Progress(total=total, desc='Infer', unit='img')
    for img_id, fname, rgb in _source():
        pbar.update(1)
        n_images += 1
        H, W = rgb.shape[:2]
        img01, r, top, left = _letterbox(rgb, mh, mw)
        predictions = run(img01[None, ...])
        pred = _per_image_preds(predictions, 0)
        dist = predictions['distance'][0].numpy() if 'distance' in predictions else None

        nd = pred['num_detections']
        keep = [k for k in range(nd) if pred['confidence'][k] >= FLAGS.score]

        # ---- predictions JSON (bbox in the chosen coordinate space) ----
        if want_json:
            for k in keep:
                y1, x1, y2, x2 = [float(v) for v in pred['bbox'][k]]   # yxyx norm to model
                if FLAGS.draw_on == 'original':
                    px1, py1 = _inv_point(x1, y1, mw, mh, r, top, left)
                    px2, py2 = _inv_point(x2, y2, mw, mh, r, top, left)
                    px1, px2 = sorted((max(0, min(W, px1)), max(0, min(W, px2))))
                    py1, py2 = sorted((max(0, min(H, py1)), max(0, min(H, py2))))
                else:
                    px1, py1, px2, py2 = x1 * mw, y1 * mh, x2 * mw, y2 * mh
                rec = {
                    'image_id': img_id, 'file_name': fname,
                    'category_id': int(pred['classes'][k]),
                    'category_name': _name(pred['classes'][k]),
                    # Full float precision — no rounding, so the JSON is bit-comparable
                    # across runs/paths.
                    'bbox': [float(px1), float(py1), float(px2 - px1), float(py2 - py1)],
                    'score': float(pred['confidence'][k]),
                }
                if dist is not None:
                    rec['distance_m'] = float(dist[k])
                coco_preds.append(rec)

        # ---- visual ----
        if want_vis:
            if FLAGS.draw_on == 'model':
                rendered = render_summary_images(
                    [img01], [pred], draw_box=True, draw_poly=not FLAGS.no_poly,
                    class_names=class_names)
                out_img = None if rendered is None else rendered[0][..., ::-1]
            else:
                out_img = np.ascontiguousarray(rgb[..., ::-1])   # BGR copy for cv2
                for k in keep:
                    y1, x1, y2, x2 = [float(v) for v in pred['bbox'][k]]
                    p1 = tuple(int(round(v)) for v in _inv_point(x1, y1, mw, mh, r, top, left))
                    p2 = tuple(int(round(v)) for v in _inv_point(x2, y2, mw, mh, r, top, left))
                    cv2.rectangle(out_img, p1, p2, (0, 200, 0), 2)
                    label = f"{_name(pred['classes'][k])} {pred['confidence'][k]:.2f}"
                    if dist is not None:
                        label += f" {float(dist[k]):.1f}m"
                    cv2.putText(out_img, label, (p1[0], max(12, p1[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    if not FLAGS.no_poly and 'polygons' in pred:
                        cxn, cyn = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        verts = _poly_vertices_norm(pred['polygons'][k], cxn, cyn, _POLY_CONF)
                        if len(verts) >= 3:
                            opts = np.array([[int(round(a)), int(round(b))]
                                             for a, b in (_inv_point(vx, vy, mw, mh, r, top, left)
                                                          for vx, vy in verts)], np.int32)
                            cv2.polylines(out_img, [opts.reshape(-1, 1, 2)], True,
                                          (0, 220, 100), 2, cv2.LINE_AA)
            if out_img is not None:
                # fname can carry subdirectories (recursive scan) — flatten so all
                # renders land in output_dir without recreating the tree.
                stem = os.path.splitext(fname)[0].replace(os.sep, '__')
                out_path = os.path.join(FLAGS.output_dir, stem + '_pred.png')
                cv2.imwrite(out_path, out_img)
    pbar.close()

    if want_json:
        jpath = FLAGS.predictions_json or os.path.join(FLAGS.output_dir, 'predictions.json')
        with open(jpath, 'w') as fh:
            json.dump(coco_preds, fh, indent=2)
        log.info("Wrote %d detections over %d images -> %s", len(coco_preds), n_images, jpath)
    if want_vis:
        log.info("Annotated images in %s", FLAGS.output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
