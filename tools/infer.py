"""Run a trained model (checkpoint or exported SavedModel) on a folder of images.

Point it at a checkpoint (+config) or an exported SavedModel and a file/folder of
images; it produces, per image, any of:

  * **visual** — an annotated image (boxes + polygons + class/score, distance when the
    model has a distance head), drawn either on the **model input** (the exact 672² /
    416² geometry the network sees) or back on the **original** full-resolution image.
  * **predictions** — a COCO-style JSON of all detections (bbox + score + class +
    distance), in whichever coordinate space you choose.

Output size (``--draw_on`` / the JSON ``bbox`` space):
  * ``model``    — the exported input size (e.g. 672 or 416), read from the SavedModel
                   signature / config.
  * ``original`` — mapped back to the source image pixels (inverse letterbox). Default.

Usage:
    # SavedModel -> annotated images + predictions JSON, on the original-size images:
    python tools/infer.py --saved_model /export/saved_model \
        --images /path/to/images_dir --output_dir /tmp/infer_out \
        --emit both --draw_on original

    # From a training checkpoint (EMA weights preferred), visuals at model size:
    python tools/infer.py --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /run/ckpt-100000 --images /path/to/imgs --emit visual --draw_on model

Arguments:
    --saved_model           exported SavedModel dir (alternative to --checkpoint).
    --config / --checkpoint experiment YAML + checkpoint prefix (the other source).
    --images (req)          an image file or a directory of images (jpg/png/bmp/webp).
    --output_dir            where annotated images / predictions.json are written.
    --emit                  visual | json | both (default: both).
    --draw_on               original | model — output coordinate space (default: original).
    --score                 min detection confidence to keep/draw (default: 0.25).
    --input_size            override the square model input; 0 = read from config/SavedModel.
    --no_poly               disable polygon overlay.
    --predictions_json      path for the JSON (default: <output_dir>/predictions.json).
"""

import glob
import json
import logging
import math
import os

from absl import app, flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config', None, 'Experiment YAML (required with --checkpoint).')
    flags.DEFINE_string('checkpoint', None, 'Checkpoint path prefix.')
    flags.DEFINE_string('saved_model', None, 'Exported SavedModel dir (alternative to --checkpoint).')
    flags.DEFINE_string('images', None, 'Image file or directory of images.', required=True)
    flags.DEFINE_string('output_dir', '/tmp/infer_out', 'Where to write outputs.')
    flags.DEFINE_enum('emit', 'both', ['visual', 'json', 'both'], 'What to produce.')
    flags.DEFINE_enum('draw_on', 'original', ['original', 'model'],
                      'Output coordinate space: original image pixels or model input size.')
    flags.DEFINE_float('score', 0.25, 'Min detection confidence to keep/draw.')
    flags.DEFINE_integer('input_size', 0, 'Override square input size; 0 = read from config/SavedModel.')
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
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in _IMG_EXTS:
        files += glob.glob(os.path.join(path, ext))
        files += glob.glob(os.path.join(path, ext.upper()))
    return sorted(files)


def _letterbox(img_hw3_uint8, size, pad=114):
    """Aspect-preserving resize to (size, size) with gray padding.

    Returns (canvas float32 [size,size,3] in [0,255], r, top, left) where
    r/top/left are the scale and pad offsets needed to invert the letterbox.

    LEGACY-SCALE PATH (branch experiment/legacy-format-match): the model is fed
    [0, 255] pixels, so no /255 here (see train.task.normalize_images).
    """
    import cv2
    h, w = img_hw3_uint8.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_hw3_uint8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas.astype(np.float32), r, top, left


def _inv_point(xn, yn, size, r, top, left):
    """model-input-normalized (xn,yn) -> original-image pixel (x,y)."""
    return (xn * size - left) / r, (yn * size - top) / r


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
    from tools.shared.ckpt_loading import restore_eval_weights
    from tools.shared.runtime_setup import apply_eval_precision_policy

    apply_eval_precision_policy(config)
    model = build_yolov8(config.task.model)
    model.deploy = True
    model.build_and_init(config.task.model.input_size)
    kind = restore_eval_weights(model, ckpt_path)
    log.info("Restored %s weights from %s", kind, ckpt_path)
    return model


def main(_):
    tf.config.run_functions_eagerly(False)
    import cv2
    from train.viz_utils import render_summary_images
    from tools.shared.progress import Progress

    class_names = None
    try:
        from configs.class_map import DETECTION_CLASSES
        class_names = [str(DETECTION_CLASSES[i]) for i in sorted(DETECTION_CLASSES)]
    except Exception:
        pass

    def _name(c):
        return class_names[int(c)] if class_names and int(c) < len(class_names) else str(int(c))

    # --- load model (checkpoint or SavedModel) ---
    if FLAGS.saved_model:
        loaded = tf.saved_model.load(FLAGS.saved_model)
        infer = loaded.signatures['serving_default']
        size = FLAGS.input_size or int(infer.inputs[0].shape[1])

        def run(batch):
            return infer(tf.constant(batch))
    else:
        if not FLAGS.config or not FLAGS.checkpoint:
            raise SystemExit("Provide --saved_model, or both --config and --checkpoint.")
        from configs.yaml_loader import load_config
        from train.task import normalize_images
        config = load_config(FLAGS.config)
        size = FLAGS.input_size or int(config.task.model.input_size[0])
        model = _load_checkpoint_model(config, FLAGS.checkpoint)

        def run(batch):
            # batch is already [0, 255] float from _letterbox; feed the model directly.
            return model(normalize_images(tf.constant(batch.astype(np.float32))),
                         training=False)

    files = _list_images(FLAGS.images)
    if not files:
        raise SystemExit(f"No images found at {FLAGS.images}")
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    want_vis = FLAGS.emit in ('visual', 'both')
    want_json = FLAGS.emit in ('json', 'both')
    log.info("Running on %d image(s) | model=%dx%d | emit=%s | draw_on=%s -> %s",
             len(files), size, size, FLAGS.emit, FLAGS.draw_on, FLAGS.output_dir)

    coco_preds = []          # COCO-style detection list (for the JSON)
    pbar = Progress(total=len(files), desc='Infer', unit='img')
    for img_id, f in enumerate(files):
        pbar.update(1)
        bgr = cv2.imread(f)
        if bgr is None:
            log.warning("Could not read %s — skipping", f)
            continue
        H, W = bgr.shape[:2]
        rgb = bgr[..., ::-1]
        img01, r, top, left = _letterbox(rgb, size)
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
                    px1, py1 = _inv_point(x1, y1, size, r, top, left)
                    px2, py2 = _inv_point(x2, y2, size, r, top, left)
                    px1, px2 = sorted((max(0, min(W, px1)), max(0, min(W, px2))))
                    py1, py2 = sorted((max(0, min(H, py1)), max(0, min(H, py2))))
                else:
                    px1, py1, px2, py2 = x1 * size, y1 * size, x2 * size, y2 * size
                rec = {
                    'image_id': img_id, 'file_name': os.path.basename(f),
                    'category_id': int(pred['classes'][k]),
                    'category_name': _name(pred['classes'][k]),
                    'bbox': [round(px1, 2), round(py1, 2), round(px2 - px1, 2), round(py2 - py1, 2)],
                    'score': round(float(pred['confidence'][k]), 5),
                }
                if dist is not None:
                    rec['distance_m'] = round(float(dist[k]), 3)
                coco_preds.append(rec)

        # ---- visual ----
        if want_vis:
            if FLAGS.draw_on == 'model':
                rendered = render_summary_images(
                    [img01], [pred], draw_box=True, draw_poly=not FLAGS.no_poly,
                    class_names=class_names)
                out_img = None if rendered is None else rendered[0][..., ::-1]
            else:
                out_img = bgr.copy()
                for k in keep:
                    y1, x1, y2, x2 = [float(v) for v in pred['bbox'][k]]
                    p1 = tuple(int(round(v)) for v in _inv_point(x1, y1, size, r, top, left))
                    p2 = tuple(int(round(v)) for v in _inv_point(x2, y2, size, r, top, left))
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
                                             for a, b in (_inv_point(vx, vy, size, r, top, left)
                                                          for vx, vy in verts)], np.int32)
                            cv2.polylines(out_img, [opts.reshape(-1, 1, 2)], True,
                                          (0, 220, 100), 2, cv2.LINE_AA)
            if out_img is not None:
                out_path = os.path.join(
                    FLAGS.output_dir, os.path.splitext(os.path.basename(f))[0] + '_pred.png')
                cv2.imwrite(out_path, out_img)
    pbar.close()

    if want_json:
        jpath = FLAGS.predictions_json or os.path.join(FLAGS.output_dir, 'predictions.json')
        with open(jpath, 'w') as fh:
            json.dump(coco_preds, fh, indent=2)
        log.info("Wrote %d detections over %d images -> %s", len(coco_preds), len(files), jpath)
    if want_vis:
        log.info("Annotated images in %s", FLAGS.output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
