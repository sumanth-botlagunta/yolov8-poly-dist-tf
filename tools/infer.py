"""Run a trained model on arbitrary images and save box + polygon overlays.

A host-side inference / visualization tool: point it at a checkpoint (or an exported
SavedModel) and a folder of images, and it writes an annotated PNG per image. Useful for
eyeballing a model on real images without going through TFDS / the eval split.

Detections are drawn on the letterboxed model input (the geometry the model actually sees),
so boxes/polygons line up with what the network predicted. Per-detection distance (when the
model has a distance head) is printed to the console.

Usage:
    # From a training checkpoint (EMA weights preferred):
    python tools/infer.py \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /output/run/ckpt-100000 \
        --images /path/to/images_dir \
        --output_dir /tmp/infer_out --score 0.3

    # From an exported SavedModel (expects float32 [0,1] input — see tools/export_saved_model.py):
    python tools/infer.py \
        --saved_model /path/to/saved_model \
        --images /path/to/one.jpg --output_dir /tmp/infer_out
"""

import glob
import logging
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
    flags.DEFINE_string('output_dir', '/tmp/infer_out', 'Where to write annotated PNGs.')
    flags.DEFINE_float('score', 0.25, 'Min detection confidence to draw.')
    flags.DEFINE_integer('input_size', 0, 'Override square input size; 0 = read from config/SavedModel.')
    flags.DEFINE_bool('no_poly', False, 'Disable polygon overlay.')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)

_IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')


def _list_images(path: str):
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in _IMG_EXTS:
        files += glob.glob(os.path.join(path, ext))
        files += glob.glob(os.path.join(path, ext.upper()))
    return sorted(files)


def _letterbox(img_hw3_uint8, size, pad=114):
    """Aspect-preserving resize to (size, size) with gray padding. Returns float32 [0,1]."""
    import cv2
    h, w = img_hw3_uint8.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_hw3_uint8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas.astype(np.float32) / 255.0


def _per_image_preds(predictions, i):
    """Slice the batched deploy-output dict to a per-image dict for render_summary_images."""
    pred = {
        'bbox':           predictions['bbox'][i].numpy(),
        'classes':        predictions['classes'][i].numpy(),
        'confidence':     predictions['confidence'][i].numpy(),
        'num_detections': int(predictions['num_detections'][i]),
    }
    if 'polygons' in predictions:
        pred['polygons'] = predictions['polygons'][i].numpy()
    return pred


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

    class_names = None
    try:
        from configs.class_map import DETECTION_CLASSES
        class_names = [str(DETECTION_CLASSES[i]) for i in sorted(DETECTION_CLASSES)]
    except Exception:
        pass

    # --- load model (checkpoint or SavedModel) ---
    if FLAGS.saved_model:
        loaded = tf.saved_model.load(FLAGS.saved_model)
        infer = loaded.signatures['serving_default']
        in_spec = infer.inputs[0].shape
        size = FLAGS.input_size or int(in_spec[1])

        def run(batch):
            out = infer(tf.constant(batch))
            # SavedModel signature returns the same keys as deploy output.
            return out
    else:
        if not FLAGS.config or not FLAGS.checkpoint:
            raise SystemExit("Provide --saved_model, or both --config and --checkpoint.")
        from configs.yaml_loader import load_config
        config = load_config(FLAGS.config)
        size = FLAGS.input_size or int(config.task.model.input_size[0])
        model = _load_checkpoint_model(config, FLAGS.checkpoint)

        from train.task import normalize_images

        def run(batch):
            return model(normalize_images(tf.constant((batch * 255.0).astype(np.uint8))),
                         training=False)

    files = _list_images(FLAGS.images)
    if not files:
        raise SystemExit(f"No images found at {FLAGS.images}")
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    log.info("Running on %d image(s) at %dx%d -> %s", len(files), size, size, FLAGS.output_dir)

    for f in files:
        bgr = cv2.imread(f)
        if bgr is None:
            log.warning("Could not read %s — skipping", f)
            continue
        rgb = bgr[..., ::-1]
        img01 = _letterbox(rgb, size)                       # float32 [size,size,3] in [0,1]
        predictions = run(img01[None, ...])
        pred = _per_image_preds(predictions, 0)

        # Console: top detections (with distance if present).
        nd = pred['num_detections']
        order = np.argsort(-pred['confidence'][:nd])
        kept = [k for k in order if pred['confidence'][k] >= FLAGS.score]
        dist = predictions['distance'][0].numpy() if 'distance' in predictions else None
        print(f"\n{os.path.basename(f)}: {len(kept)} detection(s) >= {FLAGS.score}")
        for k in kept[:20]:
            name = class_names[int(pred['classes'][k])] if class_names else str(int(pred['classes'][k]))
            extra = f"  dist={float(dist[k]):.2f}m" if dist is not None else ""
            print(f"    {name:<18} {pred['confidence'][k]:.3f}{extra}")

        rendered = render_summary_images(
            [img01], [pred], draw_box=True, draw_poly=not FLAGS.no_poly,
            class_names=class_names,
        )
        if rendered is None:
            continue
        out_path = os.path.join(FLAGS.output_dir, os.path.splitext(os.path.basename(f))[0] + '_pred.png')
        cv2.imwrite(out_path, rendered[0][..., ::-1])       # RGB -> BGR for cv2
    log.info("Done. Annotated images in %s", FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
