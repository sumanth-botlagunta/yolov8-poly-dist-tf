"""Export a trained checkpoint to TensorFlow SavedModel (and optionally TFLite).

Sets model.deploy=True before export so NMS is baked into the forward pass.

Usage:
    python tools/export_saved_model.py \
        --config     configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/ckpt-step \
        --output_dir /tmp/saved_model

    # Also convert to TFLite:
    python tools/export_saved_model.py ... --tflite

Flags:
    --config      Path to experiment YAML.
    --checkpoint  Checkpoint path prefix.
    --output_dir  Directory to write SavedModel (and TFLite if requested).
    --tflite      Also run TFLiteConverter and save a .tflite file.

Input Schema (CONTRACT — read before serving):
    images: float32 [batch, H, W, 3], pixels in the [0, 255] range.

    LEGACY-SCALE PATH (branch experiment/legacy-format-match): the model is
    trained on [0, 255] pixels (matching the old codebase / warm-start
    checkpoint), so a served caller feeds [0, 255] floats directly — do NOT
    divide by 255. The exported SavedModel does NOT normalize internally (the
    model has no /255 layer — see models/yolo_v8.py:YoloV8.call); the
    training/eval path only casts uint8→float32 via train.task.normalize_images
    (no scaling). Feeding /255-normalized [0,1] floats produces silently wrong
    detections. Cast camera/OpenCV uint8 frames to float32 (no divide) before
    calling.

Output Schema:
    With model.deploy=True the SavedModel runs NMS in-graph and returns a dict of
    post-processed detections (see models/detection_generator.py:YoloV8Layer):

        bbox:           float32 [batch, max_boxes, 4]      yxyx, normalized [0, 1]
        classes:        int64   [batch, max_boxes]         class id in [0, num_classes)
        confidence:     float32 [batch, max_boxes]         detection score in [0, 1]
        num_detections: int32   [batch]                    valid boxes per image; rows
                                                           past this are zero padding
        polygons:       float32 [batch, max_boxes, P, 3]   per-vertex (conf, dist, angle),
                                                           already sigmoid/softplus
                                                           activated. P = output_poly_size
                                                           (= 360 // angle_step, 24 by
                                                           default). conf in [0, 1], dist
                                                           is normalized radial distance,
                                                           angle is the sub-bin offset in
                                                           [0, 1). Present only when the
                                                           model has polygon heads.
        distance:       float32 [batch, max_boxes]         estimated distance in metres,
                                                           clamped to [min_distance,
                                                           max_distance]. Present only
                                                           when the model has a distance
                                                           head.

    max_boxes defaults to 300. polygons/distance keys are emitted per the configured
    heads (with_polygons / with_distance).
"""

import logging
import os

from absl import app, flags
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',      None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('checkpoint',  None, 'Checkpoint path prefix.',          required=True)
    flags.DEFINE_string('output_dir',  None, 'Directory to write SavedModel.',    required=True)
    flags.DEFINE_bool  ('tflite',      False, 'Also export a TFLite flatbuffer.')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)


def main(_):
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8

    config    = load_config(FLAGS.config)
    model_cfg = config.task.model

    # Activate the trainer's precision policy before building the model so the
    # exported SavedModel computes on the same dtype path the checkpoint was
    # trained/served on (bfloat16 backbone/decoder, float32 heads).
    from tools.shared.runtime_setup import apply_eval_precision_policy
    apply_eval_precision_policy(config)

    # ---- Build and restore ----
    from tools.shared.ckpt_loading import restore_eval_weights

    model = build_yolov8(model_cfg)
    model.deploy = True
    model.build_and_init(model_cfg.input_size)

    # Prefer EMA weights — exporting raw weights from a periodic checkpoint would
    # ship a worse model than the trainer validated/deployed.
    kind = restore_eval_weights(model, FLAGS.checkpoint)
    log.info("Checkpoint restored (%s weights): %s", kind, FLAGS.checkpoint)

    # ---- Concrete function for tracing ----
    H, W = model_cfg.input_size[0], model_cfg.input_size[1]

    @tf.function(input_signature=[
        # CONTRACT: `images` must be float32 in the [0, 255] range. The model has
        # no internal /255 (models/yolo_v8.py); the other call paths only cast
        # uint8→float32 via train.task.normalize_images (LEGACY-SCALE PATH, branch
        # experiment/legacy-format-match). Feeding /255-normalized [0,1] floats
        # yields silently wrong detections.
        tf.TensorSpec(shape=[None, H, W, 3], dtype=tf.float32, name='images_0_255')
    ])
    def serving_fn(images):
        return model(images, training=False)

    # ---- Save ----
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    tf.saved_model.save(
        model,
        FLAGS.output_dir,
        signatures={'serving_default': serving_fn},
    )
    log.info("SavedModel written to %s", FLAGS.output_dir)

    # Verify it can be loaded back
    loaded = tf.saved_model.load(FLAGS.output_dir)
    log.info("SavedModel load verification: OK (type=%s)", type(loaded).__name__)

    # ---- Optional TFLite ----
    if FLAGS.tflite:
        _export_tflite(FLAGS.output_dir, H, W)


def _export_tflite(saved_model_dir: str, H: int, W: int) -> None:
    """Convert the SavedModel at saved_model_dir to a TFLite flatbuffer."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_model = converter.convert()
    except Exception as e:
        log.error("TFLite conversion failed: %s", e)
        return

    tflite_path = os.path.join(saved_model_dir, 'model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    log.info("TFLite model written to %s (%d KB)",
             tflite_path, len(tflite_model) // 1024)


if __name__ == '__main__':
    app.run(main)
