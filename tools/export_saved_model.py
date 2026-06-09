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

    # ---- Build and restore ----
    from tools.ckpt_loading import restore_eval_weights

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
        tf.TensorSpec(shape=[None, H, W, 3], dtype=tf.float32, name='images')
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
