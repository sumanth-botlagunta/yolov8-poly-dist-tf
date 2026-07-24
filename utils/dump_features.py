"""Dump backbone / decoder intermediate feature maps for one checkpoint.

Builds the model from an experiment YAML, restores a trained checkpoint (EMA
weights preferred, via common.ckpt_loading.restore_eval_weights — the same
loading path eval and export use), runs one image through the trunk, and writes
the output of each important layer — the per-level feature maps after the
backbone and after the decoder (FPN-PAN neck) — to ``<output_dir>/<layer>.npy``,
one file per layer, named after the layer that produced it:

    backbone_level_3.npy    [1, H/8,  W/8,  C3]   stride-8 backbone output
    backbone_level_4.npy    [1, H/16, W/16, C4]   stride-16 backbone output
    backbone_level_5.npy    [1, H/32, W/32, C5]   stride-32 backbone output
    decoder_level_3.npy     [1, H/8,  W/8,  C3']  stride-8 neck output (head input)
    decoder_level_4.npy     [1, H/16, W/16, C4']  stride-16 neck output (head input)
    decoder_level_5.npy     [1, H/32, W/32, C5']  stride-32 neck output (head input)

The input image takes the exact eval preprocessing (aspect-preserving letterbox
to the model input size with gray-114 padding, then /255), so the dumped
activations match what the deployed model computes on that image. Without
--image a deterministic synthetic input is used (fixed stateless seed), which
is enough for cross-checkpoint or cross-implementation activation diffs.

Usage:
    python -m utils.dump_features \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/run/ckpt-1000 \
        --image /path/to/image.jpg \
        --output_dir /tmp/features
"""

import logging
import os

from absl import app, flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',     None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('checkpoint', None, 'Checkpoint path prefix (e.g. /run/ckpt-1000).',
                        required=True)
    flags.DEFINE_string('image',      None, 'Input image path (jpg/png). Letterboxed to the '
                        'model input size like eval. Omit for a deterministic synthetic input.')
    flags.DEFINE_string('output_dir', None, 'Directory to write the per-layer .npy files.',
                        required=True)
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)

# Stateless seed for the synthetic-input fallback: fixed so two invocations
# (e.g. against two checkpoints) dump activations for the identical input.
_SYNTHETIC_SEED = (1000, 0)


def _load_model(config, ckpt_path: str) -> tf.keras.Model:
    """Build the model and restore eval weights (EMA preferred, raw fallback).

    The FULL model is built and restored — a bare backbone/decoder object-graph
    restore misses list-tracked C2f variables (see task.initialize) — but only
    the backbone and decoder are run afterwards.
    """
    from common.ckpt_loading import restore_eval_weights
    from models.yolo_v8 import build_yolov8

    model = build_yolov8(config.task.model)
    model.build_and_init(config.task.model.input_size)

    kind = restore_eval_weights(model, ckpt_path)
    log.info("Checkpoint restored (%s weights): %s", kind, ckpt_path)
    return model


def _prepare_input(config, image_path: str = None) -> tf.Tensor:
    """Return a float32 [1, H, W, 3] batch in [0, 1], eval-preprocessed.

    With an image: decode -> letterbox to the model input size (gray-114 pad,
    the SAME math the eval parser uses) -> /255. Without: a deterministic
    stateless-uniform synthetic image of the input size.
    """
    from data_pipeline.augmentations import letterbox_resize
    from train.task import normalize_images

    h, w = config.task.model.input_size[:2]

    if image_path:
        data = tf.io.read_file(image_path)
        image = tf.image.decode_image(data, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        # No labels to remap: pass empty boxes / sentinel-padded polygons.
        image, _, _ = letterbox_resize(
            image, tf.zeros([0, 4], tf.float32), tf.fill([0, 2], -1.0), h, w)
        log.info("Input: %s letterboxed to %dx%d", image_path, h, w)
    else:
        image = tf.cast(
            tf.random.stateless_uniform([h, w, 3], seed=_SYNTHETIC_SEED,
                                        minval=0, maxval=256, dtype=tf.int32),
            tf.uint8)
        log.info("Input: deterministic synthetic %dx%d (seed %s)", h, w,
                 _SYNTHETIC_SEED)

    return normalize_images(image[tf.newaxis, ...])


def dump_features(config, ckpt_path: str, output_dir: str,
                  image_path: str = None) -> dict:
    """Run backbone + decoder on one input and write per-layer .npy dumps.

    Args:
        config: Loaded ExperimentConfig.
        ckpt_path: Checkpoint path prefix to restore.
        output_dir: Directory for the .npy files (created if missing).
        image_path: Optional input image; synthetic input when None.

    Returns:
        {layer_name: shape tuple} for every file written.
    """
    model = _load_model(config, ckpt_path)
    images = _prepare_input(config, image_path)

    feats   = model.backbone(images, training=False)
    decoded = model.decoder(feats, training=False)

    os.makedirs(output_dir, exist_ok=True)
    written = {}
    for stage_name, stage_out in (("backbone", feats), ("decoder", decoded)):
        for level in sorted(stage_out):
            layer_name = f"{stage_name}_level_{level}"
            array = stage_out[level].numpy()
            np.save(os.path.join(output_dir, f"{layer_name}.npy"), array)
            written[layer_name] = array.shape
            log.info("Wrote %s.npy  shape=%s  dtype=%s",
                     layer_name, array.shape, array.dtype)

    log.info("Dumped %d layers to %s", len(written), output_dir)
    return written


def main(argv):
    del argv
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    from configs.yaml_loader import load_config

    config = load_config(FLAGS.config)
    dump_features(config, FLAGS.checkpoint, FLAGS.output_dir, FLAGS.image)


if __name__ == '__main__':
    app.run(main)
