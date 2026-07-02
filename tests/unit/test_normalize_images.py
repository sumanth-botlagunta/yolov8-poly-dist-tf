"""Pins train.task.normalize_images — the single uint8→[0,255] gate for every
direct model() caller (validation_step, tools/eval.py in all its modes).

LEGACY-SCALE PATH (branch experiment/legacy-format-match): the model is fed the
[0, 255] pixel range (cast only, no /255) to match the old codebase the
warm-start was trained under. The parsers emit uint8 since the GPU-colour-aug
change; the standalone eval path crashed by feeding uint8 straight to float32
conv kernels. This helper is the fix — a regression here re-breaks /eval.
"""

import numpy as np
import tensorflow as tf

from train.task import normalize_images


def test_uint8_cast_to_float_0_255():
    img = tf.constant([[[0, 128, 255]]], dtype=tf.uint8)
    out = normalize_images(img)
    assert out.dtype == tf.float32
    np.testing.assert_allclose(out.numpy(), [[[0.0, 128.0, 255.0]]], atol=1e-7)


def test_float_passthrough_unchanged():
    img = tf.constant([[[0.0, 0.5, 1.0]]], dtype=tf.float32)
    out = normalize_images(img)
    assert out is img  # no cast, no rescale — already-normalized floats untouched
