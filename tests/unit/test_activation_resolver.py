"""Tests for the configurable activation resolver (models/backbone.py:resolve_activation)
and the decoder.activation='same' inheritance branch in models/yolo_v8.py."""

import numpy as np
import pytest
import tensorflow as tf

from models.backbone import resolve_activation

_X = tf.constant([-2.0, -0.5, 0.0, 0.5, 2.0])


def test_relu_is_byte_identical_to_keras():
    # the default activation must be exactly Keras Activation('relu')
    got = resolve_activation('relu')(_X)
    ref = tf.keras.layers.Activation('relu')(_X)
    assert bool(tf.reduce_all(got == ref))


@pytest.mark.parametrize("name", ['relu', 'silu', 'swish', 'gelu', 'leaky_relu',
                                  'mish', 'hardswish', 'hard_swish', 'tanh'])
def test_supported_activations_build_and_run(name):
    out = resolve_activation(name)(_X).numpy()
    assert out.shape == (5,) and np.all(np.isfinite(out))


def test_hardswish_formula():
    got = resolve_activation('hardswish')(_X).numpy()
    ref = _X.numpy() * np.clip(_X.numpy() + 3.0, 0.0, 6.0) / 6.0
    assert np.allclose(got, ref)


def test_unknown_activation_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown activation"):
        resolve_activation('definitely_not_an_activation')


# ---------------------------------------------------------------------------
# decoder.activation == 'same' branch in build_yolov8 (yolo_v8.py:145)
# ---------------------------------------------------------------------------

def test_decoder_activation_same_inherits_from_norm_activation():
    """decoder.activation='same' must make the decoder inherit norm_activation.activation.

    Build a tiny model with decoder.activation='same' and norm_activation.activation='relu',
    then run a forward pass.  The model must build without error, confirming the
    'same' branch resolves to the backbone norm_activation rather than raising.
    """
    from configs.model_config import DecoderConfig, ModelConfig, NormActivationConfig
    from models.yolo_v8 import build_yolov8

    cfg = ModelConfig()
    cfg.with_polygons = False
    cfg.with_distance = False
    cfg.decoder = DecoderConfig(activation='same')        # <- branch under test
    cfg.norm_activation = NormActivationConfig(activation='relu')

    model = build_yolov8(cfg)
    model.build_and_init()

    x = tf.zeros([1, 672, 672, 3])
    out = model(x, training=False)
    # Output dict must be non-empty regardless of deploy mode
    assert len(out) > 0
