"""Tests for the configurable activation resolver (models/backbone.py:resolve_activation)."""

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
