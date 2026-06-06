"""Tests for ExponentialMovingAverage.

Validates:
    - Shadow variables are created with correct initial values.
    - Dynamic decay formula: min(0.9999, (1+step)/(10+step)).
    - After swap_weights + swap_weights again, live weights are unchanged.
    - apply_gradients delegates to base optimizer and updates shadows.
"""

import numpy as np
import tensorflow as tf
import unittest

from optimizers.ema import ExponentialMovingAverage


def _make_tiny_model() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(4,))
    out = tf.keras.layers.Dense(2, name='dense')(inp)
    return tf.keras.Model(inp, out)


def _make_ema(model, dynamic_decay=True) -> ExponentialMovingAverage:
    base_opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    return ExponentialMovingAverage(base_opt, model, average_decay=0.9999,
                                    dynamic_decay=dynamic_decay)


class TestExponentialMovingAverage(unittest.TestCase):

    def test_shadows_initialized_equal_to_live(self):
        """Shadow variables must start as copies of the live model variables."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))  # build

        ema = _make_ema(model)
        for var, shadow in zip(model.variables, ema._shadows):
            self.assertEqual(var.shape, shadow.shape)
            np.testing.assert_allclose(var.numpy(), shadow.numpy())

    def test_dynamic_decay_at_step_zero(self):
        """At step 0 decay = (1+0)/(10+0) = 0.1, well below 0.9999."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))
        ema = _make_ema(model, dynamic_decay=True)
        decay = float(ema._get_decay())
        self.assertAlmostEqual(decay, 1.0 / 10.0, places=5)

    def test_dynamic_decay_at_large_step(self):
        """At large step count decay saturates to average_decay=0.9999."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))
        ema = _make_ema(model, dynamic_decay=True)
        ema._ema_step.assign(100_000)
        decay = float(ema._get_decay())
        self.assertAlmostEqual(decay, 0.9999, places=4)

    def test_swap_is_invertible(self):
        """Double-swap must restore the original live weights exactly."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))
        ema = _make_ema(model)

        live_before = [v.numpy().copy() for v in model.variables]

        ema.swap_weights(model)   # live → shadow, shadow → live
        ema.swap_weights(model)   # undo

        for before, var in zip(live_before, model.variables):
            np.testing.assert_allclose(before, var.numpy())

    def test_apply_gradients_updates_shadows(self):
        """After apply_gradients, shadow ≠ initial value (decay < 1.0)."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))
        ema = _make_ema(model, dynamic_decay=True)

        shadow_before = [s.numpy().copy() for s in ema._shadows]

        # Dummy gradient step: grad = 1.0 for all trainable vars
        grads_and_vars = [
            (tf.ones_like(v), v) for v in model.trainable_variables
        ]
        ema.apply_gradients(grads_and_vars)

        # Shadows must differ from initial values after the update
        any_changed = any(
            not (s_before == s_after.numpy()).all()
            for s_before, s_after in zip(shadow_before, ema._shadows)
        )
        self.assertTrue(any_changed, "Shadows were not updated by apply_gradients")

    def test_shadow_count_matches_model_variables(self):
        """Number of shadow variables == len(model.variables)."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))
        ema = _make_ema(model)
        self.assertEqual(len(ema._shadows), len(model.variables))


if __name__ == '__main__':
    unittest.main()
