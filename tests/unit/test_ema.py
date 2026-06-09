"""Tests for ExponentialMovingAverage.

Validates:
    - Shadow variables are created with correct initial values.
    - Dynamic decay formula: min(0.9999, (1+step)/(10+step)).
    - swap_in loads EMA weights; swap_out restores the originals exactly.
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

    def test_swap_in_loads_ema_then_swap_out_restores(self):
        """swap_in must actually load the shadow weights (not a no-op), and
        swap_out must restore the original live weights exactly. Also asserts the
        shadows are NOT mutated by the swap (crash-safety contract)."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))
        ema = _make_ema(model)

        # Make shadows differ from live so a no-op swap would be detectable.
        for s in ema._shadows:
            s.assign(s + 1.0)

        live_before   = [v.numpy().copy() for v in model.variables]
        shadow_before = [s.numpy().copy() for s in ema._shadows]

        ema.swap_in(model)        # model must now hold the shadow (EMA) weights
        for shadow, var in zip(shadow_before, model.variables):
            np.testing.assert_allclose(shadow, var.numpy())
        # Shadows themselves must be untouched by swap_in.
        for before, s in zip(shadow_before, ema._shadows):
            np.testing.assert_allclose(before, s.numpy())

        ema.swap_out(model)       # model must be back to the original live weights
        for before, var in zip(live_before, model.variables):
            np.testing.assert_allclose(before, var.numpy())

    def test_swap_out_is_idempotent_noop_when_not_swapped(self):
        """swap_out without a prior swap_in is a safe no-op (finally-block use)."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))
        ema = _make_ema(model)
        live_before = [v.numpy().copy() for v in model.variables]
        ema.swap_out(model)  # should not raise or change anything
        for before, var in zip(live_before, model.variables):
            np.testing.assert_allclose(before, var.numpy())

    def test_double_swap_in_raises(self):
        """Nested swap_in must raise rather than clobber the live-weight backup."""
        model = _make_tiny_model()
        model(tf.zeros([1, 4]))
        ema = _make_ema(model)
        ema.swap_in(model)
        with self.assertRaises(RuntimeError):
            ema.swap_in(model)
        ema.swap_out(model)  # cleanup

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
