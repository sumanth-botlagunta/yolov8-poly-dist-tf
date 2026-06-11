"""Pinning test: EMA.apply_gradients guards against a stale var snapshot.

The EMA wrapper snapshots model.variables in __init__ (`_model_vars`) and creates
one shadow per snapshotted var. If the model later gains a variable (e.g. a layer
built lazily, or a monkey-patched variable), apply_gradients used to zip the stale
snapshot against the shadows and silently skip the new variable — corruption that
only surfaced (as a count mismatch) at swap_in time during eval. apply_gradients
now raises immediately if the snapshot/shadow lengths diverge.
"""

import unittest

import tensorflow as tf

from optimizers.ema import ExponentialMovingAverage


def _make_model():
    inp = tf.keras.Input(shape=(4,))
    out = tf.keras.layers.Dense(2, name="dense")(inp)
    return tf.keras.Model(inp, out)


class TestEmaVarMismatchGuard(unittest.TestCase):
    def _ema(self, model):
        return ExponentialMovingAverage(
            tf.keras.optimizers.SGD(0.01), model, dynamic_decay=True
        )

    def test_normal_apply_gradients_ok(self):
        model = _make_model()
        model(tf.zeros([1, 4]))
        ema = self._ema(model)
        gv = [(tf.zeros_like(v), v) for v in model.trainable_variables]
        ema.apply_gradients(gv)  # must not raise

    def test_mismatch_raises_immediately(self):
        model = _make_model()
        model(tf.zeros([1, 4]))
        ema = self._ema(model)
        # Simulate the model gaining a variable after EMA construction by
        # corrupting the snapshot length the same way a lazily-built layer would.
        ema._model_vars = ema._model_vars + [tf.Variable(0.0)]
        gv = [(tf.zeros_like(v), v) for v in model.trainable_variables]
        with self.assertRaises(ValueError):
            ema.apply_gradients(gv)


if __name__ == "__main__":
    unittest.main()
