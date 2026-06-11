"""Pinning test: EMA.apply_gradients guards against the LIVE model growing vars.

The EMA wrapper snapshots model.variables in __init__ and creates one shadow per
snapshotted var. If the model later gains a variable (e.g. a layer built lazily,
or a monkey-patched variable), apply_gradients used to zip the stale snapshot
against the shadows and silently skip the new variable — corruption that only
surfaced (as a count mismatch) at swap_in time during eval. apply_gradients now
raises immediately if the LIVE model variable count diverges from the shadow count.

Regression guard: the check MUST read `self._model.variables` (re-read every step),
not compare two same-length __init__ snapshots (`len(self._model_vars)` vs
`len(self._shadows)`) — those are equal by construction and the guard could never
fire. ``test_grown_model_raises`` actually grows the live model so a snapshot-vs-
snapshot guard would NOT catch it; this test would fail against the no-op form.
"""

import unittest

import tensorflow as tf

from optimizers.ema import ExponentialMovingAverage


def _make_model():
    inp = tf.keras.Input(shape=(4,))
    out = tf.keras.layers.Dense(2, name="dense")(inp)
    return tf.keras.Model(inp, out)


class _GrowableModule(tf.Module):
    """tf.Module whose ``.variables`` re-reads dynamically, so attaching a new
    Variable after construction grows the live variable count — the real
    "model grows variables after EMA construction" scenario."""

    def __init__(self):
        super().__init__()
        self.v0 = tf.Variable(tf.zeros([2]), name="v0")

    def grow(self):
        self.v1 = tf.Variable(tf.zeros([3]), name="v1")


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

    def test_grown_model_raises(self):
        # The LIVE model grows a variable after EMA construction. A guard that
        # only compared the two __init__ snapshots (equal by construction) would
        # NOT fire here; the live-count guard must.
        model = _GrowableModule()
        ema = self._ema(model)
        gv = [(tf.zeros_like(v), v) for v in model.trainable_variables]
        ema.apply_gradients(gv)  # baseline: no growth yet, must not raise

        model.grow()  # live model now has one more variable than there are shadows
        with self.assertRaises(ValueError):
            ema.apply_gradients(gv)

    def test_snapshot_corruption_alone_does_not_falsely_pass(self):
        # Corrupting only the __init__ snapshot (without the live model growing)
        # must NOT trip the guard: the guard reads live model.variables, so a
        # stale `_model_vars` list is irrelevant to whether shadows still cover
        # the live model. (Pins that the guard is keyed on the live model.)
        model = _make_model()
        model(tf.zeros([1, 4]))
        ema = self._ema(model)
        ema._model_vars = ema._model_vars + [tf.Variable(0.0)]
        gv = [(tf.zeros_like(v), v) for v in model.trainable_variables]
        ema.apply_gradients(gv)  # must not raise — live count still matches shadows


if __name__ == "__main__":
    unittest.main()
