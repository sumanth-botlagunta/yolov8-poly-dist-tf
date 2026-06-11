"""Pinning test: SGDTorch.lr_for_last_step reports the LR used for the step.

apply_gradients increments `iterations` at its end, but the weight update inside
it uses the LR for the pre-increment iteration. `self.lr` read afterward is one
step ahead — an off-by-one in TensorBoard. `lr_for_last_step` evaluates the
schedule at `iterations - 1` so the logged LR matches the LR that moved the
weights.
"""

import unittest

import tensorflow as tf

from optimizers.sgd_warmup import SGDTorch


def _lr_fn(step):
    # Strictly increasing schedule so off-by-one is detectable.
    return 0.01 + 0.001 * tf.cast(step, tf.float32)


class TestLrForLastStep(unittest.TestCase):
    def _opt(self):
        return SGDTorch(lr_fn=_lr_fn, momentum=0.9)

    def test_lr_for_last_step_matches_applied_lr(self):
        opt = self._opt()
        v = tf.Variable([1.0, 2.0])
        opt.build([v])

        # Before any step: iterations==0, last-step LR clamps to schedule(0).
        self.assertAlmostEqual(float(opt.lr_for_last_step), float(_lr_fn(0)), places=6)

        # Capture the LR the step *will* use (pre-increment iteration == 0).
        lr_used = float(opt.lr)
        opt.apply_gradients([(tf.constant([0.1, 0.1]), v)])

        # iterations is now 1; plain .lr is one ahead, lr_for_last_step matches.
        self.assertAlmostEqual(float(opt.lr_for_last_step), lr_used, places=6)
        self.assertNotAlmostEqual(float(opt.lr), lr_used, places=6)

    def test_clamps_below_zero(self):
        opt = self._opt()
        self.assertGreaterEqual(float(opt.lr_for_last_step), 0.0)


if __name__ == "__main__":
    unittest.main()
