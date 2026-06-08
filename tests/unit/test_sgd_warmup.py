"""Tests for SGDTorch optimizer.

Validates:
    - Param-group classification: BN vars get zero WD, kernel vars get WD applied.
    - Momentum warmup: at step 0 momentum ≈ momentum_start; at large step ≈ momentum.
    - Nesterov update reduces a simple quadratic loss.
    - iterations counter increments on each apply_gradients call.
    - Velocity slots are created lazily and attached to the correct variable.
"""

import numpy as np
import tensorflow as tf
import unittest

from optimizers.sgd_warmup import SGDTorch, _classify_var


class TestClassifyVar(unittest.TestCase):
    def test_gamma_is_bn_group(self):
        self.assertEqual(_classify_var('batch_norm/gamma:0'), 0)

    def test_moving_mean_is_bn_group(self):
        self.assertEqual(_classify_var('bn/moving_mean:0'), 0)

    def test_bias_is_bias_group(self):
        self.assertEqual(_classify_var('dense/bias:0'), 1)

    def test_kernel_is_weight_group(self):
        self.assertEqual(_classify_var('conv/kernel:0'), 2)

    def test_unknown_falls_into_weight_group(self):
        self.assertEqual(_classify_var('mystery_param:0'), 2)


def _make_sgd(warmup_steps=100, weight_decay=0.01) -> SGDTorch:
    lr_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.01, decay_steps=10_000, alpha=0.01
    )
    return SGDTorch(
        lr_fn=lr_fn,
        momentum=0.937,
        momentum_start=0.8,
        nesterov=True,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )


class TestSGDTorch(unittest.TestCase):

    def test_iterations_starts_at_zero(self):
        sgd = _make_sgd()
        self.assertEqual(int(sgd.iterations), 0)

    def test_iterations_increments_each_call(self):
        sgd = _make_sgd()
        w = tf.Variable([1.0, 2.0])
        for _ in range(3):
            sgd.apply_gradients([(tf.ones_like(w), w)])
        self.assertEqual(int(sgd.iterations), 3)

    def test_momentum_at_step_zero_is_momentum_start(self):
        sgd = _make_sgd(warmup_steps=1000)
        mu = float(sgd._current_momentum())
        self.assertAlmostEqual(mu, 0.8, places=5)

    def test_momentum_after_warmup_is_target(self):
        sgd = _make_sgd(warmup_steps=10)
        sgd.iterations.assign(100)
        mu = float(sgd._current_momentum())
        self.assertAlmostEqual(mu, 0.937, places=5)

    def test_weight_decay_applied_to_kernel_group(self):
        """A kernel variable should be shrunk by weight decay (after warmup).

        The weight group's effective LR ramps UP from 0 over warmup_steps, so WD has
        no effect at step 0 (eff_lr=0). Past warmup the full schedule LR applies, so
        decoupled WD shrinks the weights: w_new = w * (1 - lr * wd).
        """
        sgd = _make_sgd(warmup_steps=100, weight_decay=0.1)
        sgd.iterations.assign(200)   # past warmup → weight-group LR is full base_lr
        # Variable named 'kernel' → group 2 → WD applies
        w = tf.Variable(tf.ones([2, 2]), name='kernel_wd_test')
        val_before = w.numpy().copy()
        grad = tf.zeros_like(w)   # zero grad so only WD effect is visible
        sgd.apply_gradients([(grad, w)])
        self.assertTrue((w.numpy() < val_before).all(),
                        "WD should shrink kernel weights after warmup")

    def test_no_weight_decay_for_bn_group(self):
        """BN (moving_mean) variable must NOT be shrunk by weight decay."""
        sgd = _make_sgd(weight_decay=0.1)
        w = tf.Variable(tf.ones([4]), name='bn/moving_mean')
        val_before = w.numpy().copy()
        grad = tf.zeros_like(w)
        sgd.apply_gradients([(grad, w)])
        # With zero grad and no WD, value should be unchanged
        np.testing.assert_allclose(val_before, w.numpy())

    def test_loss_decreases_with_gradient_descent(self):
        """SGDTorch must reduce a simple L2 loss over multiple steps."""
        sgd = _make_sgd(warmup_steps=0)
        sgd._momentum_start = sgd._momentum  # disable warmup for clean test
        w = tf.Variable([5.0, -5.0])         # target: w → 0

        def loss_fn():
            return tf.reduce_sum(tf.square(w))

        initial_loss = float(loss_fn())
        for _ in range(50):
            with tf.GradientTape() as tape:
                loss = loss_fn()
            grads = tape.gradient(loss, [w])
            sgd.apply_gradients(zip(grads, [w]))

        self.assertLess(float(loss_fn()), initial_loss)

    def test_velocity_slot_created_lazily(self):
        """Velocity slot must not exist before first apply_gradients call."""
        sgd = _make_sgd()
        self.assertEqual(len(sgd._velocities), 0)
        w = tf.Variable([1.0, 2.0])
        sgd.apply_gradients([(tf.ones_like(w), w)])
        self.assertEqual(len(sgd._velocities), 1)

    def test_build_precreates_zero_slots(self):
        """build() eagerly creates zero-initialized slots (for tf.distribute)."""
        sgd = _make_sgd()
        w1 = tf.Variable([1.0, 2.0], name='a/kernel')
        w2 = tf.Variable([3.0], name='b/bias')
        sgd.build([w1, w2])
        self.assertEqual(len(sgd._velocities), 2)
        for vel in sgd._velocities:
            self.assertAlmostEqual(float(tf.reduce_sum(tf.abs(vel))), 0.0)

    def test_all_reduce_gradients_noop_single_replica(self):
        """Outside a multi-replica context, gradients pass through unchanged."""
        sgd = _make_sgd()
        w = tf.Variable([1.0, 2.0])
        g = tf.constant([0.5, -0.5])
        out = sgd._all_reduce_gradients([(g, w)])
        np.testing.assert_array_equal(out[0][0].numpy(), g.numpy())
        self.assertIs(out[0][1], w)

    def test_get_config_round_trip(self):
        """get_config returns all constructor hyperparameters."""
        sgd = _make_sgd(warmup_steps=500, weight_decay=0.001)
        cfg = sgd.get_config()
        self.assertEqual(cfg['warmup_steps'], 500)
        self.assertAlmostEqual(cfg['weight_decay'], 0.001)
        self.assertTrue(cfg['nesterov'])


if __name__ == '__main__':
    unittest.main()
