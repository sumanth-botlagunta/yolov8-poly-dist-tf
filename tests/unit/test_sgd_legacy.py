"""SGDTorchLegacy (legacy Keras optimizer base) must match SGDTorch step-for-step.

The two classes implement the same update rule on different apply machinery
(hand-rolled tf.Module loop vs tf.keras.optimizers.legacy.Optimizer). Their
purpose is an A/B probe for framework-stack effects, which is only valid if
the MATH is pinned identical — these tests train the same variables with both
and require byte-level agreement through the warmup region, across all three
parameter groups (kernel with coupled WD, bias, BN).
"""

import numpy as np
import tensorflow as tf

from optimizers.sgd_legacy import SGDTorchLegacy
from optimizers.sgd_warmup import SGDTorch


def _make_vars(prefix):
    return [
        tf.Variable(tf.constant([[1.0, -2.0], [0.5, 3.0]]), name=f"{prefix}/conv/kernel"),
        tf.Variable(tf.constant([0.3, -0.7]), name=f"{prefix}/conv/bias"),
        tf.Variable(tf.constant([1.1, 0.9]), name=f"{prefix}/bn/gamma"),
    ]


def _grads_for(step):
    g = tf.constant([[0.1, 0.2], [-0.3, 0.05]]) * (1.0 + 0.1 * step)
    return [g, tf.constant([0.05, -0.02]) * (1.0 + 0.1 * step),
            tf.constant([0.01, 0.03]) * (1.0 + 0.1 * step)]


def _run(optimizer, variables, steps):
    for s in range(steps):
        optimizer.apply_gradients(list(zip(_grads_for(s), variables)))
    return [v.numpy() for v in variables]


def test_legacy_base_matches_sgd_torch_through_warmup():
    lr_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.01, decay_steps=100, alpha=0.01)
    kwargs = dict(momentum=0.937, momentum_start=0.8, nesterov=True,
                  weight_decay=0.0005, warmup_steps=6, bias_lr_scale=0.1)

    ref_vars = _make_vars("ref")
    new_vars = _make_vars("new")
    # warmup_steps=6, 10 steps → covers the ramp AND the post-warmup regime.
    ref = _run(SGDTorch(lr_fn=lr_fn, **kwargs), ref_vars, steps=10)
    new = _run(SGDTorchLegacy(lr_fn=lr_fn, **kwargs), new_vars, steps=10)

    for r, n, v in zip(ref, new, ref_vars):
        np.testing.assert_allclose(
            n, r, rtol=0, atol=1e-7,
            err_msg=f"legacy-base update diverged from SGDTorch for {v.name}")


def test_legacy_base_weight_decay_only_on_kernels():
    # With zero gradients, only the kernel group (coupled WD) may move.
    lr_fn = lambda step: tf.constant(0.01)
    opt = SGDTorchLegacy(lr_fn=lr_fn, momentum=0.9, momentum_start=0.9,
                         nesterov=False, weight_decay=0.01, warmup_steps=0,
                         bias_lr_scale=0.0)
    variables = _make_vars("wd")
    before = [v.numpy().copy() for v in variables]
    zero_grads = [tf.zeros_like(v) for v in variables]
    opt.apply_gradients(list(zip(zero_grads, variables)))

    assert not np.allclose(variables[0].numpy(), before[0]), "kernel must decay"
    np.testing.assert_array_equal(variables[1].numpy(), before[1])  # bias untouched
    np.testing.assert_array_equal(variables[2].numpy(), before[2])  # BN untouched


def test_factory_builds_legacy_type():
    from configs.model_config import LrScheduleConfig, OptimizerConfig
    from optimizers import factory
    lr = factory.build_lr_schedule(LrScheduleConfig())
    core = factory.build_core_optimizer(OptimizerConfig(type="sgd_legacy"), lr, 0.1)
    assert isinstance(core, SGDTorchLegacy)


def test_runtime_disable_onednn_parses():
    from configs.yaml_loader import _build_runtime_config
    assert _build_runtime_config({}).disable_onednn is False
    assert _build_runtime_config({"disable_onednn": True}).disable_onednn is True
