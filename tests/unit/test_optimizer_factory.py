"""Tests for the config-selectable optimizer / LR-schedule factory (optimizers/factory.py).

The cardinal requirement: the DEFAULTS ('sgd' / 'cosine') must reproduce the previous
hardcoded path exactly, so an existing config trains byte-identically.
"""

import pytest
import tensorflow as tf

from configs.model_config import LrScheduleConfig, OptimizerConfig
from optimizers import factory
from optimizers.sgd_warmup import SGDTorch


_STEPS = [0, 1, 1000, 100000, 635400, 700000]


def test_default_cosine_is_byte_identical():
    cfg = LrScheduleConfig()  # type='cosine'
    mine = factory.build_lr_schedule(cfg)
    ref = tf.keras.optimizers.schedules.CosineDecay(
        cfg.initial_learning_rate, cfg.decay_steps, alpha=cfg.alpha)
    for s in _STEPS:
        assert float(mine(s)) == float(ref(s))


def test_default_optimizer_is_sgdtorch():
    lr = factory.build_lr_schedule(LrScheduleConfig())
    core = factory.build_core_optimizer(OptimizerConfig(), lr, bias_lr_scale=0.1)
    assert isinstance(core, SGDTorch)


def test_alternative_optimizers_build():
    lr = factory.build_lr_schedule(LrScheduleConfig())
    assert type(factory.build_core_optimizer(
        OptimizerConfig(type='adamw'), lr, 0.1)).__name__ == 'AdamW'
    assert type(factory.build_core_optimizer(
        OptimizerConfig(type='adam'), lr, 0.1)).__name__ == 'Adam'


def test_linear_decay_endpoints():
    cfg = LrScheduleConfig(type='linear', initial_learning_rate=0.01,
                           decay_steps=1000, alpha=0.1)
    s = factory.build_lr_schedule(cfg)
    assert abs(float(s(0)) - 0.01) < 1e-9          # starts at initial
    assert abs(float(s(1000)) - 0.001) < 1e-9      # ends at initial*alpha
    assert abs(float(s(5000)) - 0.001) < 1e-9      # holds after decay_steps


def test_constant_is_flat():
    s = factory.build_lr_schedule(LrScheduleConfig(type='constant', initial_learning_rate=0.007))
    assert float(s(0)) == float(s(99999))          # flat
    assert abs(float(s(0)) - 0.007) < 1e-7         # at the configured value (float32)


def test_unknown_optimizer_type_fails_loud():
    # 'sgd_torch' (a removed alias) and any other unknown type must raise a
    # clear error instead of silently falling back to a default optimizer.
    lr = factory.build_lr_schedule(LrScheduleConfig())
    with pytest.raises((KeyError, ValueError)):
        factory.build_core_optimizer(OptimizerConfig(type='sgd_torch'), lr, 0.1)


def test_step_decay_staircase():
    s = factory.build_lr_schedule(LrScheduleConfig(
        type='step', initial_learning_rate=0.01, step_size=100, gamma=0.5))
    assert abs(float(s(0)) - 0.01) < 1e-9
    assert abs(float(s(100)) - 0.005) < 1e-9
    assert abs(float(s(250)) - 0.0025) < 1e-9


def test_warmup_ramps_then_matches_base():
    base_cfg = LrScheduleConfig(type='cosine')
    base = factory.build_lr_schedule(base_cfg)
    warm = factory.build_lr_schedule(LrScheduleConfig(type='cosine', warmup_steps=100,
                                                      warmup_init_lr=0.0))
    assert float(warm(0)) < float(warm(50)) < float(warm(100))   # ramps up
    assert abs(float(warm(100)) - float(base(100))) < 1e-6       # then == base
    assert abs(float(warm(5000)) - float(base(5000))) < 1e-6


def test_warmup_zero_is_passthrough():
    base = factory.build_lr_schedule(LrScheduleConfig(type='cosine'))
    warm = factory.build_lr_schedule(LrScheduleConfig(type='cosine', warmup_steps=0))
    for s in _STEPS:
        assert float(warm(s)) == float(base(s))


def test_polynomial_decay_endpoints():
    """PolynomialDecay: starts at initial_learning_rate, ends at initial*alpha."""
    cfg = LrScheduleConfig(type='polynomial', initial_learning_rate=0.01,
                           decay_steps=1000, alpha=0.1, power=2.0)
    s = factory.build_lr_schedule(cfg)
    assert abs(float(s(0)) - 0.01) < 1e-6          # starts at initial_learning_rate
    assert abs(float(s(1000)) - 0.001) < 1e-6      # ends at initial * alpha = 0.001


def test_unknown_type_raises():
    import pytest
    with pytest.raises(KeyError):
        factory.build_lr_schedule(LrScheduleConfig(type='nope'))
    with pytest.raises(KeyError):
        factory.build_core_optimizer(OptimizerConfig(type='nope'),
                                     factory.build_lr_schedule(LrScheduleConfig()), 0.1)
