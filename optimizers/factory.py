"""Config-selectable optimizers and LR schedules (registry-backed).

A ``type`` key selects the optimizer and LR schedule; the defaults are
byte-identical to the SGDTorch + CosineDecay path.

    optimizers:    sgd (default), adamw, adam
    lr schedules:  cosine (default), linear, step, polynomial, constant
    warmup:        optional linear LR warmup wrapper (off by default; SGD keeps
                   its own momentum/bias warmup)

Each builder takes the parsed config object and returns the constructed instance.
"""

from __future__ import annotations

import tensorflow as tf

from configs.registry import Registry

OPTIMIZERS = Registry("OPTIMIZERS")
LR_SCHEDULES = Registry("LR_SCHEDULES")


# Custom LR schedules (tf.keras schedules, so they compose with any optimizer).

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear decay from ``initial`` to ``initial*alpha`` over ``decay_steps``.

    Ultralytics ``lf`` schedule: ``lr = initial * ((1 - t)(1 - alpha) + alpha)``
    with ``t = min(step/decay_steps, 1)``. Holds at ``initial*alpha`` afterwards.
    """

    def __init__(self, initial_learning_rate, decay_steps, alpha):
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = int(decay_steps)
        self.alpha = float(alpha)

    def __call__(self, step):
        t = tf.minimum(tf.cast(step, tf.float32) / float(max(self.decay_steps, 1)), 1.0)
        factor = (1.0 - t) * (1.0 - self.alpha) + self.alpha
        return self.initial_learning_rate * factor

    def get_config(self):
        return {'initial_learning_rate': self.initial_learning_rate,
                'decay_steps': self.decay_steps, 'alpha': self.alpha}


class ConstantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Flat learning rate (ignores step)."""

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = float(initial_learning_rate)

    def __call__(self, step):
        return tf.constant(self.initial_learning_rate, tf.float32)

    def get_config(self):
        return {'initial_learning_rate': self.initial_learning_rate}


class StepDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Multiply the LR by ``gamma`` every ``step_size`` steps (staircase)."""

    def __init__(self, initial_learning_rate, step_size, gamma):
        self.initial_learning_rate = float(initial_learning_rate)
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def __call__(self, step):
        n = tf.floor(tf.cast(step, tf.float32) / float(max(self.step_size, 1)))
        return self.initial_learning_rate * tf.pow(self.gamma, n)

    def get_config(self):
        return {'initial_learning_rate': self.initial_learning_rate,
                'step_size': self.step_size, 'gamma': self.gamma}


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Wrap a base schedule with a linear LR warmup over the first ``warmup_steps``.

    The LR ramps linearly from ``warmup_init_lr`` to the base schedule value,
    then is exactly the base schedule. ``warmup_steps == 0`` is a pass-through.
    """

    def __init__(self, base, warmup_steps, warmup_init_lr=0.0):
        self.base = base
        self.warmup_steps = int(warmup_steps)
        self.warmup_init_lr = float(warmup_init_lr)

    def __call__(self, step):
        base_lr = self.base(step)
        if self.warmup_steps <= 0:
            return base_lr
        s = tf.cast(step, tf.float32)
        w = float(self.warmup_steps)
        frac = tf.minimum((s + 1.0) / w, 1.0)
        warm_lr = self.warmup_init_lr + (base_lr - self.warmup_init_lr) * frac
        return tf.where(s < w, warm_lr, base_lr)

    def get_config(self):
        return {'warmup_steps': self.warmup_steps, 'warmup_init_lr': self.warmup_init_lr}


# LR-schedule registry (builder per type).

@LR_SCHEDULES.register('cosine')
def _build_cosine(cfg):
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.initial_learning_rate,
        decay_steps=cfg.decay_steps,
        alpha=cfg.alpha,
    )


@LR_SCHEDULES.register('linear')
def _build_linear(cfg):
    return LinearDecay(cfg.initial_learning_rate, cfg.decay_steps, cfg.alpha)


@LR_SCHEDULES.register('polynomial')
def _build_polynomial(cfg):
    return tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=cfg.initial_learning_rate,
        decay_steps=cfg.decay_steps,
        end_learning_rate=cfg.initial_learning_rate * cfg.alpha,
        power=cfg.power,
    )


@LR_SCHEDULES.register('step')
def _build_step(cfg):
    return StepDecay(cfg.initial_learning_rate, cfg.step_size, cfg.gamma)


@LR_SCHEDULES.register('constant')
def _build_constant(cfg):
    return ConstantSchedule(cfg.initial_learning_rate)


def build_lr_schedule(lr_cfg):
    """Build the LR schedule for ``lr_cfg`` (``type`` selects the builder),
    optionally wrapped with a linear LR warmup when ``lr_cfg.warmup_steps > 0``
    (default 0 = none, so the cosine path is unwrapped)."""
    base = LR_SCHEDULES.get(getattr(lr_cfg, 'type', 'cosine'))(lr_cfg)
    warmup_steps = int(getattr(lr_cfg, 'warmup_steps', 0) or 0)
    if warmup_steps > 0:
        base = LinearWarmup(base, warmup_steps,
                            warmup_init_lr=getattr(lr_cfg, 'warmup_init_lr', 0.0))
    return base


# Optimizer registry (builder per type).

@OPTIMIZERS.register('sgd')
def _build_sgd(opt_cfg, lr_fn, bias_lr_scale, clip_norm=0.0):
    # SGDTorch clips via the clip_norm kwarg forwarded through apply_gradients.
    from optimizers.sgd_warmup import SGDTorch
    return SGDTorch(
        lr_fn=lr_fn,
        momentum=opt_cfg.momentum,
        momentum_start=opt_cfg.momentum_start,
        nesterov=opt_cfg.nesterov,
        weight_decay=opt_cfg.weight_decay,
        warmup_steps=opt_cfg.warmup_steps,
        bias_lr_scale=bias_lr_scale,
    )


def _keras_clip(clip_norm):
    # keras optimizers clip at construction via global_clipnorm (None = no clipping).
    return {'global_clipnorm': clip_norm} if clip_norm and clip_norm > 0.0 else {}


@OPTIMIZERS.register('adamw')
def _build_adamw(opt_cfg, lr_fn, bias_lr_scale, clip_norm=0.0):
    # Decoupled weight decay. LR warmup, if any, is folded into lr_fn.
    return tf.keras.optimizers.AdamW(
        learning_rate=lr_fn,
        weight_decay=opt_cfg.weight_decay,
        beta_1=opt_cfg.beta_1,
        beta_2=opt_cfg.beta_2,
        **_keras_clip(clip_norm),
    )


@OPTIMIZERS.register('adam')
def _build_adam(opt_cfg, lr_fn, bias_lr_scale, clip_norm=0.0):
    return tf.keras.optimizers.Adam(
        learning_rate=lr_fn,
        beta_1=opt_cfg.beta_1,
        beta_2=opt_cfg.beta_2,
        **_keras_clip(clip_norm),
    )


def build_core_optimizer(opt_cfg, lr_fn, bias_lr_scale, clip_norm=0.0):
    """Build the (pre-EMA) optimizer for ``opt_cfg`` (``type`` selects the builder).

    ``clip_norm`` (from ``task.gradient_clip_norm``) is applied per optimizer:
    SGDTorch clips per-call in apply_gradients (ignoring this arg), keras
    optimizers set ``global_clipnorm`` at construction.
    """
    return OPTIMIZERS.get(getattr(opt_cfg, 'type', 'sgd'))(
        opt_cfg, lr_fn, bias_lr_scale, clip_norm=clip_norm)
