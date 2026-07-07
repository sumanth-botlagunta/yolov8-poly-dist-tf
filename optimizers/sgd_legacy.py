"""SGDTorch math on the Keras *legacy* optimizer base class.

Same update rule as ``optimizers.sgd_warmup.SGDTorch`` (per-param-group weight
decay, momentum warmup, bias/BN warmup LR ramp, Nesterov, COUPLED weight
decay), but implemented as a ``tf.keras.optimizers.legacy.Optimizer`` subclass
so the gradient-application machinery (slot management, iteration counter,
cross-replica aggregation, dtype handling) is the legacy base class's rather
than the hand-rolled ``tf.Module`` loop.

Purpose: an A/B probe for framework-version effects. If a run with
``optimizer.type: sgd_legacy`` (plus ``runtime.disable_onednn: true``) diverges
from the default ``sgd`` run, the difference lives in the optimizer/apply stack
of the framework, not in the update math — both classes implement the same
formula, pinned against each other by tests.

Trade-offs vs SGDTorch (acceptable for the probe):
  * No per-call ``clip_norm`` (the recipe runs gradient_clip_norm=0.0; pass
    ``global_clipnorm`` at construction if clipping is ever needed).
  * Slots are created by the legacy base inside apply_gradients — fine under
    one_device; MirroredStrategy would need eager slot creation.
"""

from typing import Callable

import tensorflow as tf

from optimizers.sgd_warmup import _classify_var


class SGDTorchLegacy(tf.keras.optimizers.legacy.Optimizer):
    """SGDTorch update rule on the legacy Keras optimizer base.

    Update (identical to SGDTorch):
        g ← g + wd·w                     # group-2 (kernel) variables only
        v ← μ(t)·v + g                   # μ warms up momentum_start → momentum
        w ← w − lr_group(t)·(μ(t)·v + g) # Nesterov (or v without)
    where lr_group ramps bias/BN DOWN from ``bias_lr_scale`` and weights UP
    from 0 during warmup, all groups equal to the schedule LR afterwards.
    """

    def __init__(
        self,
        lr_fn: Callable,
        momentum: float = 0.937,
        momentum_start: float = 0.8,
        nesterov: bool = True,
        weight_decay: float = 0.0005,
        warmup_steps: int = 7164,
        bias_lr_scale: float = 0.1,
        name: str = "sgd_torch_legacy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._lr_fn          = lr_fn
        self._momentum       = momentum
        self._momentum_start = momentum_start
        self._nesterov       = nesterov
        self._weight_decay   = weight_decay
        self._warmup_steps   = warmup_steps
        self._bias_lr_scale  = bias_lr_scale

    # -- slots ----------------------------------------------------------
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum")

    # -- shared warmup helpers (same formulas as SGDTorch) ---------------
    def _warmup_t(self):
        step   = tf.cast(self.iterations, tf.float32)
        warmup = tf.maximum(tf.cast(self._warmup_steps, tf.float32), 1.0)
        return tf.minimum(step / warmup, 1.0)

    def _current_momentum(self) -> tf.Tensor:
        t = self._warmup_t()
        return tf.cast(
            self._momentum_start + t * (self._momentum - self._momentum_start),
            tf.float32,
        )

    def _effective_lr(self, base_lr, t, group):
        if self._bias_lr_scale <= 0.0:
            return base_lr
        if group == 2:
            return tf.where(t < 1.0, t * base_lr, base_lr)
        start = tf.cast(self._bias_lr_scale, base_lr.dtype)
        return tf.where(t < 1.0, start + t * (base_lr - start), base_lr)

    # -- logging compatibility with the trainer --------------------------
    @property
    def lr(self) -> tf.Tensor:
        return tf.cast(self._lr_fn(self.iterations), tf.float32)

    @property
    def lr_for_last_step(self) -> tf.Tensor:
        prev = tf.maximum(self.iterations - 1, 0)
        return tf.cast(self._lr_fn(prev), tf.float32)

    def group_lrs_for_last_step(self):
        prev   = tf.maximum(self.iterations - 1, 0)
        base   = tf.cast(self._lr_fn(prev), tf.float32)
        warmup = tf.maximum(tf.cast(self._warmup_steps, tf.float32), 1.0)
        t      = tf.minimum(tf.cast(prev, tf.float32) / warmup, 1.0)
        return self._effective_lr(base, t, 1), self._effective_lr(base, t, 2)

    # -- the update -------------------------------------------------------
    def _resource_apply_dense(self, grad, var, apply_state=None):
        t       = self._warmup_t()
        mu      = tf.cast(self._current_momentum(), var.dtype)
        base_lr = tf.cast(self._lr_fn(self.iterations), var.dtype)
        group   = _classify_var(var.name)
        eff_lr  = self._effective_lr(base_lr, tf.cast(t, var.dtype), group)

        if group == 2 and self._weight_decay > 0.0:
            grad = grad + tf.cast(self._weight_decay, var.dtype) * var

        vel     = self.get_slot(var, "momentum")
        new_vel = vel.assign(mu * vel + grad, use_locking=self._use_locking)
        update  = mu * new_vel + grad if self._nesterov else new_vel
        return var.assign_sub(eff_lr * update, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Dense-model training path only; no embedding/sparse variables exist here.
        raise NotImplementedError("SGDTorchLegacy does not support sparse gradients.")

    def get_config(self):
        config = super().get_config()
        config.update({
            "momentum":       self._momentum,
            "momentum_start": self._momentum_start,
            "nesterov":       self._nesterov,
            "weight_decay":   self._weight_decay,
            "warmup_steps":   self._warmup_steps,
            "bias_lr_scale":  self._bias_lr_scale,
        })
        return config
