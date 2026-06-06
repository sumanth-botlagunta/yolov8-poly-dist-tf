"""SGD optimizer with per-param-group weight decay and momentum warmup.

Replicates PyTorch-style SGD behavior for three parameter groups:
    Group 0 — BN params  (gamma, beta, moving_mean, moving_variance): WD=0
    Group 1 — Biases     (bias):                                       WD=0
    Group 2 — Weights    (kernel):                                     WD=weight_decay

Momentum is linearly warmed up from momentum_start → momentum over warmup_steps,
then held constant. Weight decay is applied as L2 regularization before the
gradient step: w ← w * (1 − lr * wd).

Classes:
    SGDTorch: SGD optimizer compatible with tf.train.Checkpoint via tf.Module.
"""

from typing import Callable, List, Optional, Tuple

import tensorflow as tf


_NORM_KEYS   = ('gamma', 'beta', 'moving_mean', 'moving_variance')
_BIAS_KEYS   = ('bias',)
_WEIGHT_KEYS = ('kernel',)


def _classify_var(name: str) -> int:
    """Return param-group index: 0=BN, 1=bias, 2=weight."""
    n = name.lower()
    if any(k in n for k in _NORM_KEYS):
        return 0
    if any(k in n for k in _BIAS_KEYS):
        return 1
    return 2


class SGDTorch(tf.Module):
    """SGD with Nesterov momentum, per-param-group weight decay, and momentum warmup.

    Args:
        lr_fn: Callable(step) → learning rate scalar (e.g. CosineDecay schedule).
        momentum: Target momentum (reached after warmup_steps).
        momentum_start: Initial momentum at step 0.
        nesterov: Use Nesterov look-ahead correction.
        weight_decay: L2 weight-decay coefficient applied to group-2 variables.
        warmup_steps: Number of steps to linearly ramp momentum to target value.
    """

    def __init__(
        self,
        lr_fn: Callable,
        momentum: float = 0.937,
        momentum_start: float = 0.8,
        nesterov: bool = True,
        weight_decay: float = 0.0005,
        warmup_steps: int = 7164,
    ):
        super().__init__(name='sgd_torch')
        self._lr_fn        = lr_fn
        self._momentum     = momentum
        self._momentum_start = momentum_start
        self._nesterov     = nesterov
        self._weight_decay = weight_decay
        self._warmup_steps = warmup_steps

        self.iterations = tf.Variable(0, trainable=False, dtype=tf.int64,
                                      name='sgd_step')
        # Velocity slots are created lazily on first apply_gradients call.
        self._velocities: List[tf.Variable] = []
        self._var_refs: List = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lr(self) -> tf.Tensor:
        return tf.cast(self._lr_fn(self.iterations), tf.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[Optional[tf.Tensor], tf.Variable]],
        **kwargs,
    ) -> None:
        lr = self.lr
        mu = self._current_momentum()

        for grad, var in grads_and_vars:
            if grad is None:
                continue

            group = _classify_var(var.name)

            if group == 2:
                # Decoupled weight decay applied before gradient step
                var.assign(var * (1.0 - lr * self._weight_decay))

            vel = self._get_or_create_velocity(var)

            # v ← μ·v + g
            new_vel = mu * vel + grad
            vel.assign(new_vel)

            # Nesterov: effective update = μ·v_new + g; plain momentum: v_new
            update = mu * new_vel + grad if self._nesterov else new_vel
            var.assign_sub(lr * update)

        self.iterations.assign_add(1)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return {
            'momentum':       self._momentum,
            'momentum_start': self._momentum_start,
            'nesterov':       self._nesterov,
            'weight_decay':   self._weight_decay,
            'warmup_steps':   self._warmup_steps,
        }

    @classmethod
    def from_config(cls, config: dict, lr_fn: Callable) -> 'SGDTorch':
        return cls(lr_fn=lr_fn, **config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_momentum(self) -> tf.Tensor:
        """Linear warmup: momentum_start → momentum over warmup_steps."""
        step    = tf.cast(self.iterations, tf.float32)
        warmup  = tf.cast(self._warmup_steps, tf.float32)
        t       = tf.minimum(step / tf.maximum(warmup, 1.0), 1.0)
        return tf.cast(
            self._momentum_start + t * (self._momentum - self._momentum_start),
            tf.float32,
        )

    def _get_or_create_velocity(self, var) -> tf.Variable:
        """Return (or lazily create) the momentum slot for *var*."""
        for stored_var, vel in zip(self._var_refs, self._velocities):
            if stored_var is var:
                return vel
        vel = tf.Variable(tf.zeros_like(var), trainable=False,
                          name=f'vel_{len(self._velocities)}')
        self._var_refs.append(var)
        self._velocities.append(vel)
        return vel
