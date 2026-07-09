"""SGD optimizer with per-param-group weight decay and momentum warmup.

PyTorch-style SGD over three parameter groups:
    Group 0 — BN params  (gamma, beta, moving_mean, moving_variance): wd=0
    Group 1 — Biases     (bias):                                      wd=0
    Group 2 — Weights    (kernel):                                    wd=weight_decay

Momentum warms up linearly from momentum_start → momentum over warmup_steps,
then holds. Weight decay is coupled into the gradient before the momentum
update (PyTorch / TF-model-garden SGDTorch semantics):

    g ← g + wd·w          # group-2 (kernel) variables only
    v ← μ·v + g
    w ← w − lr·(μ·v + g)  # Nesterov  (or w − lr·v without)

The wd·w term accumulates in the velocity buffer, so the steady-state shrink is
≈ lr·wd/(1−μ). Train-semantics: changing the coupling requires a fresh run.
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
        bias_lr_scale: Initial LR scale for bias/BN params during warmup.
            Bias/BN groups start at this absolute LR and ramp down to the
            schedule LR; the weight group starts at 0 and ramps up. After
            warmup_steps all groups use the schedule LR.
            Set to 0.0 to disable (all groups start at schedule LR).
    """

    # Consumes a per-call ``clip_norm`` kwarg in apply_gradients (forwarded by
    # the EMA wrapper); keras optimizers clip via global_clipnorm instead.
    accepts_clip_norm = True

    def __init__(
        self,
        lr_fn: Callable,
        momentum: float = 0.937,
        momentum_start: float = 0.8,
        nesterov: bool = True,
        weight_decay: float = 0.0005,
        warmup_steps: int = 7164,
        bias_lr_scale: float = 0.1,
    ):
        super().__init__(name='sgd_torch')
        self._lr_fn          = lr_fn
        self._momentum       = momentum
        self._momentum_start = momentum_start
        self._nesterov       = nesterov
        self._weight_decay   = weight_decay
        self._warmup_steps   = warmup_steps
        self._bias_lr_scale  = bias_lr_scale

        self.iterations = tf.Variable(0, trainable=False, dtype=tf.int64,
                                      name='sgd_step')
        # Velocity slots are created lazily on first apply_gradients, or eagerly
        # via build() — required under tf.distribute, where variables cannot be
        # created inside a replica context (strategy.run).
        self._velocities: List[tf.Variable] = []
        self._var_refs: List = []

    def build(self, variables) -> None:
        """Pre-create zero-initialized momentum slots for *variables*.

        Call in cross-replica context (inside strategy.scope, before the
        training loop) so no variable is created inside strategy.run. The slots
        are zeros, identical to lazy creation.
        """
        for var in variables:
            self._get_or_create_velocity(var)

    @property
    def lr(self) -> tf.Tensor:
        return tf.cast(self._lr_fn(self.iterations), tf.float32)

    @property
    def lr_for_last_step(self) -> tf.Tensor:
        """LR applied to the most recent ``apply_gradients``.

        ``apply_gradients`` increments ``iterations`` at its end, so reading
        ``self.lr`` afterwards reports the next step's LR. Evaluating at
        ``iterations - 1`` (clamped to 0) matches the LR that moved the weights.
        """
        prev = tf.maximum(self.iterations - 1, 0)
        return tf.cast(self._lr_fn(prev), tf.float32)

    def group_lrs_for_last_step(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """(bias_group_lr, weight_group_lr) — effective per-group LRs at the last
        applied step, for TensorBoard. During warmup the bias/BN group ramps down
        from ``bias_lr_scale`` and the weight group ramps up from 0; after warmup
        both equal the schedule LR."""
        prev   = tf.maximum(self.iterations - 1, 0)
        base   = tf.cast(self._lr_fn(prev), tf.float32)
        warmup = tf.cast(self._warmup_steps, tf.float32)
        t      = tf.minimum(tf.cast(prev, tf.float32) / tf.maximum(warmup, 1.0), 1.0)
        return self._effective_lr(base, t, 1), self._effective_lr(base, t, 2)

    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[Optional[tf.Tensor], tf.Variable]],
        clip_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        # Sum gradients across replicas so every replica applies the same update
        # and mirrored variables stay in sync (the loss is normalized by the
        # global object count). No-op under a single replica.
        grads_and_vars = self._all_reduce_gradients(list(grads_and_vars))

        # Clip after the cross-replica sum so it acts on the full-batch gradient
        # (clipping per-replica 1/N gradients would under-clip and diverge from
        # single-GPU). No-op when clip_norm is None/<=0; None grads pass through.
        if clip_norm is not None and clip_norm > 0.0:
            _grads = [g for g, _ in grads_and_vars]
            _vars  = [v for _, v in grads_and_vars]
            _grads, _ = tf.clip_by_global_norm(_grads, clip_norm)
            grads_and_vars = list(zip(_grads, _vars))

        base_lr = self.lr
        mu      = self._current_momentum()
        t       = self._warmup_progress()

        for grad, var in grads_and_vars:
            if grad is None:
                continue

            group  = _classify_var(var.name)
            eff_lr = self._effective_lr(base_lr, t, group)

            if group == 2 and self._weight_decay > 0.0:
                # Coupled weight decay: add wd·w to the gradient before the
                # momentum update so it compounds through the velocity buffer
                # (steady-state shrink ≈ lr·wd/(1−μ)).
                grad = grad + self._weight_decay * tf.cast(var, grad.dtype)

            vel = self._get_or_create_velocity(var)

            # v ← μ·v + g
            new_vel = mu * vel + grad
            vel.assign(new_vel)

            # Nesterov: effective update = μ·v_new + g; plain momentum: v_new
            update = mu * new_vel + grad if self._nesterov else new_vel
            var.assign_sub(eff_lr * update)

        self.iterations.assign_add(1)

    def get_config(self) -> dict:
        return {
            'momentum':       self._momentum,
            'momentum_start': self._momentum_start,
            'nesterov':       self._nesterov,
            'weight_decay':   self._weight_decay,
            'warmup_steps':   self._warmup_steps,
            'bias_lr_scale':  self._bias_lr_scale,
        }

    @classmethod
    def from_config(cls, config: dict, lr_fn: Callable) -> 'SGDTorch':
        return cls(lr_fn=lr_fn, **config)

    @staticmethod
    def _all_reduce_gradients(grads_and_vars):
        """SUM-reduce gradients across replicas (no-op under a single replica)."""
        ctx = tf.distribute.get_replica_context()
        if ctx is None or ctx.num_replicas_in_sync == 1:
            return grads_and_vars
        out = [[g, v] for g, v in grads_and_vars]
        idx   = [i for i, (g, _) in enumerate(out) if g is not None]
        grads = [out[i][0] for i in idx]
        reduced = ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        for j, i in enumerate(idx):
            out[i][0] = reduced[j]
        return [(g, v) for g, v in out]

    def _current_momentum(self) -> tf.Tensor:
        """Linear warmup: momentum_start → momentum over warmup_steps."""
        step    = tf.cast(self.iterations, tf.float32)
        warmup  = tf.cast(self._warmup_steps, tf.float32)
        t       = tf.minimum(step / tf.maximum(warmup, 1.0), 1.0)
        return tf.cast(
            self._momentum_start + t * (self._momentum - self._momentum_start),
            tf.float32,
        )

    def _warmup_progress(self) -> tf.Tensor:
        """Return t in [0.0, 1.0]: fraction of warmup completed."""
        step   = tf.cast(self.iterations, tf.float32)
        warmup = tf.cast(self._warmup_steps, tf.float32)
        return tf.minimum(step / tf.maximum(warmup, 1.0), 1.0)

    def _effective_lr(self, base_lr: tf.Tensor, t: tf.Tensor, group: int) -> tf.Tensor:
        """Per-param-group effective LR during warmup.

        group 0 (BN) / 1 (bias): bias_lr_scale → base_lr  (ramps DOWN)
        group 2 (weights):        0             → base_lr  (ramps UP)
        After warmup (t == 1.0): all groups return base_lr.
        """
        if self._bias_lr_scale <= 0.0:
            return base_lr
        if group == 2:
            return tf.where(t < 1.0, t * base_lr, base_lr)
        else:
            start = tf.cast(self._bias_lr_scale, tf.float32)
            return tf.where(t < 1.0, start + t * (base_lr - start), base_lr)

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
