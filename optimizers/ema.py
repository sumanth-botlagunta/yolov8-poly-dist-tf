"""Exponential Moving Average optimizer wrapper.

Maintains shadow weights that track a running average of the model weights,
used during evaluation only (swap in before eval, swap back after).

Dynamic decay (dynamic_decay=True) is the YOLOv5-style exponential ramp:
    decay = average_decay * (1 - exp(-step / 2000))
Decay starts at 0 (shadow copies the live weights) and rises with a 2000-step
time constant, reaching within 1% of average_decay=0.9999 by ~step 10k.
"""

import tensorflow as tf


class ExponentialMovingAverage(tf.Module):
    """Optimizer wrapper that maintains EMA shadow weights.

    Inherits tf.Module (not tf.keras.optimizers.Optimizer) so shadow variables
    are trackable by tf.train.Checkpoint without the Keras optimizer build()
    protocol.

    Usage:
        ema_opt = ExponentialMovingAverage(base_optimizer, model)
        ema_opt.apply_gradients(grads_and_vars)   # updates real + shadow
        ema_opt.swap_in(model)                     # eval: load shadow weights
        ...evaluate...
        ema_opt.swap_out(model)                    # restore real weights
    """

    def __init__(
        self,
        optimizer,
        model: tf.keras.Model,
        average_decay: float = 0.9999,
        dynamic_decay: bool = True,
        **kwargs,
    ):
        super().__init__(name='ema')
        self._optimizer = optimizer
        self._average_decay = average_decay
        self._dynamic_decay = dynamic_decay
        self._ema_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='ema_step')
        # Live model reference lets apply_gradients detect variables added after
        # construction by comparing the live count against the shadow snapshot.
        self._model = model
        self._model_vars = list(model.variables)
        # tf.identity copies the value for both tf.Variable and keras.Variable.
        self._shadows = [
            tf.Variable(tf.identity(v), trainable=False, name=f'ema_shadow_{i}')
            for i, v in enumerate(self._model_vars)
        ]
        # Full snapshot of the live weights while shadow weights are swapped in
        # for eval; None otherwise. Transient eval-only state, not checkpointed.
        self._backup = None

    def build(self, variables) -> None:
        """Pre-create the inner optimizer's slots in cross-replica context.

        Required under tf.distribute so no variable is created inside
        strategy.run. Shadows are already created in ``__init__``.
        """
        if hasattr(self._optimizer, 'build'):
            self._optimizer.build(variables)

    def _get_decay(self) -> tf.Tensor:
        """Effective decay for the current step.

        Dynamic decay is ``average_decay * (1 - exp(-step/2000))``, asymptotic
        to average_decay so no min() clamp is needed.
        """
        if self._dynamic_decay:
            step = tf.cast(self._ema_step, tf.float32)
            return tf.cast(
                self._average_decay * (1.0 - tf.math.exp(-step / 2000.0)),
                tf.float32,
            )
        return tf.constant(self._average_decay, dtype=tf.float32)

    def swap_in(self, model: tf.keras.Model) -> None:
        """Load shadow weights into the model for evaluation.

        Crash-safe: the shadow weights are never mutated here (so the
        checkpointed EMA state survives an interruption), and the full live
        snapshot is taken before any variable is overwritten (so an interrupted
        assign loop can still be fully restored by ``swap_out``). Pair every
        ``swap_in`` with a ``swap_out`` via try/finally.
        """
        model_vars = model.variables
        if len(model_vars) != len(self._shadows):
            raise ValueError(
                "EMA shadow/model variable count mismatch: "
                f"{len(self._shadows)} shadows vs {len(model_vars)} model variables. "
                "The model must be fully built before constructing the EMA wrapper; "
                "otherwise zip() truncates and skips swapping some weights at eval."
            )
        if self._backup is not None:
            raise RuntimeError(
                "EMA.swap_in() called while already swapped in; call swap_out() "
                "first. Nesting swap_in would overwrite the live-weight backup."
            )
        # Full live snapshot first (completes or raises before any var is touched).
        self._backup = [tf.identity(v) for v in model_vars]
        for var, shadow in zip(model_vars, self._shadows):
            var.assign(tf.identity(shadow))

    def swap_out(self, model: tf.keras.Model) -> None:
        """Restore the live weights saved by ``swap_in``.

        A no-op when not currently swapped in, so a ``finally`` block can call
        it unconditionally.
        """
        if self._backup is None:
            return
        for var, backup in zip(model.variables, self._backup):
            var.assign(backup)
        self._backup = None

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply gradients to real weights, then update all shadow weights."""
        # Catch the model gaining variables after construction (e.g. a lazily
        # built layer): the shadows are a fixed __init__ snapshot, so a grown
        # model would zip() against a stale subset. Compare the live variable
        # count (re-read each step) against the snapshot, not two __init__ lists
        # that are equal by construction.
        if len(self._model.variables) != len(self._shadows):
            raise ValueError(
                "EMA shadow/model-var snapshot mismatch: "
                f"{len(self._shadows)} shadows vs {len(self._model.variables)} live "
                "model variables. The model must be fully built before the EMA "
                "wrapper is constructed."
            )
        # Only SGDTorch consumes a per-call ``clip_norm`` kwarg; keras optimizers
        # clip via ``global_clipnorm`` at construction and raise on an unexpected
        # kwarg. Drop it for optimizers that don't advertise support.
        if not getattr(self._optimizer, 'accepts_clip_norm', False):
            kwargs.pop('clip_norm', None)
        result = self._optimizer.apply_gradients(grads_and_vars, **kwargs)
        # Increment before computing decay (the reference reads iterations after
        # the update): the first averaging update uses step=1 -> near-zero decay,
        # so the shadow starts as a copy of the live weights and smooths in.
        self._ema_step.assign_add(1)
        decay = self._get_decay()
        for var, shadow in zip(self._model_vars, self._shadows):
            shadow.assign(decay * shadow + (1.0 - decay) * tf.identity(var))
        return result
