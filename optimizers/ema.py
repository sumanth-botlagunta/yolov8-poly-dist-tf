"""Exponential Moving Average optimizer wrapper.

EMA maintains shadow weights that are a running average of the model weights
and are used exclusively during evaluation (swap in before eval, swap back after).

Dynamic decay formula (dynamic_decay=True):
    decay = min(average_decay, (1 + step) / (10 + step))

This starts near 0 and gradually approaches average_decay=0.9999, giving
early training steps less influence than later ones.

Classes:
    ExponentialMovingAverage: Wraps any tf.keras optimizer with EMA tracking.
"""

import tensorflow as tf


class ExponentialMovingAverage(tf.Module):
    """Optimizer wrapper that maintains EMA shadow weights.

    Inherits tf.Module (not tf.keras.optimizers.Optimizer) so that shadow
    tf.Variables are automatically trackable by tf.train.Checkpoint without
    requiring compatibility with the Keras 2/3 optimizer build() protocol.

    Usage:
        ema_opt = ExponentialMovingAverage(base_optimizer, model)
        ema_opt.apply_gradients(grads_and_vars)   # updates both real + shadow
        ema_opt.swap_in(model)                     # eval: load EMA (shadow) weights
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
        self._model_vars = list(model.variables)
        # tf.identity copies the tensor value; works with both tf.Variable and
        # keras.Variable (Keras 3 removes .read_value()).
        self._shadows = [
            tf.Variable(tf.identity(v), trainable=False, name=f'ema_shadow_{i}')
            for i, v in enumerate(self._model_vars)
        ]
        # Holds a full snapshot of the live weights while EMA weights are swapped
        # in for evaluation. None when not swapped in. Not a tf.Variable (and so
        # not checkpointed): it is transient eval-only state.
        self._backup = None

    def build(self, variables) -> None:
        """Pre-create the inner optimizer's slots (cross-replica context).

        Required under tf.distribute so no variable is created inside strategy.run.
        EMA shadows are already created in __init__ (also cross-replica context).
        """
        if hasattr(self._optimizer, 'build'):
            self._optimizer.build(variables)

    def _get_decay(self) -> tf.Tensor:
        """Compute effective decay for the current step."""
        if self._dynamic_decay:
            step = tf.cast(self._ema_step, tf.float32)
            return tf.cast(
                tf.minimum(self._average_decay, (1.0 + step) / (10.0 + step)),
                tf.float32,
            )
        return tf.constant(self._average_decay, dtype=tf.float32)

    def swap_in(self, model: tf.keras.Model) -> None:
        """Load EMA (shadow) weights into the model for evaluation.

        Crash-safe by construction, unlike a symmetric live<->shadow swap:
          * The shadow (EMA) weights are NEVER mutated here, so even if this is
            interrupted the EMA state — which is also what gets checkpointed —
            stays intact.
          * The full live snapshot is taken BEFORE any model variable is
            overwritten. If the snapshot raises, no variable was touched; if the
            assign loop is interrupted, ``swap_out`` still restores every
            variable from the complete snapshot.
        Pair every ``swap_in`` with a ``swap_out`` (use try/finally).
        """
        model_vars = model.variables
        if len(model_vars) != len(self._shadows):
            raise ValueError(
                "EMA shadow/model variable count mismatch: "
                f"{len(self._shadows)} shadows vs {len(model_vars)} model variables. "
                "The model must be fully built BEFORE constructing the EMA wrapper "
                "(otherwise zip() would silently truncate and skip swapping some "
                "weights during evaluation)."
            )
        if self._backup is not None:
            raise RuntimeError(
                "EMA.swap_in() called while already swapped in; call swap_out() "
                "first. Nesting swap_in would overwrite the live-weight backup "
                "with EMA weights and lose the originals."
            )
        # Full live snapshot first (completes or raises before any var is touched).
        self._backup = [tf.identity(v) for v in model_vars]
        for var, shadow in zip(model_vars, self._shadows):
            var.assign(tf.identity(shadow))

    def swap_out(self, model: tf.keras.Model) -> None:
        """Restore the live weights saved by ``swap_in``.

        Idempotent: a no-op when not currently swapped in, so a ``finally`` block
        can call it unconditionally without risking a double restore.
        """
        if self._backup is None:
            return
        for var, backup in zip(model.variables, self._backup):
            var.assign(backup)
        self._backup = None

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply gradients to real weights, then update all shadow weights."""
        # Guard against the model gaining/losing variables AFTER the EMA wrapper
        # was constructed (e.g. a layer built lazily on first call, or a variable
        # monkey-patched on). `_model_vars`/`_shadows` are a fixed snapshot taken
        # in __init__; if the model has since grown, the zip() below would silently
        # average a stale subset and never track the new variables — a corruption
        # that otherwise only surfaces (as a count mismatch) much later in
        # swap_in() at eval time. This check is O(1) and runs every step.
        if len(self._model_vars) != len(self._shadows):
            raise ValueError(
                "EMA shadow/model-var snapshot mismatch: "
                f"{len(self._shadows)} shadows vs {len(self._model_vars)} tracked "
                "model variables. The model must be fully built BEFORE the EMA "
                "wrapper is constructed."
            )
        result = self._optimizer.apply_gradients(grads_and_vars, **kwargs)
        # Increment BEFORE computing decay (matches Ultralytics ModelEMA): the first
        # averaging update therefore uses decay = (1+1)/(10+1), not (1+0)/(10+0).
        self._ema_step.assign_add(1)
        decay = self._get_decay()
        for var, shadow in zip(self._model_vars, self._shadows):
            shadow.assign(decay * shadow + (1.0 - decay) * tf.identity(var))
        return result
