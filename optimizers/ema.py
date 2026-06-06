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
        ema_opt.swap_weights(model)               # eval: swap to shadow
        ...evaluate...
        ema_opt.swap_weights(model)               # restore real weights
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

    def _get_decay(self) -> tf.Tensor:
        """Compute effective decay for the current step."""
        if self._dynamic_decay:
            step = tf.cast(self._ema_step, tf.float32)
            return tf.cast(
                tf.minimum(self._average_decay, (1.0 + step) / (10.0 + step)),
                tf.float32,
            )
        return tf.constant(self._average_decay, dtype=tf.float32)

    def update_average(self, var: tf.Variable, value: tf.Tensor) -> None:
        """Update one shadow variable: shadow = decay * shadow + (1-decay) * value."""
        for v, shadow in zip(self._model_vars, self._shadows):
            if v is var:
                decay = self._get_decay()
                shadow.assign(decay * shadow + (1.0 - decay) * value)
                return

    def swap_weights(self, model: tf.keras.Model) -> None:
        """Swap all model variables with their shadow counterparts in-place."""
        for var, shadow in zip(model.variables, self._shadows):
            live = tf.identity(var)
            var.assign(tf.identity(shadow))
            shadow.assign(live)

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply gradients to real weights, then update all shadow weights."""
        result = self._optimizer.apply_gradients(grads_and_vars, **kwargs)
        self._ema_step.assign_add(1)
        decay = self._get_decay()
        for var, shadow in zip(self._model_vars, self._shadows):
            shadow.assign(decay * shadow + (1.0 - decay) * tf.identity(var))
        return result
