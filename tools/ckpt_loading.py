"""Shared checkpoint-loading helper for eval / export tools.

The trainer writes two kinds of checkpoints:
  - periodic ``ckpt-N``: ``model/`` holds the RAW (live) weights and the EMA
    shadow weights live under ``optimizer/`` (the EMA wrapper).
  - ``best_*/ckpt``: the trainer swaps EMA in before writing, so ``model/``
    already holds the EMA weights and there is no ``optimizer/`` subtree.

EMA weights are what the trainer validates with and what should be deployed.
A plain ``Checkpoint(model=model).restore(...)`` only reads ``model/``, so it
silently loads RAW weights from a periodic checkpoint (wrong) but EMA weights
from a best_ checkpoint (right) — same call, opposite correctness.

``restore_eval_weights`` removes that footgun: it detects whether the checkpoint
carries EMA shadows and, if so, restores the EMA wrapper and swaps the shadows
into the model. Otherwise it restores ``model/`` directly. Both eval and export
use this so they cannot diverge.
"""

import logging

import tensorflow as tf

log = logging.getLogger(__name__)


def _checkpoint_has_ema(ckpt_path: str) -> bool:
    """True if the checkpoint stores EMA shadow variables (a periodic ckpt-N)."""
    try:
        names = [n for n, _ in tf.train.list_variables(ckpt_path)]
    except Exception as e:  # pragma: no cover - malformed/missing path
        log.warning("Could not list checkpoint variables (%s); assuming no EMA.", e)
        return False
    return any(('_shadows' in n) or ('ema_step' in n) for n in names)


def restore_eval_weights(model: tf.keras.Model, ckpt_path: str) -> str:
    """Restore weights for evaluation/export, preferring EMA weights.

    Args:
        model: a built YoloV8 model (build_and_init already called).
        ckpt_path: checkpoint path prefix.

    Returns:
        'ema' if EMA shadow weights were restored and swapped into the model,
        'raw' if the model/ slot was restored directly (best_ export or a plain
        model checkpoint that already holds the intended weights).
    """
    if _checkpoint_has_ema(ckpt_path):
        # Periodic checkpoint: reconstruct the EMA wrapper object graph so the
        # shadows restore, then swap them into the live model.
        from optimizers.sgd_warmup import SGDTorch
        from optimizers.ema import ExponentialMovingAverage

        sgd = SGDTorch(lr_fn=lambda step: tf.constant(0.0), warmup_steps=0)
        ema = ExponentialMovingAverage(optimizer=sgd, model=model)
        tf.train.Checkpoint(model=model, optimizer=ema).restore(ckpt_path).expect_partial()
        ema.swap_weights(model)   # model/ (raw) <-> shadows (EMA) → model now holds EMA
        log.info("Restored EMA weights from periodic checkpoint: %s", ckpt_path)
        return 'ema'

    tf.train.Checkpoint(model=model).restore(ckpt_path).expect_partial()
    log.info("Restored model weights directly (no EMA shadows present): %s", ckpt_path)
    return 'raw'
