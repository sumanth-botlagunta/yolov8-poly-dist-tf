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
    """True if the checkpoint stores EMA shadow variables (a periodic ckpt-N).

    Raises if the checkpoint has an ``optimizer/`` subtree but no recognizable
    EMA markers — that means a periodic checkpoint whose EMA wrapper attribute
    names drifted, and silently falling back to the RAW weights would deploy the
    worse, non-averaged weights without any signal (fail-closed-but-wrong).
    """
    try:
        names = [n for n, _ in tf.train.list_variables(ckpt_path)]
    except Exception as e:  # pragma: no cover - malformed/missing path
        log.warning("Could not list checkpoint variables (%s); assuming no EMA.", e)
        return False
    has_ema = any(('_shadows' in n) or ('ema_step' in n) for n in names)
    has_optimizer = any('optimizer/' in n for n in names)
    if has_optimizer and not has_ema:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has an 'optimizer/' subtree but no EMA "
            "markers ('_shadows'/'ema_step'). Refusing to silently load RAW "
            "(non-EMA) weights — the EMA wrapper variable names may have changed. "
            "Update _checkpoint_has_ema's markers to match optimizers/ema.py."
        )
    return has_ema


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
        # expect_partial: the SGD slot variables are intentionally not rebuilt
        # here (we only need the EMA shadows + model graph). assert_existing_
        # objects_matched() is NOT used — it false-positives on the unbuilt
        # optimizer slots even for a valid checkpoint. A missing/typo'd path
        # still fails loudly: restore() raises NotFoundError on a nonexistent
        # checkpoint, and _checkpoint_has_ema guards the renamed-EMA case.
        tf.train.Checkpoint(model=model, optimizer=ema).restore(ckpt_path).expect_partial()
        ema.swap_in(model)   # load EMA shadows into the live model for eval/export
        log.info("Restored EMA weights from periodic checkpoint: %s", ckpt_path)
        return 'ema'

    tf.train.Checkpoint(model=model).restore(ckpt_path).expect_partial()
    log.info("Restored model weights directly (no EMA shadows present): %s", ckpt_path)
    return 'raw'
