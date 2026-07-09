"""Shared runtime setup for offline tools (eval / export).

The offline tools build the same model from the same config as the trainer
(``train/run_train.py:_apply_runtime_config``), so they must activate the same global
Keras mixed-precision policy before building it. Otherwise a bfloat16-trained checkpoint
computes in float32 — a different numerical path than training/serving (the weights
still restore, since dtypes cast on assign). This helper centralizes that step so every
tool matches the trainer.

It sets only the precision policy (the part that affects model numerics); XLA, threading,
and distribution strategy are trainer-loop concerns irrelevant to single-process offline
inference.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def apply_eval_precision_policy(config) -> str:
    """Set the global Keras mixed-precision policy from ``config.runtime``.

    Mirrors the bfloat16/float32 branch of
    ``train/run_train.py:_apply_runtime_config`` so offline model construction
    matches the trained checkpoint's compute dtype. Returns the normalized precision
    string actually applied (for logging/tests). float16 is rejected as the trainer
    rejects it (no loss scaling); since these tools are inference-only it falls back to
    the float32 policy with a warning rather than raising.
    """
    import tensorflow as tf

    runtime = getattr(config, "runtime", None)
    raw = getattr(runtime, "mixed_precision_dtype", None) if runtime else None
    precision = (raw or "float32").strip().lower()

    if precision in ("bfloat16", "bf16", "mixed_bfloat16"):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        log.info("Mixed precision: bfloat16 policy active (matches training).")
        return "bfloat16"
    if precision in ("float32", "fp32", ""):
        return "float32"
    if precision in ("float16", "fp16", "half", "mixed_float16"):
        # Inference does not need loss scaling, but the trained checkpoint was
        # produced under float32/bfloat16 (the trainer forbids float16), so
        # honor float32 here rather than introduce a mismatch.
        log.warning(
            "mixed_precision_dtype=%r is not used for eval; defaulting to "
            "float32 policy.", raw,
        )
        return "float32"
    log.warning(
        "Unknown mixed_precision_dtype=%r; defaulting to float32 policy.", raw
    )
    return "float32"
