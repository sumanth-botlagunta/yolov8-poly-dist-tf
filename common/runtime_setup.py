"""Shared runtime setup for offline tools (eval / export).

The trainer (`train/run_train.py:_apply_runtime_config`) activates the global
Keras mixed-precision policy from the experiment config before building the model.
The offline tools build the *same* model from the *same* config but historically
skipped this step, so a bfloat16-trained checkpoint was loaded into a float32
graph. Weights still restore (dtypes are cast on assign), but the backbone/decoder
then compute in float32 — a different numerical path than training/serving. This
helper centralizes the policy application so every tool matches the trainer.

This intentionally only sets the *precision policy* (the part that affects model
numerics). It does not touch XLA, threading, or distribution strategy, which are
trainer-loop concerns irrelevant to single-process offline inference.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def apply_eval_precision_policy(config) -> str:
    """Set the global Keras mixed-precision policy from ``config.runtime``.

    Mirrors the bfloat16/float32 branch of
    ``train/run_train.py:_apply_runtime_config`` so offline model construction
    matches the trained checkpoint's compute dtype. Returns the normalized
    precision string actually applied (for logging/tests). float16 is rejected for
    the same reason the trainer rejects it (no loss scaling), but since these tools
    are inference-only we simply fall back to the default policy with a warning
    rather than raising.
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
