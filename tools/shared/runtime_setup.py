"""Shared runtime setup for offline tools (eval / export).

The trainer (`scripts/run_train.py:_apply_runtime_config`) activates the global
Keras mixed-precision policy from the experiment config before building the model.
The offline tools build the *same* model from the *same* config but historically
skipped this step, so a bfloat16-trained checkpoint was loaded into a float32
graph. Weights still restore (dtypes are cast on assign), but the backbone/decoder
then compute in float32 — a different numerical path than training/serving. This
helper centralizes the policy application so every tool matches the trainer.

Besides the precision policy, this also applies the trainer's **thread-pool caps**
(`apply_eval_thread_config`). On cgroup-capped hosts TF sizes its inter/intra-op
pools to the visible core count (e.g. 128) while the process may only use a few
cores — hundreds of threads thrash, which starves the GPU (util ~3%) and makes
offline eval far slower than the in-training validation that *does* apply the caps.
XLA and distribution strategy remain trainer-loop concerns not set here.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def apply_eval_precision_policy(config) -> str:
    """Set the global Keras mixed-precision policy from ``config.runtime``.

    Mirrors the bfloat16/float32 branch of
    ``scripts/run_train.py:_apply_runtime_config`` so offline model construction
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


def apply_eval_thread_config(config) -> None:
    """Apply the trainer's inter/intra-op thread-pool caps for offline tools.

    Mirrors the threading block of ``scripts/run_train.py:_apply_runtime_config``.
    Reads ``config.runtime.inter_op_threads`` / ``intra_op_threads`` and caps the
    TF thread pools. MUST run before any TF op executes (before the model is built
    or any dataset op runs), or the caps are ignored — TF has already sized its
    pools. Without this, eval on a cgroup-capped host oversubscribes threads
    (visible cores >> usable cores) and thrashes, leaving the GPU ~idle. No-op when
    the config leaves the values at 0 (unset).
    """
    import tensorflow as tf

    runtime = getattr(config, "runtime", None)
    inter = getattr(runtime, "inter_op_threads", 0) if runtime else 0
    intra = getattr(runtime, "intra_op_threads", 0) if runtime else 0
    if inter and inter > 0:
        tf.config.threading.set_inter_op_parallelism_threads(inter)
        log.info("inter_op_parallelism_threads = %d (matches training)", inter)
    if intra and intra > 0:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        log.info("intra_op_parallelism_threads = %d (matches training)", intra)
