"""Pinning test: the best checkpoint stores RAW model weights + EMA shadows.

Regression guard for the EMA-corruption-on-resume bug. ``_save_best_checkpoint``
used to ``swap_in`` the EMA (shadow) weights into the model before writing, so
``model/`` held EMA weights while ``optimizer/`` held SGD velocity slots computed
against the RAW (pre-EMA) weights. A *training* resume from that checkpoint loads
EMA weights into a model whose restored momentum was computed for raw weights — an
incoherent (weights, momentum) pair that corrupts the trajectory and the EMA
shadow update.

The fix: save the model with its RAW weights (no swap_in). The EMA shadows are
tf.Variables tracked inside the EMA wrapper and are serialized under
``optimizer/`` regardless, so eval/export recover them via
``tools/shared/ckpt_loading.restore_eval_weights`` and a training resume stays coherent.

This test drives the real trainer loop until raw and EMA weights diverge, forces
a best-checkpoint write, and asserts:
  1. ``model/`` holds RAW weights (NOT the swapped-in EMA weights).
  2. The EMA shadows are present under ``optimizer/`` and equal the live shadows.
  3. ``restore_eval_weights`` recovers the EMA weights for inference.
"""

import os

import numpy as np
import tensorflow as tf

from tests.unit.test_trainer_resume import _make_config, _make_trainer


def _flat(vs):
    return np.concatenate([np.asarray(v).ravel() for v in vs])


def test_best_checkpoint_stores_raw_weights_and_ema_shadows(tmp_path):
    cfg = _make_config()
    trainer = _make_trainer(tmp_path, cfg)

    # Drive the real trainer loop so the optimizer + EMA shadows are populated.
    trainer.train(cfg.trainer.train_epochs)
    assert int(trainer._global_step) > 0

    model = trainer._model
    ema = trainer._optimizer

    # The stub task's train_step is a no-op (no apply_gradients), so raw and EMA
    # weights are still identical. Apply a few REAL gradient steps through the
    # actual EMA optimizer to diverge them — this is the exact mechanism the real
    # trainer relies on, and is what makes the swap_in bug observable.
    x = tf.constant([[1.0]])
    for _ in range(8):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(model(x) ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        ema.apply_gradients(zip(grads, model.trainable_variables))

    raw_weights = _flat(model.variables)
    shadow_weights = _flat(ema._shadows)
    # Precondition: the bug only manifests when raw != shadow.
    assert not np.allclose(raw_weights, shadow_weights, atol=1e-5), (
        "precondition: raw and EMA shadow weights must differ to exercise the bug"
    )

    # Force a best-checkpoint write.
    trainer._best_metric.assign(0.5)
    trainer._save_best_checkpoint(epoch=1, step=int(trainer._global_step))

    # The model's live weights must be UNCHANGED by the save (no leaked swap_in).
    assert np.allclose(_flat(model.variables), raw_weights, atol=1e-7), (
        "_save_best_checkpoint mutated the live model weights (swap_in leaked)"
    )

    best_dir = next(
        os.path.join(tmp_path, n) for n in os.listdir(tmp_path) if n.startswith("best_")
    )
    ckpt_prefix = os.path.join(best_dir, "ckpt")

    # 1. model/ slot holds RAW weights, not the EMA weights.
    restored_model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    restored_model.build((None, 1))
    tf.train.Checkpoint(model=restored_model).restore(ckpt_prefix).expect_partial()
    restored_flat = _flat(restored_model.variables)
    assert np.allclose(restored_flat, raw_weights, atol=1e-6), (
        "best ckpt model/ slot does not hold RAW weights"
    )
    assert not np.allclose(restored_flat, shadow_weights, atol=1e-5), (
        "best ckpt model/ slot holds EMA weights — swap_in regressed"
    )

    # 2. EMA shadows are serialized under optimizer/.
    names = {n for n, _ in tf.train.list_variables(ckpt_prefix)}
    assert any("_shadows" in n for n in names), (
        f"best ckpt has no EMA shadow variables: {sorted(names)}"
    )

    # 3. restore_eval_weights recovers the EMA weights for inference.
    from tools.shared.ckpt_loading import restore_eval_weights

    eval_model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    eval_model.build((None, 1))
    kind = restore_eval_weights(eval_model, ckpt_prefix)
    assert kind == "ema", f"expected EMA weights to be restored for eval, got {kind!r}"
    assert np.allclose(_flat(eval_model.variables), shadow_weights, atol=1e-6), (
        "restore_eval_weights did not load the EMA shadow weights"
    )
