"""Pinning test: the best checkpoint is resumable (carries optimizer + counters).

`_save_best_checkpoint` used to write `tf.train.Checkpoint(model=...)` only — no
optimizer, no global_step. Restoring that to *continue training* would lose the
SGD `iterations` variable (the cosine LR schedule reads it), snapping the LR back
to its initial value and resetting momentum slots. The best checkpoint now also
saves optimizer / global_step / completed_epochs / best_metric, so it is a valid
training-resume source as well as an inference checkpoint.

Reuses the stub task/trainer harness from test_trainer_resume.
"""

import tensorflow as tf

from tests.unit.test_trainer_resume import _make_config, _make_trainer


def test_best_checkpoint_contains_optimizer_and_counters(tmp_path):
    cfg = _make_config()
    trainer = _make_trainer(tmp_path, cfg)

    # Drive the real trainer loop so global_step / optimizer state are populated.
    trainer.train(cfg.trainer.train_epochs)
    trained_step = int(trainer._global_step)
    assert trained_step > 0

    # Force a best-checkpoint write.
    trainer._best_metric.assign(0.5)
    trainer._save_best_checkpoint(epoch=1, step=trained_step)

    import os
    best_dir = None
    for name in os.listdir(tmp_path):
        if name.startswith("best_"):
            best_dir = os.path.join(tmp_path, name)
    assert best_dir is not None, "no best_<metric> directory written"

    ckpt_prefix = os.path.join(best_dir, "ckpt")

    # The variable names in the checkpoint must include optimizer + counters,
    # not just model weights.
    reader = tf.train.load_checkpoint(ckpt_prefix)
    keys = set(reader.get_variable_to_shape_map().keys())
    joined = " ".join(keys)
    assert "optimizer" in joined, f"best ckpt has no optimizer state: {sorted(keys)}"
    assert "global_step" in joined, "best ckpt missing global_step"
    assert "completed_epochs" in joined, "best ckpt missing completed_epochs"

    # And a fresh full Checkpoint must restore the SGD iterations from it,
    # proving the LR schedule would resume correctly.
    step_var = tf.Variable(0, dtype=tf.int64)
    restore_ck = tf.train.Checkpoint(global_step=step_var)
    restore_ck.restore(ckpt_prefix).expect_partial()
    assert int(step_var) == trained_step
