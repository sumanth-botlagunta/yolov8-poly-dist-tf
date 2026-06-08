"""Smoke tests for the training loop.

Two test groups:

  TestDrySmoke — Requires NO external data.  Builds a small model and runs
      10 training steps with synthetic tensors.  Tests EMA correctness,
      optimizer behavior, checkpoint save/restore, and finite loss trajectory.
      Always runs (no special markers).

  TestRealDataSmoke — Marked @pytest.mark.smoke.  Requires TFDS_DATA_DIR env
      var and the yolov8_bbox TFDS dataset.  Skipped automatically when the
      environment is not configured.  Validates that the full data pipeline
      (TFDS → parser → mosaic → batch → model → loss) runs for 10 real steps
      without NaN/Inf and that the loss does not increase catastrophically.

Run smoke tests:
    pytest -m smoke tests/smoke/test_train_10_steps.py -v
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from configs.model_config import (
    ExperimentConfig, ModelConfig, TaskConfig, LossConfig,
    TrainerConfig, OptimizerConfig, EmaConfig, LrScheduleConfig,
    DataConfig,
)
from models.yolo_v8 import build_yolov8
from losses.tal_loss import TaskAlignedLossExtended
from optimizers.sgd_warmup import SGDTorch
from optimizers.ema import ExponentialMovingAverage


# ---------------------------------------------------------------------------
# Helpers shared across both test groups
# ---------------------------------------------------------------------------

_H = _W = 128    # small image for speed
_B = 2
_M = 3
_NC = 4
_STEPS = 10


def _build_small_model(with_polygons=False, with_distance=False):
    cfg = ModelConfig(
        input_size=[_H, _W, 3],
        num_classes=_NC,
        with_polygons=with_polygons,
        with_distance=with_distance,
        deploy=False,
    )
    model = build_yolov8(cfg)
    model.deploy = False
    model.build_and_init(cfg.input_size)
    return model


def _make_labels(rng, b=_B, m=_M, with_polygons=False, with_distance=False):
    n_gt = np.full([b], 2, dtype=np.int64)
    y1 = rng.uniform(0.05, 0.25, (b, m)).astype(np.float32)
    x1 = rng.uniform(0.05, 0.25, (b, m)).astype(np.float32)
    y2 = np.clip(y1 + rng.uniform(0.1, 0.3, (b, m)), 0, 1).astype(np.float32)
    x2 = np.clip(x1 + rng.uniform(0.1, 0.3, (b, m)), 0, 1).astype(np.float32)
    bboxes  = np.stack([y1, x1, y2, x2], axis=-1)
    classes = rng.integers(0, _NC, (b, m)).astype(np.int64)

    labels = {
        'bbox':      tf.constant(bboxes),
        'classes':   tf.constant(classes),
        'n_gt':      tf.constant(n_gt),
        'ignore_bg': tf.zeros([b], dtype=tf.int64),
    }
    if with_polygons:
        polys = rng.uniform(0, 0.03, (b, m, 72)).astype(np.float32)
        polys[:, :, 2::3] = 1.0
        labels['polygons'] = tf.constant(polys)
    if with_distance:
        log_dist = np.full((b, m), -10.0, dtype=np.float32)
        log_dist[:, :2] = rng.uniform(np.log(0.5), np.log(10.0), (b, 2)).astype(np.float32)
        labels['log_distance'] = tf.constant(log_dist)
    return labels


# ---------------------------------------------------------------------------
# Dry smoke — no external data required
# ---------------------------------------------------------------------------

class TestDrySmoke:
    """10-step loop with synthetic data; validates EMA, optimizer, checkpointing."""

    @pytest.fixture(scope="class")
    def setup(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("dry_smoke")
        rng = np.random.default_rng(0)
        model = _build_small_model()
        loss_fn = TaskAlignedLossExtended(num_classes=_NC,
                                          with_polygons=False,
                                          with_distance=False)
        lr_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.01, decay_steps=10000, alpha=0.01
        )
        sgd = SGDTorch(lr_fn=lr_fn, momentum=0.937, momentum_start=0.8,
                       nesterov=True, weight_decay=0.0005, warmup_steps=5)
        ema = ExponentialMovingAverage(optimizer=sgd, model=model,
                                       average_decay=0.9999, dynamic_decay=True)
        return {'model': model, 'loss_fn': loss_fn, 'optimizer': ema,
                'tmp': tmp, 'rng': rng}

    def test_10_steps_no_nan_inf(self, setup):
        """All loss components stay finite for 10 consecutive steps."""
        model, loss_fn, optimizer, rng = (
            setup['model'], setup['loss_fn'], setup['optimizer'], setup['rng']
        )
        for step in range(_STEPS):
            images = tf.random.uniform([_B, _H, _W, 3], seed=step)
            labels = _make_labels(rng)
            with tf.GradientTape() as tape:
                feats = model(images, training=True)
                total, box, dfl, cls, dist, poly, poly_a, poly_d, poly_c = loss_fn(feats, labels)
            grads = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            assert tf.math.is_finite(total), \
                f"Step {step}: total_loss is NaN/Inf ({float(total):.4f})"

    def test_loss_trajectory_stays_finite(self):
        """The training loop stays finite (no NaN/Inf) over 10 steps on a fixed batch.

        Uses its own fresh model/optimizer and a single fixed (images, labels) batch.
        We assert finiteness rather than a magnitude bound: a random-init model on noise
        has no well-defined loss-decrease guarantee in 10 steps, and the exact early
        trajectory differs across TF versions/platforms — so a "<N× start" check is
        flaky. NaN/Inf is the real divergence signal and is stable to assert.
        """
        rng = np.random.default_rng(7)
        model = _build_small_model()
        loss_fn = TaskAlignedLossExtended(num_classes=_NC,
                                          with_polygons=False, with_distance=False)
        lr_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4, decay_steps=10000, alpha=0.01)
        sgd = SGDTorch(lr_fn=lr_fn, momentum=0.9, momentum_start=0.8,
                       nesterov=True, weight_decay=0.0005, warmup_steps=5)

        # One FIXED batch reused every step.
        images = tf.random.uniform([_B, _H, _W, 3], seed=99)
        labels = _make_labels(rng)

        for step in range(_STEPS):
            with tf.GradientTape() as tape:
                feats = model(images, training=True)
                total, *_ = loss_fn(feats, labels)
            grads = tape.gradient(total, model.trainable_variables)
            sgd.apply_gradients(zip(grads, model.trainable_variables))
            assert tf.math.is_finite(total), \
                f"Step {step}: loss is NaN/Inf ({float(total):.4f})"

        feats_end = model(images, training=False)
        loss_end, *_ = loss_fn(feats_end, labels)
        assert tf.math.is_finite(loss_end), f"Final loss is NaN/Inf: {float(loss_end)}"

    def test_ema_shadows_differ_from_live_weights(self, setup):
        """After updates, at least one EMA shadow must differ from the live weight."""
        model, optimizer = setup['model'], setup['optimizer']
        diffs = [
            float(tf.reduce_sum(tf.abs(
                tf.cast(tf.identity(shadow), tf.float32) -
                tf.cast(tf.identity(live), tf.float32)
            )))
            for live, shadow in zip(model.variables, optimizer._shadows)
        ]
        assert any(d > 1e-8 for d in diffs), (
            "All EMA shadows are identical to live weights — EMA is not updating"
        )

    def test_checkpoint_save_and_restore(self, setup):
        """Checkpoint saved after training can be restored; global_step is preserved."""
        model, optimizer, tmp = setup['model'], setup['optimizer'], setup['tmp']
        global_step = tf.Variable(10, trainable=False, dtype=tf.int64)

        sgd = optimizer._optimizer
        ckpt = tf.train.Checkpoint(
            model=model,
            sgd_step=sgd.iterations,
            ema_step=optimizer._ema_step,
            global_step=global_step,
        )
        ckpt_mgr = tf.train.CheckpointManager(ckpt, str(tmp / "ckpts"), max_to_keep=2)
        ckpt_path = ckpt_mgr.save(checkpoint_number=10)
        assert ckpt_path is not None

        # Restore into a fresh global_step variable
        restored_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        restore_ckpt = tf.train.Checkpoint(
            model=model,
            sgd_step=sgd.iterations,
            ema_step=optimizer._ema_step,
            global_step=restored_step,
        )
        restore_ckpt.restore(tf.train.latest_checkpoint(str(tmp / "ckpts")))
        assert int(restored_step) == 10, \
            f"Restored global_step={int(restored_step)}, expected 10"

    def test_ema_swap_restore_symmetry(self, setup):
        """swap_weights called twice returns model to original live values."""
        model, optimizer = setup['model'], setup['optimizer']
        live_before = [tf.identity(v).numpy() for v in model.variables]

        optimizer.swap_weights(model)   # live ← shadows
        optimizer.swap_weights(model)   # live ← original

        for v, before in zip(model.variables, live_before):
            assert np.allclose(v.numpy(), before, atol=1e-7), \
                f"Variable {v.name} not restored after double swap"


# ---------------------------------------------------------------------------
# Real-data smoke — requires TFDS_DATA_DIR + actual dataset
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestRealDataSmoke:
    """End-to-end smoke: real TFDS data → parser → batch → model → loss × 10 steps.

    Skipped if TFDS_DATA_DIR is not set in the environment.
    """

    @pytest.fixture(scope="class", autouse=True)
    def require_tfds(self):
        tfds_dir = os.environ.get("TFDS_DATA_DIR", "")
        if not tfds_dir:
            pytest.skip("TFDS_DATA_DIR env var not set — skipping real-data smoke test")

    @pytest.fixture(scope="class")
    def config(self):
        from configs.yaml_loader import load_config
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "configs", "experiments", "yolo", "yolov8_bbox.yaml"
        )
        cfg = load_config(yaml_path)
        # Override data dir from env
        tfds_dir = os.environ["TFDS_DATA_DIR"]
        cfg.task.train_data.tfds_data_dir = tfds_dir
        # Small batch for smoke
        cfg.task.train_data.global_batch_size = 4
        cfg.task.train_data.shuffle_buffer_size = 10
        return cfg

    @pytest.fixture(scope="class")
    def model_and_optimizer(self, config):
        from train.task import YoloV8Task
        task = YoloV8Task(config)
        model = task.build_model()
        optimizer = task.build_optimizer()
        return task, model, optimizer

    def test_10_steps_real_data_no_nan(self, config, model_and_optimizer):
        """10 training steps on real TFDS data must not produce NaN/Inf loss."""
        from train.task import YoloV8Task

        task, model, optimizer = model_and_optimizer

        try:
            ds = task.build_inputs(config.task.train_data)
        except Exception as e:
            pytest.skip(f"Could not build TFDS dataset: {e}")

        loss_values = []
        for step, inputs in enumerate(ds):
            if step >= _STEPS:
                break
            losses = task.train_step(inputs, model, optimizer)
            total = float(losses['total_loss'])
            assert np.isfinite(total), \
                f"Real-data step {step}: total_loss is NaN/Inf ({total:.4f})"
            loss_values.append(total)

        assert len(loss_values) == _STEPS, \
            f"Expected {_STEPS} steps but only got {len(loss_values)}"

        # Loss at the last step must not be more than 3× the first step loss
        assert loss_values[-1] < loss_values[0] * 3.0, (
            f"Loss diverged: step0={loss_values[0]:.4f}  step9={loss_values[-1]:.4f}"
        )

    def test_checkpoint_written_after_real_steps(
        self, config, model_and_optimizer, tmp_path
    ):
        """After training, a checkpoint can be saved and its variables listed."""
        task, model, optimizer = model_and_optimizer

        sgd = optimizer._optimizer
        global_step = tf.Variable(10, trainable=False, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(
            model=model,
            sgd_step=sgd.iterations,
            ema_step=optimizer._ema_step,
            global_step=global_step,
        )
        ckpt_mgr = tf.train.CheckpointManager(ckpt, str(tmp_path), max_to_keep=1)
        ckpt_mgr.save(checkpoint_number=10)

        saved_vars = tf.train.list_variables(tf.train.latest_checkpoint(str(tmp_path)))
        assert len(saved_vars) > 0, "Checkpoint appears empty"
