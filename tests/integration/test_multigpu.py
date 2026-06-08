"""Multi-replica (MirroredStrategy) correctness tests on virtual CPU devices.

These validate the distributed-training machinery without a real GPU by splitting
the CPU into 2 logical devices and running a real 2-replica MirroredStrategy:

    - _replica_sum all-reduces the loss normalizers to a GLOBAL count.
    - SGDTorch all-reduces gradients across replicas (mirrored vars stay in sync).
    - The EMA wrapper + lazily-built optimizer slots work under strategy.run
      (no variable created inside the replica context).
    - A full distributed train step produces finite losses.

The single-replica path of every one of these is a no-op, so single-device
training stays numerically identical (covered by the rest of the suite).
"""

import pytest
import numpy as np
import tensorflow as tf


def _make_two_logical_cpus():
    cpus = tf.config.list_physical_devices('CPU')
    try:
        tf.config.set_logical_device_configuration(
            cpus[0],
            [tf.config.LogicalDeviceConfiguration(),
             tf.config.LogicalDeviceConfiguration()],
        )
    except RuntimeError:
        # Context already initialized; whatever logical devices exist are final.
        pass
    return tf.config.list_logical_devices('CPU')


_LOGICAL_CPUS = _make_two_logical_cpus()
_HAVE_TWO = len(_LOGICAL_CPUS) >= 2

pytestmark = pytest.mark.skipif(
    not _HAVE_TWO, reason="requires 2 logical CPU devices for MirroredStrategy"
)

_DEVICES = [d.name for d in _LOGICAL_CPUS[:2]]


def _strategy():
    return tf.distribute.MirroredStrategy(devices=_DEVICES)


# ---------------------------------------------------------------------------
# Normalizer + gradient all-reduce primitives
# ---------------------------------------------------------------------------

class TestReplicaSum:
    def test_replica_sum_doubles_under_two_replicas(self):
        from losses.tal_loss import _replica_sum

        strat = _strategy()

        @tf.function
        def per_replica():
            # Each replica contributes 3.0 → global sum should be 6.0.
            return _replica_sum(tf.constant(3.0))

        out = strat.run(per_replica)
        local = strat.experimental_local_results(out)
        # Every replica sees the same all-reduced global value.
        for v in local:
            assert float(v) == pytest.approx(6.0)

    def test_replica_sum_identity_outside_strategy(self):
        from losses.tal_loss import _replica_sum
        # No replica context (or single replica) → unchanged value.
        assert float(_replica_sum(tf.constant(7.0))) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Full distributed train step with the real model + loss + EMA optimizer
# ---------------------------------------------------------------------------

_B = 4
_H = _W = 128
_M = 4
_NC = 4


def _labels(b=_B, m=_M):
    rng = np.random.default_rng(0)
    n_gt = np.full([b], 2, dtype=np.int64)
    y1 = rng.uniform(0.05, 0.25, (b, m)).astype(np.float32)
    x1 = rng.uniform(0.05, 0.25, (b, m)).astype(np.float32)
    y2 = np.clip(y1 + rng.uniform(0.1, 0.3, (b, m)), 0, 1).astype(np.float32)
    x2 = np.clip(x1 + rng.uniform(0.1, 0.3, (b, m)), 0, 1).astype(np.float32)
    bboxes = np.stack([y1, x1, y2, x2], -1)
    classes = rng.integers(0, _NC, (b, m)).astype(np.int64)
    polygons = rng.uniform(0.0, 0.03, (b, m, 72)).astype(np.float32)
    polygons[:, :, 2::3] = 1.0
    log_dist = np.full((b, m), -10.0, np.float32)
    log_dist[:, :2] = rng.uniform(np.log(0.5), np.log(10.0), (b, 2)).astype(np.float32)
    return {
        'bbox': tf.constant(bboxes), 'classes': tf.constant(classes),
        'n_gt': tf.constant(n_gt), 'polygons': tf.constant(polygons),
        'log_distance': tf.constant(log_dist),
        'ignore_bg': tf.zeros([b], tf.int64),
    }


class TestDistributedTrainStep:
    def _build(self, strat):
        from configs.model_config import ModelConfig
        from models.yolo_v8 import build_yolov8
        from losses.tal_loss import TaskAlignedLossExtended
        from optimizers.sgd_warmup import SGDTorch
        from optimizers.ema import ExponentialMovingAverage

        with strat.scope():
            cfg = ModelConfig(input_size=[_H, _W, 3], num_classes=_NC,
                              with_polygons=True, with_distance=True, deploy=False)
            model = build_yolov8(cfg)
            model.deploy = False
            model.build_and_init(cfg.input_size)
            loss_fn = TaskAlignedLossExtended(
                num_classes=_NC, with_polygons=True, with_distance=True)
            sgd = SGDTorch(lr_fn=lambda s: tf.constant(1e-3), warmup_steps=0)
            opt = ExponentialMovingAverage(optimizer=sgd, model=model)
            # Pre-create slots in cross-replica context (required for strategy.run).
            opt.build(model.trainable_variables)
        return model, loss_fn, opt

    def test_two_replica_step_finite_and_synced(self):
        strat = _strategy()
        model, loss_fn, opt = self._build(strat)

        images = tf.random.uniform([_B, _H, _W, 3], seed=1)
        labels = _labels()

        def train_step(imgs, lbls):
            with tf.GradientTape() as tape:
                feats = model(imgs, training=True)
                total, *_ = loss_fn(feats, lbls)
            grads = tape.gradient(total, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            return total

        @tf.function
        def dist_step(imgs, lbls):
            per_replica = strat.run(train_step, args=(imgs, lbls))
            return strat.reduce(tf.distribute.ReduceOp.SUM, per_replica, axis=None)

        # Broadcast the same global batch to both replicas (enough to exercise the
        # all-reduce wiring); two steps to confirm slots/velocity persist.
        for _ in range(2):
            total = dist_step(images, labels)
            assert bool(tf.math.is_finite(total)), f"non-finite total loss: {float(total)}"

        # Mirrored trainable variables must be identical across replicas after the
        # gradient all-reduce — otherwise replicas have diverged.
        for var in model.trainable_variables[:8]:
            locals_ = strat.experimental_local_results(var)
            if len(locals_) >= 2:
                np.testing.assert_allclose(
                    locals_[0].numpy(), locals_[1].numpy(), rtol=1e-5, atol=1e-6,
                    err_msg=f"replica divergence in {var.name}",
                )
