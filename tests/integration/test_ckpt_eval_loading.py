"""Tests for tools/shared/ckpt_loading.restore_eval_weights.

Pins the contract that eval/export load EMA weights from a periodic checkpoint
(model/ = raw, optimizer/ = EMA shadows) and raw weights from a best_-style
checkpoint (model/ already = EMA). A regression to a plain
`Checkpoint(model=model).restore(...)` would silently load raw weights from a
periodic checkpoint and this test would fail.
"""

import os
import tempfile

import numpy as np
import tensorflow as tf

from configs.model_config import ModelConfig
from models.yolo_v8 import build_yolov8
from optimizers.sgd_warmup import SGDTorch
from optimizers.ema import ExponentialMovingAverage
from tools.shared.ckpt_loading import restore_eval_weights, _checkpoint_has_ema


_H = _W = 64
_NC = 4


def _build_model():
    cfg = ModelConfig(input_size=[_H, _W, 3], num_classes=_NC,
                      with_polygons=False, with_distance=False, deploy=False)
    m = build_yolov8(cfg)
    m.build_and_init(cfg.input_size)
    return m


def test_periodic_checkpoint_loads_ema_weights():
    """model/ = raw, optimizer/ = EMA shadows → restore must yield the EMA values."""
    model = _build_model()
    sgd = SGDTorch(lr_fn=lambda s: tf.constant(0.0), warmup_steps=0)
    ema = ExponentialMovingAverage(optimizer=sgd, model=model)

    # Make EMA shadows distinct from the live weights: shadow = live + 1.0.
    for s in ema._shadows:
        s.assign(s + 1.0)

    # Track one variable's live value to compare against after restore.
    probe_idx = 0
    live_value = model.variables[probe_idx].numpy().copy()

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, 'ckpt')
        tf.train.Checkpoint(model=model, optimizer=ema).write(prefix)

        assert _checkpoint_has_ema(prefix), "periodic checkpoint should report EMA shadows"

        fresh = _build_model()
        kind = restore_eval_weights(fresh, prefix)

        assert kind == 'ema'
        # The model now holds the EMA (shadow) values = live + 1, NOT the raw live.
        np.testing.assert_allclose(
            fresh.variables[probe_idx].numpy(), live_value + 1.0, rtol=1e-5, atol=1e-6,
        )


def test_best_style_checkpoint_loads_raw():
    """model/ only (no optimizer/) → restore directly, no swap."""
    model = _build_model()
    probe_idx = 0
    # Set a known value so we can verify exact restore.
    target = model.variables[probe_idx].numpy() + 3.0
    model.variables[probe_idx].assign(target)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, 'ckpt')
        tf.train.Checkpoint(model=model).write(prefix)

        assert not _checkpoint_has_ema(prefix), "best_ checkpoint has no EMA shadows"

        fresh = _build_model()
        kind = restore_eval_weights(fresh, prefix)

        assert kind == 'raw'
        np.testing.assert_allclose(
            fresh.variables[probe_idx].numpy(), target, rtol=1e-5, atol=1e-6,
        )
