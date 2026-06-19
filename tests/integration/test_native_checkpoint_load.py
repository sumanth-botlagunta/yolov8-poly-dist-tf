"""Integration tests for warm-starting from a checkpoint produced by THIS codebase.

The trainer writes checkpoints with the EMA optimizer, so the complete weights live in
the EMA shadows (``optimizer/_shadows``) rather than a plain ``model/`` subtree — the
``model/`` object graph omits the list-tracked C2f block variables (a Keras quirk). The
``native`` migration strategy must therefore load via the EMA path (reusing
``restore_eval_weights``) so that EVERY backbone/decoder variable — including the C2f
blocks — is warm-started, which the structural object-graph walk cannot guarantee.

These tests build a small (4-class, no polygon/distance) model for speed; the C2f blocks
that expose the completeness gap are identical across tiers.
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from configs.model_config import ModelConfig
from models.yolo_v8 import build_yolov8
from optimizers.sgd_warmup import SGDTorch
from optimizers.ema import ExponentialMovingAverage
from tools.checkpoint_migration import migrate_checkpoint, _detect_strategy

_H = _W = 128
_NC = 4

_RAW_FILL = 0.3      # raw (model) weights
_EMA_FILL = 0.7      # EMA shadow weights — what a warm-start should load


def _build_model():
    cfg = ModelConfig(
        input_size=[_H, _W, 3],
        num_classes=_NC,
        with_polygons=False,
        with_distance=False,
        deploy=False,
    )
    model = build_yolov8(cfg)
    model.build_and_init(cfg.input_size)
    return model


def _write_trainer_like_checkpoint(tmpdir):
    """Save a periodic-style checkpoint: raw weights = _RAW_FILL, EMA shadows = _EMA_FILL.

    Distinct fills let a test prove the EMA (complete) weights were loaded, not the raw
    ``model/`` subtree.
    """
    src = _build_model()
    for v in src.variables:
        v.assign(tf.fill(v.shape, _RAW_FILL))

    sgd = SGDTorch(lr_fn=lambda step: tf.constant(0.0), warmup_steps=0)
    ema = ExponentialMovingAverage(optimizer=sgd, model=src)
    for shadow in ema._shadows:
        shadow.assign(tf.fill(shadow.shape, _EMA_FILL))

    path = os.path.join(tmpdir, "ckpt")
    tf.train.Checkpoint(
        model=src, optimizer=ema, global_step=tf.Variable(1)
    ).write(path)
    return path


def _frac_close(module, value):
    return float(np.mean([np.allclose(v.numpy(), value) for v in module.variables]))


def test_auto_detects_native_for_trainer_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        path = _write_trainer_like_checkpoint(d)
        assert _detect_strategy(tf.train.load_checkpoint(path)) == "native"


def test_native_warm_start_loads_all_backbone_decoder_vars():
    """Every backbone+decoder var — including the list-tracked C2f blocks — is loaded.

    This is the guarantee the structural object-graph walk cannot give: it would only
    align the ~55% of backbone vars that appear in the plain object graph.
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write_trainer_like_checkpoint(d)
        dst = _build_model()
        stats = migrate_checkpoint(
            old_ckpt_path=path,
            new_model=dst,
            output_ckpt_path=os.path.join(d, "migrated", "ckpt"),
            modules=["backbone", "decoder"],
            strategy="auto",
        )
        assert stats["mode"] == "ema"
        # EMA (complete) weights loaded into EVERY backbone + decoder variable.
        assert _frac_close(dst.backbone, _EMA_FILL) == 1.0
        assert _frac_close(dst.decoder, _EMA_FILL) == 1.0
        assert stats["loaded"] == len(dst.backbone.variables) + len(dst.decoder.variables)
        assert stats["loaded_by_module"]["backbone"] == len(dst.backbone.variables)


def test_native_excludes_head_when_not_requested():
    with tempfile.TemporaryDirectory() as d:
        path = _write_trainer_like_checkpoint(d)
        dst = _build_model()
        head_before = [v.numpy().copy() for v in dst.head.variables]
        migrate_checkpoint(
            old_ckpt_path=path,
            new_model=dst,
            output_ckpt_path=os.path.join(d, "migrated", "ckpt"),
            modules=["backbone", "decoder"],
            strategy="auto",
        )
        # Head was snapshotted and restored to its fresh init (not warm-started).
        assert _frac_close(dst.head, _EMA_FILL) == 0.0
        for before, after in zip(head_before, dst.head.variables):
            assert np.allclose(before, after.numpy())


def test_native_loads_head_when_requested():
    with tempfile.TemporaryDirectory() as d:
        path = _write_trainer_like_checkpoint(d)
        dst = _build_model()
        migrate_checkpoint(
            old_ckpt_path=path,
            new_model=dst,
            output_ckpt_path=os.path.join(d, "migrated", "ckpt"),
            modules=["backbone", "decoder", "head"],
            strategy="auto",
        )
        assert _frac_close(dst.head, _EMA_FILL) == 1.0
