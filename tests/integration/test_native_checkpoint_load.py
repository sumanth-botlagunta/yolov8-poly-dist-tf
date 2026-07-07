"""Integration tests for warm-starting from a checkpoint produced by this codebase.

The trainer writes checkpoints with the EMA optimizer, so the complete weights live in
the EMA shadows (``optimizer/_shadows``) rather than a plain ``model/`` subtree — the
``model/`` object graph omits the list-tracked C2f block variables (a Keras quirk).
``task.initialize``'s transfer-init therefore loads the FULL model via
``restore_eval_weights`` (the same loader eval/export use) and then restores the
non-selected modules from a pre-load snapshot, so EVERY backbone/decoder variable —
including the C2f blocks — is warm-started while the head keeps its fresh init.

These tests build a small (4-class, no polygon/distance) model for speed; the C2f blocks
that expose the completeness gap are identical across tiers.
"""

import os
import tempfile

import numpy as np
import tensorflow as tf

from configs.model_config import ModelConfig
from configs.yaml_loader import load_config
from models.yolo_v8 import build_yolov8
from optimizers.sgd_warmup import SGDTorch
from optimizers.ema import ExponentialMovingAverage
from train.task import YoloV8Task

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


def _make_task(ckpt_path, modules):
    cfg = load_config("configs/experiments/yolo/yolov8_bbox.yaml")
    cfg.task.finetune_from = None
    cfg.task.init_checkpoint = ckpt_path
    cfg.task.init_checkpoint_modules = list(modules)
    return YoloV8Task(cfg)


def _frac_close(module, value):
    return float(np.mean([np.allclose(v.numpy(), value) for v in module.variables]))


def test_transfer_init_loads_all_backbone_decoder_vars():
    """Every backbone+decoder var — including the list-tracked C2f blocks — is loaded.

    A plain object-graph restore would only align the subset of backbone vars that
    appear in the bare ``model/`` graph; the EMA-shadow path covers all of them.
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write_trainer_like_checkpoint(d)
        dst = _build_model()
        _make_task(path, ["backbone", "decoder"]).initialize(dst)
        assert _frac_close(dst.backbone, _EMA_FILL) == 1.0
        assert _frac_close(dst.decoder, _EMA_FILL) == 1.0


def test_transfer_init_excludes_head_when_not_requested():
    with tempfile.TemporaryDirectory() as d:
        path = _write_trainer_like_checkpoint(d)
        dst = _build_model()
        head_before = [v.numpy().copy() for v in dst.head.variables]
        _make_task(path, ["backbone", "decoder"]).initialize(dst)
        # Head was snapshotted and restored to its fresh init (not warm-started).
        assert _frac_close(dst.head, _EMA_FILL) == 0.0
        for before, after in zip(head_before, dst.head.variables):
            assert np.allclose(before, after.numpy())


def test_transfer_init_loads_head_when_requested():
    with tempfile.TemporaryDirectory() as d:
        path = _write_trainer_like_checkpoint(d)
        dst = _build_model()
        _make_task(path, ["backbone", "decoder", "head"]).initialize(dst)
        assert _frac_close(dst.head, _EMA_FILL) == 1.0


def test_transfer_init_rejects_unknown_module():
    with tempfile.TemporaryDirectory() as d:
        path = _write_trainer_like_checkpoint(d)
        dst = _build_model()
        try:
            _make_task(path, ["backbone", "neck"]).initialize(dst)
        except ValueError as e:
            assert "neck" in str(e)
        else:
            raise AssertionError("unknown module name must raise")
