"""Tests for module freezing (task.freeze_modules / YoloV8Task.apply_freezing)."""

import types

import numpy as np
import pytest
import tensorflow as tf

import train.task as T
from configs.model_config import ModelConfig
from models.yolo_v8 import build_yolov8


def _model():
    m = build_yolov8(ModelConfig(input_size=[64, 64, 3], num_classes=5,
                                 with_polygons=False, with_distance=False))
    m.build_and_init([64, 64, 3])
    return m


def _apply(model, freeze=None, backbone_layers=0):
    cfg = types.SimpleNamespace(task=types.SimpleNamespace(
        freeze_modules=freeze or [], freeze_backbone_layers=backbone_layers))
    task = T.YoloV8Task.__new__(T.YoloV8Task)
    task._config = cfg
    task.apply_freezing(model)


def test_freeze_first_n_backbone_layers():
    m = _model()
    first3 = m.backbone.layers[:3]
    n_all = len(m.trainable_variables)
    n_first3 = sum(len(l.trainable_variables) for l in first3)
    _apply(m, backbone_layers=3)
    assert len(m.trainable_variables) == n_all - n_first3
    assert not m.backbone.layers[0].trainable          # stem frozen
    assert m.backbone.layers[3].trainable               # down1 still trains
    assert m.head.trainable                             # head trains


def test_freeze_backbone_layers_zero_is_noop():
    m = _model()
    n = len(m.trainable_variables)
    _apply(m, backbone_layers=0)
    assert len(m.trainable_variables) == n


def test_freeze_too_many_backbone_layers_raises():
    m = _model()
    with pytest.raises(ValueError, match="exceeds the backbone"):
        _apply(m, backbone_layers=999)


def test_frozen_backbone_layer_bn_holds_stats():
    m = _model()
    _apply(m, backbone_layers=3)
    bns = [l for l in m.backbone.layers[2].submodules
           if isinstance(l, tf.keras.layers.BatchNormalization)]
    if not bns:
        pytest.skip("no BN")
    before = bns[0].moving_mean.numpy().copy()
    _ = m(tf.random.uniform([2, 64, 64, 3]), training=True)
    assert np.allclose(before, bns[0].moving_mean.numpy())


def test_freeze_backbone_excludes_its_vars_from_trainable():
    m = _model()
    n_all, n_bb = len(m.trainable_variables), len(m.backbone.trainable_variables)
    _apply(m, ['backbone'])
    assert len(m.trainable_variables) == n_all - n_bb
    assert not m.backbone.trainable and m.head.trainable and m.decoder.trainable


def test_freeze_backbone_decoder():
    m = _model()
    _apply(m, ['backbone', 'decoder'])
    assert not m.backbone.trainable and not m.decoder.trainable and m.head.trainable
    # only head variables remain trainable
    head_vars = {v.ref() for v in m.head.trainable_variables}
    assert all(v.ref() in head_vars for v in m.trainable_variables)


def test_frozen_batchnorm_holds_running_stats():
    m = _model()
    _apply(m, ['backbone'])
    bns = [l for l in m.backbone.submodules
           if isinstance(l, tf.keras.layers.BatchNormalization)]
    if not bns:
        pytest.skip("no BatchNorm in backbone")
    before = bns[0].moving_mean.numpy().copy()
    _ = m(tf.random.uniform([2, 64, 64, 3]), training=True)   # training mode!
    assert np.allclose(before, bns[0].moving_mean.numpy())     # stats NOT updated


def test_empty_freeze_is_noop():
    m = _model()
    n = len(m.trainable_variables)
    _apply(m, [])
    assert len(m.trainable_variables) == n


def test_unknown_module_raises():
    m = _model()
    with pytest.raises(ValueError, match="unknown module"):
        _apply(m, ['neck'])


def test_freezing_everything_raises():
    m = _model()
    with pytest.raises(ValueError, match="nothing left to train"):
        _apply(m, ['backbone', 'decoder', 'head'])
