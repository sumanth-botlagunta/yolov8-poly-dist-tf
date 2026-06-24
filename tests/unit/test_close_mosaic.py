"""Unit test for the close_mosaic boundary logic (train/trainer.py:_maybe_close_mosaic).

Drives the method on a stubbed trainer (no real training) to verify it rebuilds the
train stream with mosaic/mixup disabled exactly once, at the right epoch, and only when
``close_mosaic_epochs > 0``.
"""

import dataclasses
import types

import tensorflow as tf

from configs.model_config import MosaicConfig, ParserConfig, DataConfig, TaskConfig
from train.trainer import YoloV8Trainer


def _make_fake(close_epochs):
    """A minimal object carrying just what _maybe_close_mosaic touches."""
    mosaic = MosaicConfig(mosaic_frequency=0.5, mixup_frequency=0.3,
                          close_mosaic_epochs=close_epochs)
    parser = ParserConfig(mosaic=mosaic)
    train_data = DataConfig(parser=parser)
    config = types.SimpleNamespace(task=types.SimpleNamespace(train_data=train_data))

    calls = {}

    class _Task:
        def build_inputs(self, data_cfg):
            calls['data_cfg'] = data_cfg          # capture the rebuilt config
            return tf.data.Dataset.range(4)

    fake = types.SimpleNamespace(
        _config=config, _task=_Task(), _distributed=False, _strategy=None,
        _mosaic_closed=False, _train_ds=None, _train_iter='ORIGINAL',
    )
    return fake, calls


def test_noop_when_disabled():
    fake, calls = _make_fake(close_epochs=0)
    YoloV8Trainer._maybe_close_mosaic(fake, epoch=99, total_epochs=100)
    assert fake._mosaic_closed is False
    assert fake._train_iter == 'ORIGINAL'        # iterator untouched
    assert 'data_cfg' not in calls               # build_inputs never called


def test_noop_before_boundary():
    fake, calls = _make_fake(close_epochs=10)
    YoloV8Trainer._maybe_close_mosaic(fake, epoch=80, total_epochs=100)  # 80 < 90
    assert fake._mosaic_closed is False
    assert fake._train_iter == 'ORIGINAL'


def test_closes_at_boundary_once():
    fake, calls = _make_fake(close_epochs=10)
    # epoch 90 == total(100) - 10 -> close fires
    YoloV8Trainer._maybe_close_mosaic(fake, epoch=90, total_epochs=100)
    assert fake._mosaic_closed is True
    assert fake._train_iter != 'ORIGINAL'        # iterator rebuilt
    # the rebuilt config has mosaic + mixup disabled
    m = calls['data_cfg'].parser.mosaic
    assert m.mosaic_frequency == 0.0 and m.mixup_frequency == 0.0
    # other mosaic settings preserved
    assert m.close_mosaic_epochs == 10

    # a second call (next epoch) is a no-op — does not rebuild again
    calls.clear()
    fake._train_iter = 'SECOND'
    YoloV8Trainer._maybe_close_mosaic(fake, epoch=91, total_epochs=100)
    assert fake._train_iter == 'SECOND'
    assert 'data_cfg' not in calls
