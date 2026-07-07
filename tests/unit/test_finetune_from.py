"""Tests for the two seed-init paths on ``task.initialize``:

* ``finetune_from`` — load the FULL model from a trained checkpoint's EMA/deployed
  weights into a fresh optimizer.
* ``init_checkpoint`` — transfer-init: full restore via ``restore_eval_weights``,
  then non-selected modules (e.g. the head) are put back to their fresh init.
"""

import types

import tensorflow as tf

import train.task as T
from configs.model_config import TaskConfig


def _task_with(finetune_from=None, init_checkpoint=None):
    cfg = types.SimpleNamespace(task=types.SimpleNamespace(
        finetune_from=finetune_from, init_checkpoint=init_checkpoint,
        init_checkpoint_modules=['backbone', 'decoder']))
    task = T.YoloV8Task.__new__(T.YoloV8Task)
    task._config = cfg
    return task


def _fake_model():
    """Stub with backbone/decoder/head modules carrying one variable each."""
    def module():
        return types.SimpleNamespace(variables=[tf.Variable([0.0])])
    return types.SimpleNamespace(
        backbone=module(), decoder=module(), head=module())


def test_default_finetune_from_is_none():
    assert TaskConfig().finetune_from is None


def test_finetune_loads_full_model(monkeypatch):
    calls = {}
    monkeypatch.setattr('tools.shared.ckpt_loading.restore_eval_weights',
                        lambda model, path: calls.setdefault('restore', path) or 'ema')
    _task_with(finetune_from='/run/ckpt-100').initialize(object())
    assert calls.get('restore') == '/run/ckpt-100'   # EMA-aware full-model restore


def test_init_checkpoint_uses_full_restore_and_keeps_head(monkeypatch):
    calls = {}

    def fake_restore(model, path):
        calls['restore'] = path
        # Simulate a full-model load overwriting every module's weights.
        for m in (model.backbone, model.decoder, model.head):
            for v in m.variables:
                v.assign([7.0])
        return 'ema'

    monkeypatch.setattr('tools.shared.ckpt_loading.restore_eval_weights', fake_restore)
    model = _fake_model()
    _task_with(init_checkpoint='/pretrained/ckpt').initialize(model)
    assert calls.get('restore') == '/pretrained/ckpt'
    # Selected modules keep the loaded weights; the head is restored to fresh init.
    assert float(model.backbone.variables[0][0]) == 7.0
    assert float(model.decoder.variables[0][0]) == 7.0
    assert float(model.head.variables[0][0]) == 0.0


def test_neither_set_is_noop(monkeypatch):
    calls = {}
    monkeypatch.setattr('tools.shared.ckpt_loading.restore_eval_weights',
                        lambda *a: calls.setdefault('restore', True))
    _task_with().initialize(object())
    assert not calls       # from-scratch: nothing loaded


def test_finetune_precedes_init_checkpoint(monkeypatch):
    # if both somehow set, fine-tune wins in initialize() (validation also rejects both)
    calls = {}
    monkeypatch.setattr('tools.shared.ckpt_loading.restore_eval_weights',
                        lambda model, path: calls.setdefault('restore', path) or 'ema')
    _task_with(finetune_from='/a/ckpt', init_checkpoint='/b/ckpt').initialize(object())
    assert calls.get('restore') == '/a/ckpt'
