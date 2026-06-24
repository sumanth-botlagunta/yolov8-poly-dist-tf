"""Tests for the fine-tune path (task.finetune_from): load the full model from a trained
checkpoint's EMA/deployed weights into a fresh optimizer, distinct from transfer-init."""

import types

import pytest

import train.task as T
from configs.model_config import TaskConfig


def _task_with(finetune_from=None, init_checkpoint=None):
    cfg = types.SimpleNamespace(task=types.SimpleNamespace(
        finetune_from=finetune_from, init_checkpoint=init_checkpoint,
        init_checkpoint_modules=['backbone', 'decoder']))
    task = T.YoloV8Task.__new__(T.YoloV8Task)
    task._config = cfg
    return task


def test_default_finetune_from_is_none():
    assert TaskConfig().finetune_from is None


def test_finetune_loads_ema_and_skips_migration(monkeypatch):
    calls = {}
    monkeypatch.setattr('tools.shared.ckpt_loading.restore_eval_weights',
                        lambda model, path: calls.setdefault('restore', path) or 'ema')
    monkeypatch.setattr('tools.checkpoint_migration.migrate_checkpoint',
                        lambda **kw: calls.setdefault('migrate', True))
    _task_with(finetune_from='/run/ckpt-100').initialize(object())
    assert calls.get('restore') == '/run/ckpt-100'   # EMA-aware full-model restore
    assert 'migrate' not in calls                     # transfer-init path skipped


def test_init_checkpoint_path_unchanged(monkeypatch):
    calls = {}
    monkeypatch.setattr('tools.shared.ckpt_loading.restore_eval_weights',
                        lambda model, path: calls.setdefault('restore', path))
    monkeypatch.setattr('tools.checkpoint_migration.migrate_checkpoint',
                        lambda **kw: calls.setdefault('migrate', kw.get('modules')))
    _task_with(init_checkpoint='/pretrained/ckpt').initialize(object())
    assert calls.get('migrate') == ['backbone', 'decoder']   # transfer-init still works
    assert 'restore' not in calls


def test_neither_set_is_noop(monkeypatch):
    calls = {}
    monkeypatch.setattr('tools.shared.ckpt_loading.restore_eval_weights',
                        lambda *a: calls.setdefault('restore', True))
    monkeypatch.setattr('tools.checkpoint_migration.migrate_checkpoint',
                        lambda **kw: calls.setdefault('migrate', True))
    _task_with().initialize(object())
    assert not calls       # from-scratch: nothing loaded


def test_finetune_precedes_init_checkpoint(monkeypatch):
    # if both somehow set, fine-tune wins in initialize() (validation also rejects both)
    calls = {}
    monkeypatch.setattr('tools.shared.ckpt_loading.restore_eval_weights',
                        lambda model, path: calls.setdefault('restore', path) or 'ema')
    monkeypatch.setattr('tools.checkpoint_migration.migrate_checkpoint',
                        lambda **kw: calls.setdefault('migrate', True))
    _task_with(finetune_from='/a/ckpt', init_checkpoint='/b/ckpt').initialize(object())
    assert calls.get('restore') == '/a/ckpt' and 'migrate' not in calls
