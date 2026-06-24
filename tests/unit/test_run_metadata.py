"""Tests for run provenance (train/run_metadata.py)."""

import json
import types

from train import run_metadata


def _cfg(train_tfds="a:1.0.0,b:2.3.0", val_tfds="a:1.0.0", cnp="c:1.1.0",
         dist_tfds="d:0.9.0", finetune=None, freeze=None):
    td = types.SimpleNamespace(
        tfds_name=train_tfds, tfds_for_cnp=cnp, tfds_data_dir="/no/such/dir",
        distance_data=types.SimpleNamespace(tfds_name=dist_tfds))
    vd = types.SimpleNamespace(tfds_name=val_tfds)
    task = types.SimpleNamespace(
        train_data=td, validation_data=vd, finetune_from=finetune,
        init_checkpoint=None, init_checkpoint_modules=['backbone', 'decoder'],
        freeze_modules=freeze or [])
    trainer = types.SimpleNamespace(grad_accum_steps=1)
    return types.SimpleNamespace(task=task, trainer=trainer)


def test_dataset_specs_parse_name_and_version():
    specs = run_metadata._dataset_specs(_cfg())
    by_name = {(s['role'], s['name']): s for s in specs}
    assert by_name[('train', 'a')]['requested'] == '1.0.0'
    assert by_name[('train', 'b')]['requested'] == '2.3.0'
    assert by_name[('val', 'a')]['requested'] == '1.0.0'
    assert by_name[('copy_paste', 'c')]['requested'] == '1.1.0'
    assert by_name[('distance', 'd')]['requested'] == '0.9.0'


def test_unpinned_name_has_no_requested_version():
    specs = run_metadata._dataset_specs(_cfg(train_tfds="onlyname"))
    train = [s for s in specs if s['role'] == 'train']
    assert train[0]['name'] == 'onlyname' and train[0]['requested'] is None


def test_write_metadata_structure(tmp_path):
    p = run_metadata.write_run_metadata(
        str(tmp_path), _cfg(finetune="/run/ckpt-100", freeze=['backbone']),
        resume_from=None, started_at="2026-01-01T00:00:00")
    assert p is not None
    m = json.load(open(p))
    assert set(m) >= {'started_at', 'command', 'git', 'env', 'run', 'datasets'}
    assert m['run']['finetune_from'] == '/run/ckpt-100'
    assert m['run']['freeze_modules'] == ['backbone']
    assert len(m['datasets']) == 5
    assert isinstance(m['git'], dict) and 'available' in m['git']


def test_write_never_raises_on_bad_config(tmp_path):
    # a config missing fields must not crash the run — returns None or a partial file
    bad = types.SimpleNamespace(task=types.SimpleNamespace(), trainer=types.SimpleNamespace())
    # should not raise
    run_metadata.write_run_metadata(str(tmp_path), bad, started_at="x")
