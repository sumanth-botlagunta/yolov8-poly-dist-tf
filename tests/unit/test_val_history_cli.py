"""Unit tests for the merged val_history CLI (utils/reports/val_history.py).

Covers input-type detection (jsonl vs single report JSON), --format dispatch, and csv
output correctness. No TF dependency; xlsx/parquet checks skip when pandas is absent.
"""

import csv
import io
import json

import pytest

from eval import val_history as store
from utils.reports import val_history as cli


def _report(f1, epoch, step):
    """Minimal report dict shaped like eval/metrics_report.build_report output."""
    return {
        'epoch': epoch, 'step': step,
        'iou_thresh': 0.5, 'area': 'all', 'max_dets': 10,
        'metrics': {'F1score50': f1, 'mAP': f1 / 2},
        'mean': {'f1': f1, 'precision': 1.0, 'recall': f1},
        'best_conf': [
            {'category': 3, 'f1': f1, 'precision': 1.0, 'recall': f1,
             'conf_threshold': 0.5, 'valid': True},
            {'category': 7, 'f1': f1 / 2, 'precision': 0.8, 'recall': f1 / 2,
             'conf_threshold': 0.4, 'valid': True},
        ],
        'all_conf': [
            {'category': 3, 'thresh': 0.5, 'f1': f1, 'precision': 1.0, 'recall': f1},
        ],
        'per_category_ap': [],
    }


def _history(tmp_path):
    p = str(tmp_path / 'val_history.jsonl')
    store.append_record(p, _report(0.4, 1, 1000), epoch=1, step=1000,
                        metrics={'F1score50': 0.40})
    store.append_record(p, _report(0.7, 2, 2000), epoch=2, step=2000,
                        metrics={'F1score50': 0.70})
    return p


# --- input-type detection ------------------------------------------------------------

def test_load_input_detects_jsonl(tmp_path):
    p = _history(tmp_path)
    records, is_single = cli._load_input(p)
    assert is_single is False
    assert len(records) == 2


def test_load_input_detects_run_dir(tmp_path):
    _history(tmp_path)
    records, is_single = cli._load_input(str(tmp_path))
    assert is_single is False and len(records) == 2


def test_load_input_detects_single_report_json(tmp_path):
    fp = tmp_path / 'ckpt-99000_val.json'
    fp.write_text(json.dumps(_report(0.55, 5, 5000)))
    records, is_single = cli._load_input(str(fp))
    assert is_single is True
    assert len(records) == 1 and records[0]['mean']['f1'] == 0.55


def test_load_input_rejects_non_report_json(tmp_path):
    fp = tmp_path / 'metrics.json'
    fp.write_text(json.dumps({'mAP': 0.3}))   # not a report (no best_conf/all_conf/mean)
    with pytest.raises(SystemExit):
        cli._load_input(str(fp))


# --- format dispatch -----------------------------------------------------------------

def test_render_json_roundtrips(tmp_path):
    rec = _report(0.7, 2, 2000)
    out = cli._render_selected(rec, 'json', best_only=False)
    assert json.loads(out)['mean']['f1'] == 0.7


def test_render_csv_correctness():
    rec = _report(0.7, 2, 2000)
    out = cli._render_selected(rec, 'csv', best_only=False)
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == ['category', 'name', 'f1', 'precision', 'recall', 'conf_threshold']
    # one row per best_conf entry, in order
    assert len(rows) == 3
    assert rows[1][0] == '3' and rows[1][2] == '0.700000' and rows[1][5] == '0.5000'
    assert rows[2][0] == '7' and rows[2][3] == '0.800000'


def test_render_txt_is_nonempty(tmp_path):
    rec = _report(0.7, 2, 2000)
    txt = cli._render_selected(rec, 'txt', best_only=False)
    assert isinstance(txt, str) and txt.strip()


# --- selection helpers ---------------------------------------------------------------

class _Args:
    def __init__(self, **kw):
        self.best = kw.get('best', False)
        self.metric = kw.get('metric', 'F1score50')
        self.epoch = kw.get('epoch')
        self.step = kw.get('step')
        self.checkpoint = kw.get('checkpoint')


def test_select_best_and_by_epoch(tmp_path):
    records, _ = cli._load_input(_history(tmp_path))
    assert cli._select_record(records, _Args(best=True))['epoch'] == 2
    assert cli._select_record(records, _Args(epoch=1))['step'] == 1000


def test_select_no_match_exits(tmp_path):
    records, _ = cli._load_input(_history(tmp_path))
    with pytest.raises(SystemExit):
        cli._select_record(records, _Args(epoch=99))


# --- xlsx / parquet export (skip when pandas missing) --------------------------------

def test_export_xlsx(tmp_path):
    pytest.importorskip('pandas')
    pytest.importorskip('openpyxl')
    records, _ = cli._load_input(_history(tmp_path))
    out = str(tmp_path / 'metrics.xlsx')
    cli._export_tabular(records, 'xlsx', out)
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_export_parquet_writes_two_files(tmp_path):
    pytest.importorskip('pandas')
    pytest.importorskip('pyarrow')
    records, _ = cli._load_input(_history(tmp_path))
    out = str(tmp_path / 'run.parquet')
    cli._export_tabular(records, 'parquet', out)
    import os
    assert os.path.exists(str(tmp_path / 'run_best_conf.parquet'))
    assert os.path.exists(str(tmp_path / 'run_all_conf.parquet'))
