"""Unit tests for the append-only JSONL validation-history store (eval/val_history.py)."""

import json
import os

from eval import val_history


def _report(f1):
    """Minimal report dict shaped like eval/metrics_report.build_report output."""
    return {
        'iou_thresh': 0.5, 'area': 'all', 'max_dets': 10,
        'mean': {'f1': f1, 'precision': 1.0, 'recall': f1},
        'best_conf': [{'category': 3, 'f1': f1, 'precision': 1.0,
                       'recall': f1, 'conf_threshold': 0.5, 'valid': True}],
        'all_conf': [],
        'per_category_ap': [],
    }


def test_append_is_one_line_per_call(tmp_path):
    p = str(tmp_path / 'val_history.jsonl')
    val_history.append_record(p, _report(0.4), epoch=1, step=1000)
    val_history.append_record(p, _report(0.6), epoch=2, step=2000)
    lines = [l for l in open(p).read().splitlines() if l.strip()]
    assert len(lines) == 2
    # each line is valid standalone JSON (true JSONL)
    for l in lines:
        json.loads(l)


def test_record_is_report_superset_with_stamps(tmp_path):
    p = str(tmp_path / 'val_history.jsonl')
    rec = val_history.append_record(
        p, _report(0.5), epoch=7, step=7000,
        checkpoint='ckpt-7000', metrics={'F1score50': 0.5, 'mAP': 0.3, 'bad': True})
    # report keys survive (so it round-trips through metrics_report.write_txt)
    assert rec['mean']['f1'] == 0.5 and 'best_conf' in rec
    # stamps added
    assert rec['epoch'] == 7 and rec['step'] == 7000 and rec['checkpoint'] == 'ckpt-7000'
    # metrics coerced to float, bool dropped
    assert rec['metrics'] == {'F1score50': 0.5, 'mAP': 0.3}


def test_load_skips_partial_trailing_line(tmp_path):
    p = str(tmp_path / 'val_history.jsonl')
    val_history.append_record(p, _report(0.4), epoch=1, step=1000)
    with open(p, 'a') as f:
        f.write('{"epoch": 2, "step": 20')   # simulate an interrupted write
    recs = val_history.load_records(p)
    assert len(recs) == 1 and recs[0]['epoch'] == 1


def test_best_and_select(tmp_path):
    p = str(tmp_path / 'val_history.jsonl')
    val_history.append_record(p, _report(0.4), epoch=1, step=1000,
                              metrics={'F1score50': 0.40})
    val_history.append_record(p, _report(0.7), epoch=2, step=2000,
                              metrics={'F1score50': 0.70})
    val_history.append_record(p, _report(0.5), epoch=3, step=3000,
                              metrics={'F1score50': 0.50})
    recs = val_history.load_records(p)
    assert val_history.best_record(recs, 'F1score50')['epoch'] == 2
    assert val_history.select(recs, epoch=3)['step'] == 3000
    assert val_history.select(recs, step=1000)['epoch'] == 1
    assert val_history.select(recs, epoch=99) is None


def test_metric_of_falls_back_to_mean_f1(tmp_path):
    # a record with no 'metrics' still yields F1score50 from mean.f1
    rec = _report(0.66)
    assert abs(val_history.metric_of(rec, 'F1score50') - 0.66) < 1e-9


def test_revalidation_collapses_to_latest(tmp_path):
    # validating the same epoch twice appends two lines; views collapse to the latest.
    p = str(tmp_path / 'val_history.jsonl')
    val_history.append_record(p, _report(0.40), epoch=5, step=5000,
                              metrics={'F1score50': 0.40})
    val_history.append_record(p, _report(0.55), epoch=5, step=5000,
                              metrics={'F1score50': 0.55})   # re-validation
    recs = val_history.load_records(p)
    assert len(recs) == 2                                    # append-only log keeps both
    collapsed = val_history.latest_per_epoch(recs)
    assert len(collapsed) == 1                               # one canonical row
    assert collapsed[0]['metrics']['F1score50'] == 0.55      # the latest wins
    # select() already returns the latest; best ignores the stale duplicate
    assert val_history.select(recs, epoch=5)['metrics']['F1score50'] == 0.55
    assert val_history.best_record(recs)['metrics']['F1score50'] == 0.55


def test_best_prefers_latest_not_stale_higher(tmp_path):
    # a stale high duplicate must NOT beat the epoch's own re-validated (lower) value
    p = str(tmp_path / 'val_history.jsonl')
    val_history.append_record(p, _report(0.90), epoch=1, step=1000,
                              metrics={'F1score50': 0.90})   # stale, later corrected
    val_history.append_record(p, _report(0.50), epoch=1, step=1000,
                              metrics={'F1score50': 0.50})   # re-validation
    val_history.append_record(p, _report(0.70), epoch=2, step=2000,
                              metrics={'F1score50': 0.70})
    recs = val_history.load_records(p)
    best = val_history.best_record(recs)
    assert best['epoch'] == 2 and best['metrics']['F1score50'] == 0.70


def test_resolve_path_accepts_dir(tmp_path):
    assert val_history.resolve_path(str(tmp_path)).endswith('val_history.jsonl')
    f = str(tmp_path / 'val_history.jsonl')
    assert val_history.resolve_path(f) == f
