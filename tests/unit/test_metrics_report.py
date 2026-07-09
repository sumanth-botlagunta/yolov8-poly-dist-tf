"""Tests for the per-category F1 validation report (eval/metrics_report.py) and the
COCOEvaluator table methods. The writer tests are dependency-free; the full evaluator
path is skipped when pycocotools is unavailable."""

import json
import zipfile

import numpy as np
import pytest

from eval import metrics_report as mr


def _synthetic_report():
    return {
        'epoch': 7, 'step': 1234, 'iou_thresh': 0.5, 'area': 'all', 'max_dets': 100,
        'conf_grid': [0.05, 0.10, 0.15],
        'mean': {'f1': 0.69, 'precision': 0.78, 'recall': 0.62},
        'best_conf': [
            {'category': 0, 'f1': 0.66, 'precision': 0.84, 'recall': 0.55, 'conf_threshold': 0.35},
            {'category': 1, 'f1': 0.61, 'precision': 0.67, 'recall': 0.56, 'conf_threshold': 0.10},
        ],
        'all_conf': [
            {'category': 0, 'thresh': 0.05, 'f1': 0.59, 'precision': 0.54, 'recall': 0.65},
            {'category': 0, 'thresh': 0.10, 'f1': 0.62, 'precision': 0.62, 'recall': 0.63},
            {'category': 1, 'thresh': 0.05, 'f1': 0.61, 'precision': 0.60, 'recall': 0.56},
        ],
        'per_category_ap': [{'category': 0, 'ap': 0.40, 'ap50': 0.61, 'num_gt': 575, 'dontcare': 3}],
    }


def test_json_roundtrip(tmp_path):
    rep = _synthetic_report()
    p = mr.write_json(rep, str(tmp_path / 'r.json'))
    assert mr.load_json(p) == rep


def test_save_canonical_writes_json_and_txt(tmp_path):
    paths = mr.save_canonical(_synthetic_report(), str(tmp_path), 'epoch_0007')
    assert (tmp_path / 'epoch_0007.json').exists()
    assert (tmp_path / 'epoch_0007.txt').exists()
    txt = (tmp_path / 'epoch_0007.txt').read_text()
    assert 'MEAN over categories' in txt and 'best confidence per category' in txt
    assert 'all confidence thresholds' in txt


def test_txt_envelope_header_gated_on_sweep_source(tmp_path):
    """write_txt prints the COCO-interpolated caveat above the all-conf table only when
    sweep_source == 'coco_envelope' (dependency-free writer check)."""
    rep = _synthetic_report()
    rep['sweep_source'] = 'raw'
    raw_txt = (tmp_path / 'raw.txt')
    mr.write_txt(rep, str(raw_txt))
    assert 'COCO-interpolated' not in raw_txt.read_text()

    rep['sweep_source'] = 'coco_envelope'
    env_txt = (tmp_path / 'env.txt')
    mr.write_txt(rep, str(env_txt))
    assert 'COCO-interpolated' in env_txt.read_text()


def test_csv_two_files(tmp_path):
    out = mr.write_csv(_synthetic_report(), str(tmp_path), 'm')
    assert (tmp_path / 'm_best_conf.csv').exists()
    assert (tmp_path / 'm_all_conf.csv').exists()
    head = (tmp_path / 'm_best_conf.csv').read_text().splitlines()[0]
    assert head.split(',') == ['category', 'name', 'f1', 'precision', 'recall', 'conf_threshold']


def test_xlsx_is_valid_two_sheet_workbook(tmp_path):
    p = mr.write_xlsx(_synthetic_report(), str(tmp_path / 'm.xlsx'))
    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        assert {'[Content_Types].xml', 'xl/workbook.xml',
                'xl/worksheets/sheet1.xml', 'xl/worksheets/sheet2.xml'} <= names
        wb = z.read('xl/workbook.xml').decode()
        assert 'best_conf' in wb and 'all_conf' in wb
    # If openpyxl is present, confirm it actually opens and has the two sheets.
    try:
        import openpyxl
        wb = openpyxl.load_workbook(p)
        assert wb.sheetnames[:2] == ['best_conf', 'all_conf']
    except ImportError:
        pass


def test_xlsx_numbers_are_numeric_cells(tmp_path):
    """f1/precision values must be numeric cells (so Excel can sort/sum), not strings."""
    p = mr.write_xlsx(_synthetic_report(), str(tmp_path / 'm.xlsx'))
    with zipfile.ZipFile(p) as z:
        s1 = z.read('xl/worksheets/sheet1.xml').decode()
    # numeric cells render as <c r="..."><v>0.66</v></c> (no t="inlineStr")
    assert '<v>0.66</v>' in s1


# --- full evaluator path (needs pycocotools) ---

def test_cocoevaluator_metrics_tables():
    pytest.importorskip('pycocotools')
    from eval.coco_metrics import COCOEvaluator

    nc = 3
    ev = COCOEvaluator(num_classes=nc, image_size=(100, 100))
    rng = np.random.RandomState(0)
    for _ in range(6):
        M = 3
        gt_b = np.clip(rng.uniform(0, 0.6, [1, M, 4]), 0, 1).astype('float32')
        gt_b[..., 2:] = gt_b[..., :2] + 0.3
        gt_c = rng.randint(0, nc, [1, M]).astype('int64')
        pb = gt_b + rng.uniform(-0.02, 0.02, [1, M, 4]).astype('float32')
        ps = rng.uniform(0.3, 0.95, [1, M]).astype('float32')
        ev.update({'bbox': pb, 'classes': gt_c, 'confidence': ps,
                   'num_detections': np.array([M], 'int32')},
                  {'bbox': gt_b, 'classes': gt_c, 'n_gt': np.array([M], 'int32')})
    m = ev.evaluate()
    assert {'F1score50', 'precision50', 'recall50'} <= set(m)

    t = ev.metrics_tables()
    # report mean F1 must equal the logged F1score50 (same operating point)
    assert abs(t['mean']['f1'] - m['F1score50']) < 1e-9
    assert len(t['best_conf']) == nc
    assert len(t['all_conf']) == nc * len(t['conf_grid'])
    # build_report + write everything
    rep = mr.build_report(ev, epoch=1, step=10)
    assert rep['mean']['f1'] == t['mean']['f1']
