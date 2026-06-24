"""Tests for eval failure mining (eval/failure_mining.py)."""

import os

import numpy as np

from eval.failure_mining import FailureCollector, _iou_yxyx


def test_iou():
    assert abs(_iou_yxyx(np.array([0, 0, .5, .5]), np.array([0, 0, .5, .5])) - 1.0) < 1e-9
    assert _iou_yxyx(np.array([0, 0, .4, .4]), np.array([.6, .6, 1, 1])) == 0.0


def _img():
    return (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)


def test_records_fp_fn_lowiou():
    fc = FailureCollector(class_names=[f"c{i}" for i in range(10)],
                          match_iou=0.5, lowiou_below=0.7)
    pred = {'num_detections': 3,
            'bbox': np.array([[0.1, 0.1, 0.5, 0.5],     # TP (not a failure)
                              [0.6, 0.6, 0.9, 0.9],     # FP (class 5, no GT)
                              [0.1, 0.1, 0.41, 0.41]],  # low-IoU (~0.6 with gt1)
                             np.float32),
            'classes': np.array([3, 5, 3]),
            'confidence': np.array([0.9, 0.8, 0.6], np.float32)}
    gt = {'n_gt': 3,
          'bbox': np.array([[0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5],
                            [0.2, 0.2, 0.6, 0.6]], np.float32),
          'classes': np.array([3, 3, 7])}               # gt2 (class 7) -> missed
    fc.update(_img(), pred, gt)
    s = fc.summary()
    assert s['fp'] == 1 and s['fn'] == 1 and s['lowiou'] == 1


def test_low_confidence_fp_ignored():
    fc = FailureCollector(score_thresh=0.25)
    pred = {'num_detections': 1, 'bbox': np.array([[0.6, 0.6, 0.9, 0.9]], np.float32),
            'classes': np.array([5]), 'confidence': np.array([0.1], np.float32)}  # < thresh
    gt = {'n_gt': 0, 'bbox': np.zeros((0, 4), np.float32), 'classes': np.zeros((0,), int)}
    fc.update(_img(), pred, gt)
    assert fc.summary()['fp'] == 0


def test_per_class_cap():
    fc = FailureCollector(per_class=3, score_thresh=0.0)
    # 5 distinct FPs of class 5 -> only the 3 highest-score kept
    for k in range(5):
        pred = {'num_detections': 1, 'bbox': np.array([[0.6, 0.6, 0.9, 0.9]], np.float32),
                'classes': np.array([5]), 'confidence': np.array([0.1 * (k + 1)], np.float32)}
        gt = {'n_gt': 0, 'bbox': np.zeros((0, 4), np.float32), 'classes': np.zeros((0,), int)}
        fc.update(_img(), pred, gt)
    assert fc.summary()['fp'] == 3
    kept_scores = sorted(r.score for r in fc._kept[(5, 'fp')])
    assert np.allclose(kept_scores, [0.3, 0.4, 0.5], atol=1e-6)   # 3 highest-confidence FPs


def test_write_creates_per_class_dirs(tmp_path):
    fc = FailureCollector(class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    pred = {'num_detections': 1, 'bbox': np.array([[0.6, 0.6, 0.9, 0.9]], np.float32),
            'classes': np.array([5]), 'confidence': np.array([0.9], np.float32)}
    gt = {'n_gt': 0, 'bbox': np.zeros((0, 4), np.float32), 'classes': np.zeros((0,), int)}
    fc.update(_img(), pred, gt)
    n = fc.write(str(tmp_path))
    assert n == 1 and os.path.isdir(tmp_path / '05_f')
