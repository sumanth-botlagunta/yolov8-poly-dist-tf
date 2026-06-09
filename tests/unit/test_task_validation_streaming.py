"""Tests for streaming validation aggregation in YoloV8Task.

reduce_aggregated_logs used to buffer every batch's raw prediction/label tensors
until end-of-epoch (host-memory risk on large val sets). It now builds the
evaluators once and streams each batch into them via aggregate_logs. These tests
pin that behavior:
    - aggregate_logs holds evaluators, NOT buffered prediction/label lists.
    - the streamed pipeline still produces finite COCO metrics.
    - reduce_aggregated_logs(None) returns {} (no validation batches).
"""

import numpy as np
import tensorflow as tf

from configs.yaml_loader import load_config
from train.task import YoloV8Task


def _task():
    cfg = load_config('configs/experiments/yolo/yolov8_bbox.yaml')
    return YoloV8Task(cfg)


def _batch(match=True):
    """One synthetic batch (B=2): one detection per image, matching the GT box."""
    box = [0.3, 0.3, 0.7, 0.7]
    pred_box = box if match else [0.0, 0.0, 0.05, 0.05]
    preds = {
        'bbox':           tf.constant([[box], [box]], tf.float32),       # [2,1,4]
        'classes':        tf.constant([[0], [0]], tf.int64),             # [2,1]
        'confidence':     tf.constant([[0.9], [0.9]], tf.float32),       # [2,1]
        'num_detections': tf.constant([1, 1], tf.int32),                 # [2]
    }
    labels = {
        'bbox':    tf.constant([[pred_box], [pred_box]], tf.float32),    # [2,1,4]
        'classes': tf.constant([[0], [0]], tf.int64),                    # [2,1]
        'n_gt':    tf.constant([1, 1], tf.int64),                        # [2]
    }
    return {'predictions': preds, 'labels': labels}


def test_aggregate_logs_streams_not_buffers():
    """State holds evaluators, not raw prediction/label lists."""
    task = _task()
    state = None
    for _ in range(2):
        state = task.aggregate_logs(state, _batch())
    # Streaming contract: evaluators present, no buffered tensors.
    assert set(state.keys()) == {'coco', 'dist', 'poly'}
    assert 'predictions' not in state and 'labels' not in state
    # bbox-only config → no distance/polygon evaluators.
    assert state['dist'] is None
    assert state['poly'] is None


def test_streamed_metrics_finite_and_present():
    """Two streamed matching batches produce finite COCO metrics."""
    task = _task()
    state = None
    for _ in range(2):
        state = task.aggregate_logs(state, _batch(match=True))
    metrics = task.reduce_aggregated_logs(state)
    for key in ('mAP', 'mAP50', 'AR100', 'F1score50'):
        assert key in metrics, f"missing metric {key}"
        assert np.isfinite(metrics[key]), f"{key} not finite: {metrics[key]}"
    # Perfectly-matched detections → non-trivial AP50.
    assert metrics['mAP50'] > 0.0


def test_reduce_none_returns_empty():
    """No validation batches → empty metrics, no crash."""
    task = _task()
    assert task.reduce_aggregated_logs(None) == {}
