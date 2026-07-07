"""Pinning tests for training-critical behavior contracts.

These pin behaviors that affect training semantics; keep them green so the
bugs cannot silently regress:

  - copy_paste polygon validity uses the > -1.0 sentinel (-1.0 is the only
    reserved padding value; a >= 0.0 gate drops real negative-coordinate
    vertices and corrupts the polygon GT)
  - TAL assigner fallback poly_size derives from angle_step (not hardcoded 72)
  - polygon_conf_loss does not train conf=0 on distance-stream fg anchors
    (ignore_bg guard in _polygon_loss)
  - use_acsl=True fails loud instead of silently no-op'ing
"""

import inspect

import numpy as np
import pytest
import tensorflow as tf

from data_pipeline.copy_paste import CopyAndPasteModule
from losses.tal_assigner import TaskAlignedAssigner
from losses.tal_loss import TaskAlignedLossExtended


# ---------------------------------------------------------------------------
# copy_paste sentinel
# ---------------------------------------------------------------------------

def test_copy_paste_uses_minus_one_sentinel():
    # The compositing body lives in _paste (_copy_and_paste is the min-size
    # gate wrapper around it).
    src = inspect.getsource(CopyAndPasteModule._paste)
    assert "> -1.0" in src
    assert "pts[:, 0] >= 0.0" not in src


def test_copy_paste_keeps_negative_vertex():
    bg = {
        'image': tf.zeros([100, 100, 3], dtype=tf.uint8),
        'height': tf.constant(100, dtype=tf.int32),
        'width': tf.constant(100, dtype=tf.int32),
        'groundtruth_boxes': tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
        'groundtruth_classes': tf.constant([1], dtype=tf.int64),
        'groundtruth_polygons': tf.fill([1, 6], -1.0),
        'groundtruth_is_crowd': tf.constant([False]),
        'groundtruth_area': tf.constant([0.16], dtype=tf.float32),
        'groundtruth_dontcare': tf.constant([0], dtype=tf.int64),
    }
    obj = {
        'image': tf.ones([10, 10, 4], dtype=tf.uint8) * 255,
        'orig_bbox': tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32),
        'label': tf.constant(2, dtype=tf.int64),
        'points': tf.constant([-0.3, 0.5, 0.7, 0.5, -1.0, -1.0], dtype=tf.float32),
    }
    module = CopyAndPasteModule(
        prob=1.0, min_height=1, min_width=1,
        max_resize_ratio=1.0, min_resize_ratio=1.0, height_limit=0.6,
    )
    counts = []
    for seed in range(10):
        tf.random.set_seed(seed)
        res = module._copy_and_paste(dict(bg), dict(obj))
        pairs = res['groundtruth_polygons'][-1].numpy().reshape(-1, 2)
        real_x = pairs[pairs[:, 0] > -1.0, 0]
        counts.append(len(set(np.round(real_x, 4))))
    assert max(counts) >= 2, f"negative vertex dropped; counts={counts}"


# ---------------------------------------------------------------------------
# TAL assigner poly_size from angle_step
# ---------------------------------------------------------------------------

def _run_assigner(assigner, B=2, A=64, M=5, C=39):
    return assigner(
        tf.random.uniform([B, A, C]),
        tf.random.uniform([B, A, 4], 0, 672),
        tf.random.uniform([A, 2], 0, 672),
        tf.random.uniform([B, M], 0, C, dtype=tf.int64),
        tf.random.uniform([B, M, 4], 0, 672),
        tf.ones([B, M], dtype=tf.bool),
        gt_polys=None, gt_dists=None,
    )


def test_assigner_poly_size_default_15():
    a = TaskAlignedAssigner(topk=10)
    assert a.poly_size == 72
    assert _run_assigner(a)[3].shape[-1] == 72


def test_assigner_poly_size_angle_step_10():
    a = TaskAlignedAssigner(topk=10, angle_step=10)
    assert a.poly_size == 108
    assert _run_assigner(a)[3].shape[-1] == 108


def test_loss_wires_angle_step_to_assigner():
    loss = TaskAlignedLossExtended(angle_step=10, with_polygons=True, with_distance=False)
    assert loss._assigner_fn.poly_size == 108
    assert loss.num_vertices == 36


# ---------------------------------------------------------------------------
# polygon conf ignore_bg guard
# ---------------------------------------------------------------------------

_A = 4
_V = 24


def _poly_targets():
    target = np.zeros((2, _A, 72), dtype=np.float32)
    fg = np.zeros((2, _A), dtype=bool)
    fg[0, 1] = True
    fg[1, 2] = True
    for b in range(8):
        target[0, 1, b * 3 + 2] = 1.0
    return tf.constant(target), tf.constant(fg)


def test_polygon_loss_has_ignore_bg_param():
    assert "ignore_bg" in inspect.signature(TaskAlignedLossExtended._polygon_loss).parameters


def test_distance_stream_fg_gets_zero_conf_gradient():
    loss = TaskAlignedLossExtended(with_polygons=True, with_distance=False)
    target, fg = _poly_targets()
    pd_conf = tf.Variable(0.5 * np.ones((2, _A, _V), dtype=np.float32))
    zeros = tf.constant(np.zeros((2, _A, _V), dtype=np.float32))
    with tf.GradientTape() as tape:
        poly_total, *_ = loss._polygon_loss(
            zeros, zeros, pd_conf, target, fg, tf.constant(2.0),
            tf.constant([0, 1], dtype=tf.int64),
        )
    grad = tape.gradient(poly_total, pd_conf)
    assert np.max(np.abs(grad[1, 2, :].numpy())) < 1e-9   # distance-stream fg: no grad
    assert np.max(np.abs(grad[0, 1, :].numpy())) > 1e-6   # detection fg: still trained


# ---------------------------------------------------------------------------
# use_acsl fail-loud
# ---------------------------------------------------------------------------

def test_use_acsl_true_raises():
    with pytest.raises(NotImplementedError, match="ACSL"):
        TaskAlignedLossExtended(use_acsl=True, with_polygons=False, with_distance=False)


def test_use_acsl_false_ok():
    assert TaskAlignedLossExtended(
        use_acsl=False, with_polygons=False, with_distance=False
    ).use_acsl is False


def test_build_losses_passes_acsl_and_angle_step():
    from train.task import YoloV8Task
    src = inspect.getsource(YoloV8Task.build_losses)
    assert "use_acsl=loss_cfg.acsl.use_acsl" in src
    assert "angle_step=task_cfg.model.angle_step" in src
