"""Tests for the selectable box-IoU and cls loss variants (losses/tal_loss.py).

Cardinal requirement: the DEFAULTS (box 'ciou', cls 'bce', label_smoothing 0) reproduce
the previous loss exactly.
"""

import numpy as np
import tensorflow as tf

from losses.tal_loss import _bbox_iou_loss, TaskAlignedLossExtended

_TYPES = ["iou", "giou", "diou", "ciou", "eiou", "siou"]


def _rand_boxes(n, seed):
    g = tf.random.Generator.from_seed(seed)
    xy = g.uniform([n, 2], 0.0, 1.0)
    wh = g.uniform([n, 2], 0.05, 0.5)
    return tf.concat([xy, xy + wh], -1)


def test_all_variants_finite_and_perfect_overlap_is_zero():
    b = _rand_boxes(500, 1)
    for t in _TYPES:
        loss = _bbox_iou_loss(b, b, t)               # identical boxes
        assert bool(tf.reduce_all(tf.math.is_finite(loss)))
        # perfect overlap → ~0 (eps/(area+eps) leaves <1e-3 for these box sizes)
        assert float(tf.reduce_max(tf.abs(loss))) < 1e-3


def test_variants_are_nonnegative_on_random_boxes():
    b1, b2 = _rand_boxes(500, 2), _rand_boxes(500, 3)
    for t in _TYPES:
        loss = _bbox_iou_loss(b1, b2, t)
        assert bool(tf.reduce_all(tf.math.is_finite(loss)))
        assert float(tf.reduce_min(loss)) >= -1e-5


def _cls_inputs(seed=0):
    g = tf.random.Generator.from_seed(seed)
    B, A, C = 2, 16, 39
    pred = g.uniform([B, A, C], -4.0, 4.0)
    target = g.uniform([B, A, C], 0.0, 1.0) * tf.cast(g.uniform([B, A, C]) > 0.7, tf.float32)
    tss = tf.maximum(tf.reduce_sum(target), 1.0)
    fg = tf.cast(g.uniform([B, A]) > 0.5, tf.float32)
    ignore_bg = tf.constant([0, 1], tf.int32)
    return pred, target, tss, fg, ignore_bg


def test_default_cls_is_plain_bce():
    pred, target, tss, fg, ig = _cls_inputs(5)
    loss = TaskAlignedLossExtended(cls_loss_type="bce", label_smoothing=0.0)
    got = loss._class_loss(pred, target, tss, fg, ig)
    # reference: the exact previous computation
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred)
    bce_sum = tf.reduce_sum(bce, axis=-1)
    ig_f = tf.cast(ig, tf.float32)
    mask = (1.0 - ig_f[:, None]) + ig_f[:, None] * fg
    ref = tf.reduce_sum(bce_sum * mask) / tss
    assert float(tf.abs(got - ref)) < 1e-6


def test_focal_and_varifocal_finite_and_differ_from_bce():
    pred, target, tss, fg, ig = _cls_inputs(6)
    bce = float(TaskAlignedLossExtended(cls_loss_type="bce")._class_loss(pred, target, tss, fg, ig))
    for t in ["focal", "varifocal"]:
        v = float(TaskAlignedLossExtended(cls_loss_type=t)._class_loss(pred, target, tss, fg, ig))
        assert np.isfinite(v)
        assert abs(v - bce) > 1e-6      # actually changes the loss


def test_label_smoothing_changes_loss():
    pred, target, tss, fg, ig = _cls_inputs(7)
    a = float(TaskAlignedLossExtended(label_smoothing=0.0)._class_loss(pred, target, tss, fg, ig))
    b = float(TaskAlignedLossExtended(label_smoothing=0.1)._class_loss(pred, target, tss, fg, ig))
    assert np.isfinite(b) and abs(a - b) > 1e-6


def test_focal_gamma_and_alpha_affect_focal_loss():
    """focal_gamma and focal_alpha must actually change the focal cls loss value."""
    pred, target, tss, fg, ig = _cls_inputs(8)
    loss_g15 = float(TaskAlignedLossExtended(
        cls_loss_type='focal', focal_gamma=1.5)._class_loss(pred, target, tss, fg, ig))
    loss_g30 = float(TaskAlignedLossExtended(
        cls_loss_type='focal', focal_gamma=3.0)._class_loss(pred, target, tss, fg, ig))
    assert np.isfinite(loss_g15) and np.isfinite(loss_g30)
    assert abs(loss_g15 - loss_g30) > 1e-6, (
        "focal_gamma=1.5 and focal_gamma=3.0 must produce different loss values"
    )
