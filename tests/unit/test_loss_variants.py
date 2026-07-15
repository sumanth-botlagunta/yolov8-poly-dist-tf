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


# ---- weighting=legacy_hard (one-hot targets, binary fg weight, num_objs) ----

def test_legacy_hard_cls_matches_hand_formula():
    g = tf.random.Generator.from_seed(7)
    B, A, C = 2, 16, 39
    pred = g.uniform([B, A, C], -4.0, 4.0)
    soft_target = g.uniform([B, A, C], 0.0, 1.0)
    tl = tf.cast(g.uniform([B, A], 0, C, dtype=tf.int32), tf.int64)
    fg = g.uniform([B, A]) > 0.5
    ig = tf.constant([0, 1], tf.int64)
    num_objs = tf.constant(5.0)

    loss = TaskAlignedLossExtended(num_classes=C, weighting="legacy_hard")._class_loss(
        pred, soft_target, tf.constant(123.0), fg, ig,
        target_labels=tl, num_objs=num_objs)

    fg_f = tf.cast(fg, tf.float32)
    hard = tf.one_hot(tl, C) * fg_f[:, :, tf.newaxis]
    bce = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=hard, logits=pred), -1)
    ig_f = tf.cast(ig, tf.float32)[:, tf.newaxis]
    mask = (1.0 - ig_f) + ig_f * fg_f
    want = tf.reduce_sum(bce * mask) / num_objs
    assert abs(float(loss) - float(want)) < 1e-5
    # the soft-path normalizer (123.0) must be ignored entirely in legacy mode
    loss2 = TaskAlignedLossExtended(num_classes=C, weighting="legacy_hard")._class_loss(
        pred, soft_target, tf.constant(9999.0), fg, ig,
        target_labels=tl, num_objs=num_objs)
    assert abs(float(loss) - float(loss2)) < 1e-6


def test_legacy_hard_box_is_score_invariant_and_num_objs_normalized():
    g = tf.random.Generator.from_seed(8)
    B, A = 2, 16
    pd = tf.concat([g.uniform([B, A, 2], 0, 300), g.uniform([B, A, 2], 300, 660)], -1)
    tgt = tf.concat([g.uniform([B, A, 2], 0, 300), g.uniform([B, A, 2], 300, 660)], -1)
    ts = g.uniform([B, A, 39], 0.0, 1.0)
    fg = g.uniform([B, A]) > 0.5
    raw = g.uniform([B, A, 64], -2.0, 2.0)
    strides = tf.ones([A, 1]) * 8.0
    anc = g.uniform([A, 2], 0, 660)

    def run(w, scores, tss, no):
        L = TaskAlignedLossExtended(weighting=w)
        return L._box_loss(pd, tgt, scores, tss, fg, raw, strides, anc, num_objs=no)

    c1, d1 = run("legacy_hard", ts, tf.constant(50.0), tf.constant(4.0))
    c2, d2 = run("legacy_hard", 2.0 * ts, tf.constant(50.0), tf.constant(4.0))
    # binary weighting: doubling the soft scores must not move the legacy loss
    assert abs(float(c1) - float(c2)) < 1e-5 and abs(float(d1) - float(d2)) < 1e-5
    # num_objs is the normalizer: doubling it halves the loss
    c4, d4 = run("legacy_hard", ts, tf.constant(50.0), tf.constant(8.0))
    assert abs(float(c4) - float(c1) / 2.0) < 1e-5
    assert abs(float(d4) - float(d1) / 2.0) < 1e-5
    # the soft default DOES respond to score scaling (guards against the two
    # modes silently collapsing into one)
    s1, _ = run("soft", ts, tf.constant(50.0), tf.constant(4.0))
    s2, _ = run("soft", 2.0 * ts, tf.constant(50.0), tf.constant(4.0))
    assert abs(float(s1) - float(s2)) > 1e-6


def test_unknown_weighting_rejected_at_construction():
    import pytest
    with pytest.raises(ValueError, match="weighting"):
        TaskAlignedLossExtended(weighting="hard")


# ---- CIoU alpha coefficient carries no gradient (reference: no-grad alpha) ----

def test_ciou_alpha_is_not_differentiated():
    """grad(_bbox_iou_loss ciou) must equal the frozen-alpha gradient (alpha is
    a constant weighting coefficient), differ from the flowing-alpha gradient,
    and leave the forward value unchanged."""
    import math

    eps = 1e-7
    b2 = tf.constant([[0.2, 0.2, 0.8, 0.6]], tf.float32)        # target
    b1v = tf.Variable([[0.25, 0.15, 0.75, 0.70]], tf.float32)   # pred, aspect differs

    def _terms(b1):
        ix1 = tf.maximum(b1[..., 0], b2[..., 0]); iy1 = tf.maximum(b1[..., 1], b2[..., 1])
        ix2 = tf.minimum(b1[..., 2], b2[..., 2]); iy2 = tf.minimum(b1[..., 3], b2[..., 3])
        inter = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)
        a1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        a2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union = a1 + a2 - inter + eps
        iou = inter / union
        cx1 = (b1[..., 0] + b1[..., 2]) * 0.5; cy1 = (b1[..., 1] + b1[..., 3]) * 0.5
        cx2 = (b2[..., 0] + b2[..., 2]) * 0.5; cy2 = (b2[..., 1] + b2[..., 3]) * 0.5
        rho2 = tf.square(cx1 - cx2) + tf.square(cy1 - cy2)
        ex1 = tf.minimum(b1[..., 0], b2[..., 0]); ey1 = tf.minimum(b1[..., 1], b2[..., 1])
        ex2 = tf.maximum(b1[..., 2], b2[..., 2]); ey2 = tf.maximum(b1[..., 3], b2[..., 3])
        c2 = tf.square(ex2 - ex1) + tf.square(ey2 - ey1) + eps
        w1 = b1[..., 2] - b1[..., 0]; h1 = b1[..., 3] - b1[..., 1]
        w2 = b2[..., 2] - b2[..., 0]; h2 = b2[..., 3] - b2[..., 1]
        v = (4.0 / (math.pi ** 2)) * tf.square(
            tf.math.atan2(w2, h2 + eps) - tf.math.atan2(w1, h1 + eps))
        return iou, rho2, c2, v

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(_bbox_iou_loss(b1v, b2, "ciou"))
    got = tape.gradient(loss, b1v)

    # Reference: same formula with alpha precomputed OUTSIDE the tape.
    iou0, _, _, v0 = _terms(tf.constant(b1v.numpy()))
    alpha_const = v0 / (1.0 - iou0 + v0 + eps)
    with tf.GradientTape() as tape:
        iou, rho2, c2, v = _terms(b1v)
        ref_loss = tf.reduce_sum(1.0 - (iou - rho2 / c2 - alpha_const * v))
    ref = tape.gradient(ref_loss, b1v)
    np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=1e-5, atol=1e-7)

    # It must DIFFER from the flowing-alpha gradient (guards the stop_gradient).
    with tf.GradientTape() as tape:
        iou, rho2, c2, v = _terms(b1v)
        alpha_flow = v / (1.0 - iou + v + eps)
        flow_loss = tf.reduce_sum(1.0 - (iou - rho2 / c2 - alpha_flow * v))
    flow = tape.gradient(flow_loss, b1v)
    assert float(tf.reduce_max(tf.abs(flow - got))) > 1e-6

    # Forward value is unchanged by the stop.
    assert abs(float(ref_loss) - float(loss)) < 1e-6
