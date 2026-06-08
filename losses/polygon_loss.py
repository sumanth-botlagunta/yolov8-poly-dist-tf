"""PolyYOLO polygon loss functions.

Implements the three per-vertex loss components:
    angle:  cross-entropy over 360/angle_step bins per vertex
    dist:   L2 regression on (target - softplus(pred))^2, mean over vertices
    conf:   binary cross-entropy on vertex validity, mean over vertices

All losses are computed only on foreground anchors.

Functions:
    polygon_angle_loss: angle bin classification loss.
    polygon_dist_loss:  radial distance regression loss.
    polygon_conf_loss:  vertex confidence loss.
"""

import tensorflow as tf


def polygon_angle_loss(
    pd_angle: tf.Tensor,
    target_angle: tf.Tensor,
    fg_mask: tf.Tensor,
    target_scores_sum: tf.Tensor,
) -> tf.Tensor:
    """Cross-entropy loss over per-vertex angle bins.

    Args:
        pd_angle:          float32 [batch, anchors, num_vertices]  logits
        target_angle:      float32 [batch, anchors, num_vertices]  one-hot
        fg_mask:           bool    [batch, anchors]
        target_scores_sum: float32 scalar normalizer

    Returns:
        Scalar loss.
    """
    # BCE per bin (independent per vertex), averaged over 24 bins — matches legacy
    # binary_crossentropy(reduction='mean'). Using reduce_sum here would be 24×
    # too large relative to the poly_angle_gain=0.4 calibration.
    ce = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=target_angle, logits=pd_angle),
        axis=-1,
    )  # [B, A]
    fg_float = tf.cast(fg_mask, tf.float32)
    return tf.reduce_sum(ce * fg_float) / target_scores_sum


def polygon_dist_loss(
    pd_dist: tf.Tensor,
    target_dist: tf.Tensor,
    fg_mask: tf.Tensor,
    num_objs: tf.Tensor,
) -> tf.Tensor:
    """L2 regression loss for per-vertex radial distances.

    Applies softplus to the prediction before computing (target - softplus(pred))^2.
    Averages over the V=24 vertices per anchor, sums over foreground anchors,
    and normalizes by num_objs (total GT count in the batch). Matches old codebase.

    Args:
        pd_dist:    float32 [batch, anchors, num_vertices]  raw predicted distances
        target_dist: float32 [batch, anchors, num_vertices] target radial distances
        fg_mask:    bool    [batch, anchors]
        num_objs:   float32 scalar  total valid GT object count in batch

    Returns:
        Scalar loss.
    """
    l2 = tf.square(target_dist - tf.math.softplus(pd_dist))   # [B, A, V]
    per_anchor = tf.reduce_mean(l2, axis=-1)                   # [B, A] — mean over V
    fg_float = tf.cast(fg_mask, tf.float32)                    # [B, A]
    return tf.reduce_sum(per_anchor * fg_float) / num_objs


def polygon_conf_loss(
    pd_conf: tf.Tensor,
    target_conf: tf.Tensor,
    fg_mask: tf.Tensor,
    num_objs: tf.Tensor,
) -> tf.Tensor:
    """BCE loss for per-vertex validity confidence.

    Averages BCE over the V=24 vertices per anchor, sums over foreground anchors,
    and normalizes by num_objs (total GT count in the batch). Matches old codebase.

    Args:
        pd_conf:    float32 [batch, anchors, num_vertices]  logits
        target_conf: float32 [batch, anchors, num_vertices]  0 or 1
        fg_mask:    bool    [batch, anchors]
        num_objs:   float32 scalar  total valid GT object count in batch

    Returns:
        Scalar loss.
    """
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_conf, logits=pd_conf
    )  # [B, A, V]
    bce_mean = tf.reduce_mean(bce, axis=-1)    # [B, A] — mean over V
    fg_float = tf.cast(fg_mask, tf.float32)    # [B, A]
    return tf.reduce_sum(bce_mean * fg_float) / num_objs
