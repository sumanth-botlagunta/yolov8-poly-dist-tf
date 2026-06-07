"""PolyYOLO polygon loss functions.

Implements the three per-vertex loss components:
    angle:  cross-entropy over 360/angle_step bins per vertex
    dist:   L1 / smooth-L1 regression of radial distance
    conf:   binary cross-entropy on vertex validity

All losses are computed only on foreground anchors and weighted
by the corresponding TAL alignment scores.

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
    target_scores_sum: tf.Tensor,
) -> tf.Tensor:
    """L1 regression loss for per-vertex radial distances.

    Args:
        pd_dist:           float32 [batch, anchors, num_vertices]
        target_dist:       float32 [batch, anchors, num_vertices]
        fg_mask:           bool    [batch, anchors]
        target_scores_sum: float32 scalar normalizer

    Returns:
        Scalar loss.
    """
    # CONVENTION: this SUMS the L1 error over the V=24 vertices (reduce_sum), whereas
    # polygon_angle_loss AVERAGES over vertices (reduce_mean). dist/conf are therefore
    # ~24× larger than angle before gains; the configured poly gains (dist=0.45,
    # angle=0.4, conf=0.2) bake in that factor. Changing the vertex count rescales this
    # loss — re-check the gains if you do. Tracked for unification; see plan Part 2.4.
    fg_float = tf.cast(fg_mask, tf.float32)[:, :, tf.newaxis]    # [B, A, 1]
    l1 = tf.abs(pd_dist - target_dist) * fg_float                 # [B, A, V]
    return tf.reduce_sum(l1) / target_scores_sum


def polygon_conf_loss(
    pd_conf: tf.Tensor,
    target_conf: tf.Tensor,
    fg_mask: tf.Tensor,
    target_scores_sum: tf.Tensor,
) -> tf.Tensor:
    """BCE loss for per-vertex validity confidence.

    Args:
        pd_conf:           float32 [batch, anchors, num_vertices]  logits
        target_conf:       float32 [batch, anchors, num_vertices]  0 or 1
        fg_mask:           bool    [batch, anchors]
        target_scores_sum: float32 scalar normalizer

    Returns:
        Scalar loss.
    """
    # CONVENTION: SUMS BCE over the V=24 vertices (like polygon_dist_loss, unlike the
    # mean used by polygon_angle_loss). See the note in polygon_dist_loss.
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_conf, logits=pd_conf
    )  # [B, A, V]
    fg_float = tf.cast(fg_mask, tf.float32)[:, :, tf.newaxis]    # [B, A, 1]
    return tf.reduce_sum(bce * fg_float) / target_scores_sum
