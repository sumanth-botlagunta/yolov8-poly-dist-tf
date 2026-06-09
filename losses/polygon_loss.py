"""PolyYOLO polygon loss functions.

Implements the three per-vertex loss components:
    angle:  BCE on the sub-bin angular offset (continuous target in [0, 1)),
            averaged over the VALID vertices of each anchor.
    dist:   L2 regression on (target - softplus(pred))^2, averaged over the
            VALID vertices of each anchor.
    conf:   binary cross-entropy on per-bin vertex validity, averaged over ALL
            bins (it must see empty bins to learn to predict 0 there).

All three normalize by num_objs (total GT object count in the batch) and are
computed only on foreground anchors.

"Valid vertex" = a bin that received a ground-truth vertex; supplied as a
per-bin mask (vertex_mask). dist and angle only carry a meaningful target on
those bins, so they average over the valid count, not over all 24 bins.

Functions:
    polygon_angle_loss: sub-bin angular-offset BCE (masked to valid vertices).
    polygon_dist_loss:  radial distance L2 (masked to valid vertices).
    polygon_conf_loss:  per-bin vertex confidence BCE (over all bins).
"""

import tensorflow as tf


def _masked_vertex_mean(per_vertex: tf.Tensor, vertex_mask: tf.Tensor) -> tf.Tensor:
    """Mean of a [B, A, V] per-vertex tensor over the VALID vertices only.

    Returns [B, A]. Anchors with no valid vertex divide by 1 (their masked
    numerator is 0, so they contribute 0) — avoids NaN.
    """
    m = tf.cast(vertex_mask, per_vertex.dtype)            # [B, A, V]
    num_valid = tf.maximum(tf.reduce_sum(m, axis=-1), 1.0)  # [B, A]
    return tf.reduce_sum(per_vertex * m, axis=-1) / num_valid


def polygon_angle_loss(
    pd_angle: tf.Tensor,
    target_angle: tf.Tensor,
    vertex_mask: tf.Tensor,
    fg_mask: tf.Tensor,
    num_objs: tf.Tensor,
) -> tf.Tensor:
    """BCE on the per-vertex sub-bin angular offset, over valid vertices.

    target_angle is the continuous offset (vertex_angle - bin_start) / angle_step
    in [0, 1); BCE(sigmoid(pred), target) drives sigmoid(pred) toward it.
    Averaged over the valid vertices of each anchor, summed over foreground
    anchors, normalized by num_objs.

    Args:
        pd_angle:     float32 [batch, anchors, num_vertices]  logits
        target_angle: float32 [batch, anchors, num_vertices]  offset in [0, 1)
        vertex_mask:  float32 [batch, anchors, num_vertices]  1.0 on valid bins
        fg_mask:      bool    [batch, anchors]
        num_objs:     float32 scalar  total valid GT object count in batch

    Returns:
        Scalar loss.
    """
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_angle, logits=pd_angle
    )  # [B, A, V]
    per_anchor = _masked_vertex_mean(bce, vertex_mask)   # [B, A]
    fg_float = tf.cast(fg_mask, tf.float32)
    return tf.reduce_sum(per_anchor * fg_float) / num_objs


def polygon_dist_loss(
    pd_dist: tf.Tensor,
    target_dist: tf.Tensor,
    vertex_mask: tf.Tensor,
    fg_mask: tf.Tensor,
    num_objs: tf.Tensor,
) -> tf.Tensor:
    """L2 regression loss for per-vertex radial distances, over valid vertices.

    Applies softplus to the prediction before computing (target - softplus(pred))^2,
    averages over the VALID vertices of each anchor (so empty bins do not dilute
    the mean), sums over foreground anchors, and normalizes by num_objs.

    Args:
        pd_dist:     float32 [batch, anchors, num_vertices]  raw predicted distances
        target_dist: float32 [batch, anchors, num_vertices]  target radial distances
        vertex_mask: float32 [batch, anchors, num_vertices]  1.0 on valid bins
        fg_mask:     bool    [batch, anchors]
        num_objs:    float32 scalar  total valid GT object count in batch

    Returns:
        Scalar loss.
    """
    l2 = tf.square(target_dist - tf.math.softplus(pd_dist))   # [B, A, V]
    per_anchor = _masked_vertex_mean(l2, vertex_mask)         # [B, A]
    fg_float = tf.cast(fg_mask, tf.float32)
    return tf.reduce_sum(per_anchor * fg_float) / num_objs


def polygon_conf_loss(
    pd_conf: tf.Tensor,
    target_conf: tf.Tensor,
    fg_mask: tf.Tensor,
    num_objs: tf.Tensor,
) -> tf.Tensor:
    """BCE loss for per-vertex validity confidence, over ALL bins.

    Averages BCE over all V=24 bins per anchor (NOT masked: the conf head must
    learn to output 0 on empty bins), sums over foreground anchors, and
    normalizes by num_objs.

    Args:
        pd_conf:     float32 [batch, anchors, num_vertices]  logits
        target_conf: float32 [batch, anchors, num_vertices]  0 or 1
        fg_mask:     bool    [batch, anchors]
        num_objs:    float32 scalar  total valid GT object count in batch

    Returns:
        Scalar loss.
    """
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_conf, logits=pd_conf
    )  # [B, A, V]
    bce_mean = tf.reduce_mean(bce, axis=-1)    # [B, A] — mean over all V bins
    fg_float = tf.cast(fg_mask, tf.float32)    # [B, A]
    return tf.reduce_sum(bce_mean * fg_float) / num_objs
