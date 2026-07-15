"""PolyYOLO per-vertex polygon loss components.

    angle: BCE on the sub-bin angular offset (continuous target in [0, 1)),
           averaged over the valid vertices of each anchor.
    dist:  L2 on (target - softplus(pred))^2, averaged over the valid vertices.
    conf:  BCE on per-bin vertex validity, averaged over all 24 bins (occupied
           -> 1, empty -> 0). Conf is the decode gate, so empty bins need a
           negative target or their confidence drifts past the 0.4 threshold.

All three normalize by num_objs (total GT object count in the batch) and run on
foreground anchors only. A valid vertex is a bin holding a GT vertex, supplied as
vertex_mask; angle/dist average over the valid count (their targets are undefined
on empty bins), conf averages over all bins.
"""

import tensorflow as tf


def _masked_vertex_mean(per_vertex: tf.Tensor, vertex_mask: tf.Tensor) -> tf.Tensor:
    """Mean of a [B, A, V] per-vertex tensor over the valid vertices only.

    Returns [B, A]. Anchors with no valid vertex divide by 1 (their masked
    numerator is 0, so they contribute 0), avoiding NaN.
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
    in [0, 1); BCE(sigmoid(pred), target) drives sigmoid(pred) toward it. Averaged
    over the valid vertices of each anchor, summed over foreground anchors,
    normalized by num_objs.

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
    averages over the valid vertices of each anchor (so empty bins do not dilute
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


def polygon_angle_mae(
    pd_angle: tf.Tensor,
    target_angle: tf.Tensor,
    vertex_mask: tf.Tensor,
    fg_mask: tf.Tensor,
) -> tf.Tensor:
    """Diagnostic mean |sigmoid(pred) - target| over valid vertices of fg anchors.

    Not a training loss — a TensorBoard instrument. The BCE angle loss carries a
    large irreducible entropy floor (BCE of a continuous target is nonzero even at
    a perfect prediction), so its curve looks flat while the head learns; this MAE
    floors at 0 and reads ~0.25 at an untrained head (sigmoid ~ 0.5 vs ~uniform
    targets). Averaged per anchor over valid vertices, then over foreground anchors
    (not summed / num_objs, so the value stays in [0, ~0.5] regardless of
    anchors-per-GT).

    Args:
        pd_angle:     float32 [batch, anchors, num_vertices]  logits
        target_angle: float32 [batch, anchors, num_vertices]  offset in [0, 1)
        vertex_mask:  float32 [batch, anchors, num_vertices]  1.0 on valid bins
        fg_mask:      bool    [batch, anchors]

    Returns:
        Scalar diagnostic value.
    """
    err = tf.abs(tf.sigmoid(pd_angle) - target_angle)      # [B, A, V]
    per_anchor = _masked_vertex_mean(err, vertex_mask)     # [B, A]
    fg_float = tf.cast(fg_mask, tf.float32)
    n_fg = tf.maximum(tf.reduce_sum(fg_float), 1.0)
    return tf.reduce_sum(per_anchor * fg_float) / n_fg


def polygon_conf_loss(
    pd_conf: tf.Tensor,
    target_conf: tf.Tensor,
    vertex_mask: tf.Tensor,
    fg_mask: tf.Tensor,
    num_objs: tf.Tensor,
) -> tf.Tensor:
    """BCE loss for per-vertex validity confidence, over all bins.

    Sums BCE over all ``num_vertices`` bins of each anchor (occupied bins get
    target 1, empty bins get target 0) and divides by that anchor's VALID
    vertex count (the reference normalization: an object with few vertices
    weighs its conf bins more strongly than ``/num_bins`` would), then sums
    over foreground anchors and normalizes by num_objs. Unlike angle/dist
    (which stay masked — their regression targets are undefined on empty
    bins), conf must see negatives: it is the gate that tells decode/viz which
    bins to keep, so empty bins need a 0 target or their confidence drifts
    past the 0.4 threshold and produces spiky polygons.

    Args:
        pd_conf:     float32 [batch, anchors, num_vertices]  logits
        target_conf: float32 [batch, anchors, num_vertices]  0 or 1
        vertex_mask: float32 [batch, anchors, num_vertices]  1.0 on valid bins
            (the per-anchor normalizer; anchors with zero valid vertices
            contribute zero via divide_no_nan)
        fg_mask:     bool    [batch, anchors]
        num_objs:    float32 scalar  total valid GT object count in batch

    Returns:
        Scalar loss.
    """
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_conf, logits=pd_conf
    )  # [B, A, V]
    m = tf.cast(vertex_mask, bce.dtype)
    num_valid = tf.reduce_sum(m, axis=-1)                # [B, A]
    per_anchor = tf.math.divide_no_nan(
        tf.reduce_sum(bce, axis=-1), num_valid
    )  # [B, A] — all-bins sum ÷ valid-vertex count
    fg_float = tf.cast(fg_mask, tf.float32)              # [B, A]
    return tf.reduce_sum(per_anchor * fg_float) / num_objs
