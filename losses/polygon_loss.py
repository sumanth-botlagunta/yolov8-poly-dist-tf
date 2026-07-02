"""PolyYOLO polygon loss functions.

Implements the three per-vertex loss components:
    angle:  BCE on the sub-bin angular offset (continuous target in [0, 1)),
            averaged over the VALID vertices of each anchor.
    dist:   L2 regression on (target - softplus(pred))^2, averaged over the
            VALID vertices of each anchor.
    conf:   binary cross-entropy on per-bin vertex validity, averaged over ALL
            bins (occupied → 1, empty → 0) — conf is the decode gate and must
            see negatives; see polygon_conf_loss for the rationale
            and the preserved masked form.

All three normalize by num_objs (total GT object count in the batch) and are
computed only on foreground anchors.

"Valid vertex" = a bin that received a ground-truth vertex; supplied as a
per-bin mask (vertex_mask). Angle/dist average over the valid count (their
regression targets are undefined on empty bins); conf averages over all bins.

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


def polygon_angle_mae(
    pd_angle: tf.Tensor,
    target_angle: tf.Tensor,
    vertex_mask: tf.Tensor,
    fg_mask: tf.Tensor,
) -> tf.Tensor:
    """Diagnostic: mean |sigmoid(pred) − target| over valid vertices of fg anchors.

    NOT a training loss — a TensorBoard instrument. The BCE angle loss carries a
    large irreducible entropy floor (BCE of a continuous target is nonzero even at
    perfect prediction), so its curve looks flat while the head is in fact
    learning. This MAE floors at 0 and reads ~0.25 at an untrained head
    (sigmoid≈0.5 vs ~uniform targets), making convergence legible. Averaged per
    anchor over valid vertices, then averaged over foreground anchors (NOT
    summed / num_objs, so the value stays in [0, ~0.5] regardless of
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
    """BCE loss for per-vertex validity confidence, over ALL bins.

    Averages BCE over all ``num_vertices`` bins of each anchor (occupied bins
    get target 1, EMPTY bins get target 0), sums over foreground anchors, and
    normalizes by num_objs. Unlike angle/dist (which stay masked — their
    regression targets are undefined on empty bins), conf MUST see negatives:
    it is the gate that tells decode/viz which bins to keep.

    Why all bins: with the masked form, empty bins received zero
    gradient ever, so their conf output drifted with the shared features (bias
    init → sigmoid ≈ 0.5, above the 0.4 decode/viz threshold) while their dist
    was equally untrained — producing the "star/spiky polygon" artifacts
    observed in validation overlays (e.g. the doorway class). Training conf on
    all 24 bins restores the negative signal so empty bins are pushed to 0.

    The previous (masked, legacy-aligned) form is preserved here;
    restore it by swapping the per_anchor line:

        # MASKED FORM — mean over the valid vertices only (vertex_mask equals
        # target_conf, so the masked BCE only ever saw positive targets and the
        # conf head was never trained to reject empty bins):
        # per_anchor = _masked_vertex_mean(bce, vertex_mask)

    Args:
        pd_conf:     float32 [batch, anchors, num_vertices]  logits
        target_conf: float32 [batch, anchors, num_vertices]  0 or 1
        vertex_mask: float32 [batch, anchors, num_vertices]  1.0 on valid bins
            (unused by the all-bins form; kept so the signature matches and the
            masked form above remains a one-line swap)
        fg_mask:     bool    [batch, anchors]
        num_objs:    float32 scalar  total valid GT object count in batch

    Returns:
        Scalar loss.
    """
    del vertex_mask  # only the preserved masked form (docstring) uses it
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_conf, logits=pd_conf
    )  # [B, A, V]
    per_anchor = tf.reduce_mean(bce, axis=-1)            # [B, A] — mean over ALL bins
    fg_float = tf.cast(fg_mask, tf.float32)              # [B, A]
    return tf.reduce_sum(per_anchor * fg_float) / num_objs
