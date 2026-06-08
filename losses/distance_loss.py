"""Distance regression loss for the per-object distance estimation branch.

The distance head predicts log-scale distances.  Ground-truth log distances
are encoded during parsing: log_distance = log(clip(d, min_meter, max_meter)).
Invalid samples carry the sentinel value -10.0 and are excluded from loss.

Functions:
    distance_l1_loss: L1 loss on valid (non-sentinel) foreground predictions.
"""

import tensorflow as tf

INVALID_DISTANCE_SENTINEL = -10.0


def distance_l1_loss(
    pd_log_dist: tf.Tensor,
    target_log_dist: tf.Tensor,
    fg_mask: tf.Tensor,
    normalizer: tf.Tensor,
) -> tf.Tensor:
    """L1 loss on log-scale distances, masked to valid GT entries.

    A GT entry is valid when target_log_dist > INVALID_DISTANCE_SENTINEL.
    The combined validity mask is: fg_mask AND (target_log_dist > -10.0).

    Args:
        pd_log_dist:     float32 [batch, anchors, 1]   predicted log distance
        target_log_dist: float32 [batch, anchors, 1]   GT log distance
        fg_mask:         bool    [batch, anchors]       TAL foreground mask
        normalizer:      float32 scalar                 divisor (num_objs from caller)

    Returns:
        Scalar loss value.
    """
    valid = target_log_dist > INVALID_DISTANCE_SENTINEL           # [B, A, 1]
    fg_expanded = tf.expand_dims(fg_mask, axis=-1)                # [B, A, 1]
    mask = tf.cast(valid & fg_expanded, tf.float32)               # [B, A, 1]
    l1 = tf.abs(pd_log_dist - target_log_dist) * mask
    return tf.reduce_sum(l1) / normalizer
