"""Vectorized, per-batch colour augmentation for the YOLOv8 data pipeline.

The colour pipeline runs once per batch inside the compiled ``train_step`` (on
the GPU, off the CPU-capped tf.data workers). The parsers emit uint8 images and
carry uint8 through batching; this module casts /255 and applies HSV +
albumentations across the batch.

The per-image randomness distribution matches a per-sample application exactly:

  * HSV: a separate (dh, ds, dv) triple is drawn per image (``[B]`` vectors), as
    in per-image ``tf.image.random_hue/saturation/brightness``. Hue and
    saturation are fused into one rgb→hsv→rgb round trip, which is mathematically
    identical (see ``apply_hsv_deltas``).
  * Albumentations: the per-image enter gate (u < freq) and the four independent
    p=0.01 transform gates are drawn per image, in the same order as the
    per-sample ``apply_albumentations`` ``tf.cond`` chain, so the induced
    distribution is identical.

``_box_blur_tf`` is ported to batch form from
``data_pipeline.augmentations._box_blur_tf`` (kept local rather than imported so
this module stands alone).
"""

from __future__ import annotations

import tensorflow as tf


# ---------------------------------------------------------------------------
# HSV jitter (per-image deltas applied to the whole batch in one pass)
# ---------------------------------------------------------------------------

def apply_hsv_deltas(
    images: tf.Tensor,
    dh: tf.Tensor,
    ds: tf.Tensor,
    dv: tf.Tensor,
) -> tf.Tensor:
    """Apply per-image HSV deltas to a float batch in one rgb→hsv→rgb round trip.

    This is mathematically identical to applying, per image,
    ``tf.image.adjust_hue(img, dh) → adjust_saturation(img, ds) →
    adjust_brightness(img, dv) → clip(0, 1)`` — adjust_hue adds dh to the H
    channel (mod 1), adjust_saturation multiplies the S channel (TF clips S to
    [0, 1] internally), adjust_brightness adds dv to RGB. Fusing hue and
    saturation into a single hsv round trip (instead of two) changes nothing
    because both operate on the same HSV representation.

    Args:
        images: float32 [B, H, W, 3] in [0, 1].
        dh:     float32 [B] additive hue delta (fraction of the hue circle).
        ds:     float32 [B] multiplicative saturation gain.
        dv:     float32 [B] additive brightness delta (applied on RGB).

    Returns:
        float32 [B, H, W, 3] in [0, 1].
    """
    hsv = tf.image.rgb_to_hsv(images)            # [B, H, W, 3]
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h = tf.math.floormod(h + dh[:, tf.newaxis, tf.newaxis], 1.0)
    s = tf.clip_by_value(s * ds[:, tf.newaxis, tf.newaxis], 0.0, 1.0)

    rgb = tf.image.hsv_to_rgb(tf.stack([h, s, v], axis=-1))
    rgb = rgb + dv[:, tf.newaxis, tf.newaxis, tf.newaxis]
    return tf.clip_by_value(rgb, 0.0, 1.0)


def batch_hsv_augment(
    images: tf.Tensor,
    hue: float = 0.015,
    sat: float = 0.7,
    val: float = 0.4,
) -> tf.Tensor:
    """Draw per-image HSV deltas and apply them across the batch.

    Mirrors the per-sample ``hsv_augment`` (augmentations.py) draw ranges:
        dh ~ U(-hue, hue)              (hue == 0 → no-op, dh = 0)
        ds ~ U(max(0, 1-sat), 1+sat)   (sat == 0 → no-op, ds = 1)
        dv ~ U(-val, val)              (val == 0 → no-op, dv = 0)

    Args:
        images: float32 [B, H, W, 3] in [0, 1].
        hue:    max hue delta magnitude.
        sat:    saturation range half-width (gain ∈ [1-sat, 1+sat]).
        val:    max brightness delta magnitude.

    Returns:
        float32 [B, H, W, 3] in [0, 1]. Returns the input unchanged when all
        three components are disabled.
    """
    if hue <= 0.0 and sat <= 0.0 and val <= 0.0:
        return images

    b = tf.shape(images)[0]
    if hue > 0.0:
        dh = tf.random.uniform([b], -hue, hue)
    else:
        dh = tf.zeros([b], dtype=images.dtype)
    if sat > 0.0:
        sat_lower = max(0.0, 1.0 - sat)
        sat_upper = 1.0 + sat
        ds = tf.random.uniform([b], sat_lower, sat_upper)
    else:
        ds = tf.ones([b], dtype=images.dtype)
    if val > 0.0:
        dv = tf.random.uniform([b], -val, val)
    else:
        dv = tf.zeros([b], dtype=images.dtype)

    return apply_hsv_deltas(images, dh, ds, dv)


# ---------------------------------------------------------------------------
# Albumentations-style colour/filter transforms (batched, mask-gated)
# ---------------------------------------------------------------------------

def _box_blur_batch(images: tf.Tensor, kernel_size: int) -> tf.Tensor:
    """Separable box blur via depthwise conv2d over a whole batch.

    Batch form of ``data_pipeline.augmentations._box_blur_tf`` (copied + adapted;
    that file's helper is private and intentionally not imported here).

    Args:
        images:      float32 [B, H, W, 3] in [0, 1].
        kernel_size: odd box kernel side length.

    Returns:
        float32 [B, H, W, 3].
    """
    k = tf.cast(kernel_size, tf.float32)
    k_h = tf.ones([kernel_size, 1, 3, 1], dtype=tf.float32) / k
    k_w = tf.ones([1, kernel_size, 3, 1], dtype=tf.float32) / k
    out = tf.nn.depthwise_conv2d(images, k_w, strides=[1, 1, 1, 1], padding='SAME')
    out = tf.nn.depthwise_conv2d(out, k_h, strides=[1, 1, 1, 1], padding='SAME')
    return out


def apply_albumentations_masks(
    images: tf.Tensor,
    m_blur: tf.Tensor,
    m_median: tf.Tensor,
    m_gray: tf.Tensor,
    m_clahe: tf.Tensor,
) -> tf.Tensor:
    """Apply the four albumentations transforms, each gated by a per-image mask.

    Transforms are applied in the same order as the per-sample
    ``apply_albumentations`` ``tf.cond`` chain, so each transform sees the output
    of the previous one (sequential per-image semantics):
        Blur        3×3 box blur                       (m_blur)
        MedianBlur  3×3 box blur (mean≈median at k=3)  (m_median)
        ToGray      rgb_to_grayscale tiled to 3ch      (m_gray)
        CLAHE       unsharp local-contrast boost       (m_clahe)

    Args:
        images:   float32 [B, H, W, 3] in [0, 1].
        m_blur/m_median/m_gray/m_clahe: bool [B] per-image apply masks.

    Returns:
        float32 [B, H, W, 3] in [0, 1].
    """
    def _sel(mask, transformed, x):
        return tf.where(mask[:, tf.newaxis, tf.newaxis, tf.newaxis], transformed, x)

    x = images
    # Blur (p=0.01)
    x = _sel(m_blur, _box_blur_batch(x, 3), x)
    # MedianBlur (p=0.01) — 3×3 box blur ≈ median at this scale
    x = _sel(m_median, _box_blur_batch(x, 3), x)
    # ToGray (p=0.01)
    gray = tf.tile(tf.image.rgb_to_grayscale(x), [1, 1, 1, 3])
    x = _sel(m_gray, gray, x)
    # CLAHE (p=0.01) — local contrast boost via unsharp mask at tile scale (~33px).
    # The 33×33 box blur is the most expensive op here, yet m_clahe is true with
    # probability ~0.01·freq, so for almost every batch no image is selected. Guard
    # the CLAHE branch behind tf.cond(reduce_any(m_clahe)) so the common (all-False)
    # case skips the 33-px blur. The result is identical to the unconditional form
    # (tf.where with an all-False mask returns x); this only avoids wasted compute.
    def _apply_clahe():
        local_mean = _box_blur_batch(x, 33)
        clahe = tf.clip_by_value(x + 0.5 * (x - local_mean), 0.0, 1.0)
        return _sel(m_clahe, clahe, x)

    x = tf.cond(tf.reduce_any(m_clahe), _apply_clahe, lambda: x)
    return x


def batch_albumentations(
    images: tf.Tensor,
    freq: float,
    row_mask: tf.Tensor,
) -> tf.Tensor:
    """Draw per-image albumentations gates and apply the transforms.

    Reproduces the per-image distribution of the per-sample
    ``apply_albumentations`` (augmentations.py): each image first draws an
    enter gate ``u < freq``; on entry it draws four independent ``u < 0.01``
    gates (Blur, MedianBlur, ToGray, CLAHE) in that order. Rows excluded by
    ``row_mask`` never enter (the distance stream gets no albumentations).

    Args:
        images:   float32 [B, H, W, 3] in [0, 1].
        freq:     probability of entering the albumentations pipeline per image.
        row_mask: bool [B]; rows where albumentations is allowed (e.g. detection
                  rows, ignore_bg == 0).

    Returns:
        float32 [B, H, W, 3] in [0, 1]. Returns the input unchanged when
        freq <= 0.
    """
    if freq <= 0.0:
        return images

    b = tf.shape(images)[0]
    enter = (tf.random.uniform([b]) < freq) & row_mask
    # Four independent p=0.01 gates, drawn in the same order as the per-sample chain.
    m_blur = enter & (tf.random.uniform([b]) < 0.01)
    m_median = enter & (tf.random.uniform([b]) < 0.01)
    m_gray = enter & (tf.random.uniform([b]) < 0.01)
    m_clahe = enter & (tf.random.uniform([b]) < 0.01)
    return apply_albumentations_masks(images, m_blur, m_median, m_gray, m_clahe)


# ---------------------------------------------------------------------------
# Top-level entry point used by the training step
# ---------------------------------------------------------------------------

def batch_color_augment(
    images,
    hue: float = 0.015,
    sat: float = 0.7,
    val: float = 0.4,
    albu_freq: float = 1.0,
    albu_row_mask: tf.Tensor = None,
) -> tf.Tensor:
    """Full per-batch colour pipeline: cast→/255 → HSV → albumentations.

    Call once per batch inside the compiled training step so the colour work runs
    on the accelerator. The parsers emit uint8 and apply no colour augmentation
    themselves.

    Args:
        images:        uint8 OR float32 [B, H, W, 3]. uint8 is cast and divided
                       by 255; float input is assumed already in [0, 1].
        hue, sat, val: HSV jitter ranges (see ``batch_hsv_augment``).
        albu_freq:     albumentations enter probability per image.
        albu_row_mask: bool [B] rows allowed to receive albumentations
                       (e.g. ``ignore_bg == 0``). When None, all rows are
                       allowed.

    Returns:
        float32 [B, H, W, 3] in [0, 1].
    """
    if images.dtype == tf.uint8:
        images = tf.cast(images, tf.float32) / 255.0
    else:
        images = tf.cast(images, tf.float32)

    images = batch_hsv_augment(images, hue=hue, sat=sat, val=val)

    if albu_freq > 0.0:
        if albu_row_mask is None:
            albu_row_mask = tf.ones([tf.shape(images)[0]], dtype=tf.bool)
        images = batch_albumentations(images, freq=albu_freq, row_mask=albu_row_mask)

    return images
