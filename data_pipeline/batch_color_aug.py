"""Vectorized, per-batch colour augmentation for the YOLOv8 data pipeline.

The colour pipeline runs once per batch inside the compiled ``train_step`` (on
the GPU, off the CPU-capped tf.data workers). The parsers emit uint8 images and
carry uint8 through batching; this module casts /255 and applies HSV +
albumentations across the batch.

The per-image randomness distribution matches a per-sample application exactly:

  * HSV: a separate 3-vector of gains is drawn per image (``[B, 3]``), applied
    with the PyTorch-YOLO formulation — multiplicative gains in the quantized
    [180, 255, 255] HSV domain (see ``apply_hsv_gains``); identical math to the
    per-sample ``hsv_augment`` in ``data_pipeline.augmentations``.
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

_HSV_SCALE = (180.0, 255.0, 255.0)


def apply_hsv_gains(
    images: tf.Tensor,
    r: tf.Tensor,
) -> tf.Tensor:
    """Apply per-image HSV gains, PyTorch-YOLO form (quantized domain).

    Reference math, vectorized over the batch::

        x = floor(rgb_to_hsv(images) * [180, 255, 255])
        x = floor(x * r)                       # r broadcasts per image
        h %= 180; s, v clipped to [0, 255]
        images = hsv_to_rgb(x / [180, 255, 255])

    All three channels — including hue — are scaled multiplicatively.

    Args:
        images: float32 [B, H, W, 3] in [0, 1].
        r:      float32 [B, 3] per-image (hue, sat, val) gains around 1.0.

    Returns:
        float32 [B, H, W, 3] in [0, 1].
    """
    scale = tf.constant(_HSV_SCALE, tf.float32)
    x = tf.image.rgb_to_hsv(images)                      # [B, H, W, 3]
    x = tf.math.floor(x * scale)
    x = tf.math.floor(x * r[:, tf.newaxis, tf.newaxis, :])
    h = x[..., 0] % 180.0
    s = tf.clip_by_value(x[..., 1], 0.0, 255.0)
    v = tf.clip_by_value(x[..., 2], 0.0, 255.0)
    x = tf.stack([h, s, v], axis=-1) / scale
    return tf.image.hsv_to_rgb(x)


def batch_hsv_augment(
    images: tf.Tensor,
    hue: float = 0.015,
    sat: float = 0.7,
    val: float = 0.4,
) -> tf.Tensor:
    """Draw per-image HSV gains and apply them across the batch.

    Gain draw (per image): ``r = 1 + U(-1, 1) · [hue, sat, val]`` — the same
    ranges and multiplicative semantics as the per-sample ``hsv_augment``
    (augmentations.py).

    Args:
        images: float32 [B, H, W, 3] in [0, 1].
        hue:    hue gain half-range.
        sat:    saturation gain half-range.
        val:    brightness (value) gain half-range.

    Returns:
        float32 [B, H, W, 3] in [0, 1]. Returns the input unchanged when all
        three components are disabled.
    """
    if hue <= 0.0 and sat <= 0.0 and val <= 0.0:
        return images

    b = tf.shape(images)[0]
    gen_range = tf.constant([hue, sat, val], tf.float32)
    r = tf.random.uniform([b, 3], -1.0, 1.0) * gen_range + 1.0
    return apply_hsv_gains(images, r)


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
