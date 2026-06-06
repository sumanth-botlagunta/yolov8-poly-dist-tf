"""Standalone augmentation functions for the YOLOv8 data pipeline.

All functions operate on individual examples (not batches).
TF-native functions work in graph mode and tf.function.
The Albumentations wrapper uses tf.py_function and must run on CPU.

Polygon format expected here: [N, max_vertices] flat xy-coordinate pairs,
padded with -1 for missing vertices.  (x, y) interleaved: x0, y0, x1, y1, …
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import tensorflow as tf


# ---------------------------------------------------------------------------
# Albumentations (CPU-only via tf.py_function)
# ---------------------------------------------------------------------------

def apply_albumentations(image: tf.Tensor, freq: float = 1.0) -> tf.Tensor:
    """Apply Albumentations colour/filter transforms with probability *freq*.

    Wraps Albumentations in tf.py_function so it runs eagerly on CPU.
    Constructs the Compose pipeline inside the call to remain stateless.
    Falls back to the original image on any exception.

    Args:
        image: float32 [H, W, 3] in [0, 1].
        freq: probability of applying the transform pipeline.

    Returns:
        float32 [H, W, 3] in [0, 1].
    """

    def _aug_fn(img_np):
        import numpy as np
        try:
            import albumentations as A
            transform = A.Compose([
                A.Blur(blur_limit=3, p=0.01),
                A.MedianBlur(blur_limit=3, p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
            ])
            img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            result = transform(image=img_uint8)['image']
            return (result.astype(np.float32) / 255.0).clip(0.0, 1.0)
        except Exception:
            return img_np

    do_aug = tf.random.uniform([]) < freq
    image = tf.cond(
        do_aug,
        lambda: tf.py_function(_aug_fn, [image], tf.float32),
        lambda: image,
    )
    image = tf.ensure_shape(image, [None, None, 3])
    return image


# ---------------------------------------------------------------------------
# Horizontal flip
# ---------------------------------------------------------------------------

def random_horizontal_flip(
    image: tf.Tensor,
    boxes: tf.Tensor,
    polygons: tf.Tensor,
    max_vertices: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Flip image left-right with 50% probability.

    Boxes (yxyx normalized): xmin ↔ 1 − xmax.
    Polygons (flat xy pairs, -1 padded): x ↔ 1 − x for valid vertices.

    Args:
        image:    uint8 or float32 [H, W, 3].
        boxes:    float32 [N, 4] yxyx normalized.
        polygons: float32 [N, max_vertices] flat xy pairs, -1 padded.
        max_vertices: static column count of *polygons*.

    Returns:
        (image, boxes, polygons) – possibly flipped.
    """
    do_flip = tf.random.uniform([]) > 0.5

    image = tf.cond(
        do_flip,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )

    # Flip x coordinates of boxes: xmin_new = 1 - xmax, xmax_new = 1 - xmin
    ymin, xmin, ymax, xmax = (
        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    )
    boxes_flipped = tf.stack([ymin, 1.0 - xmax, ymax, 1.0 - xmin], axis=1)
    boxes = tf.cond(do_flip, lambda: boxes_flipped, lambda: boxes)

    # Flip x coords of polygons: x_new = 1 - x  (y unchanged, -1 padding kept)
    N = tf.shape(polygons)[0]
    pts = tf.reshape(polygons, [N, max_vertices // 2, 2])  # [N, n_pairs, (x, y)]
    valid_x = pts[:, :, 0] >= 0.0  # [N, n_pairs]
    x_flipped = tf.where(valid_x, 1.0 - pts[:, :, 0], pts[:, :, 0])
    pts_flipped = tf.stack([x_flipped, pts[:, :, 1]], axis=-1)
    poly_flipped = tf.reshape(pts_flipped, [N, max_vertices])
    polygons = tf.cond(do_flip, lambda: poly_flipped, lambda: polygons)

    return image, boxes, polygons


# ---------------------------------------------------------------------------
# Random affine (scale + translate letterbox)
# ---------------------------------------------------------------------------

def random_affine(
    image: tf.Tensor,
    boxes: tf.Tensor,
    polygons: tf.Tensor,
    translate: float = 0.1,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    output_size: Optional[List[int]] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Random scale and translate with gray letterbox fill.

    Uses tf.image.crop_and_resize to apply the transform in one pass.
    Out-of-bounds regions are filled with gray (114).

    Args:
        image:    uint8 [H, W, 3].
        boxes:    float32 [N, 4] yxyx normalized.
        polygons: float32 [N, max_vertices] flat xy pairs, -1 padded.
        translate: max translation as fraction of output size.
        scale_min / scale_max: random scale range.
        output_size: [H_out, W_out]. Defaults to input size.

    Returns:
        (image, boxes, polygons) after transform.
    """
    h_in = tf.shape(image)[0]
    w_in = tf.shape(image)[1]

    if output_size is None:
        h_out = h_in
        w_out = w_in
    else:
        h_out, w_out = output_size[0], output_size[1]

    # Random scale and translate
    s = tf.random.uniform([], scale_min, scale_max)
    ty = tf.random.uniform([], -translate, translate)
    tx = tf.random.uniform([], -translate, translate)

    # Crop region in normalised INPUT coordinates that maps to the output.
    # Output pixel (y, x) ↦ input pixel ((y/H − 0.5 − ty)/s + 0.5, ...)
    # Equivalently the crop box in input normalised coords is:
    y_start = 0.5 + ty - 0.5 / s
    y_end   = 0.5 + ty + 0.5 / s
    x_start = 0.5 + tx - 0.5 / s
    x_end   = 0.5 + tx + 0.5 / s

    # Resize via crop_and_resize (extrapolation_value=114 fills letterbox)
    image_f = tf.cast(image, tf.float32)
    crop_box = tf.reshape(tf.stack([y_start, x_start, y_end, x_end]), [1, 4])
    image_out = tf.image.crop_and_resize(
        image_f[tf.newaxis],
        crop_box,
        [0],
        [h_out, w_out],
        extrapolation_value=114.0,
    )  # [1, H_out, W_out, 3]
    image_out = tf.cast(tf.squeeze(image_out, 0), tf.uint8)
    image_out.set_shape([None, None, 3])

    # Transform boxes: y_out = (y_in − y_start) / (y_end − y_start)
    dy_range = y_end - y_start
    dx_range = x_end - x_start

    ymin_out = (boxes[:, 0] - y_start) / dy_range
    xmin_out = (boxes[:, 1] - x_start) / dx_range
    ymax_out = (boxes[:, 2] - y_start) / dy_range
    xmax_out = (boxes[:, 3] - x_start) / dx_range
    boxes_out = tf.stack([ymin_out, xmin_out, ymax_out, xmax_out], axis=1)

    # Transform polygons: same linear mapping
    N = tf.shape(polygons)[0]
    max_v = tf.shape(polygons)[1]
    pts = tf.reshape(polygons, [N, max_v // 2, 2])  # [N, n_pairs, (x, y)]

    valid_x = pts[:, :, 0] >= 0.0  # [N, n_pairs]

    x_out = (pts[:, :, 0] - x_start) / dx_range
    y_out = (pts[:, :, 1] - y_start) / dy_range
    x_out = tf.where(valid_x, x_out, tf.fill(tf.shape(x_out), -1.0))
    y_out = tf.where(valid_x, y_out, tf.fill(tf.shape(y_out), -1.0))

    pts_out = tf.stack([x_out, y_out], axis=-1)
    polygons_out = tf.reshape(pts_out, [N, max_v])

    return image_out, boxes_out, polygons_out


# ---------------------------------------------------------------------------
# Clip boxes (after affine some boxes may extend outside [0, 1])
# ---------------------------------------------------------------------------

def clip_boxes(
    boxes: tf.Tensor,
    min_side: float = 0.005,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Clip boxes to [0, 1] and return a validity mask.

    Args:
        boxes:    float32 [N, 4] yxyx.
        min_side: minimum side length (normalised) to keep a box.

    Returns:
        (clipped_boxes [N, 4], keep_mask [N] bool)
    """
    boxes_clipped = tf.clip_by_value(boxes, 0.0, 1.0)
    h = boxes_clipped[:, 2] - boxes_clipped[:, 0]
    w = boxes_clipped[:, 3] - boxes_clipped[:, 1]
    keep = tf.logical_and(h >= min_side, w >= min_side)
    return boxes_clipped, keep


def clip_polygon_coords(polygons: tf.Tensor) -> tf.Tensor:
    """Clip polygon xy values to [0, 1], preserving -1 padding.

    Args:
        polygons: float32 [N, max_vertices] flat xy pairs, -1 padded.

    Returns:
        float32 [N, max_vertices].
    """
    valid = polygons >= 0.0
    clipped = tf.clip_by_value(polygons, 0.0, 1.0)
    return tf.where(valid, clipped, polygons)


# ---------------------------------------------------------------------------
# HSV augmentation
# ---------------------------------------------------------------------------

def hsv_augment(
    image: tf.Tensor,
    hue: float = 0.015,
    sat: float = 0.7,
    val: float = 0.4,
) -> tf.Tensor:
    """Random HSV jitter.

    Args:
        image: float32 [H, W, 3] in [0, 1].
        hue:   max hue delta (fraction of full hue circle).
        sat:   saturation multiplicative range: [1−sat, 1+sat].
        val:   brightness additive range: ±val.

    Returns:
        float32 [H, W, 3] in [0, 1].
    """
    if hue > 0.0:
        image = tf.image.random_hue(image, hue)
    if sat > 0.0:
        lower = max(0.0, 1.0 - sat)
        upper = 1.0 + sat
        image = tf.image.random_saturation(image, lower, upper)
    if val > 0.0:
        image = tf.image.random_brightness(image, val)
    return tf.clip_by_value(image, 0.0, 1.0)
