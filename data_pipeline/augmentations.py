"""Standalone augmentation ops for the YOLOv8 data pipeline.

Each function operates on a single example (not a batch) and is a pure TF op
(graph-mode / tf.function safe; no tf.py_function or Albumentations).

Polygon format: [N, max_vertices] flat xy pairs (x0, y0, x1, y1, ...), padded
with the -1.0 sentinel for missing vertices.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import tensorflow as tf


def _box_blur_tf(image: tf.Tensor, kernel_size: int) -> tf.Tensor:
    """Separable box blur via depthwise conv2d. Input: float32 [H, W, 3] in [0, 1]."""
    k_h = tf.ones([kernel_size, 1, 3, 1], dtype=tf.float32) / tf.cast(kernel_size, tf.float32)
    k_w = tf.ones([1, kernel_size, 3, 1], dtype=tf.float32) / tf.cast(kernel_size, tf.float32)
    img4 = image[tf.newaxis]
    img4 = tf.nn.depthwise_conv2d(img4, k_w, strides=[1, 1, 1, 1], padding='SAME')
    img4 = tf.nn.depthwise_conv2d(img4, k_h, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(img4, 0)


def apply_albumentations(image: tf.Tensor, freq: float = 1.0) -> tf.Tensor:
    """Colour/filter augmentations as pure TF ops (albumentations-equivalent).

    Transforms and probabilities:
        Blur        3x3 box blur                         p=0.01
        MedianBlur  3x3 box blur (mean ~ median at k=3)  p=0.01
        ToGray      rgb_to_grayscale tiled to 3ch        p=0.01
        CLAHE       unsharp-mask local contrast boost    p=0.01

    Args:
        image: float32 [H, W, 3] in [0, 1].
        freq:  probability of entering the augmentation pipeline.

    Returns:
        float32 [H, W, 3] in [0, 1].
    """
    if freq <= 0.0:
        return image

    static_shape = image.shape

    def _augment(img0):
        # Blur (p=0.01)
        img1 = tf.cond(
            tf.random.uniform([]) < 0.01,
            lambda: _box_blur_tf(img0, 3),
            lambda: img0,
        )
        # MedianBlur (p=0.01): 3x3 box blur approximates median at this scale.
        img2 = tf.cond(
            tf.random.uniform([]) < 0.01,
            lambda: _box_blur_tf(img1, 3),
            lambda: img1,
        )
        # ToGray (p=0.01)
        img3 = tf.cond(
            tf.random.uniform([]) < 0.01,
            lambda: tf.tile(tf.image.rgb_to_grayscale(img2), [1, 1, 3]),
            lambda: img2,
        )
        # CLAHE (p=0.01): local contrast boost via unsharp mask at tile scale (~33px).
        def _clahe_approx(img):
            local_mean = _box_blur_tf(img, 33)
            return tf.clip_by_value(img + 0.5 * (img - local_mean), 0.0, 1.0)

        img4 = tf.cond(
            tf.random.uniform([]) < 0.01,
            lambda: _clahe_approx(img3),
            lambda: img3,
        )
        return img4

    do_aug = tf.random.uniform([]) < freq
    image = tf.cond(do_aug, lambda: _augment(image), lambda: image)
    image = tf.ensure_shape(image, static_shape)
    return image


def letterbox_geometry(h_in, w_in, out_h: int, out_w: int):
    """Return the letterbox placement geometry for an (h_in, w_in) image.

    Aspect-preserving fit into (out_h, out_w): the long side is scaled to fit and
    the content is centered, gray-padded to the output. Returns the scalars every
    consumer needs to map coordinates in either direction:

        scale = min(out_h/h_in, out_w/w_in)
        new_h/new_w = round(dim * scale)  (>= 1)
        pad_top  = (out_h - new_h) // 2
        pad_left = (out_w - new_w) // 2

    Args:
        h_in, w_in: source dims (python ints or scalar tensors).
        out_h, out_w: output dims (python ints).

    Returns:
        (scale, new_h, new_w, pad_top, pad_left); scale is a float32 scalar, the
        rest int32 scalars.
    """
    h_in_f = tf.cast(h_in, tf.float32)
    w_in_f = tf.cast(w_in, tf.float32)
    scale = tf.minimum(out_h / h_in_f, out_w / w_in_f)
    new_h = tf.maximum(tf.cast(tf.round(h_in_f * scale), tf.int32), 1)
    new_w = tf.maximum(tf.cast(tf.round(w_in_f * scale), tf.int32), 1)
    pad_top = (out_h - new_h) // 2
    pad_left = (out_w - new_w) // 2
    return scale, new_h, new_w, pad_top, pad_left


def letterbox_resize(
    image: tf.Tensor,
    boxes: tf.Tensor,
    polygons: tf.Tensor,
    out_h: int,
    out_w: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Letterbox-resize image to (out_h, out_w) with gray-114 padding.

    Boxes and polygon vertices are remapped through the same scale + pad so they
    stay aligned in output-normalized space; the -1.0 polygon sentinel is
    preserved. Single source of the letterbox math shared by the eval parser
    (V8ParserExtended._letterbox_resize) and the mosaic-stage pre-resize
    (input_reader), so the two cannot diverge.

    Args:
        image:    uint8 or float32 [H, W, 3].
        boxes:    float32 [N, 4] yxyx normalized to the INPUT size.
        polygons: float32 [N, max_vertices] flat xy pairs (INPUT-normalized), -1 padded.
        out_h, out_w: output size.

    Returns:
        (image uint8 [out_h, out_w, 3], boxes [N,4] output-normalized,
         polygons [N, max_vertices] output-normalized).
    """
    _scale, new_h, new_w, pad_top, pad_left = letterbox_geometry(
        tf.shape(image)[0], tf.shape(image)[1], out_h, out_w
    )

    image = tf.cast(
        tf.image.resize(tf.cast(image, tf.float32), [new_h, new_w], method='bilinear'),
        tf.uint8,
    )
    pad_bottom = out_h - new_h - pad_top
    pad_right = out_w - new_w - pad_left
    image = tf.pad(
        image,
        [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=114,
    )
    image.set_shape([out_h, out_w, 3])

    new_h_f = tf.cast(new_h, tf.float32)
    new_w_f = tf.cast(new_w, tf.float32)
    out_h_f = tf.cast(out_h, tf.float32)
    out_w_f = tf.cast(out_w, tf.float32)
    pad_top_f = tf.cast(pad_top, tf.float32)
    pad_lft_f = tf.cast(pad_left, tf.float32)

    ymin = boxes[:, 0] * new_h_f / out_h_f + pad_top_f / out_h_f
    xmin = boxes[:, 1] * new_w_f / out_w_f + pad_lft_f / out_w_f
    ymax = boxes[:, 2] * new_h_f / out_h_f + pad_top_f / out_h_f
    xmax = boxes[:, 3] * new_w_f / out_w_f + pad_lft_f / out_w_f
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

    n_inst = tf.shape(polygons)[0]
    pts = tf.reshape(polygons, [n_inst, -1, 2])       # [N, P, (x, y)]
    valid = pts[:, :, 0] > -1.0                        # reserved sentinel is exactly -1.0
    px = pts[:, :, 0] * new_w_f / out_w_f + pad_lft_f / out_w_f
    py = pts[:, :, 1] * new_h_f / out_h_f + pad_top_f / out_h_f
    neg1 = tf.fill(tf.shape(px), -1.0)
    px = tf.where(valid, px, neg1)
    py = tf.where(valid, py, neg1)
    polygons = tf.reshape(tf.stack([px, py], axis=-1), tf.shape(polygons))

    return image, boxes, polygons


_RESIZE_METHODS = ('bilinear', 'lanczos3', 'lanczos5', 'bicubic', 'gaussian',
                   'nearest', 'area', 'mitchellcubic')


def resize_with_random_method(
    image: tf.Tensor,
    new_height: tf.Tensor,
    new_width: tf.Tensor,
) -> tf.Tensor:
    """Resize with a uniformly drawn interpolation kernel.

    Draws one of the eight ``tf.image.resize`` methods per call, an implicit
    regularizer against interpolation-artifact overfitting (train-only; the
    reference pipeline applies it to mosaic-branch resizes).

    Args:
        image: [H, W, C] any numeric dtype.
        new_height: scalar int target height.
        new_width: scalar int target width.

    Returns:
        float32 [new_height, new_width, C].
    """
    image = tf.cast(image, tf.float32)
    lo = tf.reduce_min(image)
    hi = tf.reduce_max(image)
    method_index = tf.random.uniform(
        shape=[], minval=0, maxval=len(_RESIZE_METHODS), dtype=tf.int32)
    branches = [
        (tf.equal(method_index, i),
         lambda m=m: tf.image.resize(image, [new_height, new_width], method=m))
        for i, m in enumerate(_RESIZE_METHODS)
    ]
    out = tf.case(branches, exclusive=True)
    # Overshooting kernels (bicubic/lanczos/mitchellcubic) can ring past the
    # input range; a later cast to uint8 WRAPS (mod 256) rather than
    # saturating, turning overshot bright edges into near-black speckle.
    # Clamp to the input's own range so every kernel is safe to cast.
    return tf.clip_by_value(out, lo, hi)


def random_horizontal_flip(
    image: tf.Tensor,
    boxes: tf.Tensor,
    polygons: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Flip image left-right with 50% probability.

    Boxes (yxyx normalized): xmin <-> 1 - xmax.
    Polygons (flat xy pairs, -1 padded): x <-> 1 - x for valid vertices.
    The polygon width is read from the tensor itself, so any vertex count
    (raw stored width or decode-time resampled) works unchanged.

    Args:
        image:    uint8 or float32 [H, W, 3].
        boxes:    float32 [N, 4] yxyx normalized.
        polygons: float32 [N, max_vertices] flat xy pairs, -1 padded.

    Returns:
        (image, boxes, polygons), possibly flipped.
    """
    do_flip = tf.random.uniform([]) > 0.5

    image = tf.cond(
        do_flip,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )

    # Flip x coordinates of boxes: xmin_new = 1 - xmax, xmax_new = 1 - xmin.
    ymin, xmin, ymax, xmax = (
        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    )
    boxes_flipped = tf.stack([ymin, 1.0 - xmax, ymax, 1.0 - xmin], axis=1)
    boxes = tf.cond(do_flip, lambda: boxes_flipped, lambda: boxes)

    # Flip x coords of polygons: x_new = 1 - x (y unchanged, -1 padding kept).
    N = tf.shape(polygons)[0]
    max_v = tf.shape(polygons)[1]
    pts = tf.reshape(polygons, [N, -1, 2])  # [N, n_pairs, (x, y)]
    valid_x = pts[:, :, 0] > -1.0  # [N, n_pairs]; reserved sentinel is exactly -1.0
    x_flipped = tf.where(valid_x, 1.0 - pts[:, :, 0], pts[:, :, 0])
    pts_flipped = tf.stack([x_flipped, pts[:, :, 1]], axis=-1)
    poly_flipped = tf.reshape(pts_flipped, [N, max_v])
    polygons = tf.cond(do_flip, lambda: poly_flipped, lambda: polygons)

    return image, boxes, polygons


def _transform_points_px(xy_px: tf.Tensor, M: tf.Tensor) -> tf.Tensor:
    """Apply a 3x3 (input->output) matrix to [..., 2] (x, y) pixel points.

    Returns [..., 2] output pixel points (perspective divide applied).
    """
    flat = tf.reshape(xy_px, [-1, 2])                       # [P, 2]
    ones = tf.ones([tf.shape(flat)[0], 1], flat.dtype)
    hom  = tf.concat([flat, ones], axis=1)                  # [P, 3]
    out  = tf.matmul(hom, M, transpose_b=True)              # [P, 3]
    w    = out[:, 2:3]
    w    = tf.where(tf.abs(w) < 1e-12, tf.ones_like(w), w)
    out_xy = out[:, :2] / w                                 # [P, 2]
    return tf.reshape(out_xy, tf.shape(xy_px))


def make_perspective_matrix(
    h_in,
    w_in,
    target_h,
    target_w,
    degrees: float = 10.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 2.0,
    perspective: float = 0.0,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    rotate_prob: float = 1.0,
) -> tf.Tensor:
    """Build the random 3x3 (INPUT-px -> OUTPUT-px) perspective matrix.

    Matrix construction: center on (w_in/2, h_in/2) -> perspective ->
    rotation*scale -> shear -> translate-to-output-center. Fixed draw order
    (perspective px/py, rotation angle, scale gain, shear x/y, translate x/y)
    so seeded streams shift minimally when params change.

    Args:
        h_in, w_in: INPUT image dims (scalar tensors or python ints).
        target_h, target_w: OUTPUT size.
        degrees / translate / scale / shear / perspective: augmentation params.
        scale_min / scale_max: explicit scale-gain bounds; when both are given
            the gain is drawn from [scale_min, scale_max] and ``scale`` is
            ignored. Required for asymmetric config bounds (e.g. [0.4, 1.9]),
            which the symmetric ``scale`` magnitude form cannot represent.
        rotate_prob: probability rotation is applied at all. With probability
            ``1 - rotate_prob`` the angle is forced to 0 (upright output);
            otherwise it is drawn from [-degrees, degrees]. Default 1.0 always
            rotates via a single unconditional draw. Mosaic passes a small value
            so most outputs stay upright with rare +/- rotation.

    Returns:
        float32 [3, 3] input->output affine/perspective matrix M.
    """
    h_in = tf.cast(h_in, tf.float32)
    w_in = tf.cast(w_in, tf.float32)
    th_f = tf.cast(target_h, tf.float32)
    tw_f = tf.cast(target_w, tf.float32)
    deg2rad = math.pi / 180.0

    def _mat(rows):
        return tf.reshape(tf.stack([tf.cast(v, tf.float32) for v in rows]), [3, 3])

    one  = tf.constant(1.0)
    zero = tf.constant(0.0)

    # Center input on the origin.
    C = _mat([one, zero, -w_in / 2.0,
              zero, one, -h_in / 2.0,
              zero, zero, one])
    # Perspective.
    px = tf.random.uniform([], -perspective, perspective) if perspective > 0 else zero
    py = tf.random.uniform([], -perspective, perspective) if perspective > 0 else zero
    P = _mat([one, zero, zero,
              zero, one, zero,
              px,  py,  one])
    # Rotation + scale (combined). rotate_prob gates rotation: with probability
    # (1 - rotate_prob) the angle is forced to 0 (upright). rotate_prob >= 1.0
    # keeps the single unconditional draw.
    ang = tf.random.uniform([], -degrees, degrees)
    if rotate_prob < 1.0:
        ang = tf.where(tf.random.uniform([]) < rotate_prob, ang, tf.zeros([]))
    ang = ang * deg2rad
    if scale_min is not None and scale_max is not None:
        sgn = tf.random.uniform([], scale_min, scale_max)
    else:
        sgn = tf.random.uniform([], 1.0 - scale, 1.0 + scale)
    ca = tf.cos(ang) * sgn
    sa = tf.sin(ang) * sgn
    R = _mat([ca, -sa, zero,
              sa,  ca, zero,
              zero, zero, one])
    # Shear (degrees -> tan).
    shx = tf.tan(tf.random.uniform([], -shear, shear) * deg2rad)
    shy = tf.tan(tf.random.uniform([], -shear, shear) * deg2rad)
    Sh = _mat([one, shx, zero,
               shy, one, zero,
               zero, zero, one])
    # Translate to output centre + random offset.
    tx = (0.5 + tf.random.uniform([], -translate, translate)) * tw_f
    ty = (0.5 + tf.random.uniform([], -translate, translate)) * th_f
    T = _mat([one, zero, tx,
              zero, one, ty,
              zero, zero, one])

    return T @ Sh @ R @ P @ C   # input -> output


def apply_perspective_image(
    image: tf.Tensor,
    M: tf.Tensor,
    target_h: int,
    target_w: int,
) -> tf.Tensor:
    """Warp ``image`` by the input->output matrix ``M`` to (target_h, target_w).

    Uses ImageProjectiveTransformV3 with the normalized inverse map, gray-114
    CONSTANT fill, BILINEAR interpolation. Returns uint8 [target_h, target_w, 3].
    """
    image_f = tf.cast(image, tf.float32)
    th_i = tf.cast(target_h, tf.int32)
    tw_i = tf.cast(target_w, tf.int32)

    # Image warp uses the inverse (output -> input) map, normalized so M_inv[2,2]=1.
    M_inv = tf.linalg.inv(M)
    M_inv = M_inv / M_inv[2, 2]
    transforms = tf.stack([
        M_inv[0, 0], M_inv[0, 1], M_inv[0, 2],
        M_inv[1, 0], M_inv[1, 1], M_inv[1, 2],
        M_inv[2, 0], M_inv[2, 1],
    ])[tf.newaxis]   # [1, 8]

    image_out = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_f[tf.newaxis],
        transforms=transforms,
        output_shape=tf.stack([th_i, tw_i]),
        fill_value=114.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )
    image_out = tf.cast(tf.squeeze(image_out, 0), tf.uint8)
    image_out.set_shape([target_h, target_w, 3])
    return image_out


def transform_boxes_polygons(
    boxes: tf.Tensor,
    polygons: tf.Tensor,
    M: tf.Tensor,
    h_in,
    w_in,
    target_h: int,
    target_w: int,
    area_thresh: float = 0.1,
    min_side: float = 0.003,
    max_aspect_ratio: float = 20.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Transform boxes/polygons (INPUT-normalized) by ``M`` to OUTPUT-normalized.

    Boxes: transform 4 corners, re-fit AABB, clip to edge, keep by the reference
    candidate filter: visible-area fraction >= area_thresh, both sides >=
    min_side (0.003 ~ 2px at 672, the reference wh threshold), and aspect
    ratio max(h/w, w/h) < max_aspect_ratio (drops degenerate clipped slivers
    while keeping genuine partials). Polygons: transform vertices, clip to
    [0,1], keep -1 padding.

    Returns:
        (boxes_clip [N,4] normalized to OUTPUT, keep_mask [N] bool,
         polygons_out [N, max_vertices] normalized to OUTPUT).
    """
    h_in = tf.cast(h_in, tf.float32)
    w_in = tf.cast(w_in, tf.float32)
    th_f = tf.cast(target_h, tf.float32)
    tw_f = tf.cast(target_w, tf.float32)

    # ---- Boxes: transform 4 corners, re-fit AABB, clip to edge ----
    ymin, xmin, ymax, xmax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # Corners in INPUT pixels: [N, 4, 2] (x, y)
    cx = tf.stack([xmin, xmax, xmin, xmax], axis=1) * w_in   # [N, 4]
    cy = tf.stack([ymin, ymin, ymax, ymax], axis=1) * h_in   # [N, 4]
    corners = tf.stack([cx, cy], axis=-1)                    # [N, 4, 2]
    corners_out = _transform_points_px(corners, M)           # [N, 4, 2] output px
    ox = corners_out[..., 0] / tw_f                          # [N, 4] normalized
    oy = corners_out[..., 1] / th_f
    bx_min = tf.reduce_min(ox, axis=1); bx_max = tf.reduce_max(ox, axis=1)
    by_min = tf.reduce_min(oy, axis=1); by_max = tf.reduce_max(oy, axis=1)
    boxes_raw = tf.stack([by_min, bx_min, by_max, bx_max], axis=1)
    boxes_clip = tf.clip_by_value(boxes_raw, 0.0, 1.0)

    area_before = (boxes_raw[:, 2] - boxes_raw[:, 0]) * (boxes_raw[:, 3] - boxes_raw[:, 1])
    h_c = boxes_clip[:, 2] - boxes_clip[:, 0]
    w_c = boxes_clip[:, 3] - boxes_clip[:, 1]
    area_after = h_c * w_c
    # Aspect ratio on the clipped box; output is square in all configs, so the
    # normalized ratio equals the pixel ratio. eps guards the h_c/w_c=0 case
    # (those rows are already dropped by the min_side term).
    ar = tf.maximum(h_c / (w_c + 1e-9), w_c / (h_c + 1e-9))
    keep = tf.logical_and(
        tf.logical_and(area_before > 1e-9, area_after >= area_thresh * area_before),
        tf.logical_and(
            tf.logical_and(h_c >= min_side, w_c >= min_side),
            ar < max_aspect_ratio,
        ),
    )

    # ---- Polygons: transform vertices, clip to edge (keep -1 padding) ----
    N = tf.shape(polygons)[0]
    max_v = tf.shape(polygons)[1]
    pts = tf.reshape(polygons, [N, max_v // 2, 2])           # [N, P, (x, y)]
    # Source validity keys off the -1.0 sentinel: any vertex with x > -1.0 is a
    # real vertex, including a slightly-negative coordinate from mosaic-canvas
    # overflow. `> -1.0` (not `>= 0.0`) transforms + clips those to the edge
    # instead of dropping them, keeping polygon GT consistent with the box GT.
    valid = pts[:, :, 0] > -1.0                             # source validity
    pts_px = tf.stack([pts[:, :, 0] * w_in, pts[:, :, 1] * h_in], axis=-1)
    pts_out = _transform_points_px(pts_px, M)                # [N, P, 2] output px
    x_out = tf.clip_by_value(pts_out[..., 0] / tw_f, 0.0, 1.0)
    y_out = tf.clip_by_value(pts_out[..., 1] / th_f, 0.0, 1.0)
    neg1 = tf.fill(tf.shape(x_out), -1.0)
    x_out = tf.where(valid, x_out, neg1)
    y_out = tf.where(valid, y_out, neg1)
    polygons_out = tf.reshape(tf.stack([x_out, y_out], axis=-1), [N, max_v])

    return boxes_clip, keep, polygons_out


def random_perspective(
    image: tf.Tensor,
    boxes: tf.Tensor,
    polygons: tf.Tensor,
    target_h: int,
    target_w: int,
    degrees: float = 10.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 2.0,
    perspective: float = 0.0,
    area_thresh: float = 0.1,
    min_side: float = 0.003,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    rotate_prob: float = 1.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Full-affine geometric augmentation (Ultralytics random_perspective).

    Composes center -> perspective -> rotation*scale -> shear -> translate and warps
    the input to a (target_h, target_w) output (gray 114 fill on voids). When the
    input is larger than the target (e.g. a 2x mosaic canvas) this center-crops to
    the target while applying the affine; when input == target it warps in place.

    Boxes are transformed by their 4 corners then re-fit to an axis-aligned box and
    clipped to the edge (kept by visible-area fraction + min side). Polygon vertices
    are transformed and clipped to the edge (remapped onto the border, per the
    project's clip-to-edge convention); originally-padded (-1) vertices stay -1.

    Thin wrapper over ``make_perspective_matrix`` / ``apply_perspective_image`` /
    ``transform_boxes_polygons``.

    Args:
        image:    uint8 [H, W, 3].
        boxes:    float32 [N, 4] yxyx normalized to the INPUT size.
        polygons: float32 [N, max_vertices] flat xy pairs (normalized to INPUT), -1 padded.
        target_h, target_w: output size.
        degrees:  max rotation magnitude (degrees, +/-).
        translate: max translation as a fraction of the output size (+/-).
        scale:    scale-gain magnitude; the gain is drawn from [1-scale, 1+scale].
        shear:    max shear magnitude (degrees, +/-).
        perspective: max perspective coefficient (+/-); 0 disables.
        area_thresh: min visible-area fraction (after-clip / before-clip) to keep a box.
        min_side: min normalized side length to keep a box.
        scale_min / scale_max: explicit scale-gain bounds (override ``scale``);
            see ``make_perspective_matrix``.
        rotate_prob: probability rotation is applied; ``1 - rotate_prob`` of the
            time the angle is forced to 0 (upright). See ``make_perspective_matrix``.

    Returns:
        (image_out uint8 [target_h, target_w, 3], boxes_out [N,4] normalized to OUTPUT,
         keep_mask [N] bool, polygons_out [N, max_vertices] normalized to OUTPUT).
    """
    h_in = tf.shape(image)[0]
    w_in = tf.shape(image)[1]

    M = make_perspective_matrix(
        h_in=h_in, w_in=w_in,
        target_h=target_h, target_w=target_w,
        degrees=degrees, translate=translate, scale=scale,
        shear=shear, perspective=perspective,
        scale_min=scale_min, scale_max=scale_max,
        rotate_prob=rotate_prob,
    )
    image_out = apply_perspective_image(image, M, target_h, target_w)
    boxes_clip, keep, polygons_out = transform_boxes_polygons(
        boxes, polygons, M, h_in=h_in, w_in=w_in,
        target_h=target_h, target_w=target_w,
        area_thresh=area_thresh, min_side=min_side,
    )
    return image_out, boxes_clip, keep, polygons_out


# Random affine (scale + translate); superseded by random_perspective, kept for
# existing callers.

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

    # Random scale and translate.
    s = tf.random.uniform([], scale_min, scale_max)
    ty = tf.random.uniform([], -translate, translate)
    tx = tf.random.uniform([], -translate, translate)

    # Crop region in normalised INPUT coordinates that maps to the output.
    # Output pixel (y, x) -> input pixel ((y/H - 0.5 - ty)/s + 0.5, ...).
    # Equivalently the crop box in input normalised coords is:
    y_start = 0.5 + ty - 0.5 / s
    y_end   = 0.5 + ty + 0.5 / s
    x_start = 0.5 + tx - 0.5 / s
    x_end   = 0.5 + tx + 0.5 / s

    # Resize via crop_and_resize (extrapolation_value=114 fills letterbox).
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
    if output_size is not None:
        image_out.set_shape([output_size[0], output_size[1], 3])
    else:
        image_out.set_shape([None, None, 3])

    # Transform boxes: y_out = (y_in - y_start) / (y_end - y_start).
    dy_range = y_end - y_start
    dx_range = x_end - x_start

    ymin_out = (boxes[:, 0] - y_start) / dy_range
    xmin_out = (boxes[:, 1] - x_start) / dx_range
    ymax_out = (boxes[:, 2] - y_start) / dy_range
    xmax_out = (boxes[:, 3] - x_start) / dx_range
    boxes_out = tf.stack([ymin_out, xmin_out, ymax_out, xmax_out], axis=1)

    # Transform polygons: same linear mapping.
    N = tf.shape(polygons)[0]
    max_v = tf.shape(polygons)[1]
    pts = tf.reshape(polygons, [N, max_v // 2, 2])  # [N, n_pairs, (x, y)]

    valid_x = pts[:, :, 0] > -1.0  # [N, n_pairs]; reserved sentinel is exactly -1.0

    x_out = (pts[:, :, 0] - x_start) / dx_range
    y_out = (pts[:, :, 1] - y_start) / dy_range
    # Invalidate points that were originally -1 OR that fall outside [0, 1] after transform.
    in_bounds = tf.logical_and(
        tf.logical_and(x_out >= 0.0, x_out <= 1.0),
        tf.logical_and(y_out >= 0.0, y_out <= 1.0),
    )
    final_valid = tf.logical_and(valid_x, in_bounds)
    x_out = tf.where(final_valid, x_out, tf.fill(tf.shape(x_out), -1.0))
    y_out = tf.where(final_valid, y_out, tf.fill(tf.shape(y_out), -1.0))

    pts_out = tf.stack([x_out, y_out], axis=-1)
    polygons_out = tf.reshape(pts_out, [N, max_v])

    return image_out, boxes_out, polygons_out


def clip_boxes(
    boxes: tf.Tensor,
    min_side: float = 0.003,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Clip boxes to [0, 1] and return a validity mask.

    Args:
        boxes:    float32 [N, 4] yxyx.
        min_side: minimum side length (normalised, strict >) to keep a box.
                  0.0 keeps every box with positive area on both sides and
                  drops only degenerate zero-size rows.

    Returns:
        (clipped_boxes [N, 4], keep_mask [N] bool)
    """
    boxes_clipped = tf.clip_by_value(boxes, 0.0, 1.0)
    h = boxes_clipped[:, 2] - boxes_clipped[:, 0]
    w = boxes_clipped[:, 3] - boxes_clipped[:, 1]
    # Strict > so min_side=0.0 means "degenerate rows only": zero-size rows
    # (notably the mosaic stage's padded_batch zero-padding) are dropped while
    # every box with positive area on both sides survives.
    keep = tf.logical_and(h > min_side, w > min_side)
    return boxes_clipped, keep


def clip_polygon_coords(polygons: tf.Tensor) -> tf.Tensor:
    """Clip polygon xy values to [0, 1], preserving -1 padding.

    Args:
        polygons: float32 [N, max_vertices] flat xy pairs, -1 padded.

    Returns:
        float32 [N, max_vertices].
    """
    # Validity keys off the -1.0 sentinel (`> -1.0`, not `>= 0.0`): a real vertex
    # can be slightly negative (mosaic overflow near an edge) and must be clipped
    # into [0, 1], while the exact -1.0 padding is left untouched.
    valid = polygons > -1.0
    clipped = tf.clip_by_value(polygons, 0.0, 1.0)
    return tf.where(valid, clipped, polygons)


def resample_polygons(
    polygons: tf.Tensor, max_points: int, compact: bool = False
) -> tf.Tensor:
    """Resample each polygon to a fixed ``max_points`` vertices (flat xy pairs).

    Intended for decode time (before augmentation), where the valid vertices of
    each polygon are a contiguous prefix and the rest are -1 padding. Shrinking
    the polygon width here makes every downstream op (copy-paste, mosaic,
    random_perspective, the parser) process a much smaller tensor than the raw
    stored width.

    Uniform arc-length resampling along the closed contour (fully vectorized,
    graph-safe): the valid vertices form a closed loop (wraparound edge
    ``v_{c-1} -> v_0``); ``K`` samples are placed at arc positions ``t_k = k*L/K``
    (``t_0 = 0`` keeps the first vertex), interpolating along edges via
    ``tf.searchsorted`` over cumulative segment lengths. Because samples land on
    edges, long edges spanning several angular bins populate those bins, so the
    24-bin radial target is well-formed even for sparse-vertex polygons (a
    4-corner rectangle does not collapse to a diamond). This alters the radial
    target for sparse polygons; for dense contours it is within sampling
    tolerance of plain index-subsampling.

    Degenerate rows:
      - c == 0 valid vertices -> all -1.
      - c == 1                -> that point repeated K times (no NaN).
      - c >= 2                -> K points spaced by arc length around the loop,
                                 each lying on an input edge.

    The sampling assumes valid vertices are a contiguous prefix. ``compact=True``
    first compacts scattered sentinels to a prefix via a stable argsort; pass it
    only from callers that can produce interior -1 holes (copy-paste, which
    invalidates out-of-bounds vertices in place). At decode time the prefix is
    guaranteed, so the default ``compact=False`` skips the O(P log P) sort.

    Args:
        polygons:   float32 [N, F] flat xy pairs, -1 padded (valid = prefix).
        max_points: target vertex count K; output width is 2*K.
        compact:    if True, compact scattered sentinels to a prefix first.

    Returns:
        float32 [N, 2*max_points].
    """
    K = max_points
    N = tf.shape(polygons)[0]
    F = tf.shape(polygons)[1]
    pts = tf.reshape(polygons, [N, F // 2, 2])                       # [N, P, 2]
    P = tf.shape(pts)[1]
    valid = pts[:, :, 0] > -1.0                                      # [N, P]; reserved sentinel is exactly -1.0

    # Compact valid vertices to a contiguous prefix. Only copy-paste (compact=True)
    # needs it: invalidating out-of-bounds vertices in place leaves -1 holes
    # interleaved with valid vertices, which would violate the prefix assumption
    # below and poison the interpolation with sentinel coords. At decode time the
    # vertices are already a prefix, so the sort is skipped.
    if compact:
        order = tf.argsort(tf.cast(~valid, tf.int32), axis=1, stable=True)  # valid first
        pts = tf.gather(pts, order, batch_dims=1)                   # [N, P, 2] compacted
        valid = pts[:, :, 0] > -1.0                                 # [N, P] recompute after sort

    counts = tf.reduce_sum(tf.cast(valid, tf.int32), axis=1)        # [N] valid count c per row
    Pf = tf.maximum(P, 1)

    # --- Closed-loop segment geometry ---------------------------------------
    # Segment i runs v_i -> next(i) over the CLOSED valid loop of c vertices:
    #   next(i) = v_{i+1}  for 0 <= i < c-1   (interior edge), and
    #   next(i) = v_0      for i == c-1        (wrap edge that closes the loop).
    # Segments with i >= c are FAKE (their start vertex is padding) and get length 0.
    # The wrap is to v_0 (loop start), NOT v_c (which is -1 padding), so next(i)
    # must be built explicitly rather than by a plain roll.
    idx_row = tf.broadcast_to(tf.range(P)[tf.newaxis, :], [N, P])  # [N, P]
    c_row = counts[:, tf.newaxis]                                   # [N, 1]
    # For each segment, the index of its END vertex in the closed valid loop.
    is_last = tf.equal(idx_row, tf.maximum(c_row - 1, 0))          # [N, P] i == c-1
    end_idx = tf.where(is_last, tf.zeros_like(idx_row), idx_row + 1)  # [N, P] wrap last -> 0
    end_idx = tf.clip_by_value(end_idx, 0, tf.maximum(P - 1, 0))
    nxt = tf.gather(pts, end_idx, batch_dims=1)                    # [N, P, 2] next(i)
    # A segment is REAL iff its start vertex is valid AND the loop has >= 2 vertices
    # (a single-vertex loop has no edges; its self-wrap stays length 0). For i < c-1
    # both endpoints are real interior vertices; for i == c-1 the end wraps to v_0.
    seg_real = tf.logical_and(idx_row < c_row, c_row >= 2)          # [N, P]
    seg_vec = nxt - pts                                             # [N, P, 2]
    seg_len = tf.sqrt(tf.reduce_sum(seg_vec * seg_vec, axis=-1) + 1e-20)  # [N, P]
    seg_len = tf.where(seg_real, seg_len, tf.zeros_like(seg_len))   # fake segments -> 0

    perim = tf.reduce_sum(seg_len, axis=1)                          # [N] L per row
    # Exclusive cumulative length: cum[i] = sum of seg_len[0..i-1] = arc dist to v_i.
    cum_incl = tf.cumsum(seg_len, axis=1)                           # [N, P] inclusive
    cum = cum_incl - seg_len                                        # [N, P] exclusive (arc start of seg i)

    # --- Target arc positions t_k = k*L/K ------------------------------------
    k = tf.cast(tf.range(K), tf.float32)[tf.newaxis, :]            # [1, K]
    t = k * (perim[:, tf.newaxis] / tf.cast(K, tf.float32))        # [N, K] in [0, L)

    # Locate each t_k's segment: largest i with cum[i] <= t_k. searchsorted on the
    # inclusive cumulative lengths (side='right') gives that segment index directly.
    seg_idx = tf.searchsorted(cum_incl, t, side="right")           # [N, K] in [0, P]
    # Clamp to the last REAL segment (c-1), never a fake/padding segment. For a
    # zero-perimeter row (c <= 1) every t_k == 0 and searchsorted on the all-zero
    # cumulative returns P; clamping to c-1 (==0 for c==1) lands on segment 0 = v_0,
    # which is exactly the "repeat v_0" degenerate behavior the spec wants. The
    # global clamp to P-1 guards the c==0 row (whose output is overwritten with -1).
    last_real = tf.clip_by_value(counts - 1, 0, tf.maximum(P - 1, 0))  # [N]
    seg_idx = tf.minimum(seg_idx, last_real[:, tf.newaxis])        # [N, K]
    seg_idx = tf.clip_by_value(seg_idx, 0, tf.maximum(P - 1, 0))    # clamp to valid range

    # Linear interpolation within the located segment.
    seg_start = tf.gather(cum, seg_idx, batch_dims=1)              # [N, K] arc at seg start
    seg_l = tf.gather(seg_len, seg_idx, batch_dims=1)             # [N, K] this segment length
    p0 = tf.gather(pts, seg_idx, batch_dims=1)                     # [N, K, 2]
    v = tf.gather(seg_vec, seg_idx, batch_dims=1)                  # [N, K, 2]
    # Safe division: zero-length (degenerate / collinear-duplicate / fake) segments
    # give frac 0 -> output stays at p0, no NaN.
    frac = tf.math.divide_no_nan(t - seg_start, seg_l)            # [N, K]
    frac = tf.clip_by_value(frac, 0.0, 1.0)
    out = p0 + frac[:, :, tf.newaxis] * v                          # [N, K, 2]

    # --- Degenerate row handling --------------------------------------------
    # c == 0 -> all -1 sentinel. c == 1 -> no real segment (perim 0), so every t_k = 0
    # lands on segment 0 with frac 0 -> out == v_0 repeated K times (the spec).
    has = counts[:, tf.newaxis, tf.newaxis] > 0                    # [N, 1, 1]
    out = tf.where(has, out, tf.fill(tf.shape(out), -1.0))         # empty rows -> -1
    _ = Pf  # P>=1 guaranteed by callers; kept explicit for graph shape clarity
    return tf.reshape(out, [N, K * 2])


def hsv_augment(
    image: tf.Tensor,
    hue: float = 0.015,
    sat: float = 0.7,
    val: float = 0.4,
) -> tf.Tensor:
    """Random HSV jitter, PyTorch-YOLO form (multiplicative, quantized domain).

    Reference math: per-channel gains ``r = 1 + U(-1, 1) * [hue, sat, val]`` are
    applied in the integer-quantized HSV domain::

        x = floor(rgb_to_hsv(image) * [180, 255, 255])
        x = floor(x * r)
        h %= 180; s, v clipped to [0, 255]
        image = hsv_to_rgb(x / [180, 255, 255])

    All three channels, including hue, are scaled multiplicatively; this is
    NOT the additive hue-rotation / additive-brightness form.

    Args:
        image: float32 [H, W, 3] in [0, 1].
        hue:   hue gain half-range.
        sat:   saturation gain half-range.
        val:   brightness (value) gain half-range.

    Returns:
        float32 [H, W, 3] in [0, 1].
    """
    if hue <= 0.0 and sat <= 0.0 and val <= 0.0:
        return image
    scale = tf.constant([180.0, 255.0, 255.0], tf.float32)
    r = tf.random.uniform([3], -1.0, 1.0) * tf.constant([hue, sat, val]) + 1.0
    x = tf.image.rgb_to_hsv(image)
    x = tf.math.floor(x * scale)
    x = tf.math.floor(x * r)
    h, s, v = tf.split(x, 3, axis=-1)
    h = h % 180.0
    s = tf.clip_by_value(s, 0.0, 255.0)
    v = tf.clip_by_value(v, 0.0, 255.0)
    x = tf.concat([h, s, v], axis=-1) / scale
    return tf.image.hsv_to_rgb(x)
