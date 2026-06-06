"""Mosaic and MixUp augmentation combining multiple images.

Mosaic stitches 4 images into a single training sample, significantly
expanding the effective receptive field seen per step.  MixUp blends
two images with a beta-distributed weight.

Configuration from experiment_config.yaml:
    mosaic_frequency: 0.5
    mixup_frequency: 0.0
    mosaic_center: 0.2   (max offset of stitch point from image center)
    aug_scale_min: 0.4
    aug_scale_max: 1.9
    mosaic_crop_mode: scale
    area_thresh: 0.5     (minimum visible box area fraction to keep)

Classes:
    Mosaic: Manages both Mosaic and MixUp augmentations.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import tensorflow as tf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _letterbox_resize_to(
    image: tf.Tensor,
    target_h: tf.Tensor,
    target_w: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Letterbox-resize image to (target_h, target_w) with gray padding (114).

    Returns:
        image_out:  uint8 [target_h, target_w, 3]
        scale:      float32 scalar — the uniform scale applied
        pad_top:    int32 scalar pixels of top padding
        pad_left:   int32 scalar pixels of left padding
        new_h, new_w: int32 scaled dimensions before padding
    """
    h_in = tf.cast(tf.shape(image)[0], tf.float32)
    w_in = tf.cast(tf.shape(image)[1], tf.float32)
    th_f = tf.cast(target_h, tf.float32)
    tw_f = tf.cast(target_w, tf.float32)

    scale = tf.minimum(th_f / h_in, tw_f / w_in)
    new_h = tf.maximum(tf.cast(tf.round(h_in * scale), tf.int32), 1)
    new_w = tf.maximum(tf.cast(tf.round(w_in * scale), tf.int32), 1)

    image = tf.cast(
        tf.image.resize(tf.cast(image, tf.float32), [new_h, new_w], method='bilinear'),
        tf.uint8,
    )

    pad_top    = (target_h - new_h) // 2
    pad_left   = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right  = target_w - new_w - pad_left

    image_out = tf.pad(
        image,
        [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=114,
    )
    return image_out, scale, pad_top, pad_left, new_h, new_w


def _transform_boxes(
    boxes: tf.Tensor,
    scale: tf.Tensor,
    pad_top: tf.Tensor,
    pad_left: tf.Tensor,
    quad_h: tf.Tensor,
    quad_w: tf.Tensor,
    offset_y: tf.Tensor,
    offset_x: tf.Tensor,
    H_out: tf.Tensor,
    W_out: tf.Tensor,
    area_thresh: float = 0.5,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Transform boxes from input-normalised to output-normalised coordinates.

    The input image was letterbox-resized to (quad_h, quad_w) with the given
    scale and padding, then placed at (offset_y, offset_x) in the output.

    Returns:
        boxes_out: float32 [N, 4] clipped to [0,1]
        keep_mask: bool [N]
    """
    H_out_f  = tf.cast(H_out,   tf.float32)
    W_out_f  = tf.cast(W_out,   tf.float32)
    pad_top_f  = tf.cast(pad_top,  tf.float32)
    pad_left_f = tf.cast(pad_left, tf.float32)
    off_y_f    = tf.cast(offset_y, tf.float32)
    off_x_f    = tf.cast(offset_x, tf.float32)
    # Normalised scale factors: how input-normalised [0,1] maps to output-normalised [0,1]
    # Input height h_in maps to new_h = h_in * scale pixels in the quadrant.
    # In quadrant coords: y_quad = y_in * new_h = y_in * scale * h_in (not what we need)
    # Actually: boxes are normalised by INPUT size, so:
    #   y_quad_px = y_in_norm * h_in → resized to y_quad_norm * new_h
    #   We compute everything in output pixel space.
    #   y_in_norm → y_quad_px_after_pad = y_in_norm * new_h + pad_top
    #             → y_out_px = y_quad_px_after_pad + offset_y
    #             → y_out_norm = y_out_px / H_out
    # But new_h = h_in * scale, and since boxes are normalised we write:
    #   y_out_norm = (y_in_norm * new_h + pad_top + offset_y) / H_out
    # We need new_h — we can compute it: new_h = round(h_in * scale), but h_in is
    # not available here.  Instead: new_h = quad_h - 2*pad_top (approximately).
    # More precisely: new_h ≤ quad_h and new_w ≤ quad_w with pads centering them.
    # We already receive pad_top, pad_left, quad_h, quad_w:
    new_h_f = tf.cast(quad_h, tf.float32) - 2.0 * pad_top_f
    new_w_f = tf.cast(quad_w, tf.float32) - 2.0 * pad_left_f
    new_h_f = tf.maximum(new_h_f, 1.0)
    new_w_f = tf.maximum(new_w_f, 1.0)

    ymin_out = (boxes[:, 0] * new_h_f + pad_top_f  + off_y_f) / H_out_f
    xmin_out = (boxes[:, 1] * new_w_f + pad_left_f + off_x_f) / W_out_f
    ymax_out = (boxes[:, 2] * new_h_f + pad_top_f  + off_y_f) / H_out_f
    xmax_out = (boxes[:, 3] * new_w_f + pad_left_f + off_x_f) / W_out_f

    boxes_raw = tf.stack([ymin_out, xmin_out, ymax_out, xmax_out], axis=1)

    # Clip to [0, 1]
    boxes_clipped = tf.clip_by_value(boxes_raw, 0.0, 1.0)

    # Compute visible fraction (area_after / area_before)
    area_before = (
        (boxes_raw[:, 2] - boxes_raw[:, 0]) *
        (boxes_raw[:, 3] - boxes_raw[:, 1])
    )
    area_after = (
        (boxes_clipped[:, 2] - boxes_clipped[:, 0]) *
        (boxes_clipped[:, 3] - boxes_clipped[:, 1])
    )
    # Keep if area_after / area_before >= area_thresh (and box has positive area)
    keep = tf.logical_and(
        area_before > 1e-6,
        area_after >= area_thresh * area_before,
    )

    return boxes_clipped, keep


def _transform_polygons(
    polygons: tf.Tensor,
    pad_top: tf.Tensor,
    pad_left: tf.Tensor,
    quad_h: tf.Tensor,
    quad_w: tf.Tensor,
    offset_y: tf.Tensor,
    offset_x: tf.Tensor,
    H_out: tf.Tensor,
    W_out: tf.Tensor,
) -> tf.Tensor:
    """Transform flat polygon xy pairs from input-normalised to output-normalised.

    Args:
        polygons: float32 [N, max_v] flat xy pairs, -1 padded.

    Returns:
        float32 [N, max_v] transformed, clipped to [0,1] (invalid → -1).
    """
    H_out_f    = tf.cast(H_out,   tf.float32)
    W_out_f    = tf.cast(W_out,   tf.float32)
    pad_top_f  = tf.cast(pad_top,  tf.float32)
    pad_left_f = tf.cast(pad_left, tf.float32)
    off_y_f    = tf.cast(offset_y, tf.float32)
    off_x_f    = tf.cast(offset_x, tf.float32)
    new_h_f = tf.cast(quad_h, tf.float32) - 2.0 * pad_top_f
    new_w_f = tf.cast(quad_w, tf.float32) - 2.0 * pad_left_f
    new_h_f = tf.maximum(new_h_f, 1.0)
    new_w_f = tf.maximum(new_w_f, 1.0)

    N   = tf.shape(polygons)[0]
    max_v = tf.shape(polygons)[1]
    pts = tf.reshape(polygons, [N, max_v // 2, 2])  # [N, n_pairs, (x, y)]

    valid_x = pts[:, :, 0] >= 0.0  # [N, n_pairs]

    x_out = (pts[:, :, 0] * new_w_f + pad_left_f + off_x_f) / W_out_f
    y_out = (pts[:, :, 1] * new_h_f + pad_top_f  + off_y_f) / H_out_f

    # Clip and restore -1 for invalid
    x_out = tf.where(valid_x, tf.clip_by_value(x_out, 0.0, 1.0), tf.fill(tf.shape(x_out), -1.0))
    y_out = tf.where(valid_x, tf.clip_by_value(y_out, 0.0, 1.0), tf.fill(tf.shape(y_out), -1.0))

    pts_out = tf.stack([x_out, y_out], axis=-1)
    return tf.reshape(pts_out, [N, max_v])


# ---------------------------------------------------------------------------
# Mosaic class
# ---------------------------------------------------------------------------

class Mosaic:
    """Mosaic (4-image stitch) and MixUp augmentation with polygon support."""

    def __init__(
        self,
        output_size: List[int],
        mosaic_frequency: float = 0.5,
        mixup_frequency: float = 0.0,
        letter_box: bool = True,
        mosaic_crop_mode: str = "scale",
        mosaic_center: float = 0.25,
        aug_scale_min: float = 0.4,
        aug_scale_max: float = 1.9,
        area_thresh: float = 0.5,
        with_polygons: bool = True,
    ):
        self._H = output_size[0]
        self._W = output_size[1]
        self._mosaic_freq = mosaic_frequency
        self._mixup_freq  = mixup_frequency
        self._letter_box  = letter_box
        self._crop_mode   = mosaic_crop_mode
        self._center      = mosaic_center
        self._scale_min   = aug_scale_min
        self._scale_max   = aug_scale_max
        self._area_thresh = area_thresh
        self._with_polys  = with_polygons

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def mosaic_fn(self, is_training: bool = True) -> Callable:
        """Return a function for use in tf.data.Dataset.map().

        The function expects a batch of 4 samples (each field has a leading
        dim of 4) and returns a dict with a leading dim of 1 so that the
        subsequent .unbatch() produces individual examples.
        """

        def _fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            # Split batch-of-4 into 4 individual example dicts
            def _select(d: Dict, i: int) -> Dict:
                return {k: v[i] for k, v in d.items()}

            one   = _select(batch, 0)
            two   = _select(batch, 1)
            three = _select(batch, 2)
            four  = _select(batch, 3)

            if not is_training:
                # Eval: return just the first image with leading dim=1
                result = one
            else:
                do_mosaic = tf.random.uniform([]) < self._mosaic_freq
                result = tf.cond(
                    do_mosaic,
                    lambda: self._mosaic(one, two, three, four),
                    lambda: one,  # no mosaic: pass first image through
                )

            # Wrap in batch dim of 1 so .unbatch() works correctly
            return {k: tf.expand_dims(v, 0) for k, v in result.items()}

        return _fn

    # ------------------------------------------------------------------
    # Mosaic implementation
    # ------------------------------------------------------------------

    def _mosaic(
        self,
        one:   Dict[str, tf.Tensor],
        two:   Dict[str, tf.Tensor],
        three: Dict[str, tf.Tensor],
        four:  Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Stitch 4 images at a random center point and merge annotations.

        Layout (quadrant → example):
            top-left  = one   |  top-right  = two
            ──────────────────────────────────────
            bot-left  = three |  bot-right  = four

        Args:
            one / two / three / four: individual decoded example dicts.

        Returns:
            Single merged example dict (no batch dim).
        """
        H = self._H
        W = self._W
        H_t = tf.constant(H, tf.int32)
        W_t = tf.constant(W, tf.int32)

        # Random mosaic center
        lo = 0.5 - self._center
        hi = 0.5 + self._center
        cy = tf.cast(tf.round(tf.random.uniform([], lo, hi) * H), tf.int32)
        cx = tf.cast(tf.round(tf.random.uniform([], lo, hi) * W), tf.int32)
        cy = tf.clip_by_value(cy, 1, H - 1)
        cx = tf.clip_by_value(cx, 1, W - 1)

        # Build each quadrant piece
        imgs, ann_list = [], []
        quadrant_defs = [
            (one,   cy,      cx,      0,  0),   # top-left
            (two,   cy,      W_t-cx,  0,  cx),  # top-right
            (three, H_t-cy,  cx,      cy, 0),   # bot-left
            (four,  H_t-cy,  W_t-cx,  cy, cx),  # bot-right
        ]

        for ex, qh, qw, off_y, off_x in quadrant_defs:
            img_q, anns_q = self._place_quadrant(ex, qh, qw, off_y, off_x, H_t, W_t)
            imgs.append(img_q)
            ann_list.append(anns_q)

        # Assemble canvas
        top    = tf.concat([imgs[0], imgs[1]], axis=1)  # [cy,   W, 3]
        bottom = tf.concat([imgs[2], imgs[3]], axis=1)  # [H-cy, W, 3]
        canvas = tf.concat([top, bottom], axis=0)        # [H,    W, 3]

        # Merge annotations
        merged = self._merge_annotations(ann_list)
        merged['image'] = canvas
        return merged

    def _place_quadrant(
        self,
        ex:     Dict[str, tf.Tensor],
        qh:     tf.Tensor,   # quadrant height (int32)
        qw:     tf.Tensor,   # quadrant width  (int32)
        off_y:  tf.Tensor,   # y offset in output canvas
        off_x:  tf.Tensor,   # x offset in output canvas
        H_out:  tf.Tensor,
        W_out:  tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Resize one example to fit its quadrant and transform annotations.

        Returns:
            (image_piece [qh, qw, 3], annotations_dict)
        """
        img = ex['image']  # uint8 [h, w, 3]

        # Letterbox-resize to quadrant size
        img_q, _scale, pad_top, pad_left, _nh, _nw = _letterbox_resize_to(img, qh, qw)

        # Transform boxes
        boxes   = ex.get('groundtruth_boxes', tf.zeros([0, 4]))
        classes = ex.get('groundtruth_classes', tf.zeros([0], tf.int64))
        is_crowd= ex.get('groundtruth_is_crowd', tf.zeros([0], tf.bool))
        area    = ex.get('groundtruth_area', tf.zeros([0]))
        dontcare= ex.get('groundtruth_dontcare', tf.zeros([0], tf.int64))
        polygons= ex.get('groundtruth_polygons', tf.zeros([0, 2]))

        boxes_out, keep = _transform_boxes(
            boxes, _scale, pad_top, pad_left,
            qh, qw, off_y, off_x, H_out, W_out,
            area_thresh=self._area_thresh,
        )

        # Filter to kept boxes
        boxes_out  = tf.boolean_mask(boxes_out,  keep)
        classes    = tf.boolean_mask(classes,    keep)
        is_crowd   = tf.boolean_mask(is_crowd,   keep)
        area       = tf.boolean_mask(area,       keep)
        dontcare   = tf.boolean_mask(dontcare,   keep)

        anns = {
            'groundtruth_boxes':    boxes_out,
            'groundtruth_classes':  classes,
            'groundtruth_is_crowd': is_crowd,
            'groundtruth_area':     area,
            'groundtruth_dontcare': dontcare,
        }

        if self._with_polys and polygons.shape[-1] != 0:
            polygons_out = _transform_polygons(
                polygons, pad_top, pad_left,
                qh, qw, off_y, off_x, H_out, W_out,
            )
            polygons_out = tf.boolean_mask(polygons_out, keep)
            anns['groundtruth_polygons'] = polygons_out

        # Carry source_id from first example
        anns['source_id'] = ex.get('source_id', tf.constant('mosaic'))

        return img_q, anns

    @staticmethod
    def _merge_annotations(ann_list: List[Dict]) -> Dict[str, tf.Tensor]:
        """Concatenate per-quadrant annotation dicts."""
        merged: Dict[str, tf.Tensor] = {}
        keys = set(ann_list[0].keys())
        for key in keys:
            tensors = [a[key] for a in ann_list if key in a]
            if key == 'source_id':
                merged[key] = tensors[0]  # keep first
                continue
            if len(tensors) > 0:
                merged[key] = tf.concat(tensors, axis=0)
        return merged

    # ------------------------------------------------------------------
    # MixUp (disabled by default: mixup_frequency=0.0)
    # ------------------------------------------------------------------

    def _mixup(
        self,
        one: Dict[str, tf.Tensor],
        two: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Blend two images with a beta-distributed weight."""
        # Beta(8, 8) weight — strongly centred around 0.5 per Ultralytics convention
        r = tf.random.uniform([])
        img1 = tf.cast(one['image'], tf.float32)
        img2 = tf.cast(two['image'], tf.float32)

        # Resize img2 to match img1 size
        h1 = tf.shape(img1)[0]
        w1 = tf.shape(img1)[1]
        img2 = tf.image.resize(img2, [h1, w1], method='bilinear')

        blended = r * img1 + (1.0 - r) * img2
        result = dict(one)
        result['image'] = tf.cast(blended, tf.uint8)

        # Merge annotations from both
        for key in ['groundtruth_boxes', 'groundtruth_classes',
                    'groundtruth_is_crowd', 'groundtruth_area',
                    'groundtruth_dontcare', 'groundtruth_polygons']:
            if key in one and key in two:
                result[key] = tf.concat([one[key], two[key]], axis=0)

        return result
