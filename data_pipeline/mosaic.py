"""Mosaic and MixUp augmentation combining multiple images.

Mosaic (Ultralytics-style) stitches 4 images into a 2×-size canvas at a random
center — each image is placed at full scale toward the center with overflow
cropped (no letterbox padding) — then a single ``random_perspective`` (rotation +
scale + shear + translate) warps and crops the canvas back to the output size.
The same ``random_perspective`` is applied to non-mosaic single images, so it is
the one geometric transform in the pipeline (the parser no longer does affine).

Configuration (parser.mosaic in the experiment YAML):
    mosaic_frequency: 0.5
    mixup_frequency: 0.0
    mosaic_center: 0.25      (half-range of the split point as a fraction; the
                              2× canvas split lands in [H(1-2c), H(1+2c)])
    aug_scale_min / aug_scale_max: per-image scale range AND the random_perspective
                              scale-gain bounds.
    degrees: 10.0            (rotation ±, degrees)
    shear: 2.0               (shear ±, degrees)
    perspective: 0.0         (perspective coefficient ±; 0 disables)
    translate: 0.1           (translation ± as a fraction of output size)
    area_thresh: 0.5         (min visible box-area fraction to keep)

Classes:
    Mosaic: Manages both Mosaic and MixUp augmentations.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import tensorflow as tf

from data_pipeline.augmentations import random_perspective


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _place_in_cell(
    R: tf.Tensor,
    cell_h: tf.Tensor,
    cell_w: tf.Tensor,
    top_y: tf.Tensor,
    top_x: tf.Tensor,
) -> tf.Tensor:
    """Place a resized image R into a (cell_h, cell_w) gray-114 cell.

    R's top-left corner is positioned at (top_y, top_x) within the cell (may be
    negative → R is cropped); regions of the cell not covered by R stay 114.
    Pure crop + pad so it runs in graph mode.
    """
    nh = tf.shape(R)[0]
    nw = tf.shape(R)[1]
    src_y0 = tf.maximum(0, -top_y)
    src_x0 = tf.maximum(0, -top_x)
    dst_y0 = tf.minimum(tf.maximum(0, top_y), cell_h)
    dst_x0 = tf.minimum(tf.maximum(0, top_x), cell_w)
    copy_h = tf.maximum(0, tf.minimum(nh - src_y0, cell_h - dst_y0))
    copy_w = tf.maximum(0, tf.minimum(nw - src_x0, cell_w - dst_x0))

    R_crop = tf.slice(R, [src_y0, src_x0, 0], [copy_h, copy_w, 3])
    pad_b = cell_h - dst_y0 - copy_h
    pad_r = cell_w - dst_x0 - copy_w
    return tf.pad(
        R_crop,
        [[dst_y0, pad_b], [dst_x0, pad_r], [0, 0]],
        constant_values=114,
    )


def _scale_box_poly_to_canvas(
    ex: Dict[str, tf.Tensor],
    nh: tf.Tensor, nw: tf.Tensor,
    padh: tf.Tensor, padw: tf.Tensor,
    H2: tf.Tensor, W2: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Map an example's boxes/polygons (input-normalized) into 2× canvas-normalized.

    canvas_px = coord_in_px * scaled_dim + pad; then / canvas size. No clipping —
    the subsequent random_perspective clips to the output edge.
    """
    nh_f = tf.cast(nh, tf.float32); nw_f = tf.cast(nw, tf.float32)
    padh_f = tf.cast(padh, tf.float32); padw_f = tf.cast(padw, tf.float32)
    H2_f = tf.cast(H2, tf.float32); W2_f = tf.cast(W2, tf.float32)

    boxes = ex.get('groundtruth_boxes', tf.zeros([0, 4]))
    ymin = (boxes[:, 0] * nh_f + padh_f) / H2_f
    xmin = (boxes[:, 1] * nw_f + padw_f) / W2_f
    ymax = (boxes[:, 2] * nh_f + padh_f) / H2_f
    xmax = (boxes[:, 3] * nw_f + padw_f) / W2_f
    boxes_c = tf.stack([ymin, xmin, ymax, xmax], axis=1)

    polys = ex.get('groundtruth_polygons', tf.zeros([0, 2]))
    N = tf.shape(polys)[0]
    max_v = tf.shape(polys)[1]
    pts = tf.reshape(polys, [N, max_v // 2, 2])
    valid = pts[:, :, 0] >= 0.0
    x_c = (pts[:, :, 0] * nw_f + padw_f) / W2_f
    y_c = (pts[:, :, 1] * nh_f + padh_f) / H2_f
    neg1 = tf.fill(tf.shape(x_c), -1.0)
    x_c = tf.where(valid, x_c, neg1)
    y_c = tf.where(valid, y_c, neg1)
    polys_c = tf.reshape(tf.stack([x_c, y_c], axis=-1), [N, max_v])
    return boxes_c, polys_c


# ---------------------------------------------------------------------------
# Mosaic class
# ---------------------------------------------------------------------------

class Mosaic:
    """Mosaic (4-image stitch) + random_perspective, with polygon support."""

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
        degrees: float = 10.0,
        shear: float = 2.0,
        perspective: float = 0.0,
        translate: float = 0.1,
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
        self._degrees     = degrees
        self._shear       = shear
        self._perspective = perspective
        self._translate   = translate

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def mosaic_fn(self, is_training: bool = True) -> Callable:
        """Return a function for tf.data.Dataset.map() over a batch of 4 samples.

        Both the mosaic branch and the single-image branch run random_perspective,
        so every training image gets the full geometric transform exactly once.
        """

        def _fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            def _select(d: Dict, i: int) -> Dict:
                return {k: v[i] for k, v in d.items()}

            one   = _select(batch, 0)
            two   = _select(batch, 1)
            three = _select(batch, 2)
            four  = _select(batch, 3)

            if not is_training:
                result = self._single(one)
            else:
                do_mosaic = tf.random.uniform([]) < self._mosaic_freq
                result = tf.cond(
                    do_mosaic,
                    lambda: self._mosaic(one, two, three, four),
                    lambda: self._single(one),
                )

            return {k: tf.expand_dims(v, 0) for k, v in result.items()}

        return _fn

    # ------------------------------------------------------------------
    # Geometric transform helpers
    # ------------------------------------------------------------------

    def _warp(
        self,
        image: tf.Tensor,
        boxes: tf.Tensor,
        polygons: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """random_perspective with this module's configured params → output size."""
        return random_perspective(
            image, boxes, polygons,
            target_h=self._H, target_w=self._W,
            degrees=self._degrees,
            translate=self._translate,
            scale=max(self._scale_max - 1.0, 1.0 - self._scale_min),
            shear=self._shear,
            perspective=self._perspective,
            area_thresh=self._area_thresh,
        )

    @staticmethod
    def _filtered_anns(
        ex: Dict[str, tf.Tensor],
        boxes: tf.Tensor,
        polygons: tf.Tensor,
        keep: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Build the output annotation dict, filtered by the perspective keep mask.

        Per-box side tensors (classes/crowd/area/dontcare/dists) are taken from ex
        and masked by keep (same ordering as the boxes fed to random_perspective).
        """
        def _side(key, default_val, dtype):
            v = ex.get(key, tf.zeros([0], dtype))
            v = tf.cast(v, dtype)
            return tf.boolean_mask(v, keep)

        return {
            'groundtruth_boxes':    tf.boolean_mask(boxes, keep),
            'groundtruth_polygons': tf.boolean_mask(polygons, keep),
            'groundtruth_classes':  _side('groundtruth_classes',  0, tf.int64),
            'groundtruth_is_crowd': _side('groundtruth_is_crowd', False, tf.bool),
            'groundtruth_area':     _side('groundtruth_area',     0.0, tf.float32),
            'groundtruth_dontcare': _side('groundtruth_dontcare', 0, tf.int64),
            'groundtruth_dists':    _side('groundtruth_dists',    0.0, tf.float32),
            'source_id':            ex.get('source_id', tf.constant('mosaic')),
        }

    def _single(self, ex: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Non-mosaic path: random_perspective on one (already output-sized) image."""
        img = ex['image']
        boxes = ex.get('groundtruth_boxes', tf.zeros([0, 4]))
        polys = ex.get('groundtruth_polygons', tf.zeros([0, 2]))
        img_out, boxes_out, keep, polys_out = self._warp(img, boxes, polys)
        anns = self._filtered_anns(ex, boxes_out, polys_out, keep)
        anns['image']  = img_out
        anns['height'] = tf.constant(self._H, tf.int32)
        anns['width']  = tf.constant(self._W, tf.int32)
        return anns

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
        """Assemble a 2× canvas (per-image scale + offset + crop) then warp to output.

        Quadrant → example: TL=one, TR=two, BL=three, BR=four. Each image is scaled
        by a per-image random factor and placed so its center-adjacent corner abuts
        the random split point (xc, yc); overflow is cropped, gaps stay 114.
        """
        H, W = self._H, self._W
        H2 = tf.constant(2 * H, tf.int32)
        W2 = tf.constant(2 * W, tf.int32)

        # Random split point on the 2× canvas: [H(1-2c), H(1+2c)] clipped.
        c = self._center
        yc = tf.cast(tf.round(tf.random.uniform([], H * (1.0 - 2.0 * c), H * (1.0 + 2.0 * c))), tf.int32)
        xc = tf.cast(tf.round(tf.random.uniform([], W * (1.0 - 2.0 * c), W * (1.0 + 2.0 * c))), tf.int32)
        yc = tf.clip_by_value(yc, 1, 2 * H - 1)
        xc = tf.clip_by_value(xc, 1, 2 * W - 1)

        examples = [one, two, three, four]
        cells, boxes_list, polys_list = [], [], []

        # (cell_h, cell_w, cell_top_y_fn, cell_top_x_fn, canvas_off_y, canvas_off_x)
        # cell_top_* depend on the per-image scaled (nh, nw).
        for i, ex in enumerate(examples):
            img = ex['image']
            h_in = tf.shape(img)[0]
            w_in = tf.shape(img)[1]
            scale = tf.random.uniform([], self._scale_min, self._scale_max)
            nh = tf.maximum(tf.cast(tf.round(tf.cast(h_in, tf.float32) * scale), tf.int32), 1)
            nw = tf.maximum(tf.cast(tf.round(tf.cast(w_in, tf.float32) * scale), tf.int32), 1)
            R = tf.cast(tf.image.resize(tf.cast(img, tf.float32), [nh, nw], method='bilinear'), tf.uint8)

            if i == 0:    # TL
                cell_h, cell_w = yc, xc
                top_y, top_x = yc - nh, xc - nw
                off_y, off_x = tf.constant(0, tf.int32), tf.constant(0, tf.int32)
            elif i == 1:  # TR
                cell_h, cell_w = yc, W2 - xc
                top_y, top_x = yc - nh, tf.constant(0, tf.int32)
                off_y, off_x = tf.constant(0, tf.int32), xc
            elif i == 2:  # BL
                cell_h, cell_w = H2 - yc, xc
                top_y, top_x = tf.constant(0, tf.int32), xc - nw
                off_y, off_x = yc, tf.constant(0, tf.int32)
            else:         # BR
                cell_h, cell_w = H2 - yc, W2 - xc
                top_y, top_x = tf.constant(0, tf.int32), tf.constant(0, tf.int32)
                off_y, off_x = yc, xc

            cell = _place_in_cell(R, cell_h, cell_w, top_y, top_x)
            cells.append(cell)

            padh = off_y + top_y
            padw = off_x + top_x
            b_c, p_c = _scale_box_poly_to_canvas(ex, nh, nw, padh, padw, H2, W2)
            boxes_list.append(b_c)
            polys_list.append(p_c)

        # Assemble 2× canvas from the 4 quadrant cells.
        top    = tf.concat([cells[0], cells[1]], axis=1)   # [yc,    2W, 3]
        bottom = tf.concat([cells[2], cells[3]], axis=1)   # [2H-yc, 2W, 3]
        canvas = tf.concat([top, bottom], axis=0)          # [2H,    2W, 3]

        boxes_all = tf.concat(boxes_list, axis=0)
        polys_all = tf.concat(polys_list, axis=0)

        # Concatenate per-box side tensors across the 4 examples (same order).
        def _cat(key, default, dtype):
            return tf.concat(
                [tf.cast(ex.get(key, tf.zeros([0], dtype)), dtype) for ex in examples],
                axis=0,
            )
        merged_src = {
            'groundtruth_classes':  _cat('groundtruth_classes',  0, tf.int64),
            'groundtruth_is_crowd': _cat('groundtruth_is_crowd', False, tf.bool),
            'groundtruth_area':     _cat('groundtruth_area',     0.0, tf.float32),
            'groundtruth_dontcare': _cat('groundtruth_dontcare', 0, tf.int64),
            'groundtruth_dists':    _cat('groundtruth_dists',    0.0, tf.float32),
            'source_id':            one.get('source_id', tf.constant('mosaic')),
        }

        # One geometric warp of the whole canvas → output size.
        img_out, boxes_out, keep, polys_out = self._warp(canvas, boxes_all, polys_all)

        anns = self._filtered_anns(merged_src, boxes_out, polys_out, keep)
        anns['image']  = img_out
        anns['height'] = tf.constant(H, tf.int32)
        anns['width']  = tf.constant(W, tf.int32)
        return anns

    # ------------------------------------------------------------------
    # MixUp (disabled by default: mixup_frequency=0.0)
    # ------------------------------------------------------------------

    def _mixup(
        self,
        one: Dict[str, tf.Tensor],
        two: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Blend two images with a beta-distributed weight."""
        r = tf.random.uniform([])
        img1 = tf.cast(one['image'], tf.float32)
        img2 = tf.cast(two['image'], tf.float32)

        h1 = tf.shape(img1)[0]
        w1 = tf.shape(img1)[1]
        img2 = tf.image.resize(img2, [h1, w1], method='bilinear')

        blended = r * img1 + (1.0 - r) * img2
        result = dict(one)
        result['image'] = tf.cast(blended, tf.uint8)

        for key in ['groundtruth_boxes', 'groundtruth_classes',
                    'groundtruth_is_crowd', 'groundtruth_area',
                    'groundtruth_dontcare', 'groundtruth_dists',
                    'groundtruth_polygons']:
            if key in one and key in two:
                result[key] = tf.concat([one[key], two[key]], axis=0)

        return result
