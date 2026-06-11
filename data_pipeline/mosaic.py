"""Mosaic and MixUp augmentation combining multiple images.

Mosaic (Ultralytics-style) stitches 4 images into a 2×-size canvas at a random
center — each image is placed at full scale toward the center with overflow
cropped (no letterbox padding) — then a single ``random_perspective`` (rotation +
scale + shear + translate) warps and crops the canvas back to the output size.
The same ``random_perspective`` is applied to non-mosaic single images, so it is
the one geometric transform in the pipeline (the parser no longer does affine).

4-in / 4-out semantics (epoch accounting)
-----------------------------------------
``mosaic_fn`` maps a ``padded_batch(4)`` group to **four** emitted samples (one
per decoded image), not one. The previous implementation emitted a single sample
per group (``tf.cond(do_mosaic, _mosaic(4 imgs), _single(1 img))`` then
``expand_dims(0)``), which in the non-mosaic branch silently *discarded three of
the four decoded images*. That broke epoch accounting: 4 raw images were consumed
per emitted sample, so an "epoch" only saw a quarter of the dataset (and the three
dropped images never contributed gradients).

Now every decoded image yields exactly one emitted sample, and per-sample mosaic
probability is still exactly ``mosaic_frequency``: one coin flip per group decides
whether the group's 4 outputs are 4 mosaics (each built from a rotated quadrant
permutation of the same 4 images) or 4 single-image warps. Downstream
``.unbatch()`` then sees 4 elements (leading dim 4 instead of 1).

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

from data_pipeline.augmentations import (
    apply_perspective_image,
    make_perspective_matrix,
    random_perspective,
    transform_boxes_polygons,
)


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


# How each annotation field is padded up to the group-max instance count before
# stacking. Per-instance (axis-0 = N) fields only. (key, pad_value)
#   - boxes pad with zero rows [0,0,0,0]: the parser's clip_boxes min_side=0.005
#     filter provably drops zero boxes (incl. after flip), so they never train.
#   - polygons pad with -1.0 rows (defense in depth; the -1 sentinel marks invalid
#     vertices). Width V is identical across the 4 results (same padded_batch group).
#   - classes/dontcare pad 0 (int64); is_crowd pad False (NEVER True — crowd
#     filtering is config-conditional in the parser); area/dists pad 0.0.
_PAD_SPEC = (
    ('groundtruth_boxes',    0.0,   tf.float32),
    ('groundtruth_polygons', -1.0,  tf.float32),
    ('groundtruth_classes',  0,     tf.int64),
    ('groundtruth_is_crowd', False, tf.bool),
    ('groundtruth_area',     0.0,   tf.float32),
    ('groundtruth_dontcare', 0,     tf.int64),
    ('groundtruth_dists',    0.0,   tf.float32),
)


def _stack_results(results: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
    """Stack a list of per-sample result dicts to a single dict with leading dim len(results).

    Per-instance annotation tensors have differing instance counts ``N_i``; each is
    padded (per ``_PAD_SPEC``) up to the group-max ``N`` before ``tf.stack`` (which
    requires equal shapes). Non-instance fields (image / height / width / source_id)
    are stacked directly. Used by both ``tf.cond`` branches, so the output dicts have
    identical keys, dtypes, and ranks.
    """
    # Group-max instance count across all results (boxes' axis-0).
    n_list = [tf.shape(r['groundtruth_boxes'])[0] for r in results]
    max_n = n_list[0]
    for n_i in n_list[1:]:
        max_n = tf.maximum(max_n, n_i)

    out: Dict[str, tf.Tensor] = {}

    for key, pad_val, dtype in _PAD_SPEC:
        padded = []
        for r, n_i in zip(results, n_list):
            t = tf.cast(r[key], dtype)
            pad_rows = max_n - n_i
            # Pad only axis 0 (instances); trailing dims unchanged.
            paddings = [[0, pad_rows]] + [[0, 0]] * (len(t.shape) - 1)
            t = tf.pad(t, paddings, constant_values=pad_val)
            padded.append(t)
        out[key] = tf.stack(padded, axis=0)

    out['image']     = tf.stack([r['image'] for r in results], axis=0)
    out['height']    = tf.stack([r['height'] for r in results], axis=0)
    out['width']     = tf.stack([r['width'] for r in results], axis=0)
    out['source_id'] = tf.stack([r['source_id'] for r in results], axis=0)
    return out


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
        """Return a function for tf.data.Dataset.map() over a ``padded_batch(4)`` group.

        Maps the 4-image group to FOUR emitted samples (leading dim 4), so every
        decoded image yields exactly one output. One coin flip per group selects
        the path; per-sample mosaic probability is still exactly ``mosaic_frequency``:

          * mosaic branch: build 4 mosaics from the same 4 examples using rotated
            quadrant permutations ``[(0,1,2,3),(1,2,3,0),(2,3,0,1),(3,0,1,2)]`` —
            each image lands in each quadrant exactly once across the 4 mosaics.
            Each ``_mosaic`` call draws its own split center / per-image scales /
            ``random_perspective`` params (4 independent draws).
          * single branch: ``_single`` on each of the 4 examples (4 independent
            ``random_perspective`` draws).

        Both branches return the SAME dict structure with every value stacked on
        axis 0 to ``[4, ...]`` (annotations zero/-1/False/0.0 padded to the group
        max instance count before stacking). Downstream ``.unbatch()`` yields 4
        single examples with static image shape ``[H, W, 3]``.

        Both the mosaic branch and the single-image branch run random_perspective,
        so every training image gets the full geometric transform exactly once.
        """

        def _fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            def _select(d: Dict, i: int) -> Dict:
                return {k: v[i] for k, v in d.items()}

            examples = [_select(batch, i) for i in range(4)]

            if not is_training:
                results = [self._single(ex) for ex in examples]
                return _stack_results(results)

            do_mosaic = tf.random.uniform([]) < self._mosaic_freq

            def _mosaic_branch():
                perms = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
                results = [
                    self._mosaic(
                        examples[p[0]], examples[p[1]],
                        examples[p[2]], examples[p[3]],
                    )
                    for p in perms
                ]
                return _stack_results(results)

            def _single_branch():
                results = [self._single(ex) for ex in examples]
                return _stack_results(results)

            return tf.cond(do_mosaic, _mosaic_branch, _single_branch)

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
        """random_perspective with this module's configured params → output size.

        The warp scale gain is drawn from the EXPLICIT [aug_scale_min,
        aug_scale_max] config bounds. (The earlier symmetric-magnitude form
        widened the configured [0.4, 1.9] to [0.1, 1.9], occasionally shrinking
        content to ~1% area — the "mostly-gray frame" bug.)
        """
        return random_perspective(
            image, boxes, polygons,
            target_h=self._H, target_w=self._W,
            degrees=self._degrees,
            translate=self._translate,
            scale_min=self._scale_min,
            scale_max=self._scale_max,
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
            # When a side field is absent, the fallback must have the SAME length
            # as ``keep`` (one entry per box fed to random_perspective), not the
            # 0-length ``tf.zeros([0], dtype)`` used previously — boolean_mask
            # requires mask and tensor to share the masked dimension, so a
            # length-0 fallback against an N-length ``keep`` raises
            # ``ValueError: Shapes (0,) and (N,) are incompatible``. Build the
            # fallback from ``keep`` so it is always N-length.
            v = ex.get(key, None)
            v = tf.zeros_like(keep, dtype=dtype) if v is None else tf.cast(v, dtype)
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
        """Mosaic-warp 4 images to the output in a single resample per source.

        Image path: assemble the 2× canvas (per-image ``tf.image.resize`` at the
        drawn scale, ``_place_in_cell`` crop/pad, concat) and apply ONE
        ``apply_perspective_image`` warp canvas→output.

        Why canvas and not composed-affine: a composed variant (per-quadrant
        affine folded into ``M``, warping each source full-frame to the output)
        was tried and measured ~95 ms·core per emitted image on the production
        CPU — ``ImageProjectiveTransformV3`` is several times slower per output
        pixel than ``tf.image.resize`` there, and the composed form pays 4 full
        warps per mosaic vs 4 cheap resizes + 1 warp here. Geometry and the
        label path are identical in both forms (labels go through
        ``_scale_box_poly_to_canvas`` → ``transform_boxes_polygons`` with the
        same single ``M``); only the image resampling chain differs.

        Quadrant → example: TL=one, TR=two, BL=three, BR=four. The warp scale
        gain is drawn from the explicit [aug_scale_min, aug_scale_max] bounds.
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
        boxes_list, polys_list = [], []

        # Draw the global canvas→output matrix ONCE (same params as self._warp;
        # scale gain from the explicit [aug_scale_min, aug_scale_max] bounds).
        M = make_perspective_matrix(
            h_in=H2, w_in=W2,
            target_h=H, target_w=W,
            degrees=self._degrees,
            translate=self._translate,
            scale_min=self._scale_min,
            scale_max=self._scale_max,
            shear=self._shear,
            perspective=self._perspective,
        )

        # Per quadrant: draw the per-image scale, resize, and place into its
        # cell (crop overflow / pad voids with gray 114) — then assemble the 2×
        # canvas and apply ONE warp canvas→output. tf.image.resize is far
        # cheaper per pixel than the warp op, so total resample cost is
        # 4 small resizes + 1 output-sized warp per emitted mosaic.
        cells = []
        for i, ex in enumerate(examples):
            img = ex['image']
            h_in = tf.shape(img)[0]
            w_in = tf.shape(img)[1]
            h_in_f = tf.cast(h_in, tf.float32)
            w_in_f = tf.cast(w_in, tf.float32)
            scale = tf.random.uniform([], self._scale_min, self._scale_max)
            nh = tf.maximum(tf.cast(tf.round(h_in_f * scale), tf.int32), 1)
            nw = tf.maximum(tf.cast(tf.round(w_in_f * scale), tf.int32), 1)
            R = tf.cast(
                tf.image.resize(tf.cast(img, tf.float32), [nh, nw], method='bilinear'),
                tf.uint8,
            )

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

            cells.append(_place_in_cell(R, cell_h, cell_w, top_y, top_x))

            padh = off_y + top_y
            padw = off_x + top_x
            b_c, p_c = _scale_box_poly_to_canvas(ex, nh, nw, padh, padw, H2, W2)
            boxes_list.append(b_c)
            polys_list.append(p_c)

        # Assemble the 2× canvas from the 4 quadrant cells and warp once.
        top_row = tf.concat([cells[0], cells[1]], axis=1)   # [yc,    2W, 3]
        bot_row = tf.concat([cells[2], cells[3]], axis=1)   # [2H-yc, 2W, 3]
        canvas  = tf.concat([top_row, bot_row], axis=0)     # [2H,    2W, 3]
        image = apply_perspective_image(canvas, M, H, W)

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

        # Annotation transform uses the SAME global M as the legacy single warp,
        # so the label math is bit-identical to canvas-then-_warp.
        boxes_out, keep, polys_out = transform_boxes_polygons(
            boxes_all, polys_all, M,
            h_in=H2, w_in=W2,
            target_h=H, target_w=W,
            area_thresh=self._area_thresh, min_side=0.005,
        )

        anns = self._filtered_anns(merged_src, boxes_out, polys_out, keep)
        anns['image']  = image
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
