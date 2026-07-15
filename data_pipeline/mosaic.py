"""Mosaic and MixUp augmentation combining multiple images.

Mosaic stitches 4 images into a 2×-size canvas at a random center, then one
``random_perspective`` warps the canvas back to the output size. The same warp
runs on non-mosaic single images, so it is the pipeline's one geometric
transform (the parser applies no affine).

Placement is upright (matching stock YOLO mosaic): each source is resized so its
long side equals the output size and placed toward its cell's center corner
(overflow cropped, no letterbox pad) — a consistent per-image scale, not a random
one. Size variety comes only from the canvas→output warp's scale gain
``[aug_scale_min, aug_scale_max]``. Rotation is rare: the warp rotates only with
probability ``rotate_prob`` by ±``degrees``. The split center shifts H+V
(``mosaic_center``), so each tile's visible crop varies and boxes/polygons are cut
at the moving edges.

Group / diversity semantics
---------------------------
``mosaic_fn`` maps a ``padded_batch(group_size)`` group to
``group_size // decodes_per_output`` samples. R = ``decodes_per_output`` is both
the reuse knob and the decode multiplier (freshly-decoded images per output):

  * Each output draws 4 sources from one per-group permutation at shifts from an
    R-keyed Sidon set (``_SIDON_SHIFTS``). At R=4 the shifts are the contiguous
    window {0,1,2,3} and tile the permutation: every output's 4 images are
    disjoint (zero reuse). At R<4 each image recurs in exactly 4/R outputs, and
    the Sidon shifts guarantee any two outputs share at most one source image.
    R<4's cost vs R=4 is the count of distinct images per epoch, not correlation.
  * ``group_size`` is the draw pool (larger = more varied combinations at the same
    R); must be a multiple of R and >= 4.
  * The mosaic/single decision is a per-output coin flip, so the per-sample mosaic
    probability is exactly ``mosaic_frequency`` with no batch clustering.

Epoch accounting is unaffected: the trainer runs a fixed number of steps and the
final ``.batch(batch_size)`` is downstream, so R only changes how many source
images are decoded per emitted sample.

Configuration (parser.mosaic in the experiment YAML):
    mosaic_frequency: 0.5
    mixup_frequency: 0.0     (per-output probability of blending with a second
                              mosaic — Ultralytics MixUp; 0 = off, the default)
    group_size: 32           (mosaic source pool per group)
    decodes_per_output: 4    (R: 4 = no reuse; lower = more reuse, less decode)
    mosaic_center: 0.25      (half-range of the split point; the 2× canvas split
                              lands in [H(1-2c), H(1+2c)])
    aug_scale_min / aug_scale_max: canvas→output warp scale-gain bounds. Per-image
                              placement scale is fixed (long side fills the output),
                              not random.
    degrees: 10.0            (rotation ± magnitude, degrees — applied only when the
                              rotate_prob coin fires)
    rotate_prob: 0.10        (probability a given output is rotated at all)
    shear: 0.0               (shear ±, degrees; 0 = no shear)
    perspective: 0.0         (perspective coefficient ±; 0 disables)
    translate: 0.1           (translation ± as a fraction of output size)
    area_thresh: 0.5         (min visible box-area fraction to keep on the
                              MOSAIC path — the legacy mosaic value; the single
                              path filters at the parser-level 0.1 instead)

Classes:
    Mosaic: manages both Mosaic and MixUp augmentations.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import tensorflow as tf

from data_pipeline.augmentations import (
    apply_perspective_image,
    letterbox_geometry,
    make_perspective_matrix,
    random_horizontal_flip,
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
    # Source validity keys off the -1.0 sentinel: any vertex with x > -1.0 is a
    # real vertex, including a slightly-negative coordinate from mosaic-cell
    # placement. `> -1.0` (not `>= 0.0`) carries it into the canvas; the
    # subsequent random_perspective clips it to the output edge.
    valid = pts[:, :, 0] > -1.0
    x_c = (pts[:, :, 0] * nw_f + padw_f) / W2_f
    y_c = (pts[:, :, 1] * nh_f + padh_f) / H2_f
    neg1 = tf.fill(tf.shape(x_c), -1.0)
    x_c = tf.where(valid, x_c, neg1)
    y_c = tf.where(valid, y_c, neg1)
    polys_c = tf.reshape(tf.stack([x_c, y_c], axis=-1), [N, max_v])
    return boxes_c, polys_c


# _PAD_SPEC (defined below): how each per-instance (axis-0 = N) annotation field
# is padded to the group-max instance count before stacking. (key, pad_value)
#   - boxes pad with zero rows: the parser's clip_boxes degenerate-row filter
#     (strict > 0 on both sides) drops zero boxes (including after flip), so
#     they never train.
#   - polygons pad with -1.0 rows (the invalid-vertex sentinel).
#   - classes/dontcare pad 0; is_crowd pad False (crowd filtering is
#     config-conditional in the parser); area/dists pad 0.0.
#
# _SIDON_SHIFTS: source-selection shift sets keyed by R = decodes_per_output.
# Output j draws its 4 sources at perm[(j*R + s) % G] for s in the set. A set is
# valid for a given R when:
#   1. Uniform reuse — the shifts cover each residue class mod R exactly 4/R
#      times, so every image is used in exactly 4/R outputs.
#   2. Sidon property (mod G, including sign) — all pairwise shift differences
#      are distinct, so any two outputs share at most one source image. A
#      difference d collides with -d' when d + d' == G (with itself when 2d == G).
# _SIDON_MIN_G is the smallest group size at which each set's difference
# collection stays collision-free mod G ({±1,±2,±3,±4,±6,±7} for R=1;
# {±1,±3,±4,±5,±8,±9} for R=2). Below it (and for R not listed) selection falls
# back to the contiguous window. R=4 uses the contiguous window by construction:
# the windows tile the permutation, so the Sidon concern is vacuous.
_SIDON_SHIFTS = {1: (0, 1, 3, 7), 2: (0, 1, 4, 9), 4: (0, 1, 2, 3)}
_SIDON_MIN_G = {1: 15, 2: 20, 4: 4}


def _window_shifts(decodes_per_output: int, group_size: int) -> Tuple[int, int, int, int]:
    """The four source-selection shifts for a given R and group size."""
    shifts = _SIDON_SHIFTS.get(decodes_per_output)
    if shifts is not None and group_size >= _SIDON_MIN_G[decodes_per_output]:
        return shifts
    return (0, 1, 2, 3)


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
        mosaic_crop_mode: str = "scale",
        mosaic_center: float = 0.25,
        aug_scale_min: float = 0.5,
        aug_scale_max: float = 1.5,
        area_thresh: float = 0.5,
        with_polygons: bool = True,
        degrees: float = 0.0,
        shear: float = 0.0,
        perspective: float = 0.0,
        translate: float = 0.1,
        rotate_prob: float = 0.0,
        group_size: int = 32,
        decodes_per_output: int = 4,
        tile_crop_min: float = 0.0,
        tile_crop_max: float = 0.0,
        single_scale_min: Optional[float] = None,
        single_scale_max: Optional[float] = None,
        single_translate: Optional[float] = None,
        single_area_thresh: Optional[float] = None,
        random_flip: bool = False,
        single_rotate: bool = False,
        single_rotate_degrees: Optional[float] = None,
        copy_paste_module=None,
    ):
        self._H = output_size[0]
        self._W = output_size[1]
        self._mosaic_freq = mosaic_frequency
        self._mixup_freq  = mixup_frequency
        self._crop_mode   = mosaic_crop_mode
        self._center      = mosaic_center
        self._scale_min   = aug_scale_min
        self._scale_max   = aug_scale_max
        self._area_thresh = area_thresh
        self._with_polys  = with_polygons
        # Mosaic tiles/canvas never rotate: `degrees` / `rotate_prob` are accepted
        # for compatibility but wired to 0 in every warp. The only rotation is the
        # optional single-path pre-warp below.
        self._degrees     = degrees
        self._shear       = shear
        self._perspective = perspective
        self._translate   = translate
        self._rotate_prob = rotate_prob
        # Per-tile RANDOM-WINDOW CROP. When tile_crop_max > 0, each mosaic tile
        # crops a random window of side fraction s ~ U[tile_crop_min, tile_crop_max]
        # of its content (random position within bounds), then scales the crop to
        # its quadrant — a zoom/translate scale-invariance signal. 0/0 disables it
        # (the content region fills its quadrant unchanged).
        if (tile_crop_max > 0.0) and not (0.0 < tile_crop_min <= tile_crop_max <= 1.0):
            raise ValueError(
                f"tile_crop bounds invalid: need 0 < min <= max <= 1, got "
                f"[{tile_crop_min}, {tile_crop_max}]"
            )
        self._tile_crop_min = tile_crop_min
        self._tile_crop_max = tile_crop_max
        # Optional single-path pre-warp rotation (parser-level rotate / rotate_degrees),
        # applied to non-mosaic images only, before flip, via the perspective-matrix
        # machinery (a pure centered rotation). Off by default.
        self._single_rotate = bool(single_rotate) and (single_rotate_degrees is not None)
        self._single_rotate_degrees = (
            single_rotate_degrees if single_rotate_degrees is not None else 0.0)
        # Copy-paste module (optional). When set, each mosaic TILE independently
        # pastes its own cnp candidate with the module's probability; the single
        # path ignores cnp fields entirely.
        self._copy_paste_module = copy_paste_module
        self._copy_paste_fn = (
            copy_paste_module.process_fn(is_training=True)
            if copy_paste_module is not None else None)
        # Non-mosaic (single) path warp params. The two paths differ: mosaics get
        # the [aug_scale_min/max] warp gain with no translate; singles get no scale
        # gain (1.0) but a small translate. None falls back to the mosaic values;
        # input_reader wires the parser-level aug_scale_min/max + aug_rand_translate.
        self._single_scale_min = (
            single_scale_min if single_scale_min is not None else aug_scale_min)
        self._single_scale_max = (
            single_scale_max if single_scale_max is not None else aug_scale_max)
        self._single_translate = (
            single_translate if single_translate is not None else translate)
        # The mosaic warp culls at area_thresh (0.5 — the legacy mosaic value);
        # the single-image warp reads the parser-level area_thresh (0.1) via
        # single_area_thresh, so the two paths remain independently configurable.
        self._single_area_thresh = (
            single_area_thresh if single_area_thresh is not None else area_thresh)
        # Flip ownership: when True, this module flips — each mosaic tile
        # independently (before placement; the assembled canvas is never mirrored)
        # and each single image once. The detection train parser's flip must then
        # be off or images would flip twice (input_reader wires that). Default
        # False so direct constructions keep deterministic geometry.
        self._random_flip = random_flip
        # Image-diversity controls (see module docstring). A group of `group_size`
        # decoded images yields `outputs_per_group = group_size // decodes_per_output`
        # emitted samples; each mosaic draws 4 source images from the group.
        if group_size < 4:
            raise ValueError(f"mosaic group_size must be >= 4, got {group_size}")
        if group_size % decodes_per_output != 0:
            raise ValueError(
                f"mosaic group_size ({group_size}) must be a multiple of "
                f"decodes_per_output ({decodes_per_output})"
            )
        self._group_size = group_size
        self._decodes_per_output = decodes_per_output
        self._outputs_per_group = group_size // decodes_per_output

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def mosaic_fn(self, is_training: bool = True) -> Callable:
        """Return a function for ``tf.data.Dataset.map()`` over a ``padded_batch(group_size)`` group.

        Maps a ``group_size``-image group to ``outputs_per_group = group_size //
        decodes_per_output`` emitted samples (leading dim = outputs_per_group), so the
        decode cost per emitted sample is exactly ``decodes_per_output`` (R) and the epoch
        step count is unchanged (the trainer runs a fixed number of steps).

        Image diversity — each output independently:
          * flips ``mosaic_frequency`` (a **per-output** coin flip — not per-group — so the
            per-sample mosaic probability is exactly ``mosaic_frequency`` with no batch
            clustering);
          * if mosaic, draws **4 distinct** source images from one per-group random
            permutation at the ``_window_shifts(R, G)`` offsets. At R=4 the shifts tile
            the permutation, so every output's 4 images are disjoint from every other
            output's — stock-YOLO, zero reuse. At R<4 each image recurs in ``4/R``
            outputs, and the Sidon shifts guarantee any two outputs share at most one
            source image (no near-duplicate outputs);
          * if single, ``_single`` on the output's first source image.

        ``_mosaic`` / ``_single`` each draw their own split center / per-image scales /
        ``random_perspective`` params, and every output runs ``random_perspective`` exactly
        once. All outputs return the SAME dict structure; ``_stack_results`` pads the
        per-instance fields to the group-max instance count and stacks to leading dim
        ``outputs_per_group``. Downstream ``.unbatch()`` yields one example per output with
        static image shape ``[H, W, 3]``.

        The ``is_training=False`` path single-warps every image in the group (no mixing),
        but the eval dataset is built without mosaic, so it is unused in practice.
        """
        G = self._group_size
        P = self._outputs_per_group

        def _fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            def _select(d: Dict, i) -> Dict:
                # i may be a Python int or a scalar tensor (dynamic gather by index).
                return {k: tf.gather(v, i) for k, v in d.items()}

            if not is_training:
                results = [self._single(_select(batch, i)) for i in range(G)]
                return _stack_results(results)

            # One random permutation of the group per call. Output j reads sources
            # at perm[(j·R + s) % G] for the four Sidon shifts s. At R=4 the shifts
            # are the contiguous window {0,1,2,3} and tile the permutation (disjoint
            # outputs, zero reuse). At R<4 the Sidon set keeps per-image reuse at
            # exactly 4/R while any two outputs share at most one source image.
            R = self._decodes_per_output
            shifts = _window_shifts(R, G)
            perm = tf.random.shuffle(tf.range(G))

            results = []
            for j in range(P):
                idx = [perm[(R * j + s) % G] for s in shifts]
                examples = [_select(batch, idx[k]) for k in range(4)]
                do_mosaic = tf.random.uniform([]) < self._mosaic_freq
                out_j = tf.cond(
                    do_mosaic,
                    lambda ex=examples: self._mosaic(ex[0], ex[1], ex[2], ex[3]),
                    lambda ex=examples: self._single(ex[0]),
                )
                # MixUp (Ultralytics): with probability mixup_frequency, blend this
                # output with a second mosaic from a distant window of the same group
                # (offset by G//2 so its 4 sources differ) and concatenate labels.
                # Gated by a Python check on the config constant, so at the default
                # mixup_frequency=0.0 the partner ops are never added to the graph.
                # The partner mosaic is built inside the true branch, so it runs only
                # when the coin fires.
                if self._mixup_freq > 0:
                    pidx = [perm[(R * j + G // 2 + s) % G] for s in shifts]
                    pex = [_select(batch, pidx[k]) for k in range(4)]
                    do_mixup = tf.random.uniform([]) < self._mixup_freq
                    out_j = tf.cond(
                        do_mixup,
                        lambda p=out_j, e=pex: self._mixup(
                            p, self._mosaic(e[0], e[1], e[2], e[3])),
                        lambda p=out_j: p,
                    )
                results.append(out_j)
            return _stack_results(results)

        return _fn

    # ------------------------------------------------------------------
    # Geometric transform helpers
    # ------------------------------------------------------------------

    def _flip_example(self, ex: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Horizontally flip an example dict with probability 0.5.

        ``random_horizontal_flip`` draws its own coin, mirrors box x
        (xmin ↔ 1 − xmax) and valid polygon x (keeping the -1 sentinel).
        Applied per mosaic TILE (before placement) and per single image, so
        tiles of one mosaic flip independently — the assembled canvas itself
        is never mirrored.
        """
        img, boxes, polys = random_horizontal_flip(
            ex['image'],
            ex.get('groundtruth_boxes', tf.zeros([0, 4])),
            ex.get('groundtruth_polygons', tf.zeros([0, 2])),
        )
        out = dict(ex)
        out['image'] = img
        out['groundtruth_boxes'] = boxes
        out['groundtruth_polygons'] = polys
        return out

    def _warp(
        self,
        image: tf.Tensor,
        boxes: tf.Tensor,
        polygons: tf.Tensor,
        scale_min: Optional[float] = None,
        scale_max: Optional[float] = None,
        translate: Optional[float] = None,
        area_thresh: Optional[float] = None,
        min_side: float = 0.003,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """random_perspective with this module's params → output size.

        The warp never rotates (degrees/rotate_prob forced to 0); single-path
        rotation is the separate optional pre-warp step. The scale gain is drawn
        from the [aug_scale_min, aug_scale_max] bounds, or the per-call override
        (the single path passes its own bounds/translate, and min_side=0.0 —
        the 2px size floor is mosaic-only).
        """
        return random_perspective(
            image, boxes, polygons,
            target_h=self._H, target_w=self._W,
            degrees=0.0,
            translate=self._translate if translate is None else translate,
            scale_min=self._scale_min if scale_min is None else scale_min,
            scale_max=self._scale_max if scale_max is None else scale_max,
            shear=self._shear,
            perspective=self._perspective,
            area_thresh=(self._area_thresh if area_thresh is None
                         else area_thresh),
            rotate_prob=0.0,
            min_side=min_side,
        )

    def _rotate_image_boxes_polys(
        self,
        image: tf.Tensor,
        boxes: tf.Tensor,
        polygons: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Pure centered rotation by uniform(-deg, +deg) via the perspective machinery.

        Rotates image + boxes + polygons about the image center by a single angle
        drawn from [-single_rotate_degrees, +single_rotate_degrees]. Built with
        ``make_perspective_matrix`` (rotate_prob=1.0, unit scale, no translate/shear/
        perspective) so no new geometry code is introduced. Coordinates are only
        transformed + clipped here (no rows dropped); the subsequent single warp's
        filter culls anything rotated out of frame.
        """
        H, W = self._H, self._W
        M = make_perspective_matrix(
            h_in=H, w_in=W,
            target_h=H, target_w=W,
            degrees=self._single_rotate_degrees,
            translate=0.0,
            scale_min=1.0, scale_max=1.0,
            shear=0.0,
            perspective=0.0,
            rotate_prob=1.0,
        )
        image_out = apply_perspective_image(image, M, H, W)
        boxes_out, _keep, polys_out = transform_boxes_polygons(
            boxes, polygons, M,
            h_in=H, w_in=W,
            target_h=H, target_w=W,
            area_thresh=0.0, min_side=0.0,
        )
        return image_out, boxes_out, polys_out

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
            # When a side field is absent, the fallback must be the same length as
            # ``keep`` (one entry per box fed to random_perspective): boolean_mask
            # requires the mask and tensor to share the masked dimension, so a
            # length-0 fallback against an N-length ``keep`` is a shape error. Build
            # it from ``keep`` so it is always N-length.
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
        """Non-mosaic path: (optional rotation) → flip → single-path warp.

        The image arrives LETTERBOXED to the output size (aspect-preserved content
        inset with gray margins), so border objects survive the pre-resize and the
        subsequent translate can no longer expel them wholesale. Optional pre-warp
        rotation (single_rotate) is applied first, then flip, then the single warp
        (single_scale_min/max + single_translate — no scale gain, a small translate).
        The single path ignores all cnp_* fields (copy-paste is mosaic-only).
        """
        img = ex['image']
        boxes = ex.get('groundtruth_boxes', tf.zeros([0, 4]))
        polys = ex.get('groundtruth_polygons', tf.zeros([0, 2]))
        # Optional pre-warp rotation (top-level, before flip). Off by default.
        if self._single_rotate:
            img, boxes, polys = self._rotate_image_boxes_polys(img, boxes, polys)
        if self._random_flip:
            img, boxes, polys = random_horizontal_flip(img, boxes, polys)
        img_out, boxes_out, keep, polys_out = self._warp(
            img, boxes, polys,
            scale_min=self._single_scale_min,
            scale_max=self._single_scale_max,
            translate=self._single_translate,
            area_thresh=self._single_area_thresh,
            # Legacy convention: the ~2px min_side floor applies on the mosaic
            # branch only; non-mosaic images keep sub-2px objects' labels. A
            # fully warped-out box still drops via the zero-area ratio term.
            min_side=0.0,
        )
        anns = self._filtered_anns(ex, boxes_out, polys_out, keep)
        anns['image']  = img_out
        anns['height'] = tf.constant(self._H, tf.int32)
        anns['width']  = tf.constant(self._W, tf.int32)
        return anns

    # ------------------------------------------------------------------
    # Per-tile content preparation (slice content, copy-paste, tile-crop)
    # ------------------------------------------------------------------

    def _content_example(self, ex: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Slice the letterbox CONTENT region out of a pre-resized tile.

        The pre-resize letterboxes each image to the output size (aspect-preserved
        content inset with gray-114 margins). For the mosaic path the gray margins
        must be removed so a tile is the object content only, scaled to fill its
        quadrant. Using the ORIGINAL capture dims ('height'/'width') the content box
        is reconstructed with the SAME letterbox geometry, the content is sliced, and
        the GT (currently letterbox-normalized) is mapped back to CONTENT-normalized
        coords (the exact inverse of the pre-resize — i.e. the original decoder
        coords). When 'height'/'width' are absent (some direct-call tests) the whole
        image is treated as content (identity), which is exact for square inputs.
        """
        H, W = self._H, self._W
        img_lb = ex['image']
        boxes = ex.get('groundtruth_boxes', tf.zeros([0, 4], tf.float32))
        polys = ex.get('groundtruth_polygons', tf.zeros([0, 2], tf.float32))

        if ('height' in ex) and ('width' in ex):
            _scale, content_h, content_w, pad_top, pad_left = letterbox_geometry(
                ex['height'], ex['width'], H, W)
            content = tf.slice(img_lb, [pad_top, pad_left, 0],
                               [content_h, content_w, -1])
            ch_f = tf.cast(content_h, tf.float32)
            cw_f = tf.cast(content_w, tf.float32)
            pt_f = tf.cast(pad_top, tf.float32)
            pl_f = tf.cast(pad_left, tf.float32)
            H_f = tf.cast(H, tf.float32)
            W_f = tf.cast(W, tf.float32)

            # letterbox-normalized -> content-normalized (inverse of the pre-resize).
            ymin = (boxes[:, 0] * H_f - pt_f) / ch_f
            xmin = (boxes[:, 1] * W_f - pl_f) / cw_f
            ymax = (boxes[:, 2] * H_f - pt_f) / ch_f
            xmax = (boxes[:, 3] * W_f - pl_f) / cw_f
            boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

            Np = tf.shape(polys)[0]
            maxv = tf.shape(polys)[1]
            pts = tf.reshape(polys, [Np, -1, 2])
            valid = pts[:, :, 0] > -1.0
            xv = (pts[:, :, 0] * W_f - pl_f) / cw_f
            yv = (pts[:, :, 1] * H_f - pt_f) / ch_f
            neg1 = tf.fill(tf.shape(xv), -1.0)
            xv = tf.where(valid, xv, neg1)
            yv = tf.where(valid, yv, neg1)
            polys = tf.reshape(tf.stack([xv, yv], axis=-1), [Np, maxv])
            h_orig = tf.cast(ex['height'], tf.int32)
            w_orig = tf.cast(ex['width'], tf.int32)
        else:
            content = img_lb
            h_orig = tf.shape(img_lb)[0]
            w_orig = tf.shape(img_lb)[1]

        cex = dict(ex)
        cex['image'] = content
        cex['groundtruth_boxes'] = boxes
        cex['groundtruth_polygons'] = polys
        cex['height'] = h_orig
        cex['width'] = w_orig
        return cex

    def _paste_tile(self, cex: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Copy-paste one cnp candidate onto a tile with the module's probability.

        No-op when the module or the cnp fields are absent. The cnp image is sliced
        back to its native size (padded_batch pads it to the group-max) so the
        object coords in orig_bbox/points stay valid. The single path never calls
        this, so pastes are mosaic-only.
        """
        if self._copy_paste_fn is None or 'cnp_image' not in cex:
            return cex
        obj = {
            'image': tf.slice(cex['cnp_image'], [0, 0, 0],
                              [cex['cnp_h'], cex['cnp_w'], -1]),
            'orig_bbox': cex['cnp_orig_bbox'],
            'label': cex['cnp_label'],
            'points': cex['cnp_points'],
        }
        return self._copy_paste_fn(cex, obj)

    def _tile_crop_example(self, cex: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Random-window crop of a tile's content (side fraction s ~ U[min, max]).

        Draws a window of side fraction ``s`` of the content dims at a uniform-random
        position, crops the image and remaps + clips the GT to the window; rows that
        fall fully outside are dropped (with the full aligned mask across every
        per-instance field). The cropped window then scales to fill the quadrant.
        """
        img = cex['image']
        ch = tf.shape(img)[0]
        cw = tf.shape(img)[1]
        ch_f = tf.cast(ch, tf.float32)
        cw_f = tf.cast(cw, tf.float32)
        s = tf.random.uniform([], self._tile_crop_min, self._tile_crop_max)
        win_h = tf.maximum(tf.cast(tf.round(ch_f * s), tf.int32), 1)
        win_w = tf.maximum(tf.cast(tf.round(cw_f * s), tf.int32), 1)
        top = tf.random.uniform([], 0, tf.maximum(ch - win_h, 0) + 1, dtype=tf.int32)
        left = tf.random.uniform([], 0, tf.maximum(cw - win_w, 0) + 1, dtype=tf.int32)
        crop = tf.slice(img, [top, left, 0], [win_h, win_w, -1])

        boxes_w, polys_w, keep = self._apply_window(
            cex['groundtruth_boxes'], cex['groundtruth_polygons'],
            ch, cw, top, left, win_h, win_w,
        )

        out = dict(cex)
        out['image'] = crop
        out['groundtruth_boxes'] = tf.boolean_mask(boxes_w, keep)
        out['groundtruth_polygons'] = tf.boolean_mask(polys_w, keep)
        for key, dtype in (
            ('groundtruth_classes', tf.int64),
            ('groundtruth_is_crowd', tf.bool),
            ('groundtruth_area', tf.float32),
            ('groundtruth_dontcare', tf.int64),
            ('groundtruth_dists', tf.float32),
        ):
            if key in cex:
                out[key] = tf.boolean_mask(tf.cast(cex[key], dtype), keep)
        return out

    @staticmethod
    def _apply_window(
        boxes: tf.Tensor,
        polygons: tf.Tensor,
        ch: tf.Tensor, cw: tf.Tensor,
        top: tf.Tensor, left: tf.Tensor,
        win_h: tf.Tensor, win_w: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Map content-normalized GT into a crop window, clip, and mark keep.

        content pixel = coord * content_dim; window-normalized =
        (content_px - window_origin) / window_dim, clipped to [0, 1]. A box with
        zero visible area after clip is dropped (keep=False). Polygon vertices are
        clipped to the window edge; the -1.0 sentinel is preserved. Pure function
        of explicit window ints so the geometry is unit-testable.
        """
        ch_f = tf.cast(ch, tf.float32)
        cw_f = tf.cast(cw, tf.float32)
        top_f = tf.cast(top, tf.float32)
        left_f = tf.cast(left, tf.float32)
        wh_f = tf.cast(win_h, tf.float32)
        ww_f = tf.cast(win_w, tf.float32)

        by0 = tf.clip_by_value((boxes[:, 0] * ch_f - top_f) / wh_f, 0.0, 1.0)
        bx0 = tf.clip_by_value((boxes[:, 1] * cw_f - left_f) / ww_f, 0.0, 1.0)
        by1 = tf.clip_by_value((boxes[:, 2] * ch_f - top_f) / wh_f, 0.0, 1.0)
        bx1 = tf.clip_by_value((boxes[:, 3] * cw_f - left_f) / ww_f, 0.0, 1.0)
        boxes_w = tf.stack([by0, bx0, by1, bx1], axis=1)
        keep = tf.logical_and((by1 - by0) > 1e-9, (bx1 - bx0) > 1e-9)

        Np = tf.shape(polygons)[0]
        maxv = tf.shape(polygons)[1]
        pts = tf.reshape(polygons, [Np, -1, 2])
        valid = pts[:, :, 0] > -1.0
        xv = tf.clip_by_value((pts[:, :, 0] * cw_f - left_f) / ww_f, 0.0, 1.0)
        yv = tf.clip_by_value((pts[:, :, 1] * ch_f - top_f) / wh_f, 0.0, 1.0)
        neg1 = tf.fill(tf.shape(xv), -1.0)
        xv = tf.where(valid, xv, neg1)
        yv = tf.where(valid, yv, neg1)
        polys_w = tf.reshape(tf.stack([xv, yv], axis=-1), [Np, maxv])
        return boxes_w, polys_w, keep

    def _prepare_tile(self, ex: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Slice content → copy-paste → per-tile flip → tile-crop for one tile."""
        cex = self._content_example(ex)
        cex = self._paste_tile(cex)
        if self._random_flip:
            # Per-TILE independent flip; the assembled canvas is never mirrored.
            cex = self._flip_example(cex)
        if self._tile_crop_max > 0.0:
            cex = self._tile_crop_example(cex)
        return cex

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

        Assemble the 2× canvas (per-image ``tf.image.resize`` at the drawn scale,
        ``_place_in_cell`` crop/pad, concat) and apply one ``apply_perspective_image``
        warp canvas→output. Per-tile resize + one warp is used because a full-frame
        per-tile warp variant is several times slower on CPU
        (``ImageProjectiveTransformV3`` costs far more per pixel than
        ``tf.image.resize``); both forms are geometrically identical, and labels go
        through ``_scale_box_poly_to_canvas`` → ``transform_boxes_polygons`` with the
        same single ``M``.

        Quadrant → example: TL=one, TR=two, BL=three, BR=four. The warp scale gain
        is drawn from the [aug_scale_min, aug_scale_max] bounds.
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

        # Per-tile preparation: slice the letterbox content, copy-paste (mosaic-only,
        # per tile), flip (per tile), tile-crop. Each returns a content example whose
        # GT is CONTENT-normalized (matching nh/nw below).
        examples = [self._prepare_tile(ex) for ex in (one, two, three, four)]
        boxes_list, polys_list = [], []

        # Draw the global canvas→output matrix once (same params as self._warp;
        # scale gain from the [aug_scale_min, aug_scale_max] bounds). The mosaic
        # canvas never rotates (degrees/rotate_prob = 0).
        M = make_perspective_matrix(
            h_in=H2, w_in=W2,
            target_h=H, target_w=W,
            degrees=0.0,
            translate=self._translate,
            scale_min=self._scale_min,
            scale_max=self._scale_max,
            shear=self._shear,
            perspective=self._perspective,
            rotate_prob=0.0,
        )

        # Per quadrant: resize the (prepared) content to fill its cell, place into
        # the cell (crop overflow / pad voids with gray 114) — then assemble the 2×
        # canvas and apply ONE warp canvas→output. tf.image.resize is far cheaper
        # per pixel than the warp op, so total resample cost is 4 small resizes +
        # 1 output-sized warp per emitted mosaic.
        cells = []
        for i, ex in enumerate(examples):
            img = ex['image']
            h_in = tf.shape(img)[0]
            w_in = tf.shape(img)[1]
            h_in_f = tf.cast(h_in, tf.float32)
            w_in_f = tf.cast(w_in, tf.float32)
            # Placement scale: long side = output size. The content (or tile-crop
            # window) is scaled to fill its quadrant; tiles anchor at the moving
            # center corner so overflow only ever falls AWAY from the other cells
            # and is cropped at the canvas edge by _place_in_cell — no cross-cell
            # label corruption. Labels map through the same nh/nw/pad and are
            # clipped/dropped by the final warp's transform_boxes_polygons.
            long_side = tf.maximum(h_in_f, w_in_f)
            place_scale = tf.cast(H, tf.float32) / long_side
            nh = tf.maximum(tf.cast(tf.round(h_in_f * place_scale), tf.int32), 1)
            nw = tf.maximum(tf.cast(tf.round(w_in_f * place_scale), tf.int32), 1)
            # Skip the resize when the source is already the placement size (the common
            # case: images are pre-resized to H² before mosaic, so place_scale == 1 and
            # the resize is a no-op that still allocates + runs a bilinear kernel over
            # ~5.4M pixels per quadrant). Returning `img` is bit-identical (a same-size
            # bilinear resize + uint8 round-trip is the identity on uint8 pixels).
            R = tf.cond(
                tf.logical_and(tf.equal(nh, h_in), tf.equal(nw, w_in)),
                lambda im=img: im,
                lambda im=img, nh_=nh, nw_=nw: tf.cast(
                    tf.image.resize(tf.cast(im, tf.float32), [nh_, nw_], method='bilinear'),
                    tf.uint8),
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

        # Annotation transform uses the same global M as the canvas→output image
        # warp, so the label math matches canvas-then-_warp exactly.
        boxes_out, keep, polys_out = transform_boxes_polygons(
            boxes_all, polys_all, M,
            h_in=H2, w_in=W2,
            target_h=H, target_w=W,
            area_thresh=self._area_thresh, min_side=0.003,
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
        """Blend two images with a Beta(32, 32) weight and concatenate their labels.

        Beta(32, 32) concentrates the mix ratio tightly around 0.5 (Ultralytics' MixUp
        recipe), sampled as ``g1/(g1+g2)`` with ``gi ~ Gamma(32)``. Both inputs are
        already at the output size (mosaic/single results), so the resize is a no-op
        safeguard. Labels (boxes/classes/polygons/dist/…) from both are concatenated;
        the downstream parser caps at ``max_num_instances`` and builds the radial
        polygon target from the union.
        """
        g1 = tf.random.gamma([], 32.0)
        g2 = tf.random.gamma([], 32.0)
        r = g1 / (g1 + g2)
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
