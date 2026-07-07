"""Mosaic and MixUp augmentation combining multiple images.

Mosaic (Ultralytics-style) stitches 4 images into a 2×-size canvas at a random
center, then a single ``random_perspective`` warps/crops the canvas back to the
output size. The same ``random_perspective`` is applied to non-mosaic single
images, so it is the one geometric transform in the pipeline (the parser no longer
does affine).

Placement is **upright** (matching the real YOLO mosaic): each source image is
resized so its long side equals the output size and placed full toward the cell's
center corner (overflow cropped, no letterbox padding) — a CONSISTENT per-image
scale, not a random one. Size variety comes only from the single canvas→output
warp's scale gain ``[aug_scale_min, aug_scale_max]`` (default stock ``[0.5, 1.5]``).
Rotation is **rare**: the warp rotates only with probability ``rotate_prob``
(default 0.10) by ±``degrees``; the rest of the time the angle is 0 so tiles read as
upright panels. The split center shifts H+V (``mosaic_center``), so the visible crop
of each tile varies and boxes/polygons are cut at the moving edges.

Group / image-diversity semantics
---------------------------------
``mosaic_fn`` maps a ``padded_batch(group_size)`` group to
``group_size // decodes_per_output`` emitted samples. ``decodes_per_output`` (R) is
both the image-diversity knob and the data-pipeline decode multiplier: it is how many
freshly-decoded images each emitted sample consumes.

  * **R = decodes_per_output** controls reuse. Each emitted output draws 4 source
    images from one per-group random permutation at shifts from an R-keyed Sidon
    set (see ``_SIDON_SHIFTS``). At **R=4** (the default — stock YOLO) the shifts
    are the contiguous window {0,1,2,3} and the windows tile the permutation:
    every output's 4 images are disjoint, zero reuse, ~4× decode work. At **R<4**
    each image is reused in exactly ``4/R`` outputs (that ratio is fixed by the
    decode budget), but the Sidon shifts guarantee any two outputs of a group
    share **at most one** source image — matching stock Ultralytics semantics,
    where images also recur ~4×/epoch but with fresh partners each time. (The
    earlier contiguous window SLID at R<4: adjacent outputs shared 4-R sources —
    3 of 4 at R=1, ~82 near-duplicate-content pairs per 128-batch measured. The
    Sidon selection removes that pathology at identical decode cost.) What R<4
    still costs vs R=4 is the count of *distinct* images consumed per epoch
    (R=1 decodes 4× fewer unique images per epoch pass).
  * **group_size** is the pool each window is drawn from; larger pools give more varied
    4-image combinations at the same R. It must be a multiple of R and >= 4.
  * The mosaic/single decision is a **per-output** coin flip (not per-group), so the
    per-sample mosaic probability is exactly ``mosaic_frequency`` with no batch
    clustering.

Epoch accounting is unaffected: the trainer runs a fixed number of steps, and the
final ``.batch(batch_size)`` is downstream, so the model still sees the same number of
training samples per epoch (R only changes how many source images are decoded to build
them).

Configuration (parser.mosaic in the experiment YAML):
    mosaic_frequency: 0.5
    mixup_frequency: 0.0     (per-output probability of blending the mosaic with a
                              second mosaic — Ultralytics MixUp; 0 = off, the default)
    group_size: 32           (mosaic source pool per group)
    decodes_per_output: 4    (R: 4 = stock YOLO / no reuse; lower = more reuse, less decode)
    mosaic_center: 0.25      (half-range of the split point as a fraction; the
                              2× canvas split lands in [H(1-2c), H(1+2c)])
    aug_scale_min / aug_scale_max: the canvas→output warp scale-gain bounds
                              (default stock [0.5, 1.5]). NOTE: per-image placement
                              scale is no longer random — it is fixed so the long
                              side fills the output (consistent upright tiles).
    degrees: 10.0            (rotation ± magnitude, degrees — applied only when the
                              rotate_prob coin fires)
    rotate_prob: 0.10        (probability a given output is rotated at all; the rest
                              stay upright)
    shear: 0.0               (shear ±, degrees; 0 = no shear, the default)
    perspective: 0.0         (perspective coefficient ±; 0 disables)
    translate: 0.1           (translation ± as a fraction of output size)
    area_thresh: 0.5         (min visible box-area fraction to keep)

Classes:
    Mosaic: Manages both Mosaic and MixUp augmentations.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import tensorflow as tf

from data_pipeline.augmentations import (
    apply_perspective_image,
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
    # Source validity: -1.0 is the reserved polygon sentinel. A vertex with x
    # strictly > -1.0 is a REAL vertex even when negative — a
    # mosaic-cell placement can legitimately map an in-view object's vertex to a
    # slightly-negative input-normalized coordinate. Using `> -1.0` (not `>= 0.0`)
    # carries that vertex into the canvas instead of overwriting it with -1.0; the
    # subsequent random_perspective clips it to the output edge.
    valid = pts[:, :, 0] > -1.0
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
# Source-selection shift sets, keyed by R = decodes_per_output. Output j of a
# group draws its 4 sources at perm[(j*R + s) % G] for s in the set. Two
# invariants make a set valid for a given R:
#   1. Uniform reuse — the shifts cover each residue class mod R exactly 4/R
#      times, so every image in the group is used in exactly 4/R outputs
#      (R=4: {0,1,2,3} has one shift per class -> each image used once;
#      R=1: any 4 shifts -> each image used 4x, same decode bill either way).
#   2. Sidon property — all pairwise shift differences are distinct, so
#      s - s' == (j'-j)*R has at most one solution and ANY two outputs of the
#      group share at most ONE source image. A contiguous window {0,1,2,3} at
#      R<4 violates this badly: it slides, and adjacent outputs share 4-R
#      sources (3 of 4 at R=1 — near-duplicate training samples back to back).
# The Sidon property must hold MODULO G, including sign: a shift difference d
# collides with -d' when d + d' == G (and with itself when 2d == G), giving two
# (s, s') pairs for one output distance and thus 2 shared images. _SIDON_MIN_G
# is the smallest group size at which each set's difference collection
# {±1,±2,±3,±4,±6,±7} (R=1) / {±1,±3,±4,±5,±8,±9} (R=2) stays collision-free
# mod G (R=1: no two differences sum to <15 and none is G/2 from 15 up;
# R=2: G=16 fails via 8 == G/2 and G=18 via 9 == G/2, safe from 20). Below the
# minimum (and for R values not listed) selection falls back to the contiguous
# window. R=4 uses the contiguous window by construction — the windows tile the
# permutation, so the Sidon concern is vacuous and the emitted indices are
# identical to the historical behavior.
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
        degrees: float = 10.0,
        shear: float = 0.0,
        perspective: float = 0.0,
        translate: float = 0.1,
        rotate_prob: float = 0.10,
        group_size: int = 32,
        decodes_per_output: int = 4,
        tile_scale_min: float = 0.0,
        tile_scale_max: float = 0.0,
        single_scale_min: Optional[float] = None,
        single_scale_max: Optional[float] = None,
        single_translate: Optional[float] = None,
        random_flip: bool = False,
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
        self._degrees     = degrees
        self._shear       = shear
        self._perspective = perspective
        self._translate   = translate
        self._rotate_prob = rotate_prob
        # Per-tile independent scale. When
        # tile_scale_max > 0, each mosaic tile's placement scale is multiplied by
        # an INDEPENDENT uniform draw from [tile_scale_min, tile_scale_max], so
        # the 4 tiles of one mosaic appear at 4 different scales (intra-image
        # scale diversity — the strongest scale-invariance signal a detector
        # gets). 0/0 disables it: placement stays at the consistent long-side
        # scale and this code path adds nothing to the graph.
        if (tile_scale_max > 0.0) and not (0.0 < tile_scale_min <= tile_scale_max):
            raise ValueError(
                f"tile_scale bounds invalid: need 0 < min <= max, got "
                f"[{tile_scale_min}, {tile_scale_max}]"
            )
        self._tile_scale_min = tile_scale_min
        self._tile_scale_max = tile_scale_max
        # Non-mosaic (single) path warp params. The two paths are augmented
        # DIFFERENTLY: mosaics get the [aug_scale_min/max] warp
        # gain with no translate, singles get NO scale gain (1.0) but a small
        # translate. None = fall back to the mosaic values (back-compat for
        # direct constructions); input_reader wires the parser-level
        # aug_scale_min/max + aug_rand_translate here.
        self._single_scale_min = (
            single_scale_min if single_scale_min is not None else aug_scale_min)
        self._single_scale_max = (
            single_scale_max if single_scale_max is not None else aug_scale_max)
        self._single_translate = (
            single_translate if single_translate is not None else translate)
        # Flip ownership: when True, this module flips — each mosaic TILE
        # independently (tiles flip before placement, the assembled canvas
        # is never mirrored) and each single
        # image once. The detection train parser's flip must then be OFF or
        # images would flip twice (input_reader wires that). Default False so
        # direct constructions (tests) keep deterministic geometry.
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

            # One random permutation of the group per call. Output j reads sources at
            # perm[(j·R + s) % G] for the four shifts s (R = decodes_per_output). At R=4
            # the shifts are the contiguous window {0,1,2,3}, so the windows tile the
            # permutation exactly (disjoint outputs — stock YOLO, zero reuse). At R<4 a
            # contiguous window would SLIDE (adjacent outputs sharing 3 of 4 sources at
            # R=1 — near-duplicate samples back to back), so a Sidon shift set is used
            # instead: per-image reuse stays exactly 4/R, but any two outputs share at
            # most ONE source image. Selection is index arithmetic only — decode count,
            # op count, and RNG draw order are identical to the windowed form.
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
                # output with a SECOND mosaic built from a distant window of the same
                # group (offset by G//2 so its 4 sources differ from the primary) and
                # concatenate their labels. Gated by a PYTHON check on the config
                # constant, so at the default mixup_frequency=0.0 the partner ops are
                # never added to the graph — byte-identical to the no-mixup pipeline.
                # The partner mosaic is built inside the true branch, so it only
                # executes when the coin fires.
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
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """random_perspective with this module's configured params → output size.

        The warp scale gain is drawn from the EXPLICIT [aug_scale_min,
        aug_scale_max] config bounds (or the per-call override — the single
        path passes its own bounds/translate). (The earlier
        symmetric-magnitude form widened the configured [0.4, 1.9] to
        [0.1, 1.9], occasionally shrinking content to ~1% area — the
        "mostly-gray frame" bug.)
        """
        return random_perspective(
            image, boxes, polygons,
            target_h=self._H, target_w=self._W,
            degrees=self._degrees,
            translate=self._translate if translate is None else translate,
            scale_min=self._scale_min if scale_min is None else scale_min,
            scale_max=self._scale_max if scale_max is None else scale_max,
            shear=self._shear,
            perspective=self._perspective,
            area_thresh=self._area_thresh,
            rotate_prob=self._rotate_prob,
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
        """Non-mosaic path: flip + random_perspective with the SINGLE-path params.

        Uses single_scale_min/max + single_translate (no scale gain and a
        small translate for non-mosaic images) rather than the mosaic warp
        bounds.
        """
        if self._random_flip:
            ex = self._flip_example(ex)
        img = ex['image']
        boxes = ex.get('groundtruth_boxes', tf.zeros([0, 4]))
        polys = ex.get('groundtruth_polygons', tf.zeros([0, 2]))
        img_out, boxes_out, keep, polys_out = self._warp(
            img, boxes, polys,
            scale_min=self._single_scale_min,
            scale_max=self._single_scale_max,
            translate=self._single_translate,
        )
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
        if self._random_flip:
            # Per-TILE independent flip (each tile draws its own coin); the
            # assembled canvas is never mirrored as a whole.
            examples = [self._flip_example(ex) for ex in examples]
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
            rotate_prob=self._rotate_prob,
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
            # Placement scale: long side = output size (consistent upright tiles),
            # optionally multiplied by an INDEPENDENT per-tile draw from
            # [tile_scale_min, tile_scale_max] (each tile lands
            # at its own scale, giving intra-image scale
            # diversity on top of the single canvas->output warp gain). Tiles
            # are anchored at the moving center corner, so an overscaled tile
            # only ever overflows AWAY from the other cells and is cropped at
            # the canvas edge by _place_in_cell; its labels map through the
            # same nh/nw/pad values and are clipped/dropped by the final warp's
            # transform_boxes_polygons + area_thresh — no cross-cell label
            # corruption is possible. tile_scale 0/0 = consistent scale only.
            long_side = tf.maximum(h_in_f, w_in_f)
            place_scale = tf.cast(H, tf.float32) / long_side
            if self._tile_scale_max > 0.0:
                place_scale *= tf.random.uniform(
                    [], self._tile_scale_min, self._tile_scale_max)
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

        # Annotation transform uses the SAME global M as the canvas→output
        # image warp, so the label math is bit-identical to canvas-then-_warp.
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
