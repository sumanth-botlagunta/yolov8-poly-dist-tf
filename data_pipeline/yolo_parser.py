"""YOLOv8-specific parser with polygon and distance support (V8ParserExtended).

Training augmentation order:
    1. Skip is_crowd annotations (if skip_crowd_during_training=True)
    2. Random horizontal flip (with polygon transformation)
    3. Clip boxes and polygons to image bounds
    4. Preprocess polygons to PolyYOLO format
    5. Build labels dictionary

The geometric affine (random_perspective: rotate/scale/shear/translate) runs
upstream in the mosaic stage (data_pipeline/mosaic.py) for both the 4-image
mosaic and non-mosaic singles, so the parser applies none here.

Colour augmentation (normalize /255 → HSV jitter → albumentations) also runs
elsewhere: the parser emits a uint8 image (the whole pipeline carries uint8 for
less host→device traffic) and the colour pipeline runs once per batch on the
accelerator inside ``train.task.train_step`` via
``data_pipeline.batch_color_aug.batch_color_augment`` (eval normalizes /255 in
``validation_step``), with the same per-image randomness distribution.

Output labels schema:
    source_id: string [batch]
    bbox: float32 [batch, max_instances, 4]       yxyx normalized
    classes: int64 [batch, max_instances]
    polygons: float32 [batch, max_instances, 72]  PolyYOLO (angle_step=15)
    n_gt: int64 [batch]
    ignore_bg: int64 [batch]                       always 0 for detection data
    log_distance: float32 [batch, max_instances]   -10.0 for invalid

Classes:
    V8ParserExtended: Full-featured parser for the detection + polygon stream.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from data_pipeline.augmentations import (
    clip_boxes,
    clip_polygon_coords,
    letterbox_resize,
    random_horizontal_flip,
)
from data_pipeline.parser import Parser


class V8ParserExtended(Parser):
    """Parser for YOLOv8 training with polygon segmentation and distance labels."""

    def __init__(
        self,
        output_size: List[int],
        expanded_strides: Dict[str, int],
        levels: List[str],
        max_vertices: int = 10938,
        angle_step: int = 15,
        with_polygons: bool = True,
        dummy_distance: bool = True,
        skip_crowd_during_training: bool = True,
        max_num_instances: int = 300,
        aug_rand_hue: float = 0.015,
        aug_rand_saturation: float = 0.7,
        aug_rand_brightness: float = 0.4,
        aug_rand_translate: float = 0.1,
        aug_scale_min: float = 1.0,
        aug_scale_max: float = 1.0,
        random_flip: bool = True,
        resize_with_random_method: bool = True,
        albumentations_frequency: float = 1.0,
        area_thresh: float = 0.1,
        eval_gray_border: bool = False,
    ):
        self._output_size = output_size          # [H, W]
        self._expanded_strides = expanded_strides
        self._levels = levels
        self._max_vertices = max_vertices
        self._angle_step = angle_step
        self._with_polygons = with_polygons
        self._dummy_distance = dummy_distance
        self._skip_crowd = skip_crowd_during_training
        self._max_num_instances = max_num_instances
        self._aug_rand_hue = aug_rand_hue
        self._aug_rand_saturation = aug_rand_saturation
        self._aug_rand_brightness = aug_rand_brightness
        self._aug_rand_translate = aug_rand_translate
        self._aug_scale_min = aug_scale_min
        self._aug_scale_max = aug_scale_max
        self._random_flip = random_flip
        self._albumentations_frequency = albumentations_frequency
        self._area_thresh = area_thresh
        self._eval_gray_border = eval_gray_border

        self._n_angles = 360 // angle_step      # = 24 for angle_step=15
        self._poly_depth = self._n_angles * 3   # = 72
        self._invalid_sentinel = -10.0

    # ------------------------------------------------------------------
    # Parser interface
    # ------------------------------------------------------------------

    def _parse_train_data(
        self, data: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict]:
        """Parse and augment a single training example."""
        image, boxes, classes, polygons, is_crowd = self._extract_fields(data)

        # 1. Filter crowd annotations
        if self._skip_crowd:
            valid = tf.logical_not(is_crowd)
            boxes    = tf.boolean_mask(boxes,    valid)
            classes  = tf.boolean_mask(classes,  valid)
            polygons = tf.boolean_mask(polygons, valid)

        # Resize to output_size so all images entering augmentation share a fixed
        # shape. When the image arrives from the mosaic stage it is already exactly
        # [h_out, w_out, 3] with a static shape (random_perspective calls set_shape),
        # so skip the cast→resize→cast round trip. The decision is made at trace time
        # via a Python `if` on the static shape. Variable-size raw inputs (tests) keep
        # the resize path.
        h_out, w_out = self._output_size[0], self._output_size[1]
        static_h = image.shape[0]
        static_w = image.shape[1]
        if static_h == h_out and static_w == w_out:
            image.set_shape([h_out, w_out, 3])
        else:
            image = tf.cast(
                tf.image.resize(
                    tf.cast(image, tf.float32), [h_out, w_out], method='bilinear'
                ),
                tf.uint8,
            )
            image.set_shape([h_out, w_out, 3])

        # 2. Random horizontal flip
        if self._random_flip:
            image, boxes, polygons = random_horizontal_flip(image, boxes, polygons)

        # 3. Geometric augmentation (rotation/scale/shear/translate) runs upstream
        #    in the mosaic stage's random_perspective for both the mosaic and
        #    single-image branches, so the parser applies no affine here (doing so
        #    would double-warp). The image already arrives at output_size.

        # 4. Clip boxes; drop degenerate rows. min_side=0.0 (strict >) removes
        #    only zero-size rows — notably the mosaic stage's padded_batch
        #    zero-padding. The 2px min_side filter applies on the MOSAIC branch
        #    only (legacy convention: non-mosaic images are not size-filtered);
        #    the area-ratio and aspect filters run in the mosaic-stage warps
        #    for both paths.
        pre_areas = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        )
        boxes, keep = clip_boxes(boxes, min_side=0.0)
        if self._area_thresh > 0.0:
            post_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            ratio_ok = (post_areas / tf.maximum(pre_areas, 1e-6)) >= self._area_thresh
            keep = tf.logical_and(keep, ratio_ok)
        boxes    = tf.boolean_mask(boxes,    keep)
        classes  = tf.boolean_mask(classes,  keep)
        polygons = tf.boolean_mask(polygons, keep)

        # Clip polygon coords to [0, 1]
        polygons = clip_polygon_coords(polygons)

        # Colour augmentation (normalize /255 → HSV → albumentations) runs once per
        # batch on the accelerator in train.task.train_step (see
        # data_pipeline.batch_color_aug). The image stays uint8 through batching.

        n_gt = tf.shape(boxes)[0]

        # 5. Preprocess polygons → PolyYOLO radial format
        if self._with_polygons:
            poly_labels = self._preprocess_polygons_v2(
                boxes, polygons, self._angle_step
            )
        else:
            poly_labels = tf.zeros([n_gt, self._poly_depth], dtype=tf.float32)

        # 6. Build labels dict
        log_dist = tf.fill([n_gt], self._invalid_sentinel)  # no distance in det stream
        boxes, classes, poly_labels, log_dist = self._pad_labels(
            boxes, classes, poly_labels, log_dist, n_gt
        )

        labels = {
            'bbox':         boxes,
            'classes':      classes,
            'polygons':     poly_labels,
            # Clamp to the padded width: _pad_labels truncates to
            # max_num_instances, so an un-clamped n_gt would make the loss build
            # a mask_gt wider than the (truncated) label tensors → shape crash.
            'n_gt':         tf.cast(tf.minimum(n_gt, self._max_num_instances), tf.int64),
            'ignore_bg':    tf.constant(0, dtype=tf.int64),
            'log_distance': log_dist,
        }
        return image, labels

    def _parse_eval_data(
        self, data: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict]:
        """Parse a single evaluation example (letterbox resize only)."""
        image, boxes, classes, polygons, is_crowd = self._extract_fields(data)

        # Letterbox resize to output size (transforms boxes AND polygons together)
        image, boxes, polygons = self._letterbox_resize(image, boxes, polygons)

        # Clip polygons to output bounds (after resize)
        polygons = clip_polygon_coords(polygons)

        # Clip GT boxes to [0, 1] — the same defense the train path gets from
        # clip_boxes. The letterbox affine cannot push a well-formed box out of
        # range, so this only fires on marginally-invalid source annotations
        # (e.g. xmax = 1.0003), which would otherwise reach the AP computation
        # unclipped while the train stream sees them clipped.
        boxes = tf.clip_by_value(boxes, 0.0, 1.0)

        # Image stays uint8; normalization /255 happens once per batch in
        # train.task.validation_step. Gray border (replacing near-black
        # letterbox-padding pixels with mid-gray) is applied here on uint8:
        # mask = all channels < 8 (DN) → fill 128 (== 0.5 after /255).
        if self._eval_gray_border:
            gray_mask = tf.reduce_all(image < 8, axis=-1, keepdims=True)
            image = tf.where(
                gray_mask,
                tf.fill(tf.shape(image), tf.constant(128, dtype=tf.uint8)),
                image,
            )

        n_gt = tf.shape(boxes)[0]

        if self._with_polygons:
            poly_labels = self._preprocess_polygons_v2(
                boxes, polygons, self._angle_step
            )
        else:
            poly_labels = tf.zeros([n_gt, self._poly_depth], dtype=tf.float32)

        log_dist = tf.fill([n_gt], self._invalid_sentinel)
        boxes, classes, poly_labels, log_dist = self._pad_labels(
            boxes, classes, poly_labels, log_dist, n_gt
        )

        m = self._max_num_instances
        pad = tf.maximum(0, m - n_gt)

        def _pad_bool(t):
            return tf.reshape(
                tf.concat([tf.cast(t[:m], tf.bool), tf.zeros([pad], tf.bool)], axis=0),
                [m],
            )

        is_crowd_out = _pad_bool(is_crowd)

        # Default from the UNPADDED per-object tensor (is_crowd is still raw
        # [N] here; `classes` was already padded to max_num_instances above, so
        # a zeros_like(classes) default would be pre-padded and _pad_bool would
        # pad it a second time → [N+pad] reshape-to-[N] crash the first time a
        # decoder omits groundtruth_dontcare).
        dontcare_raw = tf.cast(
            data.get('groundtruth_dontcare', tf.zeros_like(is_crowd, dtype=tf.int64)),
            tf.bool,
        )
        is_dontcare_out = _pad_bool(dontcare_raw)

        labels = {
            'bbox':         boxes,
            'classes':      classes,
            'polygons':     poly_labels,
            'n_gt':         tf.cast(tf.minimum(n_gt, self._max_num_instances), tf.int64),
            'ignore_bg':    tf.constant(0, dtype=tf.int64),
            'log_distance': log_dist,
            # Eval extras (not present in train labels)
            'is_crowd':     is_crowd_out,
            'is_dontcare':  is_dontcare_out,
            'source_id':    data.get('source_id', tf.constant('')),
        }
        return image, labels

    # ------------------------------------------------------------------
    # Polygon → PolyYOLO conversion
    # ------------------------------------------------------------------

    def _preprocess_polygons_v2(
        self,
        boxes: tf.Tensor,
        polygons: tf.Tensor,
        angle_step: int,
        _unused_v1_compat: bool = False,
    ) -> tf.Tensor:
        """Convert raw polygon vertices to PolyYOLO radial format.

        Output format per instance: [dist_0, angle_0, conf_0, ..., dist_23, angle_23, conf_23]
            dist:  radial distance from box center to the vertex assigned to this bin.
            angle: sub-bin angular offset (vertex_angle - bin_start) / angle_step in
                   [0, 1) on bins that hold a vertex, 0.0 on empty bins.
            conf:  1.0 if any valid vertex assigned to this bin, else 0.0.

        For each of the 24 angle bins (0°, 15°, ..., 345°):
            - Find all valid polygon vertices whose angle from box center falls in the bin.
            - Select the vertex with maximum radial distance as the bin representative.
            - dist = sqrt(dx² + dy²), conf = 1.0 if any vertex present, and
              angle = that vertex's fractional position within the bin.

        Implementation uses a flat segment formulation instead of a dense
        ``[N, P, n_angles]`` one-hot intermediate (P can be ~2000): every
        (instance, bin) pair is given a unique segment id, and per-bin maxima are
        computed with ``tf.math.unsorted_segment_max`` over the flattened
        ``[N*P]`` distance vector. The bin representative (first-max vertex, to
        match the old ``argmax`` tie-break) is recovered with a second
        ``unsorted_segment_min`` over per-bin-max vertex indices. This is exactly
        output-equivalent to the one-hot version (including ties) but avoids the
        large dense intermediate.

        Args:
            boxes:    float32 [N, 4] yxyx normalized (used for box centers).
            polygons: float32 [N, max_vertices+2] flat xy pairs, -1 padded.
            angle_step: degrees per bin (15 → 24 bins).
            _unused_v1_compat: unused parameter kept for call-site compatibility.

        Returns:
            float32 [N, n_angles * 3]
        """
        n_angles = 360 // angle_step

        N = tf.shape(boxes)[0]

        # Box centers (normalized xy)
        cy = (boxes[:, 0] + boxes[:, 2]) / 2.0  # [N]
        cx = (boxes[:, 1] + boxes[:, 3]) / 2.0  # [N]

        # Reshape polygons to [N, n_pairs, 2] (x, y); -1 auto-infers n_pairs.
        pts = tf.reshape(polygons, [N, -1, 2])

        # Valid vertices key off the -1.0 sentinel (x > -1.0). A slightly-negative
        # canvas coordinate (mosaic overflow that survived clip-to-edge) is a real
        # vertex, not padding, and must contribute to the radial target; `>= 0.0`
        # would drop it.
        valid = pts[:, :, 0] > -1.0  # [N, n_pairs]

        # Relative positions from box center
        dx = pts[:, :, 0] - cx[:, tf.newaxis]  # [N, n_pairs]
        dy = pts[:, :, 1] - cy[:, tf.newaxis]  # [N, n_pairs]
        dists = tf.sqrt(dx * dx + dy * dy)       # [N, n_pairs]

        # Angle bin for each vertex
        angles_rad = tf.math.atan2(dy, dx)       # [N, n_pairs] in (-pi, pi]
        angles_deg = angles_rad * (180.0 / math.pi)
        angles_deg = tf.math.floormod(angles_deg, 360.0)  # [0, 360)
        bins = tf.cast(
            tf.math.floor(angles_deg / angle_step), tf.int32
        )  # [N, n_pairs]
        bins = tf.clip_by_value(bins, 0, n_angles - 1)

        P = tf.shape(pts)[1]  # n_pairs

        # --- Segment formulation -------------------------------------------------
        # Give each (instance, bin) pair a unique segment id in [0, N*n_angles).
        seg_ids = tf.range(N)[:, tf.newaxis] * n_angles + bins   # [N, P] int32
        flat_seg = tf.reshape(seg_ids, [-1])                     # [N*P]

        # Invalid vertices contribute distance 0 (same as the one-hot * valid_3d).
        dist_valid = dists * tf.cast(valid, tf.float32)          # [N, P]
        flat_dist = tf.reshape(dist_valid, [-1])                 # [N*P]

        n_seg = N * n_angles
        # Max distance per bin. Empty segments yield dtype-min; clamp to 0.0 so it
        # matches the one-hot version (max over a row of zeros == 0).
        max_flat = tf.math.unsorted_segment_max(flat_dist, flat_seg, n_seg)  # [n_seg]
        max_dists = tf.maximum(tf.reshape(max_flat, [N, n_angles]), 0.0)     # [N, n_angles]

        # Confidence: 1.0 if any valid vertex was assigned to this bin
        conf_bins = tf.cast(max_dists > 0.0, tf.float32)  # [N, n_angles]

        # Bin representative = FIRST vertex (smallest p) attaining the bin max,
        # reproducing tf.argmax's first-max tie-break exactly.
        m_per_elem = tf.gather(max_flat, flat_seg)                # [N*P] this elem's bin-max
        p_idx = tf.tile(tf.range(P)[tf.newaxis, :], [N, 1])       # [N, P] vertex index
        flat_p_idx = tf.reshape(p_idx, [-1])                      # [N*P]
        BIG = tf.constant(2 ** 30, dtype=tf.int32)
        # Float equality is exact: segment_max returns one of its actual inputs.
        winner_idx = tf.where(flat_dist == m_per_elem, flat_p_idx, BIG)  # [N*P]
        first_winner = tf.math.unsorted_segment_min(winner_idx, flat_seg, n_seg)  # [n_seg]
        # Empty segments (or all-BIG) clamp to a valid index; conf==0 there so the
        # gathered offset is zeroed out below regardless.
        best_pair = tf.minimum(
            tf.reshape(first_winner, [N, n_angles]),
            tf.maximum(P - 1, 0),
        )  # [N, n_angles]

        # Sub-bin angular offset: (vertex_angle - bin_start) / angle_step in [0, 1),
        # i.e. the fractional position of the vertex within its angular bin. This
        # lets the model recover the exact vertex angle, not just the bin index.
        frac = angles_deg / angle_step - tf.math.floor(angles_deg / angle_step)  # [N, n_pairs]
        # Per bin, take the offset of the vertex that owns it (the first
        # max-radial-dist one — the same vertex whose distance is regressed).
        angle_bins = tf.gather(frac, best_pair, batch_dims=1)                # [N, n_angles]
        # Empty bins (no vertex) carry offset 0.0; the loss masks them out via conf.
        angle_bins = angle_bins * conf_bins                                  # [N, n_angles]

        # Interleave [dist0, angle0, conf0, dist1, angle1, conf1, ...].
        # Channel-order trap: the GT per-bin order here is (dist, angle, conf), but
        # the prediction tensor from the detection generator stacks (conf, dist,
        # angle) (models/detection_generator.py, poly_out). The two conventions are
        # pinned by tests/unit/test_polygon_channel_order.py; never index one with
        # the other's layout.
        result = tf.stack([max_dists, angle_bins, conf_bins], axis=-1)  # [N, n_angles, 3]
        return tf.reshape(result, [N, n_angles * 3])                     # [N, n_angles*3]

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _random_horizontal_flip(
        self,
        image: tf.Tensor,
        boxes: tf.Tensor,
        polygons: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Flip image and transform boxes/polygons accordingly."""
        return random_horizontal_flip(image, boxes, polygons)

    def _letterbox_resize(
        self, image: tf.Tensor, boxes: tf.Tensor, polygons: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Letterbox-resize image to self._output_size with gray padding.

        Delegates to the shared ``augmentations.letterbox_resize`` so the eval
        parser and the mosaic-stage pre-resize (input_reader) use byte-identical
        letterbox math. Boxes AND polygon vertices are remapped through the same
        scale + pad; the -1.0 polygon sentinel is preserved.
        """
        h_out, w_out = self._output_size[0], self._output_size[1]
        return letterbox_resize(image, boxes, polygons, h_out, w_out)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_fields(
        self, data: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Extract and cast fields from decoded data dict."""
        image    = tf.cast(data['image'], tf.uint8)
        boxes    = tf.cast(data['groundtruth_boxes'],   tf.float32)
        classes  = tf.cast(data['groundtruth_classes'], tf.int64)
        is_crowd = tf.cast(
            data.get('groundtruth_is_crowd', tf.zeros_like(classes, dtype=tf.bool)),
            tf.bool,
        )
        polygons = tf.cast(
            data.get(
                'groundtruth_polygons',
                # +2 matches the actual TFDS feature shape (max_vertices+2 columns).
                tf.fill([tf.shape(boxes)[0], self._max_vertices + 2], -1.0),
            ),
            tf.float32,
        )
        return image, boxes, classes, polygons, is_crowd

    def _pad_labels(
        self,
        boxes:      tf.Tensor,
        classes:    tf.Tensor,
        poly_labels: tf.Tensor,
        log_dists:  tf.Tensor,
        n_gt:       tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Pad or truncate all label tensors to max_num_instances."""
        m = self._max_num_instances
        pad_n = tf.maximum(0, m - n_gt)

        boxes = tf.concat(
            [boxes[:m], tf.zeros([pad_n, 4], tf.float32)], axis=0
        )
        boxes = tf.reshape(boxes, [m, 4])

        classes = tf.concat(
            [classes[:m], tf.zeros([pad_n], tf.int64)], axis=0
        )
        classes = tf.reshape(classes, [m])

        poly_labels = tf.concat(
            [poly_labels[:m],
             tf.zeros([pad_n, self._poly_depth], tf.float32)], axis=0
        )
        poly_labels = tf.reshape(poly_labels, [m, self._poly_depth])

        log_dists = tf.concat(
            [log_dists[:m],
             tf.fill([pad_n], self._invalid_sentinel)], axis=0
        )
        log_dists = tf.reshape(log_dists, [m])

        return boxes, classes, poly_labels, log_dists
