"""YOLOv8-specific parser with polygon and distance support (V8ParserExtended).

Training augmentation pipeline order:
    1. Skip is_crowd annotations (if skip_crowd_during_training=True)
    2. Random horizontal flip (with polygon transformation)
    3. Jitter and scale (affine transformation)
    4. Clip boxes and polygons to image bounds
    5. Normalize image to [0, 1]
    6. Apply HSV augmentation (hue=0.015, sat=0.7, bright=0.4)
    7. Apply Albumentations colour transforms (frequency=albumentations_frequency)
    8. Preprocess polygons to PolyYOLO format
    9. Build labels dictionary

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
    apply_albumentations,
    clip_boxes,
    clip_polygon_coords,
    hsv_augment,
    random_affine,
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
        letter_box: bool = True,
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
        self._letter_box = letter_box
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

        # Resize to output_size so all images entering augmentation have a fixed shape.
        h_out, w_out = self._output_size[0], self._output_size[1]
        image = tf.cast(
            tf.image.resize(tf.cast(image, tf.float32), [h_out, w_out], method='bilinear'),
            tf.uint8,
        )
        image.set_shape([h_out, w_out, 3])

        # 2. Random horizontal flip
        if self._random_flip:
            image, boxes, polygons = random_horizontal_flip(
                image, boxes, polygons, self._max_vertices
            )

        # 3. Random affine (translate + scale)
        image, boxes, polygons = random_affine(
            image, boxes, polygons,
            translate=self._aug_rand_translate,
            scale_min=self._aug_scale_min,
            scale_max=self._aug_scale_max,
            output_size=self._output_size,
        )

        # 4. Clip boxes; filter degenerate and too-small boxes
        pre_areas = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        )
        boxes, keep = clip_boxes(boxes)
        if self._area_thresh > 0.0:
            post_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            ratio_ok = (post_areas / tf.maximum(pre_areas, 1e-6)) >= self._area_thresh
            keep = tf.logical_and(keep, ratio_ok)
        boxes    = tf.boolean_mask(boxes,    keep)
        classes  = tf.boolean_mask(classes,  keep)
        polygons = tf.boolean_mask(polygons, keep)

        # Clip polygon coords to [0, 1]
        polygons = clip_polygon_coords(polygons)

        # 5. Normalize image to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # 6. HSV augmentation
        image = hsv_augment(
            image,
            hue=self._aug_rand_hue,
            sat=self._aug_rand_saturation,
            val=self._aug_rand_brightness,
        )

        # 7. Albumentations colour transforms
        if self._albumentations_frequency > 0.0:
            image = apply_albumentations(image, freq=self._albumentations_frequency)

        n_gt = tf.shape(boxes)[0]

        # 8. Preprocess polygons → PolyYOLO radial format
        if self._with_polygons:
            poly_labels = self._preprocess_polygons_v2(
                boxes, polygons, self._angle_step
            )
        else:
            poly_labels = tf.zeros([n_gt, self._poly_depth], dtype=tf.float32)

        # 9. Build labels dict
        log_dist = tf.fill([n_gt], self._invalid_sentinel)  # no distance in det stream
        boxes, classes, poly_labels, log_dist = self._pad_labels(
            boxes, classes, poly_labels, log_dist, n_gt
        )

        labels = {
            'bbox':         boxes,
            'classes':      classes,
            'polygons':     poly_labels,
            'n_gt':         tf.cast(n_gt, tf.int64),
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

        image = tf.cast(image, tf.float32) / 255.0

        # Gray border: replace letterbox-padding pixels (value ~0 from resize) with 0.5
        if self._eval_gray_border:
            gray_mask = tf.reduce_all(image < (8.0 / 255.0), axis=-1, keepdims=True)
            image = tf.where(gray_mask, tf.fill(tf.shape(image), 0.5), image)

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

        dontcare_raw = tf.cast(
            data.get('groundtruth_dontcare', tf.zeros_like(classes, dtype=tf.int64)),
            tf.bool,
        )
        is_dontcare_out = _pad_bool(dontcare_raw)

        labels = {
            'bbox':         boxes,
            'classes':      classes,
            'polygons':     poly_labels,
            'n_gt':         tf.cast(n_gt, tf.int64),
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

        Output format per instance: [dist_0, angle_norm_0, conf_0, ..., dist_23, angle_norm_23, conf_23]
            dist:       radial distance from box center to the vertex assigned to this bin.
            angle_norm: 1.0 for the dominant bin (max dist across all bins), 0.0 elsewhere.
            conf:       1.0 if any valid vertex assigned to this bin, else 0.0.

        For each of the 24 angle bins (0°, 15°, ..., 345°):
            - Find all valid polygon vertices whose angle from box center falls in the bin.
            - Select the vertex with maximum radial distance as the bin representative.
            - dist = sqrt(dx² + dy²), conf = 1.0 if any vertex present.
        The bin with maximum dist across all bins gets angle_norm = 1.0 (one-hot dominant bin).

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

        # Valid vertices: x >= 0 (both x and y are -1 for invalid pairs)
        valid = pts[:, :, 0] >= 0.0  # [N, n_pairs]

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

        # For each (instance, bin), find the vertex with maximum radial distance.
        bins_oh = tf.one_hot(bins, n_angles)  # [N, n_pairs, n_angles]
        valid_3d = tf.cast(valid[:, :, tf.newaxis], tf.float32)  # [N, n_pairs, 1]

        # Distance assigned to each (n_pairs, bin) cell
        dists_assigned = (
            dists[:, :, tf.newaxis] * bins_oh * valid_3d
        )  # [N, n_pairs, n_angles]

        # Max distance per bin: [N, n_angles]
        max_dists = tf.reduce_max(dists_assigned, axis=1)

        # Confidence: 1.0 if any valid vertex was assigned to this bin
        conf_bins = tf.cast(max_dists > 0.0, tf.float32)  # [N, n_angles]

        # Dominant bin (max dist across all bins) → one-hot angle label.
        # Guard: a box with no valid vertices has all-zero max_dists, so argmax
        # would return bin 0 and emit a spurious angle_norm=1.0 there — driving the
        # polygon angle loss toward bin 0 on polygon-less objects. Gate the one-hot
        # on "has at least one vertex" so empty polygons get an all-zero angle target.
        dominant_bin = tf.argmax(max_dists, axis=1, output_type=tf.int32)  # [N]
        has_vertex = tf.cast(
            tf.reduce_max(max_dists, axis=1, keepdims=True) > 0.0, tf.float32
        )  # [N, 1]
        angle_bins = tf.one_hot(dominant_bin, n_angles) * has_vertex  # [N, n_angles]

        # Interleave [dist0, angle_norm0, conf0, dist1, angle_norm1, conf1, ...]
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
        return random_horizontal_flip(image, boxes, polygons, self._max_vertices)

    def _apply_hsv_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random HSV jitter to a float32 [H, W, 3] image."""
        return hsv_augment(
            image,
            hue=self._aug_rand_hue,
            sat=self._aug_rand_saturation,
            val=self._aug_rand_brightness,
        )

    def _letterbox_resize(
        self, image: tf.Tensor, boxes: tf.Tensor, polygons: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Letterbox-resize image to self._output_size with gray padding.

        Boxes AND polygon vertices are remapped through the same scale + pad so
        they stay aligned in the output normalized space. Invalid polygon
        vertices (-1 sentinel) are preserved.
        """
        h_out, w_out = self._output_size[0], self._output_size[1]
        h_in = tf.cast(tf.shape(image)[0], tf.float32)
        w_in = tf.cast(tf.shape(image)[1], tf.float32)

        scale = tf.minimum(h_out / h_in, w_out / w_in)
        new_h = tf.maximum(tf.cast(tf.round(h_in * scale), tf.int32), 1)
        new_w = tf.maximum(tf.cast(tf.round(w_in * scale), tf.int32), 1)

        image = tf.cast(
            tf.image.resize(tf.cast(image, tf.float32), [new_h, new_w], method='bilinear'),
            tf.uint8,
        )

        pad_top    = (h_out - new_h) // 2
        pad_left   = (w_out - new_w) // 2
        pad_bottom = h_out - new_h - pad_top
        pad_right  = w_out - new_w - pad_left

        image = tf.pad(
            image,
            [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            constant_values=114,
        )
        image.set_shape([h_out, w_out, 3])

        # Adjust boxes for padding and scale
        new_h_f   = tf.cast(new_h,   tf.float32)
        new_w_f   = tf.cast(new_w,   tf.float32)
        h_out_f   = tf.cast(h_out,   tf.float32)
        w_out_f   = tf.cast(w_out,   tf.float32)
        pad_top_f = tf.cast(pad_top,  tf.float32)
        pad_lft_f = tf.cast(pad_left, tf.float32)

        ymin = boxes[:, 0] * new_h_f / h_out_f + pad_top_f / h_out_f
        xmin = boxes[:, 1] * new_w_f / w_out_f + pad_lft_f / w_out_f
        ymax = boxes[:, 2] * new_h_f / h_out_f + pad_top_f / h_out_f
        xmax = boxes[:, 3] * new_w_f / w_out_f + pad_lft_f / w_out_f
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        # Adjust polygon vertices with the same scale + pad. polygons is
        # [N, max_vertices+2] flat (x, y) pairs, -1 padded for invalid vertices.
        n_inst = tf.shape(polygons)[0]
        pts    = tf.reshape(polygons, [n_inst, -1, 2])        # [N, P, 2] (x, y)
        valid  = pts[:, :, 0] >= 0.0                           # [N, P]
        px = pts[:, :, 0] * new_w_f / w_out_f + pad_lft_f / w_out_f
        py = pts[:, :, 1] * new_h_f / h_out_f + pad_top_f / h_out_f
        neg1 = tf.fill(tf.shape(px), -1.0)
        px = tf.where(valid, px, neg1)
        py = tf.where(valid, py, neg1)
        polygons = tf.reshape(tf.stack([px, py], axis=-1), tf.shape(polygons))

        return image, boxes, polygons

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
