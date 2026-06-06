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

        # 4. Clip boxes; filter degenerate boxes
        boxes, keep = clip_boxes(boxes)
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

        # Letterbox resize to output size
        image, boxes = self._letterbox_resize(image, boxes)

        # Clip polygons to output bounds (after resize)
        polygons = clip_polygon_coords(polygons)

        image = tf.cast(image, tf.float32) / 255.0

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

        labels = {
            'bbox':         boxes,
            'classes':      classes,
            'polygons':     poly_labels,
            'n_gt':         tf.cast(n_gt, tf.int64),
            'ignore_bg':    tf.constant(0, dtype=tf.int64),
            'log_distance': log_dist,
            # Eval extras (not present in train labels)
            'is_crowd':     tf.cast(is_crowd[:self._max_num_instances], tf.bool)
                            if tf.size(is_crowd) > 0 else
                            tf.zeros([self._max_num_instances], tf.bool),
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
        legacy: bool = False,
    ) -> tf.Tensor:
        """Convert raw polygon vertices to PolyYOLO radial format.

        PolyYOLO format per instance: [dx_0, dy_0, conf_0, ..., dx_23, dy_23, conf_23]
            origin: box center (normalized) — NOT stored in the output tensor.
            dx, dy: delta from box center to vertex (normalized coords).
            conf:   1.0 if any vertex assigned to this angle bin, else 0.0.

        For each of the 24 angle bins (0°, 15°, ..., 345°):
            - Find all valid polygon vertices whose angle from box center falls
              within [bin*angle_step, (bin+1)*angle_step).
            - Select the vertex with maximum radial distance.
            - Compute (dx, dy) and set conf=1.0.
            - Bins with no valid vertex: dx=dy=0, conf=0.

        Args:
            boxes:    float32 [N, 4] yxyx normalized (used for box centers).
            polygons: float32 [N, max_vertices] flat xy pairs, -1 padded.
            angle_step: degrees per bin (15 → 24 bins).
            legacy:   unused, kept for API compatibility.

        Returns:
            float32 [N, n_angles * 3]
        """
        n_angles = 360 // angle_step
        max_v = self._max_vertices
        n_pairs = max_v // 2

        N = tf.shape(boxes)[0]

        # Box centers (normalized xy)
        cy = (boxes[:, 0] + boxes[:, 2]) / 2.0  # [N]
        cx = (boxes[:, 1] + boxes[:, 3]) / 2.0  # [N]

        # Reshape polygons to [N, n_pairs, 2] (x, y)
        pts = tf.reshape(polygons, [N, n_pairs, 2])

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
        # Build a [N, n_pairs, n_angles] assignment mask: 1 where bin matches AND valid.
        bins_oh = tf.one_hot(bins, n_angles)  # [N, n_pairs, n_angles]
        valid_3d = tf.cast(valid[:, :, tf.newaxis], tf.float32)  # [N, n_pairs, 1]

        # Distance assigned to each (n_pairs, bin) cell
        dists_assigned = (
            dists[:, :, tf.newaxis] * bins_oh * valid_3d
        )  # [N, n_pairs, n_angles]

        # Max distance per bin: [N, n_angles]
        max_dists = tf.reduce_max(dists_assigned, axis=1)

        # Index of argmax vertex per bin: [N, n_angles]
        argmax_idx = tf.argmax(dists_assigned, axis=1, output_type=tf.int32)

        # Gather dx, dy for argmax vertices
        inst_idx = tf.tile(tf.range(N)[:, tf.newaxis], [1, n_angles])  # [N, n_angles]
        gather_idx = tf.stack([inst_idx, argmax_idx], axis=-1)           # [N, n_angles, 2]
        dx_bins   = tf.gather_nd(dx, gather_idx)   # [N, n_angles]
        dy_bins   = tf.gather_nd(dy, gather_idx)   # [N, n_angles]

        # Confidence: 1.0 if any valid vertex was assigned to this bin
        conf_bins = tf.cast(max_dists > 0.0, tf.float32)  # [N, n_angles]

        # Interleave [dx0, dy0, conf0, dx1, dy1, conf1, ...]
        result = tf.stack([dx_bins, dy_bins, conf_bins], axis=-1)  # [N, n_angles, 3]
        return tf.reshape(result, [N, n_angles * 3])                # [N, n_angles*3]

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
        self, image: tf.Tensor, boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Letterbox-resize image to self._output_size with gray padding."""
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

        return image, boxes

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
                tf.fill([tf.shape(boxes)[0], self._max_vertices], -1.0),
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
