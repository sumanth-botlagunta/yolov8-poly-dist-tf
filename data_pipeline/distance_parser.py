"""Distance-only parser for the servingbot_polygon dataset (V8DistanceParser).

Produces the same label schema as V8ParserExtended but:
    - Sets ignore_bg=1 to suppress class loss on background anchors.
    - Populates log_distance from the groundtruth_dists field.
    - Does not apply Copy-Paste or Mosaic augmentation.
    - with_polygons defaults to False (distance dataset has no polygon labels).

Colour augmentation (HSV jitter + normalize /255) runs elsewhere: the parser
emits a uint8 image so the merged batch carries uint8, and the per-batch colour
pipeline runs on the accelerator in ``train.task.train_step`` via
``data_pipeline.batch_color_aug.batch_color_augment``. That step applies HSV to
all rows (the distance stream's aug_rand_hue/saturation/brightness are set equal
to the detection stream's from the same YAML, so both streams see the same HSV
distribution) and applies albumentations only to ``ignore_bg == 0`` rows, so the
distance stream receives none.

Classes:
    V8DistanceParser: lightweight parser for the distance stream.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import tensorflow as tf

from data_pipeline.parser import Parser


class V8DistanceParser(Parser):
    """Parser for the distance-only training stream (servingbot_polygon)."""

    def __init__(
        self,
        output_size: List[int],
        max_num_instances: int = 300,
        angle_step: int = 15,
        with_polygons: bool = False,
        min_meter: float = 0.5,
        max_meter: float = 10.0,
        aug_rand_hue: float = 0.015,
        aug_rand_saturation: float = 0.7,
        aug_rand_brightness: float = 0.4,
        random_flip: bool = True,
        skip_crowd_during_training: bool = True,
    ):
        self._output_size = output_size          # [H, W]
        self._max_num_instances = max_num_instances
        self._angle_step = angle_step
        self._with_polygons = with_polygons
        self._min_meter = min_meter
        self._max_meter = max_meter
        self._aug_rand_hue = aug_rand_hue
        self._aug_rand_saturation = aug_rand_saturation
        self._aug_rand_brightness = aug_rand_brightness
        self._random_flip = random_flip
        self._skip_crowd = skip_crowd_during_training

        self._poly_depth = (360 // angle_step) * 3  # 72 for angle_step=15
        # Sentinel for invalid/absent distance.
        self._invalid_sentinel = -10.0

    # ------------------------------------------------------------------
    # Parser interface
    # ------------------------------------------------------------------

    def _parse_train_data(
        self, data: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict]:
        """Parse a distance sample; sets ignore_bg=1 in returned labels."""
        image, boxes, classes, is_crowd, dists = self._extract_fields(data)

        # Filter crowd annotations during training.
        if self._skip_crowd:
            valid = tf.logical_not(is_crowd)
            boxes = tf.boolean_mask(boxes, valid)
            classes = tf.boolean_mask(classes, valid)
            dists = tf.boolean_mask(dists, valid)

        # Resize to output_size with letterbox.
        image, boxes = self._letterbox_resize(image, boxes)

        # Light augmentation (no mosaic/copy-paste for distance stream).
        if self._random_flip:
            image, boxes = self._maybe_flip(image, boxes)

        # Colour augmentation (HSV + normalize /255) runs once per batch on the
        # accelerator in train.task.train_step. Keep the image uint8.

        # Build label tensors.
        n_gt = tf.shape(boxes)[0]
        log_dists = self._encode_log_distance(dists)

        # Pad to max_num_instances.
        boxes, classes, log_dists = self._pad_labels(boxes, classes, log_dists, n_gt)

        labels = {
            'bbox': boxes,
            'classes': classes,
            'n_gt': tf.cast(tf.minimum(n_gt, self._max_num_instances), tf.int64),
            'ignore_bg': tf.constant(1, dtype=tf.int64),  # ← distance stream
            'log_distance': log_dists,
            # Polygon fields are zeros (ignored by loss when with_distance=True).
            'polygons': tf.zeros(
                [self._max_num_instances, self._poly_depth], dtype=tf.float32
            ),
        }
        return image, labels

    def _parse_eval_data(
        self, data: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict]:
        """Parse an evaluation distance sample (letterbox resize, no augmentation)."""
        image, boxes, classes, is_crowd, dists = self._extract_fields(data)
        image, boxes = self._letterbox_resize(image, boxes)
        # Image stays uint8; normalization /255 happens once per batch in
        # train.task.validation_step.

        n_gt = tf.shape(boxes)[0]
        log_dists = self._encode_log_distance(dists)
        boxes, classes, log_dists = self._pad_labels(boxes, classes, log_dists, n_gt)

        labels = {
            'bbox': boxes,
            'classes': classes,
            'n_gt': tf.cast(tf.minimum(n_gt, self._max_num_instances), tf.int64),
            'ignore_bg': tf.constant(1, dtype=tf.int64),
            'log_distance': log_dists,
            'polygons': tf.zeros(
                [self._max_num_instances, self._poly_depth], dtype=tf.float32
            ),
        }
        return image, labels

    # ------------------------------------------------------------------
    # Distance encoding
    # ------------------------------------------------------------------

    def _encode_log_distance(self, distances: tf.Tensor) -> tf.Tensor:
        """Clip distances to [min_meter, max_meter] then take log.

        Invalid distances (<= 0) are encoded as -10.0. A distance of exactly 0.0
        is physically invalid (an object at zero range), so it is treated as a
        sentinel rather than clipped up to min_meter and contributing to the loss.

        Returns:
            float32 [N]
        """
        valid_mask = distances > 0.0
        clipped = tf.clip_by_value(distances, self._min_meter, self._max_meter)
        log_dist = tf.math.log(clipped)
        # Replace invalid entries with the sentinel.
        return tf.where(valid_mask, log_dist, tf.fill(tf.shape(distances), self._invalid_sentinel))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_fields(self, data: Dict[str, tf.Tensor]):
        image = tf.cast(data['image'], tf.uint8)
        boxes = tf.cast(data['groundtruth_boxes'], tf.float32)    # [N, 4] yxyx
        classes = tf.cast(data['groundtruth_classes'], tf.int64)  # [N]
        is_crowd = tf.cast(data.get('groundtruth_is_crowd',
                                    tf.zeros_like(classes, dtype=tf.bool)), tf.bool)
        dists_raw = data.get('groundtruth_dists',
                             tf.fill([tf.shape(classes)[0]], -1.0))
        dists = tf.cast(dists_raw, tf.float32)
        return image, boxes, classes, is_crowd, dists

    def _letterbox_resize(
        self, image: tf.Tensor, boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Resize image to output_size with gray letterbox padding.

        Boxes (yxyx normalized) are adjusted to account for the padding offset
        and scale so they remain correct after resize.
        """
        h_out, w_out = self._output_size[0], self._output_size[1]
        h_in = tf.cast(tf.shape(image)[0], tf.float32)
        w_in = tf.cast(tf.shape(image)[1], tf.float32)

        scale = tf.minimum(h_out / h_in, w_out / w_in)
        # Guard against a 0-size resize on degenerate (e.g. 1px) inputs, matching
        # V8ParserExtended._letterbox_resize. tf.image.resize raises on a 0 dim.
        new_h = tf.maximum(tf.cast(tf.round(h_in * scale), tf.int32), 1)
        new_w = tf.maximum(tf.cast(tf.round(w_in * scale), tf.int32), 1)

        # Cast to float32 before resizing (matches V8ParserExtended._letterbox_resize):
        # resizing uint8 directly is ~1 DN less precise, which would make the distance
        # stream's pixels differ subtly from the detection stream after /255.
        image = tf.cast(
            tf.image.resize(tf.cast(image, tf.float32), [new_h, new_w], method='bilinear'),
            tf.uint8,
        )

        # Pad to output size with gray (114).
        pad_top = (h_out - new_h) // 2
        pad_left = (w_out - new_w) // 2
        pad_bottom = h_out - new_h - pad_top
        pad_right = w_out - new_w - pad_left
        image = tf.pad(
            image,
            [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            constant_values=114,
        )
        image.set_shape([h_out, w_out, 3])

        # Adjust boxes for padding and scale.
        new_h_f = tf.cast(new_h, tf.float32)
        new_w_f = tf.cast(new_w, tf.float32)
        h_out_f = tf.cast(h_out, tf.float32)
        w_out_f = tf.cast(w_out, tf.float32)
        pad_top_f = tf.cast(pad_top, tf.float32)
        pad_left_f = tf.cast(pad_left, tf.float32)

        ymin = boxes[:, 0] * new_h_f / h_out_f + pad_top_f / h_out_f
        xmin = boxes[:, 1] * new_w_f / w_out_f + pad_left_f / w_out_f
        ymax = boxes[:, 2] * new_h_f / h_out_f + pad_top_f / h_out_f
        xmax = boxes[:, 3] * new_w_f / w_out_f + pad_left_f / w_out_f
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        return image, boxes

    def _maybe_flip(
        self, image: tf.Tensor, boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        do_flip = tf.random.uniform(()) > 0.5
        image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        # Flip x-coordinates: xmin ↔ 1-xmax.
        xmin = tf.where(do_flip, 1.0 - boxes[:, 3], boxes[:, 1])
        xmax = tf.where(do_flip, 1.0 - boxes[:, 1], boxes[:, 3])
        boxes = tf.stack([boxes[:, 0], xmin, boxes[:, 2], xmax], axis=1)
        return image, boxes

    def _pad_labels(
        self,
        boxes: tf.Tensor,
        classes: tf.Tensor,
        log_dists: tf.Tensor,
        n_gt: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Pad/truncate tensors to max_num_instances."""
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

        log_dists = tf.concat(
            [log_dists[:m], tf.fill([pad_n], self._invalid_sentinel)], axis=0
        )
        log_dists = tf.reshape(log_dists, [m])

        return boxes, classes, log_dists
