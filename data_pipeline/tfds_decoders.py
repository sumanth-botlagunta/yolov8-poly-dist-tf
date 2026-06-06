"""TFDS decoder classes for polygon detection and distance datasets.

Decoders convert raw TFDS feature dicts (loaded with as_supervised=False)
into a normalized schema consumed by the parsers downstream.

Access pattern mirrors TF Model Garden MSCOCODecoder:
  data['image']               top-level image tensor
  data['image/id']            literal key with slash (int64 image identifier)
  data['objects']['bbox']     two-step nested access for per-object fields
  data['objects']['label']
  data['objects']['is_crowd']
  data['objects']['area']
  data['objects']['polygon_points']   (or 'points' for older datasets)

The TFDS feature schemas for the custom datasets are:

cleaner_polygon2026 / field_misrecog2026 / station_misrecog:
    image: uint8 [H, W, 3]
    image/id: int64
    objects/bbox: float32 [N, 4]  ymin/xmin/ymax/xmax normalized
    objects/label: int64 [N]
    objects/is_crowd: bool [N]
    objects/area: float32 [N]
    objects/polygon_points: float32 [N, max_pts]  xy interleaved, -1 padded

servingbot_polygon:
    Same as above + objects/distance: float32 [N]

cleaner_copy_paste:
    image: uint8 [H, W, 4]  (RGBA — 4th channel is alpha mask)
    image/id: int64
    objects/bbox: float32 [4]
    objects/label: int64
    objects/polygon_points: float32 [max_pts*2]
    objects/obj_id: int64

Classes:
    PolygonDecoder: Decodes polygon TFDS records (e.g. cleaner_polygon2026).
    ServingBotDetDecoder: Decodes servingbot_polygon records with distance field.
    CopyPasteDecoder: Decodes cleaner_copy_paste RGBA object crops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf


class PolygonDecoder:
    """Decode polygon TFDS records to a standard tensor dictionary.

    Output schema:
        image: uint8 [H, W, 3]
        source_id: string scalar
        height: int32 scalar
        width: int32 scalar
        groundtruth_boxes: float32 [N, 4] (yxyx normalized 0-1)
        groundtruth_classes: int64 [N]
        groundtruth_polygons: float32 [N, max_polygon_coords] (xy interleaved, -1 padded)
        groundtruth_is_crowd: bool [N]
        groundtruth_area: float32 [N]
        groundtruth_dontcare: int64 [N]
    """

    def __init__(
        self,
        max_vertices: int = 10938,
        class_remap_json_path: Optional[str] = None,
        num_classes: int = 39,
        with_distance: bool = False,
    ):
        self._max_vertices = max_vertices
        self._num_classes = num_classes
        self._with_distance = with_distance

        self._class_remap: Optional[List[int]] = None
        if class_remap_json_path:
            path = Path(class_remap_json_path)
            if path.exists():
                with open(path) as f:
                    remap_dict = json.load(f)
                max_id = max(int(k) for k in remap_dict) + 1
                table = list(range(max_id))
                for old, new in remap_dict.items():
                    table[int(old)] = int(new)
                self._class_remap = table

    def decode(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Normalize a TFDS feature dict to our standard schema.

        Mirrors TF Model Garden MSCOCODecoder: direct two-step access for
        nested fields (data['objects']['bbox']) and literal-slash keys for
        top-level metadata (data['image/id']).
        """
        # --- image ---
        image = data['image']
        if image.dtype == tf.string:
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
        image = tf.cast(image, tf.uint8)

        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.int32)
        width = tf.cast(shape[1], tf.int32)

        # --- source_id — 'image/id' is a literal flat key in TFDS ---
        try:
            source_id = tf.strings.as_string(tf.cast(data['image/id'], tf.int64))
        except KeyError:
            source_id = tf.strings.as_string(height * 100000 + width)

        # --- per-object fields: always two-step access ---
        objects = data['objects']

        boxes = tf.cast(objects['bbox'], tf.float32)  # [N, 4] ymin/xmin/ymax/xmax
        if len(boxes.shape) == 1:
            boxes = tf.expand_dims(boxes, 0)

        classes = tf.cast(objects['label'], tf.int64)  # [N]
        if self._class_remap is not None:
            classes = self._remap_classes(classes)

        n = tf.shape(classes)[0]

        try:
            is_crowd = tf.cast(objects['is_crowd'], tf.bool)
        except KeyError:
            is_crowd = tf.zeros([n], dtype=tf.bool)

        try:
            area = tf.cast(objects['area'], tf.float32)
        except KeyError:
            area = tf.zeros([n], dtype=tf.float32)

        polygons = self._decode_polygons(objects, n)

        # dontcare is not in standard TFDS schemas; default to zeros
        dontcare = tf.zeros([n], dtype=tf.int64)

        return {
            'image': image,
            'source_id': source_id,
            'height': height,
            'width': width,
            'groundtruth_boxes': boxes,
            'groundtruth_classes': classes,
            'groundtruth_polygons': polygons,
            'groundtruth_is_crowd': is_crowd,
            'groundtruth_area': area,
            'groundtruth_dontcare': dontcare,
        }

    def _decode_polygons(self, objects: dict, n: tf.Tensor) -> tf.Tensor:
        """Extract polygon_points from the objects sub-dict."""
        try:
            return tf.cast(objects['polygon_points'], tf.float32)
        except KeyError:
            pass
        try:
            return tf.cast(objects['points'], tf.float32)
        except KeyError:
            pass
        return tf.zeros([n, self._max_vertices + 2], dtype=tf.float32) - 1.0

    def _remap_classes(self, classes: tf.Tensor) -> tf.Tensor:
        table_tensor = tf.constant(self._class_remap, dtype=tf.int64)
        clipped = tf.clip_by_value(classes, 0, len(self._class_remap) - 1)
        return tf.gather(table_tensor, clipped)


class ServingBotDetDecoder(PolygonDecoder):
    """Decode servingbot_polygon TFDS records including per-object distances.

    Extends PolygonDecoder output with:
        groundtruth_dists: float32 [N]  — raw meter distances (-1 if absent)
    """

    def __init__(
        self,
        class_remap_json_path: Optional[str] = None,
        num_classes: int = 39,
    ):
        super().__init__(
            max_vertices=10938,
            class_remap_json_path=class_remap_json_path,
            num_classes=num_classes,
            with_distance=True,
        )

    def decode(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        result = super().decode(data)

        objects = data['objects']
        n = tf.shape(result['groundtruth_classes'])[0]

        try:
            dists = tf.cast(objects['distance'], tf.float32)
        except KeyError:
            dists = tf.fill([n], -1.0)

        result['groundtruth_dists'] = dists
        return result


class CopyPasteDecoder:
    """Decode cleaner_copy_paste TFDS records (RGBA object crops).

    The cleaner_copy_paste:1.0.0 TFDS stores individual object crops with
    an alpha-channel mask used for compositing.

    Output schema:
        image: uint8 [H, W, 4]  — RGBA with alpha mask in channel 3
        image/id: int64 scalar
        orig_bbox: float32 [4]  — ymin, xmin, ymax, xmax normalized
        label: int64 scalar
        points: float32 [max_pts]  — xy interleaved, normalized
        obj_id: int64 scalar
    """

    def __init__(self, num_classes: int = 39):
        self._num_classes = num_classes

    def decode(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Normalize a cleaner_copy_paste TFDS feature dict."""
        image = data['image']
        if image.dtype == tf.string:
            image = tf.io.decode_image(image, channels=4, expand_animations=False)
            image.set_shape([None, None, 4])
        image = tf.cast(image, tf.uint8)  # [H, W, 4]

        try:
            image_id = tf.cast(data['image/id'], tf.int64)
        except KeyError:
            image_id = tf.constant(0, dtype=tf.int64)

        objects = data['objects']

        orig_bbox = tf.cast(tf.reshape(objects['bbox'], [4]), tf.float32)
        label = tf.cast(tf.reshape(objects['label'], []), tf.int64)

        try:
            points = tf.cast(tf.reshape(objects['polygon_points'], [-1]), tf.float32)
        except KeyError:
            points = tf.zeros([0], dtype=tf.float32)

        try:
            obj_id = tf.cast(tf.reshape(objects['obj_id'], []), tf.int64)
        except KeyError:
            obj_id = tf.constant(0, dtype=tf.int64)

        return {
            'image': image,
            'image/id': image_id,
            'orig_bbox': orig_bbox,
            'label': label,
            'points': points,
            'obj_id': obj_id,
        }
