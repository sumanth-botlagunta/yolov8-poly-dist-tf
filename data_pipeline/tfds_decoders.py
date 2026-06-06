"""TFDS decoder classes for polygon detection and distance datasets.

Decoders convert raw TFDS feature dicts (loaded with as_supervised=False)
into a normalized schema consumed by the parsers downstream.

The TFDS feature schemas for the custom datasets are:

cleaner_polygon2026 / field_misrecog2026 / station_misrecog:
    image: uint8 [H, W, 3]
    image/id: int64  (or image/filename: string)
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
    objects (single object per image):
        bbox: float32 [4]
        label: int64
        polygon_points: float32 [max_pts*2]
        obj_id: int64

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


def _get_nested(data: dict, *keys: str, default=None):
    """Try each key in order; return the first that exists, or default."""
    for key in keys:
        if key in data:
            return data[key]
        # Try nested: 'objects/bbox' → data['objects']['bbox']
        parts = key.split('/')
        node = data
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if node is not None:
            return node
    return default


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
                # Build lookup table indexed by old class ID.
                max_id = max(int(k) for k in remap_dict) + 1
                table = list(range(max_id))
                for old, new in remap_dict.items():
                    table[int(old)] = int(new)
                self._class_remap = table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Normalize a TFDS feature dict to our standard schema.

        Args:
            data: Feature dict from tfds.load(as_supervised=False).

        Returns:
            Normalized tensor dict conforming to the output schema above.
        """
        image = self._decode_image(data)
        shape = tf.shape(image)
        height = tf.cast(shape[0], tf.int32)
        width = tf.cast(shape[1], tf.int32)

        source_id = self._decode_source_id(data, height, width)
        boxes, classes, is_crowd, area, polygons, dontcare = (
            self._decode_objects(data, height, width)
        )

        result = {
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
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode_image(self, data: dict) -> tf.Tensor:
        image = _get_nested(data, 'image')
        if image is None:
            raise ValueError("TFDS feature dict missing 'image' field.")
        # TFDS may return already-decoded uint8 or encoded bytes.
        if image.dtype == tf.string:
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
        return tf.cast(image, tf.uint8)

    def _decode_source_id(self, data: dict, height, width) -> tf.Tensor:
        sid = _get_nested(data, 'image/id', 'image_id', 'source_id')
        if sid is None:
            # Derive a deterministic string from image shape as fallback.
            sid = tf.strings.as_string(height * 100000 + width)
        if sid.dtype != tf.string:
            sid = tf.strings.as_string(tf.cast(sid, tf.int64))
        return sid

    def _decode_objects(self, data: dict, height, width):
        objects = _get_nested(data, 'objects') or data

        # Bounding boxes — stored as [N, 4] ymin/xmin/ymax/xmax normalized.
        boxes = tf.cast(
            _get_nested(objects, 'bbox', 'groundtruth_boxes',
                        data, 'objects/bbox'),
            tf.float32,
        )  # [N, 4]
        if len(boxes.shape) == 1:
            boxes = tf.expand_dims(boxes, 0)

        # Classes
        classes = tf.cast(
            _get_nested(objects, 'label', 'labels', 'groundtruth_classes',
                        data, 'objects/label'),
            tf.int64,
        )  # [N]
        if self._class_remap is not None:
            classes = self._remap_classes(classes)

        n = tf.shape(classes)[0]

        # is_crowd
        is_crowd_raw = _get_nested(
            objects, 'is_crowd', 'groundtruth_is_crowd',
            data, 'objects/is_crowd',
        )
        if is_crowd_raw is None:
            is_crowd = tf.zeros([n], dtype=tf.bool)
        else:
            is_crowd = tf.cast(is_crowd_raw, tf.bool)

        # area
        area_raw = _get_nested(
            objects, 'area', 'groundtruth_area',
            data, 'objects/area',
        )
        if area_raw is None:
            area = tf.zeros([n], dtype=tf.float32)
        else:
            area = tf.cast(area_raw, tf.float32)

        # polygons
        polygons = self._decode_polygons(objects, data, n)

        # dontcare
        dc_raw = _get_nested(
            objects, 'dontcare', 'groundtruth_dontcare',
            data, 'objects/dontcare',
        )
        if dc_raw is None:
            dontcare = tf.zeros([n], dtype=tf.int64)
        else:
            dontcare = tf.cast(dc_raw, tf.int64)

        return boxes, classes, is_crowd, area, polygons, dontcare

    def _decode_polygons(self, objects: dict, data: dict, n: tf.Tensor) -> tf.Tensor:
        """Extract polygon coordinates, pad to [N, max_vertices] with -1."""
        pts_raw = _get_nested(
            objects, 'polygon_points', 'polygons',
            data, 'objects/polygon_points',
        )
        if pts_raw is None:
            # No polygon annotations — return all-zeros (not -1) so the loss
            # ignores them (the parser's conf=0 will suppress them).
            return tf.zeros([n, self._max_vertices], dtype=tf.float32) - 1.0

        pts = tf.cast(pts_raw, tf.float32)  # [N, variable] or [N, max_pts]

        # Pad / truncate to exactly max_vertices columns.
        current_cols = tf.shape(pts)[1]
        pad_needed = tf.maximum(0, self._max_vertices - current_cols)
        pts = tf.pad(pts, [[0, 0], [0, pad_needed]], constant_values=-1.0)
        pts = pts[:, :self._max_vertices]  # truncate if needed
        return pts

    def _remap_classes(self, classes: tf.Tensor) -> tf.Tensor:
        table_tensor = tf.constant(self._class_remap, dtype=tf.int64)
        # Clip to valid range before lookup.
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

        objects = _get_nested(data, 'objects') or data
        n = tf.shape(result['groundtruth_classes'])[0]

        dists_raw = _get_nested(
            objects, 'distance', 'distances', 'groundtruth_dists',
            data, 'objects/distance',
        )
        if dists_raw is None:
            dists = tf.fill([n], -1.0)
        else:
            dists = tf.cast(dists_raw, tf.float32)

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
        # Image — 4 channels (RGB + alpha mask).
        image = _get_nested(data, 'image')
        if image.dtype == tf.string:
            image = tf.io.decode_image(image, channels=4, expand_animations=False)
            image.set_shape([None, None, 4])
        image = tf.cast(image, tf.uint8)  # [H, W, 4]

        image_id = _get_nested(data, 'image/id', 'image_id', 'id')
        if image_id is None:
            image_id = tf.constant(0, dtype=tf.int64)
        image_id = tf.cast(image_id, tf.int64)

        objects = _get_nested(data, 'objects') or data

        bbox = _get_nested(objects, 'bbox', 'orig_bbox', data, 'orig_bbox')
        if bbox is None:
            bbox = tf.zeros([4], dtype=tf.float32)
        orig_bbox = tf.cast(tf.reshape(bbox, [4]), tf.float32)

        label = _get_nested(objects, 'label', data, 'label')
        if label is None:
            label = tf.constant(0, dtype=tf.int64)
        label = tf.cast(tf.reshape(label, []), tf.int64)

        points = _get_nested(objects, 'polygon_points', 'points', data, 'points')
        if points is None:
            points = tf.zeros([0], dtype=tf.float32)
        points = tf.cast(tf.reshape(points, [-1]), tf.float32)

        obj_id = _get_nested(objects, 'obj_id', data, 'obj_id')
        if obj_id is None:
            obj_id = tf.constant(0, dtype=tf.int64)
        obj_id = tf.cast(tf.reshape(obj_id, []), tf.int64)

        return {
            'image': image,
            'image/id': image_id,
            'orig_bbox': orig_bbox,
            'label': label,
            'points': points,
            'obj_id': obj_id,
        }
