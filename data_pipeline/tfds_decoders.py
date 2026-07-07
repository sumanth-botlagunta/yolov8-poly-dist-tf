"""TFDS decoder classes for polygon detection and distance datasets.

Access pattern mirrors TF Model Garden MSCOCODecoder — direct access only:
  data['image']                   top-level
  data['image/id']                literal slash key
  data['objects']['bbox']         two-step nested for per-object fields
  data['objects']['label']
  data['objects']['is_crowd']
  data['objects']['area']         dtype int64 in TFDS → cast to float32
  data['objects']['points']       polygon coords (not 'polygon_points')
  data['objects']['is_dontcare']  dontcare flag (not 'dontcare')

Actual TFDS schemas used in this project:

cleaner_polygon2026:2.0.0 / field_misrecog2026:1.0.0 / station_misrecog:1.1.0
  (all identical structure)
    image:            uint8  [H, W, 3]
    image/filename:   string
    image/id:         int64
    objects/area:     int64  [N]
    objects/bbox:     float32 [N, 4]   ymin/xmin/ymax/xmax normalized
    objects/id:       int64  [N]
    objects/is_crowd: bool   [N]
    objects/is_dontcare: bool [N]
    objects/label:    int64  [N]
    objects/points:   float32 [N, 3972]   xy interleaved, -1 padded

servingbot_polygon:1.0.1
    Same as above except:
      objects/points:    float32 [N, 10940]
      objects/distance:  float32 [N]
      (no objects/is_dontcare)

cleaner_copy_paste:1.0.0   — FLAT, no nested 'objects' dict
    image:          uint8  [H, W, 4]   RGBA
    image/filename: string
    image/id:       int64
    label:          int64  scalar
    obj_id:         int64  scalar
    orig_bbox:      float32 [4]
    points:         float32 [3972]     xy interleaved, -1 padded

Classes:
    PolygonDecoder:        cleaner_polygon2026 / field_misrecog2026 / station_misrecog
    ServingBotDetDecoder:  servingbot_polygon (adds real distance values)
    CopyPasteDecoder:      cleaner_copy_paste (flat schema, RGBA image)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf


class PolygonDecoder:
    """Decode polygon TFDS records (cleaner_polygon2026 schema).

    All three detection datasets share the same schema and are decoded here.
    Distance is absent from these datasets; groundtruth_dists is filled with
    the sentinel value -1.0 so the output schema matches ServingBotDetDecoder,
    enabling zip+concat of the two streams.

    Output schema:
        image:                  uint8  [H, W, 3]
        source_id:              string scalar
        height:                 int32 scalar
        width:                  int32 scalar
        groundtruth_boxes:      float32 [N, 4]   ymin/xmin/ymax/xmax
        groundtruth_classes:    int64  [N]
        groundtruth_polygons:   float32 [N, pts]  xy interleaved, -1 padded
        groundtruth_is_crowd:   bool   [N]
        groundtruth_area:       float32 [N]
        groundtruth_dontcare:   int64  [N]
        groundtruth_dists:      float32 [N]   (-1.0 sentinel for no-distance data)
    """

    def __init__(
        self,
        max_vertices: int = 10938,
        class_remap_json_path: Optional[str] = None,
        num_classes: int = 39,
        resample_points: int = 0,
    ):
        self._max_vertices = max_vertices
        self._num_classes = num_classes
        # Optional: resample every polygon to a fixed `resample_points` vertices at
        # decode time so the whole augmentation pipeline carries [N, 2*K] instead of
        # the (often huge) raw stored width. 0 = off (raw width preserved). The
        # 24-bin radial target is preserved (see augmentations.resample_polygons).
        self._resample_points = resample_points

        self._class_remap: Optional[List[int]] = None
        if class_remap_json_path:
            path = Path(class_remap_json_path)
            if path.exists():
                with open(path) as f:
                    remap_dict = json.load(f)
                # Build a FULL identity table over [0, num_classes-1], then apply
                # overrides. A table sized to max(remap_key)+1 would clip every
                # class above that key to the last index in _remap_classes,
                # silently collapsing unrelated classes onto the override target.
                table = list(range(num_classes))
                for old, new in remap_dict.items():
                    table[int(old)] = int(new)
                self._class_remap = table

    def decode(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Decode a cleaner_polygon2026-schema TFDS feature dict."""
        # image
        image = data['image']
        if image.dtype == tf.string:
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
        image = tf.cast(image, tf.uint8)

        shape = tf.shape(image)
        # Prefer explicitly-stored ORIGINAL dims when present (pre-resized
        # dataset variants carry them): the copy-paste resolution correction
        # needs the original capture size, which tf.shape() can no longer
        # provide once images are stored small.
        try:
            height = tf.cast(data['orig_height'], tf.int32)
            width = tf.cast(data['orig_width'], tf.int32)
        except KeyError:
            height = tf.cast(shape[0], tf.int32)
            width = tf.cast(shape[1], tf.int32)

        # source_id — 'image/id' is a literal flat key in TFDS (slash in key name)
        source_id = tf.strings.as_string(tf.cast(data['image/id'], tf.int64))

        # per-object fields — always two-step: data['objects'][field]
        objects = data['objects']

        boxes = tf.cast(objects['bbox'], tf.float32)        # [N, 4]
        classes = tf.cast(objects['label'], tf.int64)       # [N]
        if self._class_remap is not None:
            classes = self._remap_classes(classes)

        n = tf.shape(classes)[0]

        try:
            is_crowd = tf.cast(objects['is_crowd'], tf.bool)
        except KeyError:
            is_crowd = tf.zeros([n], dtype=tf.bool)

        try:
            area = tf.cast(objects['area'], tf.float32)     # int64 in TFDS → float32
        except KeyError:
            area = tf.zeros([n], dtype=tf.float32)

        try:
            polygons = tf.cast(objects['points'], tf.float32)   # [N, pts]
        except KeyError:
            polygons = tf.zeros([n, self._max_vertices + 2], dtype=tf.float32) - 1.0

        # Optionally shrink polygon width up front (valid vertices are a prefix
        # here, before any transform) — big throughput win for huge max_vertices.
        if self._resample_points and self._resample_points > 0:
            from data_pipeline.augmentations import resample_polygons
            polygons = resample_polygons(polygons, self._resample_points)

        # is_dontcare exists in cleaner datasets; ServingBot has no such field
        try:
            dontcare = tf.cast(objects['is_dontcare'], tf.int64)
        except KeyError:
            dontcare = tf.zeros([n], dtype=tf.int64)

        # No distance in detection datasets — sentinel so schema matches ServingBot
        dists = tf.fill([n], -1.0)

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
            'groundtruth_dists': dists,
        }

    def _remap_classes(self, classes: tf.Tensor) -> tf.Tensor:
        table_tensor = tf.constant(self._class_remap, dtype=tf.int64)
        clipped = tf.clip_by_value(classes, 0, len(self._class_remap) - 1)
        return tf.gather(table_tensor, clipped)


class ServingBotDetDecoder(PolygonDecoder):
    """Decode servingbot_polygon:1.0.1 TFDS records.

    Identical schema to cleaner_polygon2026 except:
      - objects/points shape is [N, 10940] (vs 3972)
      - objects/distance: float32 [N] is present
      - objects/is_dontcare is absent

    Overrides groundtruth_dists with the real distance values from
    objects['distance'] instead of the -1.0 sentinel from PolygonDecoder.
    """

    def __init__(
        self,
        num_classes: int = 39,
        resample_points: int = 0,
    ):
        super().__init__(
            max_vertices=10938,
            num_classes=num_classes,
            resample_points=resample_points,
        )

        from configs.class_map import SERVINGBOT_CLASS_REMAP

        # Build identity table over the full [0, num_classes-1] range, then
        # apply per-class overrides.  Using full length ensures _remap_classes
        # clip_by_value(classes, 0, num_classes-1) never maps unexpected IDs
        # to the wrong target class.
        _table = list(range(num_classes))
        for _old, _new in SERVINGBOT_CLASS_REMAP.items():
            _table[_old] = _new
        self._class_remap = _table

    def decode(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        result = super().decode(data)
        try:
            result['groundtruth_dists'] = tf.cast(data['objects']['distance'], tf.float32)
        except KeyError:
            pass  # parent already filled -1.0 sentinel
        return result


class CopyPasteDecoder:
    """Decode cleaner_copy_paste:1.0.0 TFDS records (RGBA object crops).

    This dataset has a FLAT schema — no nested 'objects' dict.  Every field
    is a scalar or 1-D tensor at the top level of the feature dict.  Each
    TFDS example is one object crop with an alpha mask in channel 3.

    Output schema:
        image:      uint8  [H, W, 4]   RGBA (alpha mask in channel 3)
        image/id:   int64  scalar
        orig_bbox:  float32 [4]        ymin, xmin, ymax, xmax normalized
        label:      int64  scalar
        points:     float32 [3972]     xy interleaved, -1 padded
        obj_id:     int64  scalar
    """

    def __init__(self, num_classes: int = 39):
        self._num_classes = num_classes

    def decode(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Decode a cleaner_copy_paste flat feature dict."""
        image = data['image']
        if image.dtype == tf.string:
            image = tf.io.decode_image(image, channels=4, expand_animations=False)
            image.set_shape([None, None, 4])
        image = tf.cast(image, tf.uint8)  # [H, W, 4]

        # All fields are flat top-level keys — no objects sub-dict
        return {
            'image': image,
            'image/id': tf.cast(data['image/id'], tf.int64),
            'orig_bbox': tf.cast(data['orig_bbox'], tf.float32),   # [4]
            'label': tf.cast(data['label'], tf.int64),             # scalar
            'points': tf.cast(data['points'], tf.float32),         # [3972]
            'obj_id': tf.cast(data['obj_id'], tf.int64),           # scalar
        }
