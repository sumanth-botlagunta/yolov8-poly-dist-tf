"""Unit tests for TFDS decoder classes.

Tests run in eager mode (set by conftest.py).
No TFDS access is needed — decoders are tested with synthetic feature dicts
that exactly mirror the actual TFDS element_spec for each dataset.

Actual schemas:
  cleaner_polygon2026:2.0.0
    objects/points       float32 [N, 3972]  (NOT polygon_points)
    objects/area         int64  [N]         (NOT float32)
    objects/is_dontcare  bool   [N]         (NOT dontcare)

  servingbot_polygon:1.0.1
    objects/points       float32 [N, 10940]
    objects/distance     float32 [N]
    (no objects/is_dontcare)

  cleaner_copy_paste:1.0.0   — FLAT, no nested objects dict
    label, obj_id, orig_bbox, points are all top-level keys
"""

import json
import numpy as np
import pytest
import tensorflow as tf


# ---------------------------------------------------------------------------
# PolygonDecoder
# ---------------------------------------------------------------------------

class TestPolygonDecoder:

    @pytest.fixture
    def decoder(self):
        from data_pipeline.tfds_decoders import PolygonDecoder
        return PolygonDecoder(max_vertices=10938, num_classes=39)

    @pytest.fixture
    def raw_tfds_dict(self):
        """Synthetic cleaner_polygon2026-schema TFDS feature dict."""
        N = 4
        rng = np.random.RandomState(1)
        boxes = rng.uniform(0, 0.5, (N, 4)).astype(np.float32)
        boxes[:, 2:] += 0.1
        boxes = np.clip(boxes, 0.0, 1.0)
        return {
            'image':    tf.constant(rng.randint(0, 256, (320, 480, 3), dtype=np.uint8)),
            'image/id': tf.constant(42, dtype=tf.int64),
            'objects': {
                'bbox':        tf.constant(boxes),
                'label':       tf.constant(rng.randint(0, 39, (N,)), dtype=tf.int64),
                'is_crowd':    tf.constant([False, True, False, False]),
                'area':        tf.constant([1000, 2000, 1500, 800], dtype=tf.int64),
                'is_dontcare': tf.constant([False, False, False, False]),
                'points':      tf.constant(
                    rng.uniform(-1, 1, (N, 3972)).astype(np.float32)
                ),
            },
        }

    def test_output_keys(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        expected_keys = {
            'image', 'source_id', 'height', 'width',
            'groundtruth_boxes', 'groundtruth_classes',
            'groundtruth_polygons', 'groundtruth_is_crowd',
            'groundtruth_area', 'groundtruth_dontcare',
            'groundtruth_dists',   # -1.0 sentinel; enables zip+concat with distance stream
        }
        assert set(result.keys()) == expected_keys

    def test_image_shape(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        assert result['image'].shape == (320, 480, 3)
        assert result['image'].dtype == tf.uint8

    def test_image_spatial_dims(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        assert result['height'].numpy() == 320
        assert result['width'].numpy() == 480

    def test_source_id_is_string(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        assert result['source_id'].dtype == tf.string
        assert result['source_id'].numpy() == b'42'

    def test_boxes_shape_and_dtype(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        boxes = result['groundtruth_boxes']
        assert boxes.shape == (4, 4)
        assert boxes.dtype == tf.float32

    def test_boxes_normalized(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        boxes = result['groundtruth_boxes'].numpy()
        assert np.all(boxes >= 0.0)
        assert np.all(boxes <= 1.0)

    def test_classes_shape_and_dtype(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        assert result['groundtruth_classes'].shape == (4,)
        assert result['groundtruth_classes'].dtype == tf.int64

    def test_is_crowd_dtype(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        assert result['groundtruth_is_crowd'].dtype == tf.bool

    def test_area_cast_to_float32(self, decoder, raw_tfds_dict):
        """objects/area is int64 in TFDS; must be output as float32."""
        result = decoder.decode(raw_tfds_dict)
        assert result['groundtruth_area'].dtype == tf.float32

    def test_polygon_passthrough(self, decoder, raw_tfds_dict):
        """Decoder passes objects['points'] through as-is — no truncation/padding."""
        result = decoder.decode(raw_tfds_dict)
        polys = result['groundtruth_polygons']
        assert polys.dtype == tf.float32
        assert polys.shape == (4, 3972)   # exactly what TFDS provides

    def test_dontcare_from_is_dontcare(self, decoder, raw_tfds_dict):
        """groundtruth_dontcare comes from objects['is_dontcare'] (bool→int64)."""
        result = decoder.decode(raw_tfds_dict)
        assert result['groundtruth_dontcare'].dtype == tf.int64

    def test_dists_sentinel_minus_one(self, decoder, raw_tfds_dict):
        """All distances must be -1.0 sentinel for detection-only datasets."""
        result = decoder.decode(raw_tfds_dict)
        dists = result['groundtruth_dists'].numpy()
        assert dists.dtype == np.float32
        assert np.all(dists == -1.0)

    def test_missing_optional_fields(self):
        """Decoder falls back gracefully when optional fields are absent."""
        from data_pipeline.tfds_decoders import PolygonDecoder
        dec = PolygonDecoder(max_vertices=10938)
        data = {
            'image':    tf.zeros([64, 64, 3], tf.uint8),
            'image/id': tf.constant(1, dtype=tf.int64),
            'objects': {
                'bbox':  tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
                'label': tf.constant([5], dtype=tf.int64),
                # is_crowd, area, points, is_dontcare all absent
            },
        }
        result = dec.decode(data)
        assert result['groundtruth_is_crowd'].numpy().tolist() == [False]
        assert result['groundtruth_area'].numpy().tolist() == [0.0]
        assert np.all(result['groundtruth_polygons'].numpy() == -1.0)
        assert result['groundtruth_dontcare'].numpy().tolist() == [0]
        assert result['groundtruth_dists'].numpy().tolist() == [-1.0]

    def test_class_remapping(self, tmp_path):
        from data_pipeline.tfds_decoders import PolygonDecoder
        remap = {'0': 2, '1': 0, '3': 1}
        remap_file = tmp_path / 'remap.json'
        remap_file.write_text(json.dumps(remap))
        dec = PolygonDecoder(max_vertices=10938, class_remap_json_path=str(remap_file))
        data = {
            'image':    tf.zeros([64, 64, 3], tf.uint8),
            'image/id': tf.constant(1, dtype=tf.int64),
            'objects': {
                'bbox':  tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
                'label': tf.constant([0], dtype=tf.int64),
            },
        }
        result = dec.decode(data)
        assert result['groundtruth_classes'].numpy()[0] == 2  # 0 → 2


# ---------------------------------------------------------------------------
# ServingBotDetDecoder
# ---------------------------------------------------------------------------

class TestServingBotDetDecoder:

    @pytest.fixture
    def decoder(self):
        from data_pipeline.tfds_decoders import ServingBotDetDecoder
        return ServingBotDetDecoder(num_classes=39)

    def _make_servingbot_dict(self, n=3, dists=None):
        """Synthetic servingbot_polygon:1.0.1 feature dict (full schema)."""
        rng = np.random.RandomState(2)
        d = {
            'image':    tf.constant(rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)),
            'image/id': tf.constant(99, dtype=tf.int64),
            'objects': {
                'bbox':     tf.constant(rng.uniform(0, 0.8, (n, 4)).astype(np.float32)),
                'label':    tf.constant(list(range(n)), dtype=tf.int64),
                'is_crowd': tf.zeros([n], dtype=tf.bool),
                'area':     tf.ones([n], dtype=tf.int64) * 500,
                'points':   tf.zeros([n, 10940], dtype=tf.float32) - 1.0,
            },
        }
        if dists is not None:
            d['objects']['distance'] = tf.constant(dists, dtype=tf.float32)
        return d

    def test_has_distance_field(self, decoder):
        data = self._make_servingbot_dict(n=3, dists=[1.5, 5.0, -1.0])
        result = decoder.decode(data)
        assert 'groundtruth_dists' in result
        assert result['groundtruth_dists'].shape == (3,)
        assert result['groundtruth_dists'].dtype == tf.float32

    def test_distance_values_preserved(self, decoder):
        data = self._make_servingbot_dict(n=3, dists=[1.5, 5.0, 7.2])
        result = decoder.decode(data)
        vals = result['groundtruth_dists'].numpy().tolist()
        assert vals == pytest.approx([1.5, 5.0, 7.2])

    def test_polygon_shape_10940(self, decoder):
        """servingbot points tensor is [N, 10940]."""
        data = self._make_servingbot_dict(n=2, dists=[2.0, 3.0])
        result = decoder.decode(data)
        assert result['groundtruth_polygons'].shape == (2, 10940)

    def test_dontcare_zeros_when_absent(self, decoder):
        """servingbot has no is_dontcare field; decoder defaults to zeros."""
        data = self._make_servingbot_dict(n=3, dists=[1.0, 2.0, 3.0])
        result = decoder.decode(data)
        assert np.all(result['groundtruth_dontcare'].numpy() == 0)

    def test_missing_distance_defaults_to_neg1(self, decoder):
        data = self._make_servingbot_dict(n=1, dists=None)  # no distance key
        result = decoder.decode(data)
        assert result['groundtruth_dists'].numpy()[0] == pytest.approx(-1.0)

    def test_output_schema_matches_polygon_decoder(self, decoder):
        """Both decoders must produce the same output keys for zip+concat."""
        from data_pipeline.tfds_decoders import PolygonDecoder
        poly_data = {
            'image':    tf.zeros([64, 64, 3], tf.uint8),
            'image/id': tf.constant(1, dtype=tf.int64),
            'objects': {
                'bbox':  tf.zeros([1, 4], tf.float32),
                'label': tf.zeros([1], tf.int64),
            },
        }
        poly_keys = set(PolygonDecoder().decode(poly_data).keys())
        sb_keys = set(decoder.decode(self._make_servingbot_dict(dists=[1.0])).keys())
        assert poly_keys == sb_keys, "Schema mismatch prevents zip+concat"


# ---------------------------------------------------------------------------
# CopyPasteDecoder
# ---------------------------------------------------------------------------

class TestCopyPasteDecoder:

    @pytest.fixture
    def decoder(self):
        from data_pipeline.tfds_decoders import CopyPasteDecoder
        return CopyPasteDecoder(num_classes=39)

    def _make_cp_dict(self, h=128, w=128):
        """Flat schema matching cleaner_copy_paste:1.0.0 actual structure."""
        return {
            'image':     tf.constant(
                np.random.randint(0, 256, (h, w, 4), dtype=np.uint8)
            ),
            'image/id':  tf.constant(7, dtype=tf.int64),
            'label':     tf.constant(3, dtype=tf.int64),
            'obj_id':    tf.constant(99, dtype=tf.int64),
            'orig_bbox': tf.constant([0.1, 0.1, 0.6, 0.6], dtype=tf.float32),
            'points':    tf.zeros([3972], dtype=tf.float32) - 1.0,
        }

    def test_output_keys(self, decoder):
        result = decoder.decode(self._make_cp_dict())
        assert set(result.keys()) == {'image', 'image/id', 'orig_bbox', 'label', 'points', 'obj_id'}

    def test_image_has_4_channels(self, decoder):
        result = decoder.decode(self._make_cp_dict())
        assert result['image'].shape[-1] == 4
        assert result['image'].dtype == tf.uint8

    def test_bbox_shape(self, decoder):
        result = decoder.decode(self._make_cp_dict())
        assert result['orig_bbox'].shape == (4,)

    def test_label_scalar(self, decoder):
        result = decoder.decode(self._make_cp_dict())
        assert result['label'].shape == ()
        assert result['label'].numpy() == 3

    def test_points_shape(self, decoder):
        result = decoder.decode(self._make_cp_dict())
        assert result['points'].shape == (3972,)

    def test_flat_schema_no_objects_key(self, decoder):
        """cleaner_copy_paste is flat — data must have no 'objects' key."""
        data = self._make_cp_dict()
        assert 'objects' not in data
        result = decoder.decode(data)
        assert result['label'].numpy() == 3


# ---------------------------------------------------------------------------
# SkipDecoding (encoded-image) branch
# ---------------------------------------------------------------------------

class TestEncodedImageBranch:
    """The pipeline loads TFDS with SkipDecoding, so decoders receive ENCODED
    image bytes (tf.string) and must decode them in their string branch.
    These tests exercise that live path with real JPEG/PNG bytes."""

    def test_polygon_decoder_decodes_jpeg_bytes(self):
        """PolygonDecoder must accept a JPEG-encoded scalar string for 'image'."""
        from data_pipeline.tfds_decoders import PolygonDecoder
        rgb = tf.cast(tf.random.uniform([48, 64, 3], 0, 255), tf.uint8)
        ex = {
            'image':    tf.io.encode_jpeg(rgb),   # scalar tf.string
            'image/id': tf.constant(1, dtype=tf.int64),
            'objects': {
                'bbox':  tf.zeros([2, 4], tf.float32),
                'label': tf.zeros([2], tf.int64),
            },
        }
        out = PolygonDecoder(max_vertices=10938, num_classes=39).decode(ex)
        assert out['image'].dtype == tf.uint8
        assert tuple(out['image'].shape) == (48, 64, 3)
        assert int(out['height']) == 48
        assert int(out['width']) == 64

    def test_copy_paste_decoder_decodes_rgba_png_bytes(self):
        """CopyPasteDecoder must accept a PNG-encoded scalar string for 'image'."""
        from data_pipeline.tfds_decoders import CopyPasteDecoder
        rgba = tf.cast(tf.random.uniform([32, 32, 4], 0, 255), tf.uint8)
        ex = {
            'image':     tf.io.encode_png(rgba),   # scalar tf.string
            'image/id':  tf.constant(7, dtype=tf.int64),
            'label':     tf.constant(3, dtype=tf.int64),
            'obj_id':    tf.constant(99, dtype=tf.int64),
            'orig_bbox': tf.constant([0.1, 0.1, 0.6, 0.6], dtype=tf.float32),
            'points':    tf.zeros([3972], dtype=tf.float32) - 1.0,
        }
        out = CopyPasteDecoder(num_classes=39).decode(ex)
        assert out['image'].dtype == tf.uint8
        assert tuple(out['image'].shape) == (32, 32, 4)
