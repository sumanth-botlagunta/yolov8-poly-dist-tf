"""Unit tests for TFDS decoder classes.

Tests run in eager mode (set by conftest.py).
No TFDS access is needed — decoders are tested with synthetic feature dicts.
"""

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
        return PolygonDecoder(max_vertices=100, num_classes=39)

    @pytest.fixture
    def raw_tfds_dict(self):
        """Simulate a TFDS feature dict with nested objects."""
        N = 4
        rng = np.random.RandomState(1)
        boxes = rng.uniform(0, 0.5, (N, 4)).astype(np.float32)
        boxes[:, 2] += 0.1
        boxes[:, 3] += 0.1
        boxes = np.clip(boxes, 0.0, 1.0)

        return {
            'image': tf.constant(
                rng.randint(0, 256, (320, 480, 3), dtype=np.uint8)
            ),
            'image/id': tf.constant(42, dtype=tf.int64),
            'objects': {
                'bbox': tf.constant(boxes),
                'label': tf.constant(rng.randint(0, 39, (N,)), dtype=tf.int64),
                'is_crowd': tf.constant([False, True, False, False]),
                'area': tf.constant(rng.uniform(100, 1000, (N,)).astype(np.float32)),
                'polygon_points': tf.constant(
                    rng.uniform(-1, 1, (N, 60)).astype(np.float32)
                ),
            }
        }

    def test_output_keys(self, decoder, raw_tfds_dict):
        result = decoder.decode(raw_tfds_dict)
        expected_keys = {
            'image', 'source_id', 'height', 'width',
            'groundtruth_boxes', 'groundtruth_classes',
            'groundtruth_polygons', 'groundtruth_is_crowd',
            'groundtruth_area', 'groundtruth_dontcare',
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

    def test_polygon_shape_padded(self, decoder, raw_tfds_dict):
        """Polygons should be padded/truncated to max_vertices columns."""
        result = decoder.decode(raw_tfds_dict)
        polys = result['groundtruth_polygons']
        assert polys.shape == (4, 100)  # max_vertices=100
        assert polys.dtype == tf.float32

    def test_polygon_truncation(self):
        """Polygons longer than max_vertices are truncated."""
        from data_pipeline.tfds_decoders import PolygonDecoder
        dec = PolygonDecoder(max_vertices=10)
        data = {
            'image': tf.zeros([64, 64, 3], tf.uint8),
            'objects': {
                'bbox': tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
                'label': tf.constant([0], dtype=tf.int64),
                'polygon_points': tf.constant(
                    np.random.rand(1, 20).astype(np.float32)
                ),
            }
        }
        result = dec.decode(data)
        assert result['groundtruth_polygons'].shape == (1, 10)

    def test_polygon_padding_value(self):
        """Polygons shorter than max_vertices are padded with -1."""
        from data_pipeline.tfds_decoders import PolygonDecoder
        dec = PolygonDecoder(max_vertices=20)
        data = {
            'image': tf.zeros([64, 64, 3], tf.uint8),
            'objects': {
                'bbox': tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
                'label': tf.constant([0], dtype=tf.int64),
                'polygon_points': tf.constant([[0.2, 0.3]], dtype=tf.float32),
            }
        }
        result = dec.decode(data)
        polys = result['groundtruth_polygons'].numpy()
        assert polys[0, 0] == pytest.approx(0.2)
        assert polys[0, 1] == pytest.approx(0.3)
        assert np.all(polys[0, 2:] == -1.0)

    def test_missing_objects_graceful(self):
        """Decoder handles missing optional fields without error."""
        from data_pipeline.tfds_decoders import PolygonDecoder
        dec = PolygonDecoder(max_vertices=10)
        data = {
            'image': tf.zeros([64, 64, 3], tf.uint8),
            'objects': {
                'bbox': tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
                'label': tf.constant([5], dtype=tf.int64),
                # is_crowd, area, polygon_points, dontcare all absent
            }
        }
        result = dec.decode(data)
        assert result['groundtruth_is_crowd'].numpy() == [False]
        assert result['groundtruth_area'].numpy() == [0.0]
        assert np.all(result['groundtruth_polygons'].numpy() == -1.0)
        assert result['groundtruth_dontcare'].numpy() == [0]

    def test_class_remapping(self, tmp_path):
        """Class IDs are remapped when class_remap_json_path is provided."""
        import json
        from data_pipeline.tfds_decoders import PolygonDecoder

        remap = {'0': 2, '1': 0, '3': 1}
        remap_file = tmp_path / 'remap.json'
        remap_file.write_text(json.dumps(remap))

        dec = PolygonDecoder(max_vertices=10, class_remap_json_path=str(remap_file))
        data = {
            'image': tf.zeros([64, 64, 3], tf.uint8),
            'objects': {
                'bbox': tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
                'label': tf.constant([0], dtype=tf.int64),
            }
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

    def test_has_distance_field(self, decoder):
        N = 3
        rng = np.random.RandomState(2)
        data = {
            'image': tf.constant(rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)),
            'objects': {
                'bbox': tf.constant(rng.uniform(0, 0.8, (N, 4)).astype(np.float32)),
                'label': tf.constant([0, 1, 2], dtype=tf.int64),
                'distance': tf.constant([1.5, 5.0, -1.0], dtype=tf.float32),
            }
        }
        result = decoder.decode(data)
        assert 'groundtruth_dists' in result
        assert result['groundtruth_dists'].shape == (N,)
        assert result['groundtruth_dists'].dtype == tf.float32

    def test_distance_values_preserved(self, decoder):
        data = {
            'image': tf.zeros([64, 64, 3], tf.uint8),
            'objects': {
                'bbox': tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
                'label': tf.constant([0], dtype=tf.int64),
                'distance': tf.constant([3.7], dtype=tf.float32),
            }
        }
        result = decoder.decode(data)
        assert result['groundtruth_dists'].numpy()[0] == pytest.approx(3.7)

    def test_missing_distance_defaults_to_neg1(self, decoder):
        data = {
            'image': tf.zeros([64, 64, 3], tf.uint8),
            'objects': {
                'bbox': tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32),
                'label': tf.constant([0], dtype=tf.int64),
                # No distance field
            }
        }
        result = decoder.decode(data)
        assert result['groundtruth_dists'].numpy()[0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# CopyPasteDecoder
# ---------------------------------------------------------------------------

class TestCopyPasteDecoder:

    @pytest.fixture
    def decoder(self):
        from data_pipeline.tfds_decoders import CopyPasteDecoder
        return CopyPasteDecoder(num_classes=39)

    def test_output_keys(self, decoder):
        data = {
            'image': tf.constant(
                np.random.randint(0, 256, (128, 128, 4), dtype=np.uint8)
            ),
            'image/id': tf.constant(7, dtype=tf.int64),
            'objects': {
                'bbox': tf.constant([0.1, 0.1, 0.6, 0.6], dtype=tf.float32),
                'label': tf.constant(3, dtype=tf.int64),
                'polygon_points': tf.constant(
                    np.random.rand(40).astype(np.float32)
                ),
                'obj_id': tf.constant(99, dtype=tf.int64),
            }
        }
        result = decoder.decode(data)
        assert set(result.keys()) == {'image', 'image/id', 'orig_bbox', 'label', 'points', 'obj_id'}

    def test_image_has_4_channels(self, decoder):
        data = {
            'image': tf.constant(
                np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
            ),
            'objects': {
                'bbox': tf.constant([0.1, 0.1, 0.5, 0.5], dtype=tf.float32),
                'label': tf.constant(0, dtype=tf.int64),
            }
        }
        result = decoder.decode(data)
        assert result['image'].shape[-1] == 4  # RGBA
        assert result['image'].dtype == tf.uint8

    def test_bbox_shape(self, decoder):
        data = {
            'image': tf.zeros([64, 64, 4], tf.uint8),
            'objects': {
                'bbox': tf.constant([0.1, 0.2, 0.8, 0.9], dtype=tf.float32),
                'label': tf.constant(1, dtype=tf.int64),
            }
        }
        result = decoder.decode(data)
        assert result['orig_bbox'].shape == (4,)

    def test_label_scalar(self, decoder):
        data = {
            'image': tf.zeros([64, 64, 4], tf.uint8),
            'objects': {
                'bbox': tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32),
                'label': tf.constant(5, dtype=tf.int64),
            }
        }
        result = decoder.decode(data)
        assert result['label'].shape == ()
        assert result['label'].numpy() == 5
