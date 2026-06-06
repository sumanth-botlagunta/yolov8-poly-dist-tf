"""Tests for TFDS decoder classes.

Each test constructs a synthetic dict that exactly mirrors the actual TFDS
element_spec for that dataset, then calls the decoder and asserts:
  - all required output keys are present
  - dtypes and shapes match the documented output schema
  - dataset-specific behaviour (distance sentinel, flat copy-paste schema, etc.)

Actual TFDS schemas (from tfds.load / element_spec):

cleaner_polygon2026:2.0.0
    image:              uint8  [H, W, 3]
    image/filename:     string
    image/id:           int64
    objects/area:       int64  [N]
    objects/bbox:       float32 [N, 4]
    objects/id:         int64  [N]
    objects/is_crowd:   bool   [N]
    objects/is_dontcare:bool   [N]
    objects/label:      int64  [N]
    objects/points:     float32 [N, 3972]

servingbot_polygon:1.0.1
    (same as above, except)
    objects/points:     float32 [N, 10940]
    objects/distance:   float32 [N]
    (no objects/is_dontcare)

cleaner_copy_paste:1.0.0   — FLAT, no nested objects dict
    image:              uint8  [H, W, 4]   RGBA
    image/filename:     string
    image/id:           int64
    label:              int64  scalar
    obj_id:             int64  scalar
    orig_bbox:          float32 [4]
    points:             float32 [3972]
"""

import unittest
import tensorflow as tf


# ---------------------------------------------------------------------------
# Synthetic TFDS feature dicts
# ---------------------------------------------------------------------------

def _cleaner_polygon_example(n: int = 3, h: int = 480, w: int = 640) -> dict:
    """Mimics one cleaner_polygon2026:2.0.0 TFDS element."""
    return {
        'image':          tf.zeros([h, w, 3], dtype=tf.uint8),
        'image/filename': tf.constant('test.jpg', dtype=tf.string),
        'image/id':       tf.constant(123, dtype=tf.int64),
        'objects': {
            'area':        tf.ones([n], dtype=tf.int64) * 1000,
            'bbox':        tf.zeros([n, 4], dtype=tf.float32),
            'id':          tf.range(n, dtype=tf.int64),
            'is_crowd':    tf.zeros([n], dtype=tf.bool),
            'is_dontcare': tf.zeros([n], dtype=tf.bool),
            'label':       tf.zeros([n], dtype=tf.int64),
            'points':      tf.zeros([n, 3972], dtype=tf.float32) - 1.0,
        },
    }


def _servingbot_example(n: int = 3, h: int = 640, w: int = 480) -> dict:
    """Mimics one servingbot_polygon:1.0.1 TFDS element."""
    return {
        'image':          tf.zeros([h, w, 3], dtype=tf.uint8),
        'image/filename': tf.constant('sb.jpg', dtype=tf.string),
        'image/id':       tf.constant(456, dtype=tf.int64),
        'objects': {
            'area':       tf.ones([n], dtype=tf.int64) * 2000,
            'bbox':       tf.zeros([n, 4], dtype=tf.float32),
            'id':         tf.range(n, dtype=tf.int64),
            'distance':   tf.constant([1.5, 3.0, 7.2][:n], dtype=tf.float32),
            'is_crowd':   tf.zeros([n], dtype=tf.bool),
            'label':      tf.zeros([n], dtype=tf.int64),
            'points':     tf.zeros([n, 10940], dtype=tf.float32) - 1.0,
        },
    }


def _copy_paste_example(h: int = 256, w: int = 256) -> dict:
    """Mimics one cleaner_copy_paste:1.0.0 TFDS element (flat schema)."""
    return {
        'image':          tf.zeros([h, w, 4], dtype=tf.uint8),   # RGBA
        'image/filename': tf.constant('cp.png', dtype=tf.string),
        'image/id':       tf.constant(789, dtype=tf.int64),
        'label':          tf.constant(5, dtype=tf.int64),
        'obj_id':         tf.constant(42, dtype=tf.int64),
        'orig_bbox':      tf.constant([0.1, 0.2, 0.5, 0.7], dtype=tf.float32),
        'points':         tf.zeros([3972], dtype=tf.float32) - 1.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPolygonDecoder(unittest.TestCase):

    def setUp(self):
        from data_pipeline.tfds_decoders import PolygonDecoder
        self.decoder = PolygonDecoder(max_vertices=10938, num_classes=39)

    def test_output_keys_present(self):
        out = self.decoder.decode(_cleaner_polygon_example())
        required = {
            'image', 'source_id', 'height', 'width',
            'groundtruth_boxes', 'groundtruth_classes', 'groundtruth_polygons',
            'groundtruth_is_crowd', 'groundtruth_area', 'groundtruth_dontcare',
            'groundtruth_dists',
        }
        self.assertEqual(required, set(out.keys()))

    def test_image_shape_and_dtype(self):
        out = self.decoder.decode(_cleaner_polygon_example(h=480, w=640))
        self.assertEqual(out['image'].dtype, tf.uint8)
        self.assertEqual(tuple(tf.shape(out['image']).numpy()), (480, 640, 3))

    def test_source_id_from_image_id(self):
        out = self.decoder.decode(_cleaner_polygon_example())
        self.assertEqual(out['source_id'].numpy(), b'123')

    def test_boxes_dtype_and_shape(self):
        n = 4
        out = self.decoder.decode(_cleaner_polygon_example(n=n))
        self.assertEqual(out['groundtruth_boxes'].dtype, tf.float32)
        self.assertEqual(tuple(tf.shape(out['groundtruth_boxes']).numpy()), (n, 4))

    def test_area_cast_from_int64_to_float32(self):
        """objects/area is int64 in TFDS; decoder must cast to float32."""
        out = self.decoder.decode(_cleaner_polygon_example(n=3))
        self.assertEqual(out['groundtruth_area'].dtype, tf.float32)

    def test_polygons_from_objects_points(self):
        """Polygons come from objects['points'] (not 'polygon_points')."""
        n = 2
        out = self.decoder.decode(_cleaner_polygon_example(n=n))
        self.assertEqual(out['groundtruth_polygons'].dtype, tf.float32)
        # shape is [N, 3972] — whatever TFDS provides
        self.assertEqual(tuple(tf.shape(out['groundtruth_polygons']).numpy()), (n, 3972))

    def test_dontcare_from_is_dontcare(self):
        """groundtruth_dontcare comes from objects['is_dontcare'] (bool→int64)."""
        out = self.decoder.decode(_cleaner_polygon_example(n=3))
        self.assertEqual(out['groundtruth_dontcare'].dtype, tf.int64)

    def test_dists_sentinel_for_no_distance_data(self):
        """Detection datasets have no distance; decoder fills -1.0 sentinel."""
        n = 3
        out = self.decoder.decode(_cleaner_polygon_example(n=n))
        self.assertEqual(out['groundtruth_dists'].dtype, tf.float32)
        self.assertTrue(
            tf.reduce_all(out['groundtruth_dists'] == -1.0).numpy(),
            "Expected all -1.0 sentinel values for detection dataset",
        )

    def test_variable_image_sizes(self):
        """Decoder must handle any native image resolution."""
        for h, w in [(1280, 800), (340, 510), (100, 100)]:
            out = self.decoder.decode(_cleaner_polygon_example(h=h, w=w))
            actual = tuple(tf.shape(out['image']).numpy())
            self.assertEqual(actual, (h, w, 3), f"Wrong shape for ({h},{w})")


class TestServingBotDetDecoder(unittest.TestCase):

    def setUp(self):
        from data_pipeline.tfds_decoders import ServingBotDetDecoder
        self.decoder = ServingBotDetDecoder(num_classes=39)

    def test_real_distance_values(self):
        """ServingBotDetDecoder must read actual objects/distance values."""
        n = 3
        out = self.decoder.decode(_servingbot_example(n=n))
        self.assertEqual(out['groundtruth_dists'].dtype, tf.float32)
        expected = [1.5, 3.0, 7.2]
        actual = out['groundtruth_dists'].numpy().tolist()
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=4)

    def test_polygons_shape_10940(self):
        """servingbot_polygon has [N, 10940] points."""
        n = 2
        out = self.decoder.decode(_servingbot_example(n=n))
        self.assertEqual(
            tuple(tf.shape(out['groundtruth_polygons']).numpy()), (n, 10940),
        )

    def test_dontcare_zeros_when_absent(self):
        """servingbot_polygon has no is_dontcare; decoder defaults to zeros."""
        n = 3
        out = self.decoder.decode(_servingbot_example(n=n))
        self.assertTrue(tf.reduce_all(out['groundtruth_dontcare'] == 0).numpy())

    def test_output_schema_matches_polygon_decoder(self):
        """Both decoders must produce the exact same set of output keys."""
        from data_pipeline.tfds_decoders import PolygonDecoder
        poly_keys = set(PolygonDecoder().decode(_cleaner_polygon_example()).keys())
        sb_keys = set(self.decoder.decode(_servingbot_example()).keys())
        self.assertEqual(poly_keys, sb_keys,
                         "Schema mismatch prevents zip+concat of the two streams")


class TestCopyPasteDecoder(unittest.TestCase):

    def setUp(self):
        from data_pipeline.tfds_decoders import CopyPasteDecoder
        self.decoder = CopyPasteDecoder(num_classes=39)

    def test_output_keys(self):
        out = self.decoder.decode(_copy_paste_example())
        self.assertEqual(
            set(out.keys()),
            {'image', 'image/id', 'orig_bbox', 'label', 'points', 'obj_id'},
        )

    def test_image_is_rgba(self):
        """cleaner_copy_paste images have 4 channels (RGBA alpha mask)."""
        out = self.decoder.decode(_copy_paste_example(h=256, w=256))
        self.assertEqual(out['image'].dtype, tf.uint8)
        self.assertEqual(tuple(tf.shape(out['image']).numpy()), (256, 256, 4))

    def test_flat_schema_fields(self):
        """All fields are scalars or 1-D — no objects sub-dict."""
        out = self.decoder.decode(_copy_paste_example())
        self.assertEqual(out['label'].shape.rank, 0)
        self.assertEqual(out['obj_id'].shape.rank, 0)
        self.assertEqual(tuple(tf.shape(out['orig_bbox']).numpy()), (4,))
        self.assertEqual(tuple(tf.shape(out['points']).numpy()), (3972,))

    def test_image_id_value(self):
        out = self.decoder.decode(_copy_paste_example())
        self.assertEqual(out['image/id'].numpy(), 789)


if __name__ == '__main__':
    unittest.main()
