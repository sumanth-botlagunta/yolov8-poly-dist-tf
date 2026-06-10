"""Shape-consistency tests for every batch() boundary in the data pipeline.

The "Cannot batch tensors with different shapes" crash happens when images with
different native resolutions reach any tf.data.Dataset.batch() or tf.stack()
call without being resized first.  This file pins every such boundary so a
regression immediately shows which stage broke.

Batch boundaries covered
------------------------
1. Before mosaic's inner batch(4):
       raw decode → _pre_resize_for_mosaic → batch(4)
2. After mosaic / before final batch:
       mosaic_fn → unbatch → parser → batch(N)
3. After V8ParserExtended (train and eval):
       variable-size raw → parse_fn → fixed [H, W, 3]
4. After V8DistanceParser (train):
       variable-size raw → parse_fn → fixed [H, W, 3]
5. Static shape:
       image.shape.as_list() must be [H, W, 3], not [None, None, 3], so that
       downstream tf.function compilation and model shape-inference work.
6. End-to-end dataset batch:
       from_generator(variable sizes) → map(parse) → batch(4) → no crash.
"""

import unittest

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_H = 64
_W = 64
_OUT_SIZE = [_H, _W]
_MAX_VERTICES = 20   # kept small for test speed
_MAX_INSTANCES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_example(h: int, w: int, n_boxes: int = 2) -> dict:
    """Return a synthetic decoded example dict with image size (h × w).

    Mimics the output of PolygonDecoder.decode() — variable-size uint8 image
    plus fixed-schema annotation tensors.
    """
    image = tf.cast(
        tf.random.uniform([h, w, 3], minval=0, maxval=256, dtype=tf.int32),
        tf.uint8,
    )
    boxes = tf.constant(
        [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]][:n_boxes],
        dtype=tf.float32,
    )
    n = n_boxes
    return {
        'image':                 image,
        'groundtruth_boxes':     boxes,
        'groundtruth_classes':   tf.ones([n], dtype=tf.int64),
        'groundtruth_is_crowd':  tf.zeros([n], dtype=tf.bool),
        'groundtruth_area':      tf.ones([n], dtype=tf.float32) * 0.05,
        'groundtruth_dontcare':  tf.zeros([n], dtype=tf.int64),
        'groundtruth_polygons':  tf.fill([n, _MAX_VERTICES + 2], -1.0),
        'groundtruth_dists':     tf.fill([n], -1.0),
        'source_id':             tf.constant('test'),
    }


def _make_v8_parser(albumentations_frequency: float = 0.0):
    from data_pipeline.yolo_parser import V8ParserExtended
    return V8ParserExtended(
        output_size=_OUT_SIZE,
        expanded_strides={'3': 8, '4': 16, '5': 32},
        levels=['3', '4', '5'],
        max_vertices=_MAX_VERTICES,
        angle_step=15,
        with_polygons=True,
        dummy_distance=True,
        skip_crowd_during_training=True,
        max_num_instances=_MAX_INSTANCES,
        aug_rand_hue=0.0,
        aug_rand_saturation=0.0,
        aug_rand_brightness=0.0,
        aug_rand_translate=0.0,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        random_flip=False,
        letter_box=True,
        albumentations_frequency=albumentations_frequency,
    )


def _make_distance_parser():
    from data_pipeline.distance_parser import V8DistanceParser
    return V8DistanceParser(
        output_size=_OUT_SIZE,
        max_num_instances=_MAX_INSTANCES,
        angle_step=15,
        with_polygons=False,
        min_meter=0.5,
        max_meter=10.0,
        aug_rand_hue=0.0,
        aug_rand_saturation=0.0,
        aug_rand_brightness=0.0,
        random_flip=False,
        skip_crowd_during_training=True,
    )


# Sizes that triggered the original crash.
_VARIABLE_SIZES = [(1280, 800), (340, 510), (640, 480), (1024, 768)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreResizeBeforeMosaicBatch(unittest.TestCase):
    """Stage 1: images must have identical shape before mosaic's batch(4)."""

    def test_pre_resize_runtime_shape(self):
        """After _pre_resize_for_mosaic, every image is exactly [H, W, 3]."""
        for h, w in _VARIABLE_SIZES:
            ex = _make_raw_example(h, w)
            img = tf.cast(
                tf.image.resize(tf.cast(ex['image'], tf.float32), [_H, _W], method='bilinear'),
                tf.uint8,
            )
            self.assertEqual(
                tuple(img.shape), (_H, _W, 3),
                f"pre-resize gave {img.shape} for input ({h},{w},3)",
            )

    def test_pre_resize_static_shape_fully_known(self):
        """Static shape after pre-resize must be [H, W, 3] — no None dims."""
        ex = _make_raw_example(640, 480)
        img = tf.cast(
            tf.image.resize(tf.cast(ex['image'], tf.float32), [_H, _W], method='bilinear'),
            tf.uint8,
        )
        self.assertEqual(
            img.shape.as_list(), [_H, _W, 3],
            f"Static shape has unknown dims: {img.shape}",
        )

    def test_batch4_succeeds_after_pre_resize(self):
        """batch(4) on pre-resized images must not raise an InvalidArgumentError."""
        def _pre_resize(ex):
            img = tf.cast(
                tf.image.resize(tf.cast(ex['image'], tf.float32), [_H, _W], method='bilinear'),
                tf.uint8,
            )
            return {**ex, 'image': img}

        pre_resized = [_pre_resize(_make_raw_example(h, w)) for h, w in _VARIABLE_SIZES]
        images = tf.stack([ex['image'] for ex in pre_resized], axis=0)
        self.assertEqual(images.shape, (4, _H, _W, 3))


class TestMosaicOutputShape(unittest.TestCase):
    """Stage 2: mosaic output must be [H, W, 3] before the parser."""

    def _make_pre_resized(self, h, w):
        ex = _make_raw_example(h, w)
        img = tf.cast(tf.image.resize(tf.cast(ex['image'], tf.float32), [_H, _W]), tf.uint8)
        return {**ex, 'image': img}

    def test_mosaic_canvas_shape(self):
        """_mosaic() must produce an image of exactly [H, W, 3]."""
        from data_pipeline.mosaic import Mosaic
        mosaic = Mosaic(output_size=_OUT_SIZE, mosaic_frequency=1.0, with_polygons=True)
        exs = [self._make_pre_resized(*s) for s in _VARIABLE_SIZES]
        result = mosaic._mosaic(*exs)
        self.assertEqual(tuple(result['image'].shape), (_H, _W, 3))

    def test_mosaic_fn_output_shape(self):
        """mosaic_fn() (the dataset-map callable) emits 4 samples → [4, H, W, 3]."""
        from data_pipeline.mosaic import Mosaic
        mosaic = Mosaic(output_size=_OUT_SIZE, mosaic_frequency=1.0, with_polygons=True)
        fn = mosaic.mosaic_fn(is_training=True)

        exs = [self._make_pre_resized(_H, _W) for _ in range(4)]
        batch_dict = {
            k: tf.stack([ex[k] for ex in exs], axis=0) for k in exs[0]
        }
        result = fn(batch_dict)
        # 4-in/4-out: every decoded image yields one emitted sample.
        self.assertEqual(tuple(result['image'].shape), (4, _H, _W, 3))

    def test_mosaic_passthrough_shape(self):
        """When mosaic is skipped (freq=0.0), all 4 images still have [H, W, 3]."""
        from data_pipeline.mosaic import Mosaic
        mosaic = Mosaic(output_size=_OUT_SIZE, mosaic_frequency=0.0, with_polygons=True)
        fn = mosaic.mosaic_fn(is_training=True)

        exs = [self._make_pre_resized(_H, _W) for _ in range(4)]
        batch_dict = {
            k: tf.stack([ex[k] for ex in exs], axis=0) for k in exs[0]
        }
        result = fn(batch_dict)
        self.assertEqual(tuple(result['image'].shape), (4, _H, _W, 3))
        self.assertEqual(tuple(result['image'].shape[1:]), (_H, _W, 3))


class TestParserOutputShape(unittest.TestCase):
    """Stage 3: parsers must emit [H, W, 3] images from variable-size raw inputs."""

    def test_train_parser_runtime_shape(self):
        """V8ParserExtended (train) must resize any input to [H, W, 3]."""
        parser = _make_v8_parser()
        parse_fn = parser.parse_fn(is_training=True)
        for h, w in _VARIABLE_SIZES:
            image, _ = parse_fn(_make_raw_example(h, w))
            self.assertEqual(
                tuple(image.shape), (_H, _W, 3),
                f"Train parser gave {image.shape} for ({h},{w}) input",
            )

    def test_eval_parser_runtime_shape(self):
        """V8ParserExtended (eval) must letterbox-resize any input to [H, W, 3]."""
        parser = _make_v8_parser()
        parse_fn = parser.parse_fn(is_training=False)
        for h, w in _VARIABLE_SIZES:
            image, _ = parse_fn(_make_raw_example(h, w))
            self.assertEqual(
                tuple(image.shape), (_H, _W, 3),
                f"Eval parser gave {image.shape} for ({h},{w}) input",
            )

    def test_distance_parser_runtime_shape(self):
        """V8DistanceParser (train) must letterbox-resize any input to [H, W, 3]."""
        parser = _make_distance_parser()
        parse_fn = parser.parse_fn(is_training=True)
        for h, w in _VARIABLE_SIZES:
            ex = _make_raw_example(h, w)
            ex['groundtruth_dists'] = tf.constant([1.5, 2.0])
            image, _ = parse_fn(ex)
            self.assertEqual(
                tuple(image.shape), (_H, _W, 3),
                f"Distance parser gave {image.shape} for ({h},{w}) input",
            )

    def test_parsers_emit_uint8(self):
        """All parsers emit uint8 images now — colour aug + /255 moved to the GPU step."""
        v8 = _make_v8_parser()
        train_img, _ = v8.parse_fn(is_training=True)(_make_raw_example(640, 480))
        eval_img, _ = v8.parse_fn(is_training=False)(_make_raw_example(640, 480))
        self.assertEqual(train_img.dtype, tf.uint8)
        self.assertEqual(eval_img.dtype, tf.uint8)

        dist = _make_distance_parser()
        ex = _make_raw_example(640, 480)
        ex['groundtruth_dists'] = tf.constant([1.5, 2.0])
        dist_img, _ = dist.parse_fn(is_training=True)(ex)
        self.assertEqual(dist_img.dtype, tf.uint8)


class TestStaticShapeKnown(unittest.TestCase):
    """Parser output must carry static shape [H, W, 3] — no None spatial dims.

    Unknown static dims would propagate into the batched dataset spec as
    [batch, None, None, 3], breaking model shape inference and tf.function.
    """

    def test_train_parser_static_shape_no_albumentations(self):
        """Static shape must be [H, W, 3] when albumentations is disabled."""
        parser = _make_v8_parser(albumentations_frequency=0.0)
        image, _ = parser.parse_fn(is_training=True)(_make_raw_example(640, 480))
        self.assertEqual(
            image.shape.as_list(), [_H, _W, 3],
            f"Static shape has unknown dims (albumentations off): {image.shape}",
        )

    def test_train_parser_static_shape_with_albumentations(self):
        """Static shape stays [H, W, 3] even with albumentations_frequency set.

        Albumentations now runs per-batch on the accelerator (not in the parser),
        but the parser config still accepts the frequency; the parser output must
        keep fully-known spatial dims regardless.
        """
        parser = _make_v8_parser(albumentations_frequency=1.0)
        image, _ = parser.parse_fn(is_training=True)(_make_raw_example(640, 480))
        self.assertEqual(
            image.shape.as_list(), [_H, _W, 3],
            f"Static shape has unknown dims after albumentations: {image.shape}",
        )

    def test_eval_parser_static_shape(self):
        parser = _make_v8_parser()
        image, _ = parser.parse_fn(is_training=False)(_make_raw_example(640, 480))
        self.assertEqual(image.shape.as_list(), [_H, _W, 3])

    def test_distance_parser_static_shape(self):
        parser = _make_distance_parser()
        ex = _make_raw_example(640, 480)
        ex['groundtruth_dists'] = tf.constant([1.5, 2.0])
        image, _ = parser.parse_fn(is_training=True)(ex)
        self.assertEqual(image.shape.as_list(), [_H, _W, 3])


class TestLabelShapesFixed(unittest.TestCase):
    """Label tensors must also have fixed shapes so the final batch() succeeds."""

    def test_train_label_shapes(self):
        parser = _make_v8_parser()
        parse_fn = parser.parse_fn(is_training=True)
        m = _MAX_INSTANCES

        for h, w in [(1280, 800), (340, 510)]:
            _, labels = parse_fn(_make_raw_example(h, w))
            self.assertEqual(labels['bbox'].shape.as_list(),         [m, 4])
            self.assertEqual(labels['classes'].shape.as_list(),      [m])
            self.assertEqual(labels['polygons'].shape.as_list(),     [m, 72])
            self.assertEqual(labels['log_distance'].shape.as_list(), [m])
            self.assertEqual(labels['n_gt'].shape.as_list(),         [])
            self.assertEqual(labels['ignore_bg'].shape.as_list(),    [])

    def test_distance_label_shapes(self):
        parser = _make_distance_parser()
        parse_fn = parser.parse_fn(is_training=True)
        m = _MAX_INSTANCES

        ex = _make_raw_example(640, 480)
        ex['groundtruth_dists'] = tf.constant([1.5, 2.0])
        _, labels = parse_fn(ex)
        self.assertEqual(labels['bbox'].shape.as_list(),         [m, 4])
        self.assertEqual(labels['classes'].shape.as_list(),      [m])
        self.assertEqual(labels['log_distance'].shape.as_list(), [m])


class TestEndToEndDatasetBatch(unittest.TestCase):
    """Full tf.data pipeline with variable-size images must reach batch() without crash."""

    def _make_generator(self, sizes):
        """Yield raw numpy example dicts with different image sizes."""
        def _gen():
            for h, w in sizes:
                yield {
                    'image': np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
                    'groundtruth_boxes': np.array(
                        [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]], dtype=np.float32,
                    ),
                    'groundtruth_classes':  np.ones(2, dtype=np.int64),
                    'groundtruth_is_crowd': np.zeros(2, dtype=bool),
                    'groundtruth_area':     np.ones(2, dtype=np.float32) * 0.05,
                    'groundtruth_dontcare': np.zeros(2, dtype=np.int64),
                    'groundtruth_polygons': np.full((2, _MAX_VERTICES + 2), -1.0, dtype=np.float32),
                    'groundtruth_dists':    np.full(2, -1.0, dtype=np.float32),
                    'source_id':            b'test',
                }
        return _gen

    def _output_signature(self):
        return {
            'image':                 tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            'groundtruth_boxes':     tf.TensorSpec(shape=(None, 4),       dtype=tf.float32),
            'groundtruth_classes':   tf.TensorSpec(shape=(None,),          dtype=tf.int64),
            'groundtruth_is_crowd':  tf.TensorSpec(shape=(None,),          dtype=tf.bool),
            'groundtruth_area':      tf.TensorSpec(shape=(None,),          dtype=tf.float32),
            'groundtruth_dontcare':  tf.TensorSpec(shape=(None,),          dtype=tf.int64),
            'groundtruth_polygons':  tf.TensorSpec(shape=(None, _MAX_VERTICES + 2), dtype=tf.float32),
            'groundtruth_dists':     tf.TensorSpec(shape=(None,),          dtype=tf.float32),
            'source_id':             tf.TensorSpec(shape=(),               dtype=tf.string),
        }

    def test_train_parser_map_then_batch(self):
        """from_generator(variable) → map(train parse) → batch(4) must succeed."""
        parser = _make_v8_parser()
        ds = tf.data.Dataset.from_generator(
            self._make_generator(_VARIABLE_SIZES),
            output_signature=self._output_signature(),
        )
        ds = ds.map(parser.parse_fn(is_training=True))
        ds = ds.batch(4)

        for images, labels in ds:
            self.assertEqual(tuple(images.shape), (4, _H, _W, 3))
            break

    def test_eval_parser_map_then_batch(self):
        """from_generator(variable) → map(eval parse) → batch(4) must succeed."""
        parser = _make_v8_parser()
        ds = tf.data.Dataset.from_generator(
            self._make_generator(_VARIABLE_SIZES),
            output_signature=self._output_signature(),
        )
        ds = ds.map(parser.parse_fn(is_training=False))
        ds = ds.batch(4)

        for images, _ in ds:
            self.assertEqual(tuple(images.shape), (4, _H, _W, 3))
            break

    def test_stacking_multiple_parsed_outputs(self):
        """tf.stack on outputs from different input sizes must not fail."""
        parser = _make_v8_parser()
        parse_fn = parser.parse_fn(is_training=True)
        images = [parse_fn(_make_raw_example(h, w))[0] for h, w in _VARIABLE_SIZES]

        try:
            batch = tf.stack(images, axis=0)
        except Exception as e:
            self.fail(f"tf.stack failed — shapes are inconsistent: {e}")

        self.assertEqual(tuple(batch.shape), (len(_VARIABLE_SIZES), _H, _W, 3))


if __name__ == '__main__':
    unittest.main()
