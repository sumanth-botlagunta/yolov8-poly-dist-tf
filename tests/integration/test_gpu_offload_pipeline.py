"""Integration test: the full GPU-offload data path (no TFDS, no model).

Wires the real stages the offload pipeline uses —
    padded_batch(group) → mosaic_gpu.mosaic_prepare_fn → unbatch
    → V8ParserExtended(defer_warp=True).parse_fn → batch → gpu_mosaic_warp
— on a synthetic in-memory group, and asserts the payload threads through cleanly:
the parser passes canvas/warp/flip through, builds the padded PolyYOLO labels, and
the deferred GPU warp turns the canvas batch into the [B, H, W, 3] image batch the
model would consume. This is the local stand-in for the TFDS-backed
``tools/pipeline/bench_mosaic_pipeline.py`` (which needs the real datasets + a GPU).
"""

import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.mosaic import Mosaic
from data_pipeline.mosaic_gpu import (
    CANVAS_KEY,
    FLIP_KEY,
    WARP_KEY,
    gpu_mosaic_warp,
    mosaic_prepare_fn,
)
from data_pipeline.yolo_parser import V8ParserExtended


_H = _W = 64
_G = 8
_R = 4


def _group(G, h, w, n=2):
    box = tf.constant([[0.25, 0.25, 0.75, 0.75], [0.1, 0.1, 0.4, 0.4]][:n], tf.float32)
    poly = tf.constant(
        [[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0],
         [0.15, 0.15, 0.35, 0.35, -1.0, -1.0, -1.0, -1.0]][:n], tf.float32)
    images = tf.stack([tf.fill([h, w, 3], tf.constant(20 + 25 * i, tf.uint8)) for i in range(G)])
    return {
        "image":  images,
        "height": tf.constant([h] * G, tf.int32),
        "width":  tf.constant([w] * G, tf.int32),
        "groundtruth_boxes":    tf.stack([box] * G),
        "groundtruth_classes":  tf.zeros([G, n], tf.int64),
        "groundtruth_is_crowd": tf.zeros([G, n], tf.bool),
        "groundtruth_area":     tf.ones([G, n], tf.float32),
        "groundtruth_dontcare": tf.zeros([G, n], tf.int64),
        "groundtruth_dists":    tf.fill([G, n], tf.constant(-1.0)),
        "groundtruth_polygons": tf.stack([poly] * G),
        "source_id":            tf.constant([str(i) for i in range(G)]),
    }


def _mosaic():
    return Mosaic(
        output_size=[_H, _W], mosaic_frequency=0.5, with_polygons=True,
        aug_scale_min=0.6, aug_scale_max=1.4, degrees=8.0, shear=2.0,
        perspective=0.0, translate=0.1, mosaic_center=0.25, area_thresh=0.1,
        group_size=_G, decodes_per_output=_R,
    )


def _parser():
    return V8ParserExtended(
        output_size=[_H, _W],
        expanded_strides={"3": 8, "4": 16, "5": 32},
        levels=["3", "4", "5"],
        max_vertices=10938,
        angle_step=15,
        with_polygons=True,
        max_num_instances=16,
        area_thresh=0.1,
        defer_warp=True,
    )


class TestGpuOffloadPipeline(unittest.TestCase):

    def test_full_offload_path_shapes(self):
        bs = 2
        ds = (
            tf.data.Dataset.from_tensor_slices(_group(_G, _H, _W))
            .padded_batch(_G, drop_remainder=True)
            .map(mosaic_prepare_fn(_mosaic(), random_flip=True))
            .unbatch()
            .map(_parser().parse_fn(is_training=True))
            .batch(bs)
        )
        images, labels = next(iter(ds))

        # The parser passes the canvas through as the "image"; warp/flip ride in labels.
        self.assertEqual(tuple(images.shape), (bs, 2 * _H, 2 * _W, 3))
        self.assertEqual(images.dtype, tf.uint8)
        self.assertIn("mosaic_warp", labels)
        self.assertIn("mosaic_flip", labels)
        self.assertEqual(tuple(labels["mosaic_warp"].shape), (bs, 8))
        self.assertEqual(tuple(labels["mosaic_flip"].shape), (bs,))

        # Padded PolyYOLO labels (max_num_instances=16, angle_step=15 → 72).
        self.assertEqual(tuple(labels["bbox"].shape), (bs, 16, 4))
        self.assertEqual(tuple(labels["polygons"].shape), (bs, 16, 72))
        self.assertEqual(tuple(labels["classes"].shape), (bs, 16))

        # The deferred GPU warp turns the canvas batch into the model input batch.
        warped = gpu_mosaic_warp(
            images, labels["mosaic_warp"], labels["mosaic_flip"], _H, _W
        )
        self.assertEqual(tuple(warped.shape), (bs, _H, _W, 3))
        self.assertEqual(warped.dtype, tf.uint8)
        # Not all-gray (real content warped in).
        self.assertGreater(len(np.unique(warped.numpy())), 1)

    def test_defer_parser_skips_image_geometry(self):
        """defer_warp parser must NOT resize the canvas (it passes it through)."""
        out = (
            tf.data.Dataset.from_tensor_slices(_group(_G, _H, _W))
            .padded_batch(_G, drop_remainder=True)
            .map(mosaic_prepare_fn(_mosaic(), random_flip=False))
            .unbatch()
            .map(_parser().parse_fn(is_training=True))
        )
        img, labels = next(iter(out))
        # Image kept at canvas size 2H×2W (NOT resized to H×W) — proves the parser
        # deferred the geometry instead of running its standard resize/flip path.
        self.assertEqual(tuple(img.shape), (2 * _H, 2 * _W, 3))
        # random_flip=False → every flip coin is False.
        self.assertFalse(bool(labels["mosaic_flip"].numpy()))


if __name__ == "__main__":
    unittest.main()
