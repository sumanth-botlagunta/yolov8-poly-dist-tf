"""Tests for V8ParserExtended and V8DistanceParser output formats.

Validates:
    - Parsed image is float32 normalized to [0, 1] at the configured output size.
    - labels['bbox'] is padded to [max_num_instances, 4]; polygons to [*, 72].
    - labels['n_gt'] matches the number of valid ground-truth boxes.
    - Distance parser sets ignore_bg=1 and log-encodes distances (invalid → -10.0).
"""

import math
import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.yolo_parser import V8ParserExtended
from data_pipeline.distance_parser import V8DistanceParser

_OUT = [64, 64]
_STRIDES = {"3": 8, "4": 16, "5": 32}


def _det_data(n=2, h=80, w=80):
    return {
        "image": tf.constant(np.full((h, w, 3), 100, np.uint8)),
        "groundtruth_boxes": tf.constant(
            [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]][:n], dtype=tf.float32
        ),
        "groundtruth_classes":  tf.constant([2, 5][:n], dtype=tf.int64),
        "groundtruth_polygons": tf.fill([n, 8], -1.0),
        "groundtruth_is_crowd": tf.constant([False] * n),
    }


def _det_parser():
    return V8ParserExtended(
        output_size=_OUT, expanded_strides=_STRIDES, levels=["3", "4", "5"],
        angle_step=15, random_flip=False, albumentations_frequency=0.0,
        aug_rand_hue=0.0, aug_rand_saturation=0.0, aug_rand_brightness=0.0,
        aug_rand_translate=0.0, aug_scale_min=1.0, aug_scale_max=1.0,
        area_thresh=0.0, max_num_instances=300,
    )


class TestV8ParserExtended(unittest.TestCase):
    def setUp(self):
        self.image, self.labels = _det_parser()._parse_train_data(_det_data(n=2))

    def test_image_range(self):
        img = self.image.numpy()
        self.assertEqual(self.image.dtype, tf.float32)
        self.assertEqual(img.shape, (64, 64, 3))
        self.assertGreaterEqual(img.min(), 0.0)
        self.assertLessEqual(img.max(), 1.0)

    def test_label_shapes(self):
        self.assertEqual(tuple(self.labels["bbox"].shape), (300, 4))
        self.assertEqual(tuple(self.labels["classes"].shape), (300,))
        self.assertEqual(tuple(self.labels["polygons"].shape), (300, 72))

    def test_n_gt_correct(self):
        # Both boxes are valid and in-range (area_thresh=0, no flip/affine).
        self.assertEqual(int(self.labels["n_gt"]), 2)


def _dist_data(dists, h=80, w=80):
    n = len(dists)
    return {
        "image": tf.constant(np.full((h, w, 3), 100, np.uint8)),
        "groundtruth_boxes": tf.constant([[0.1, 0.1, 0.3, 0.3]] * n, dtype=tf.float32),
        "groundtruth_classes":  tf.zeros([n], tf.int64),
        "groundtruth_is_crowd": tf.constant([False] * n),
        "groundtruth_dists":    tf.constant(dists, dtype=tf.float32),
    }


class TestV8DistanceParser(unittest.TestCase):
    def _parser(self):
        return V8DistanceParser(
            output_size=_OUT, angle_step=15, random_flip=False,
            aug_rand_hue=0.0, aug_rand_saturation=0.0, aug_rand_brightness=0.0,
            min_meter=0.5, max_meter=10.0, max_num_instances=300,
        )

    def test_ignore_bg_set(self):
        _, labels = self._parser()._parse_train_data(_dist_data([1.5]))
        self.assertEqual(int(labels["ignore_bg"]), 1)

    def test_log_distance_encoding(self):
        # Valid distance → log(d); invalid (<0) → sentinel -10.0.
        _, labels = self._parser()._parse_train_data(_dist_data([2.0, -1.0]))
        log_dist = labels["log_distance"].numpy()
        self.assertAlmostEqual(log_dist[0], math.log(2.0), places=4)
        self.assertAlmostEqual(log_dist[1], -10.0, places=5)


if __name__ == "__main__":
    unittest.main()
