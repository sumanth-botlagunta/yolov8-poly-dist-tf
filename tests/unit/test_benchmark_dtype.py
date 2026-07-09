"""Pinning test: the pipeline benchmark validates image dtype (uint8).

The training pipeline emits uint8 images (normalization is the model's job). An
upstream float32 cast would 4x the measured memory/workload without any signal,
so _run_benchmark asserts uint8 and surfaces the observed dtype/shape in its
stats.
"""

import unittest

import numpy as np
import tensorflow as tf

from utils.pipeline.benchmark_pipeline import _run_benchmark, _images_of


def _ds(dtype):
    imgs = tf.zeros([2, 8, 8, 3], dtype=dtype)
    lbls = {"x": tf.zeros([2])}
    return tf.data.Dataset.from_tensors((imgs, lbls)).repeat(10)


class TestBenchmarkDtype(unittest.TestCase):
    def test_uint8_passes_and_reports_dtype(self):
        stats = _run_benchmark(_ds(tf.uint8), n_steps=3)
        self.assertEqual(stats["image_dtype"], "uint8")
        self.assertEqual(stats["batch_size"], 2)

    def test_float32_images_raise(self):
        with self.assertRaises(AssertionError):
            _run_benchmark(_ds(tf.float32), n_steps=3)

    def test_images_of_handles_tuple_and_dict(self):
        t = (tf.zeros([1, 4, 4, 3], tf.uint8), {})
        self.assertEqual(_images_of(t).dtype, tf.uint8)
        d = {"image": tf.zeros([1, 4, 4, 3], tf.uint8)}
        self.assertEqual(_images_of(d).dtype, tf.uint8)


if __name__ == "__main__":
    unittest.main()
