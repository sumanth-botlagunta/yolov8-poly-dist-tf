"""bfloat16 mixed-precision sanity: model builds, runs, and heads stay float32.

The experiment config now trains under the Keras ``mixed_bfloat16`` policy.
bfloat16 needs no loss scaling, but two invariants must hold:
  - the prediction heads are pinned to float32 (models/head.py) so the loss
    receives float32 logits without explicit casts;
  - a forward pass produces finite values.
"""

import unittest

import numpy as np
import tensorflow as tf

from configs.model_config import ModelConfig
from models.yolo_v8 import build_yolov8


class TestBf16Policy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        cfg = ModelConfig()  # all 6 heads
        cls.model = build_yolov8(cfg)
        cls.model.build_and_init()
        cls.model.deploy = False

    @classmethod
    def tearDownClass(cls):
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_raw_head_outputs_are_float32_and_finite(self):
        x = tf.random.uniform([1, 672, 672, 3], dtype=tf.float32)
        out = self.model(x, training=True)
        self.assertEqual(
            set(out.keys()),
            {"box", "cls", "poly_angle", "poly_dist", "poly_conf", "dist"},
        )
        for branch, levels in out.items():
            for level, t in levels.items():
                self.assertEqual(
                    t.dtype, tf.float32,
                    f"{branch}/{level} must be float32 under mixed_bfloat16 "
                    "(heads are pinned so the loss needs no casts)",
                )
                self.assertTrue(
                    bool(np.isfinite(t.numpy()).all()),
                    f"{branch}/{level} produced non-finite values",
                )


if __name__ == "__main__":
    unittest.main()
