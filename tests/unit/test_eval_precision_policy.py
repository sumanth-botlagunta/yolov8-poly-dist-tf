"""Pinning test: offline tools apply the trainer's mixed-precision policy.

A bfloat16-trained checkpoint evaluated under a float32 global policy computes a
different numerical path than training/serving, so the offline tools must apply
the trainer's policy. common/runtime_setup centralizes this; the tests pin that:
  * a bfloat16 runtime config activates the mixed_bfloat16 global policy;
  * float32 leaves the default policy;
  * the helper restores nothing destructive (tests reset policy afterward).
"""

import types
import unittest

import tensorflow as tf

from common.runtime_setup import apply_eval_precision_policy


def _cfg(dtype):
    return types.SimpleNamespace(
        runtime=types.SimpleNamespace(mixed_precision_dtype=dtype)
    )


class TestEvalPrecisionPolicy(unittest.TestCase):
    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_bfloat16_activates_policy(self):
        applied = apply_eval_precision_policy(_cfg("bfloat16"))
        self.assertEqual(applied, "bfloat16")
        self.assertEqual(
            tf.keras.mixed_precision.global_policy().name, "mixed_bfloat16"
        )

    def test_float32_leaves_default(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        applied = apply_eval_precision_policy(_cfg("float32"))
        self.assertEqual(applied, "float32")
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, "float32")

    def test_none_dtype_defaults_to_float32(self):
        applied = apply_eval_precision_policy(_cfg(None))
        self.assertEqual(applied, "float32")

    def test_live_config_precision_reaches_eval(self):
        # Eval must apply whatever precision the config trains with: float32
        # for the base tier, bfloat16 for the _bf16 overlay.
        from configs.yaml_loader import load_config
        base = load_config("configs/experiments/yolo/yolov8_poly_dist.yaml")
        self.assertEqual(apply_eval_precision_policy(base), "float32")
        bf16 = load_config("configs/experiments/yolo/yolov8_poly_dist_bf16.yaml")
        self.assertEqual(apply_eval_precision_policy(bf16), "bfloat16")


if __name__ == "__main__":
    unittest.main()
