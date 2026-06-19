"""Pinning test: offline tools apply the trainer's mixed-precision policy.

eval / export / continuous_eval historically built the model without setting the
global Keras precision policy, so a bfloat16-trained checkpoint computed in
float32 (a different numerical path than training/serving). tools/runtime_setup
centralizes the policy application; this pins that:
  * a bfloat16 runtime config activates the mixed_bfloat16 global policy;
  * float32 leaves the default policy;
  * the helper restores nothing destructive (tests reset policy afterward).
"""

import types
import unittest

import tensorflow as tf

from tools.shared.runtime_setup import apply_eval_precision_policy


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

    def test_live_config_yaml_is_bfloat16(self):
        # The live training config trains in bfloat16; eval must match it.
        from configs.yaml_loader import load_config
        cfg = load_config("configs/experiments/yolo/yolov8_poly_dist.yaml")
        applied = apply_eval_precision_policy(cfg)
        self.assertEqual(applied, "bfloat16")


if __name__ == "__main__":
    unittest.main()
