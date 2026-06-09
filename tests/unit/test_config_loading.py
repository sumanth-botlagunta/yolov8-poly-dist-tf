"""Config round-trip tests.

Validates that every experiment YAML loads cleanly onto the dataclass tree, that
derived fields are computed, and that the ``base:`` include + deep-merge works.
"""

import glob
import os
import unittest

from configs.yaml_loader import load_config, _deep_merge

_EXP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "experiments", "yolo"
)


class TestConfigLoading(unittest.TestCase):
    def test_all_experiment_configs_load(self):
        """Every shipped experiment YAML maps onto ExperimentConfig without error."""
        paths = sorted(glob.glob(os.path.join(_EXP_DIR, "*.yaml")))
        self.assertGreater(len(paths), 0, "no experiment configs found")
        for path in paths:
            with self.subTest(config=os.path.basename(path)):
                cfg = load_config(path)
                self.assertGreater(cfg.task.num_classes, 0)
                self.assertEqual(len(cfg.task.model.input_size), 3)

    def test_derived_fields_filled(self):
        cfg = load_config(os.path.join(_EXP_DIR, "yolov8_poly_dist.yaml"))
        # train_total_examples=305780, batch=128 → steps_per_loop=2388; ×300 epochs.
        self.assertEqual(cfg.trainer.steps_per_loop, 305780 // 128)
        self.assertEqual(
            cfg.trainer.train_steps, cfg.trainer.steps_per_loop * cfg.trainer.train_epochs
        )

    def test_base_include_overrides_and_inherits(self):
        base = load_config(os.path.join(_EXP_DIR, "yolov8_poly_dist.yaml"))
        bf16 = load_config(os.path.join(_EXP_DIR, "yolov8_poly_dist_bf16.yaml"))
        # Overridden runtime keys
        self.assertEqual(bf16.runtime.mixed_precision_dtype, "bfloat16")
        self.assertTrue(bf16.runtime.enable_xla)
        # Everything else inherited from the base
        self.assertEqual(bf16.task.num_classes, base.task.num_classes)
        self.assertEqual(bf16.task.losses.iou_gain, base.task.losses.iou_gain)
        self.assertEqual(bf16.trainer.train_epochs, base.trainer.train_epochs)

    def test_detection_generator_score_thresh_and_distance_wired(self):
        """score_thresh and the task distance range must reach the generator
        config (regression: both were previously dropped by the loader)."""
        from configs.yaml_loader import _build_model_config

        m = {"detection_generator": {"score_thresh": 0.3}}
        task = {"min_distance": 1.5, "max_distance": 22.0}
        mc = _build_model_config(m, task)
        self.assertAlmostEqual(mc.detection_generator.score_thresh, 0.3)
        self.assertAlmostEqual(mc.detection_generator.min_distance, 1.5)
        self.assertAlmostEqual(mc.detection_generator.max_distance, 22.0)

    def test_deep_merge_is_recursive(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 20, "z": 30}, "c": 4}
        merged = _deep_merge(base, override)
        self.assertEqual(merged, {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4})


if __name__ == "__main__":
    unittest.main()
