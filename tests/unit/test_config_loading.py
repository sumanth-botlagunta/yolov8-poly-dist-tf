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
        # steps_per_loop derives from whatever the YAML declares (271,166 verified
        # against the actual TFDS builders 2026-06-10; batch 128 → 2118).
        self.assertEqual(
            cfg.trainer.steps_per_loop,
            cfg.trainer.train_total_examples // cfg.task.train_data.global_batch_size,
        )
        self.assertEqual(cfg.trainer.steps_per_loop, 2118)
        self.assertEqual(
            cfg.trainer.train_steps, cfg.trainer.steps_per_loop * cfg.trainer.train_epochs
        )
        # The cosine schedule must span exactly the training run (run_train warns
        # otherwise — this pins the shipped YAML to consistency).
        self.assertEqual(
            cfg.trainer.optimizer_config.learning_rate.decay_steps,
            cfg.trainer.train_steps,
        )

    def test_validation_batch_size_consistent_across_tiers(self):
        """All three tiers must use the same validation batch size.

        bbox/poly previously shipped validation_data.global_batch_size=2 while
        poly_dist used 64, so the lower tiers ran ~32x more validation forward
        passes per epoch over the same val set — a large, silent per-epoch tax.
        """
        sizes = {
            tier: load_config(
                os.path.join(_EXP_DIR, f"yolov8_{tier}.yaml")
            ).task.validation_data.global_batch_size
            for tier in ("bbox", "poly", "poly_dist")
        }
        self.assertEqual(
            sizes["bbox"], sizes["poly_dist"],
            f"bbox val batch {sizes['bbox']} != poly_dist {sizes['poly_dist']}",
        )
        self.assertEqual(
            sizes["poly"], sizes["poly_dist"],
            f"poly val batch {sizes['poly']} != poly_dist {sizes['poly_dist']}",
        )
        # Pin the agreed value so a regression to 2 is caught.
        self.assertEqual(sizes["poly_dist"], 64)

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
