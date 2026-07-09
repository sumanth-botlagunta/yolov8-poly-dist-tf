"""Unknown-key warning behavior of configs.yaml_loader.

The loader warns (never silently drops) on any unknown key, at every nesting level,
and the shipped tier YAMLs load with zero warnings. DataConfig defaults must also
match an empty-YAML build for the copy-paste / seed knobs.
"""

import logging
import unittest

import glob
import os

from configs.yaml_loader import (
    _build_data_config,
    _build_model_config,
    _build_parser_config,
    _build_trainer_config,
    load_config,
)

_EXP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "experiments", "yolo"
)


class TestUnknownKeyWarnings(unittest.TestCase):
    def test_unknown_data_key_warns(self):
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_data_config({"verbose_eval_results_per_category": True})
        self.assertTrue(
            any("verbose_eval_results_per_category" in m for m in cm.output)
        )

    def test_unknown_trainer_key_warns(self):
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_trainer_config({"totally_bogus_trainer_key": 1})
        self.assertTrue(
            any("totally_bogus_trainer_key" in m for m in cm.output)
        )


class TestNestedUnknownKeyWarnings(unittest.TestCase):
    """Unknown keys in nested sections must surface as warnings, not be dropped."""

    def test_parser_and_mosaic_keys_warn(self):
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_parser_config(
                {"bogus_parser_key": 1, "mosaic": {"bogus_mosaic_key": 2}}
            )
        joined = " ".join(cm.output)
        self.assertIn("bogus_parser_key", joined)
        self.assertIn("bogus_mosaic_key", joined)

    def test_model_and_detection_generator_keys_warn(self):
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_model_config(
                {"bogus_model_key": 1,
                 "detection_generator": {"bogus_detgen_key": 2}},
                {},
            )
        joined = " ".join(cm.output)
        self.assertIn("bogus_model_key", joined)
        self.assertIn("bogus_detgen_key", joined)

    def test_optimizer_and_lr_keys_warn(self):
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_trainer_config({"optimizer_config": {
                "optimizer": {"type": "sgd", "sgd": {"bogus_sgd_key": 1}},
                "learning_rate": {"type": "cosine", "cosine": {"bogus_lr_key": 2}},
            }})
        joined = " ".join(cm.output)
        self.assertIn("bogus_sgd_key", joined)
        self.assertIn("bogus_lr_key", joined)

    def test_shipped_tiers_load_without_warnings(self):
        logger = logging.getLogger("configs.yaml_loader")
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture()
        logger.addHandler(handler)
        try:
            for path in sorted(glob.glob(os.path.join(_EXP_DIR, "*.yaml"))):
                records.clear()
                load_config(path)
                warns = [r.getMessage() for r in records
                         if r.levelno >= logging.WARNING]
                self.assertEqual(
                    warns, [],
                    f"{os.path.basename(path)} loaded with warnings: {warns}",
                )
        finally:
            logger.removeHandler(handler)


class TestDataConfigDefaultsMatchEmptyYaml(unittest.TestCase):
    def test_dataconfig_defaults_match_empty_yaml(self):
        """A bare DataConfig() must equal _build_data_config({}) for the
        copy-paste and seed knobs.

        If the dataclass and loader defaults diverge for tfds_for_cnp/
        tfds_for_cnp_split/seed, a directly constructed DataConfig() silently
        enables copy-paste or seeded shuffling that an empty YAML leaves off.
        """
        from configs.model_config import DataConfig

        direct = DataConfig()
        from_empty_yaml = _build_data_config({})
        for field in ("tfds_for_cnp", "tfds_for_cnp_split", "seed"):
            self.assertEqual(
                getattr(direct, field), getattr(from_empty_yaml, field),
                f"DataConfig().{field} != _build_data_config({{}}).{field}",
            )
            self.assertIsNone(getattr(direct, field))


if __name__ == "__main__":
    unittest.main()
