"""Pinning tests for dead/typo YAML key handling in configs/yaml_loader.

Pinned contracts:
  * verbose_eval_results_per_category was a silently-ignored data key; per-category
    eval is gated on task.per_category_metrics instead. The dead key is removed
    from the live YAML, and _build_data_config now warns on unknown data keys.
  * validation_interval / summary_interval / train_tf_* were dead trainer keys in
    the bbox/poly tiers; removed, and _build_trainer_config now warns on unknown
    trainer keys (while still tolerating the known TF-Vision vestigial ones).
  * iou_thresh was a dead detection_generator field never used by NMS; removed
    from the dataclass, loader, and YAMLs.
"""

import logging
import unittest

import glob
import os
import re

from configs.yaml_loader import (
    _build_data_config,
    _build_model_config,
    _build_parser_config,
    _build_trainer_config,
    load_config,
)
from configs.model_config import DetectionGeneratorConfig

_TIERS = [
    "configs/experiments/yolo/yolov8_bbox.yaml",
    "configs/experiments/yolo/yolov8_poly.yaml",
    "configs/experiments/yolo/yolov8_poly_dist.yaml",
]

_EXP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "experiments", "yolo"
)


class TestDeadKeysRemovedFromYaml(unittest.TestCase):
    def test_no_dead_trainer_or_data_keys_in_yaml_text(self):
        dead = [
            "verbose_eval_results_per_category",
            "validation_interval",
            "summary_interval",
            "train_tf_function",
            "train_tf_while_loop",
            "eval_tf_function",
            "eval_tf_while_loop",
            "iou_thresh",
        ]
        for path in _TIERS:
            with open(path) as f:
                text = f.read()
            for key in dead:
                self.assertNotIn(
                    key, text, f"dead key '{key}' still present in {path}"
                )

    def test_all_tiers_still_load(self):
        for path in _TIERS:
            cfg = load_config(path)
            self.assertEqual(cfg.task.num_classes, 39)


class TestIouThreshRemoved(unittest.TestCase):
    def test_detection_generator_config_has_no_iou_thresh(self):
        cfg = DetectionGeneratorConfig()
        self.assertFalse(
            hasattr(cfg, "iou_thresh"),
            "iou_thresh was dead (NMS uses nms_thresh) and must be removed.",
        )
        # nms_thresh is the actual NMS overlap threshold and must survive.
        self.assertEqual(cfg.nms_thresh, 0.65)


class TestUnknownKeyWarnings(unittest.TestCase):
    def test_unknown_data_key_warns(self):
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_data_config({"verbose_eval_results_per_category": True})
        self.assertTrue(
            any("verbose_eval_results_per_category" in m for m in cm.output)
        )

    def test_removed_data_key_warns(self):
        # tfds_download was removed from the YAML surface; the loader has no
        # silent-tolerance list, so a stale key must warn instead of being
        # dropped as if it did something.
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_data_config({"tfds_download": True})
        self.assertTrue(any("tfds_download" in m for m in cm.output))

    def test_unknown_trainer_key_warns(self):
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_trainer_config({"totally_bogus_trainer_key": 1})
        self.assertTrue(
            any("totally_bogus_trainer_key" in m for m in cm.output)
        )

    def test_removed_trainer_keys_warn(self):
        # Keys this trainer never honored are no longer silently tolerated —
        # they must surface in the unknown-key warning so a stale YAML is
        # visibly stale rather than quietly partially applied.
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_trainer_config({
                "validation_interval": 2118,
                "summary_interval": 2118,
                "train_tf_function": True,
                "eval_tf_while_loop": False,
            })
        joined = " ".join(cm.output)
        for k in ("validation_interval", "summary_interval", "train_tf_function"):
            self.assertIn(k, joined)


class TestRemovedDeadKeysGoneFromYaml(unittest.TestCase):
    """The optimizer/model/detection/parser keys read by nobody are deleted."""

    DEAD = [
        "clipnorm", "clipvalue", "global_clipnorm", "decay",
        "weight_keys", "bias_keys",
        "darknet_based_model", "anchor_boxes",
        "topN_per_anchor", "path_scales",
        "random_pad", "best_match_only", "use_tie_breaker", "anchor_thresh",
    ]

    def test_dead_keys_absent(self):
        # Match a key only at the start of a line (after indentation) so substrings
        # of live keys (weight_decay, average_decay, decay_steps) don't false-hit.
        for path in _TIERS:
            with open(path) as f:
                text = f.read()
            for key in self.DEAD:
                self.assertIsNone(
                    re.search(rf"(?m)^\s*{re.escape(key)}\s*:", text),
                    f"dead key '{key}' still in {path}",
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

        The dataclass previously defaulted tfds_for_cnp/tfds_for_cnp_split/seed
        to non-None values while the YAML path defaulted them to None — so
        directly constructing DataConfig() silently enabled copy-paste (truthy
        tfds_for_cnp) and seeded shuffling (seed=1000) that an empty YAML would
        leave off.
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
