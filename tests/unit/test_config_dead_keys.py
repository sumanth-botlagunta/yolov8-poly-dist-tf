"""Pinning tests for dead/typo YAML key handling in configs/yaml_loader.

Covers the bughunt fixes:
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

from configs.yaml_loader import (
    _build_data_config,
    _build_trainer_config,
    load_config,
)
from configs.model_config import DetectionGeneratorConfig

_TIERS = [
    "configs/experiments/yolo/yolov8_bbox.yaml",
    "configs/experiments/yolo/yolov8_poly.yaml",
    "configs/experiments/yolo/yolov8_poly_dist.yaml",
]


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

    def test_known_ignored_data_key_does_not_warn(self):
        logger = logging.getLogger("configs.yaml_loader")
        with self.assertLogs(logger, level="WARNING") as cm:
            # tfds_download is intentionally ignored; emit one real warning so
            # assertLogs has something, then assert tfds_download is NOT in it.
            logger.warning("sentinel")
            _build_data_config({"tfds_download": True})
        self.assertFalse(any("tfds_download" in m for m in cm.output))

    def test_unknown_trainer_key_warns(self):
        with self.assertLogs("configs.yaml_loader", level="WARNING") as cm:
            _build_trainer_config({"totally_bogus_trainer_key": 1})
        self.assertTrue(
            any("totally_bogus_trainer_key" in m for m in cm.output)
        )

    def test_known_ignored_trainer_keys_do_not_warn(self):
        logger = logging.getLogger("configs.yaml_loader")
        with self.assertLogs(logger, level="WARNING") as cm:
            logger.warning("sentinel")
            _build_trainer_config({
                "validation_interval": 2118,
                "summary_interval": 2118,
                "train_tf_function": True,
                "eval_tf_while_loop": False,
            })
        joined = " ".join(cm.output)
        for k in ("validation_interval", "summary_interval", "train_tf_function"):
            self.assertNotIn(k, joined)


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
