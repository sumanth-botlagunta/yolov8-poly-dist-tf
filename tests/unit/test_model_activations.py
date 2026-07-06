"""Activation wiring: trunk (backbone+decoder) vs head activations.

The original codebase hardcoded swish in its backbone/decoder layer specs
while the head used the config activation (relu). Here that split is
config-driven: ``norm_activation.activation`` feeds the trunk, and
``head.activation`` ("same" inherits the trunk) lets the head diverge.

These tests build the real model and inspect every ``tf.keras.layers.
Activation`` instance per module, so a regression in the factory wiring
(e.g. the head silently inheriting swish, or the decoder ignoring the
config) fails loudly.
"""

import unittest

import tensorflow as tf

from configs.model_config import ModelConfig
from models.yolo_v8 import build_yolov8


def _activation_names(module) -> set:
    names = set()
    for layer in module.submodules:
        if isinstance(layer, tf.keras.layers.Activation):
            act = layer.get_config()["activation"]
            names.add(act if isinstance(act, str) else getattr(act, "__name__", str(act)))
    return names


def _build(trunk: str, head: str):
    cfg = ModelConfig(input_size=[64, 64, 3], num_classes=3)
    cfg.norm_activation.activation = trunk
    cfg.head.activation = head
    model = build_yolov8(cfg)
    model.build_and_init(cfg.input_size)
    return model


class TestActivationWiring(unittest.TestCase):
    def test_swish_trunk_relu_head(self):
        """The production split: swish backbone+decoder, relu head."""
        model = _build(trunk="swish", head="relu")

        for name, module in (("backbone", model.backbone), ("decoder", model.decoder)):
            acts = _activation_names(module)
            self.assertTrue(acts, f"{name} has no Activation layers")
            self.assertEqual(acts, {"swish"}, f"{name} activations: {acts}")

        head_acts = _activation_names(model.head)
        self.assertTrue(head_acts, "head has no Activation layers")
        self.assertEqual(head_acts, {"relu"}, f"head activations: {head_acts}")

    def test_head_same_inherits_trunk(self):
        """head.activation='same' must follow norm_activation.activation."""
        model = _build(trunk="relu", head="same")
        self.assertEqual(_activation_names(model.head), {"relu"})

    def test_production_yaml_resolves_swish_trunk_relu_head(self):
        """The poly_dist experiment YAML pins the trunk/head activation split."""
        from configs.yaml_loader import load_config
        cfg = load_config("configs/experiments/yolo/yolov8_poly_dist.yaml")
        self.assertEqual(cfg.task.model.norm_activation.activation, "swish")
        self.assertEqual(cfg.task.model.head.activation, "relu")


if __name__ == "__main__":
    unittest.main()
