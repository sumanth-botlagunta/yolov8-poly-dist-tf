"""NMS class-mode contract: per-class vs class-agnostic suppression.

The generator supports two suppression scopes (``nms_class_mode``):

  * ``per_class`` (default): NMS runs independently per class, so two
    heavily-overlapping boxes whose argmax classes differ BOTH survive.
  * ``agnostic``: one NMS over all boxes — at each location only the
    highest-scored box survives regardless of class (the original
    codebase's mode).

The discriminating case is a cross-class duplicate: same location, different
argmax class. per_class must keep both; agnostic must keep only the higher
scored one.
"""

import unittest

import numpy as np
import tensorflow as tf

from models.detection_generator import YoloV8Layer


def _make_generator(mode: str) -> YoloV8Layer:
    return YoloV8Layer(
        input_image_size=[64, 64],
        num_classes=3,
        max_boxes=20,
        nms_thresh=0.65,
        score_thresh=0.05,
        nms_class_mode=mode,
    )


def _raw_outputs(anchor_cls_logits):
    """Raw head outputs for a 64x64 input (levels 8/4/2 anchors).

    Box logits are all-zero -> uniform DFL softmax -> every side offset is the
    bin mean (7.5 feature px), so every anchor decodes to the same-size box
    centred on itself: neighbouring level-3 anchors (8 px apart, 120 px boxes)
    overlap almost completely.

    Args:
        anchor_cls_logits: {(level, y, x, class): logit} — everything else is
            -10 (sigmoid ~4.5e-5, far below score_thresh).
    """
    raw = {"box": {}, "cls": {}}
    for level, stride in (("3", 8), ("4", 16), ("5", 32)):
        f = 64 // stride
        raw["box"][level] = tf.zeros([1, f, f, 4 * 16], tf.float32)
        cls = np.full((1, f, f, 3), -10.0, np.float32)
        for (lv, y, x, c), logit in anchor_cls_logits.items():
            if lv == level:
                cls[0, y, x, c] = logit
        raw["cls"][level] = tf.constant(cls)
    return raw


class TestNmsClassMode(unittest.TestCase):
    # Two neighbouring level-3 anchors: near-identical boxes, DIFFERENT argmax
    # classes; class 0 is scored higher than class 1.
    CROSS_CLASS = {("3", 2, 2, 0): 4.0, ("3", 2, 3, 1): 3.0}

    def test_per_class_keeps_cross_class_duplicates(self):
        out = _make_generator("per_class")(_raw_outputs(self.CROSS_CLASS))
        n = int(out["num_detections"][0])
        self.assertEqual(n, 2)
        self.assertCountEqual(out["classes"][0, :n].numpy().tolist(), [0, 1])

    def test_agnostic_suppresses_cross_class_duplicates(self):
        out = _make_generator("agnostic")(_raw_outputs(self.CROSS_CLASS))
        n = int(out["num_detections"][0])
        self.assertEqual(n, 1)
        # The higher-scored box (class 0) must be the survivor.
        self.assertEqual(int(out["classes"][0, 0]), 0)
        self.assertAlmostEqual(
            float(out["confidence"][0, 0]), float(tf.sigmoid(4.0)), places=5)

    def test_same_class_duplicates_suppressed_in_both_modes(self):
        logits = {("3", 2, 2, 0): 4.0, ("3", 2, 3, 0): 3.0}
        for mode in ("per_class", "agnostic"):
            out = _make_generator(mode)(_raw_outputs(logits))
            self.assertEqual(int(out["num_detections"][0]), 1, f"mode={mode}")

    def test_distant_boxes_survive_in_both_modes(self):
        # Opposite corners of the level-5 grid (32 px stride, 2x2 anchors are
        # 32 px apart with 120 px boxes -> still overlapping; use level 3
        # corners 40 px apart... they'd overlap too. Distant = level-3 corners.
        # (0.5+2)*8=20 px vs (0.5+5)*8=44 px centres with 120 px boxes overlap
        # heavily; IoU ~ (120-24)/(120+24) per axis ~ 0.44 area -> below 0.65
        # threshold, so both survive even agnostically.
        logits = {("3", 2, 2, 0): 4.0, ("3", 5, 5, 1): 3.0}
        for mode in ("per_class", "agnostic"):
            out = _make_generator(mode)(_raw_outputs(logits))
            self.assertEqual(int(out["num_detections"][0]), 2, f"mode={mode}")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            _make_generator("both")

    def test_config_default_and_wiring(self):
        from configs.model_config import DetectionGeneratorConfig
        self.assertEqual(DetectionGeneratorConfig().nms_class_mode, "per_class")
        # Default constructor value matches the config default.
        self.assertEqual(_make_generator("per_class").nms_class_mode, "per_class")


if __name__ == "__main__":
    unittest.main()
