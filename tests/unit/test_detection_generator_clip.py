"""Final detection boxes must be clipped to the image ([0, 1] normalized).

The DFL decode can place box edges beyond the borders (cx − l < 0, cx + r > W),
which used to leak out of the generator: boxes drew outside the frame in
TensorBoard overlays and slightly hurt IoU against edge-clipped GT in eval.
"""

import unittest

import numpy as np
import tensorflow as tf

from models.detection_generator import YoloV8Layer


class TestFinalBoxClipping(unittest.TestCase):
    def test_boxes_clipped_to_unit_range(self):
        size = 64  # input 64 → levels 8/4/2
        gen = YoloV8Layer(
            input_image_size=[size, size],
            num_classes=3,
            max_boxes=20,
            score_thresh=0.01,
        )

        # DFL logits hard-pinned to the LAST bin → ltrb = 15 × stride px on every
        # side. For border anchors that decodes far outside the image.
        raw = {"box": {}, "cls": {}}
        for level, stride in (("3", 8), ("4", 16), ("5", 32)):
            f = size // stride
            box_logits = np.full((1, f, f, 4 * 16), -1e9, np.float32)
            box_logits[..., 15::16] = 0.0  # one-hot on bin 15 per side
            raw["box"][level] = tf.constant(box_logits)
            cls_logits = np.full((1, f, f, 3), -10.0, np.float32)
            cls_logits[..., 0] = 4.0  # high score → survives thresholds/NMS
            raw["cls"][level] = tf.constant(cls_logits)

        out = gen(raw)
        n = int(out["num_detections"][0])
        self.assertGreater(n, 0, "test setup produced no detections")
        boxes = out["bbox"][0, :n].numpy()
        self.assertGreaterEqual(boxes.min(), 0.0)
        self.assertLessEqual(boxes.max(), 1.0)
        # Sanity: without clipping these decodes WOULD exceed the image
        # (15 × 8 = 120 px reach ≥ 64 px image) — assert at least one box was
        # actually clamped flush to a border.
        on_border = np.logical_or(np.isclose(boxes, 0.0), np.isclose(boxes, 1.0))
        self.assertTrue(on_border.any())


if __name__ == "__main__":
    unittest.main()
