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


class TestDistanceLogClampNoOverflow(unittest.TestCase):
    """Distance decode must clamp in LOG space before exp().

    A huge log-distance logit would overflow exp() to +inf, and the previous
    `clip(exp(x), min, max)` order turned that into max_distance silently. Clamping
    the log first (`exp(clip(x, log min, log max))`) bounds the exp input, so the
    result is finite for any input and identical for in-range values.
    """

    def _gen(self):
        size = 64
        return YoloV8Layer(
            input_image_size=[size, size],
            num_classes=2,
            max_boxes=10,
            score_thresh=0.01,
            min_distance=0.5,
            max_distance=10.0,
        ), size

    def _raw(self, size, dist_logit):
        raw = {"box": {}, "cls": {}, "dist": {},
               "poly_angle": {}, "poly_dist": {}, "poly_conf": {}}
        for level, stride in (("3", 8), ("4", 16), ("5", 32)):
            f = size // stride
            box_logits = np.zeros((1, f, f, 4 * 16), np.float32)
            box_logits[..., 0::16] = 5.0  # small finite box
            raw["box"][level] = tf.constant(box_logits)
            cls_logits = np.full((1, f, f, 2), -10.0, np.float32)
            cls_logits[..., 0] = 4.0
            raw["cls"][level] = tf.constant(cls_logits)
            raw["dist"][level] = tf.constant(
                np.full((1, f, f, 1), dist_logit, np.float32))
            raw["poly_angle"][level] = tf.constant(np.zeros((1, f, f, 24), np.float32))
            raw["poly_dist"][level] = tf.constant(np.zeros((1, f, f, 24), np.float32))
            raw["poly_conf"][level] = tf.constant(np.zeros((1, f, f, 24), np.float32))
        return raw

    def test_huge_log_distance_does_not_overflow(self):
        gen, size = self._gen()
        out = gen(self._raw(size, dist_logit=120.0))  # exp(120) = inf
        n = int(out["num_detections"][0])
        self.assertGreater(n, 0)
        d = out["distance"][0, :n].numpy()
        self.assertTrue(np.isfinite(d).all(), "distance contains inf/nan")
        self.assertLessEqual(d.max(), 10.0 + 1e-4)
        self.assertGreaterEqual(d.min(), 0.5 - 1e-4)

    def test_in_range_log_distance_round_trips(self):
        gen, size = self._gen()
        import math
        log_d = math.log(3.0)  # in range
        out = gen(self._raw(size, dist_logit=log_d))
        n = int(out["num_detections"][0])
        d = out["distance"][0, :n].numpy()
        self.assertTrue(np.allclose(d, 3.0, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
