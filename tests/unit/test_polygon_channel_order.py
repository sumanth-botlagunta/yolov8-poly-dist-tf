"""Pin the two polygon per-bin channel layouts (they are intentionally different).

GT radial target (parser):   [dist, angle, conf] interleaved -> [N, 24*3]
Prediction (detection gen):  (conf, dist, angle) stacked      -> [B, boxes, 24, 3]

Every consumer indexes one of the two; a transcription mix-up would silently
score/draw garbage. These tests fail loudly if either layout changes.
"""

import numpy as np
import tensorflow as tf

from models.detection_generator import YoloV8Layer


def test_gt_radial_layout_is_dist_angle_conf():
    from data_pipeline.yolo_parser import V8ParserExtended

    parser = V8ParserExtended.__new__(V8ParserExtended)  # no init needed for the helper
    # One axis-aligned square, box == polygon bounds, center (0.5, 0.5).
    boxes = tf.constant([[0.3, 0.3, 0.7, 0.7]], tf.float32)  # yxyx normalized
    square = tf.constant([[0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7]], tf.float32)
    target = parser._preprocess_polygons_v2(boxes, square, angle_step=15)
    target = tf.reshape(target, [1, 24, 3]).numpy()[0]

    dist, angle, conf = target[:, 0], target[:, 1], target[:, 2]
    occupied = conf > 0.5
    assert occupied.any(), "square must occupy at least the corner bins"
    # Channel 2 is the 0/1 conf gate; channel 0 the radial distance (positive on
    # occupied bins, normalized units < 1); channel 1 the sub-bin offset in [0,1).
    assert set(np.unique(conf)) <= {0.0, 1.0}
    assert (dist[occupied] > 0.0).all() and (dist[occupied] < 1.0).all()
    assert (angle >= 0.0).all() and (angle < 1.0).all()
    # Empty bins carry zeroed dist/angle (masked by conf in the loss).
    assert (dist[~occupied] == 0.0).all()


def test_prediction_layout_is_conf_dist_angle():
    gen = YoloV8Layer(input_image_size=[64, 64], num_classes=3, max_boxes=8,
                      score_thresh=0.01)
    # Distinct raw values per branch so the output channels are attributable.
    A_RAW, D_RAW, C_RAW = 2.0, 1.0, -1.0
    raw = {"box": {}, "cls": {}, "poly_angle": {}, "poly_dist": {}, "poly_conf": {}}
    for level, stride in (("3", 8), ("4", 16), ("5", 32)):
        f = 64 // stride
        raw["box"][level] = tf.zeros([1, f, f, 64], tf.float32)
        cls = np.full((1, f, f, 3), -10.0, np.float32)
        if level == "3":
            cls[0, 2, 2, 0] = 4.0
        raw["cls"][level] = tf.constant(cls)
        raw["poly_angle"][level] = tf.fill([1, f, f, 24], A_RAW)
        raw["poly_dist"][level]  = tf.fill([1, f, f, 24], D_RAW)
        raw["poly_conf"][level]  = tf.fill([1, f, f, 24], C_RAW)

    out = gen(raw)
    n = int(out["num_detections"][0])
    assert n >= 1
    poly = out["polygons"][0, 0].numpy()   # [24, 3]
    # The live detection is the level-3 (stride 8) anchor at (2, 2); poly_dist
    # is pre-scaled at the per-level flatten by stride/img_size (8/64 = 0.125)
    # to convert the grid-unit training target back to a normalized-image
    # radius. Layout (channel order) itself is unchanged: (conf, dist, angle).
    np.testing.assert_allclose(poly[:, 0], tf.sigmoid(C_RAW).numpy(),  atol=1e-6)  # conf
    np.testing.assert_allclose(
        poly[:, 1], tf.math.softplus(D_RAW).numpy() * (8.0 / 64.0), atol=1e-6)  # dist
    np.testing.assert_allclose(poly[:, 2], tf.sigmoid(A_RAW).numpy(),  atol=1e-6)  # angle
