"""Unit tests for full YoloV8 model forward pass — training and inference modes."""

import pytest
import tensorflow as tf

from configs.model_config import ModelConfig
from models.yolo_v8 import YoloV8, build_yolov8


@pytest.fixture(autouse=True)
def eager_mode():
    tf.config.run_functions_eagerly(True)


@pytest.fixture(scope="module")
def full_model():
    """Build and return a fully-initialised YoloV8 (all 6 heads)."""
    cfg = ModelConfig()   # default: with_polygons=True, with_distance=True
    model = build_yolov8(cfg)
    model.build_and_init()
    return model


@pytest.fixture(scope="module")
def bbox_only_model():
    """YoloV8 with box + cls heads only."""
    cfg = ModelConfig()
    cfg.with_polygons = False
    cfg.with_distance = False
    model = build_yolov8(cfg)
    model.build_and_init()
    return model


# ---------------------------------------------------------------------------
# Training mode (deploy=False)
# ---------------------------------------------------------------------------

class TestTrainingMode:
    def test_output_keys_full(self, full_model):
        full_model.deploy = False
        x = tf.zeros([1, 672, 672, 3])
        out = full_model(x, training=False)
        assert set(out.keys()) == {"box", "cls", "poly_angle", "poly_dist", "poly_conf", "dist"}

    def test_output_keys_bbox_only(self, bbox_only_model):
        bbox_only_model.deploy = False
        x = tf.zeros([1, 672, 672, 3])
        out = bbox_only_model(x, training=False)
        assert set(out.keys()) == {"box", "cls"}

    @pytest.mark.parametrize("level,h,w", [("3", 84, 84), ("4", 42, 42), ("5", 21, 21)])
    def test_box_shape(self, full_model, level, h, w):
        full_model.deploy = False
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out["box"][level].shape == (2, h, w, 64)   # 4 * reg_max=16

    @pytest.mark.parametrize("level,h,w", [("3", 84, 84), ("4", 42, 42), ("5", 21, 21)])
    def test_cls_shape(self, full_model, level, h, w):
        full_model.deploy = False
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out["cls"][level].shape == (2, h, w, 39)

    @pytest.mark.parametrize("head", ["poly_angle", "poly_dist", "poly_conf"])
    @pytest.mark.parametrize("level,h,w", [("3", 84, 84), ("4", 42, 42), ("5", 21, 21)])
    def test_poly_shape(self, full_model, head, level, h, w):
        full_model.deploy = False
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out[head][level].shape == (2, h, w, 24)

    @pytest.mark.parametrize("level,h,w", [("3", 84, 84), ("4", 42, 42), ("5", 21, 21)])
    def test_dist_shape(self, full_model, level, h, w):
        full_model.deploy = False
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out["dist"][level].shape == (2, h, w, 1)

    def test_no_nan_in_training_output(self, full_model):
        full_model.deploy = False
        x = tf.random.normal([1, 672, 672, 3])
        out = full_model(x, training=False)
        for head, levels in out.items():
            for lvl, t in levels.items():
                assert not tf.reduce_any(tf.math.is_nan(t)).numpy(), \
                    f"NaN in {head}[{lvl}]"


# ---------------------------------------------------------------------------
# Inference mode (deploy=True)
# ---------------------------------------------------------------------------

class TestInferenceMode:
    def test_output_keys(self, full_model):
        full_model.deploy = True
        x = tf.zeros([1, 672, 672, 3])
        out = full_model(x, training=False)
        assert set(out.keys()) == {"bbox", "classes", "confidence",
                                    "num_detections", "polygons", "distance"}

    def test_bbox_shape(self, full_model):
        full_model.deploy = True
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out["bbox"].shape == (2, 300, 4)

    def test_classes_shape(self, full_model):
        full_model.deploy = True
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out["classes"].shape == (2, 300)

    def test_num_detections_scalar_per_image(self, full_model):
        full_model.deploy = True
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out["num_detections"].shape == (2,)

    def test_polygons_shape(self, full_model):
        full_model.deploy = True
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out["polygons"].shape == (2, 300, 24, 3)

    def test_distance_shape(self, full_model):
        full_model.deploy = True
        x = tf.zeros([2, 672, 672, 3])
        out = full_model(x, training=False)
        assert out["distance"].shape == (2, 300)

    def test_confidence_in_0_1(self, full_model):
        full_model.deploy = True
        x = tf.random.normal([1, 672, 672, 3])
        out = full_model(x, training=False)
        conf = out["confidence"]
        assert tf.reduce_all(conf >= 0.0).numpy()
        assert tf.reduce_all(conf <= 1.0).numpy()


# ---------------------------------------------------------------------------
# Smart bias initialisation
# ---------------------------------------------------------------------------

class TestSmartBias:
    def test_biases_initialized_flag(self, full_model):
        assert full_model._biases_initialized

    def test_cls_bias_not_zero(self, full_model):
        # After smart bias init, cls bias should be negative (log-scale)
        full_model.deploy = False
        cls_bias = full_model.head.cls_pred_3.bias.numpy()
        assert not (cls_bias == 0).all(), "cls bias should be non-zero after init"
