"""Unit tests for CSPDarkNetV8 backbone and C2f building blocks."""

import pytest
import tensorflow as tf

from models.backbone import C2f, C2fBottleneck, CSPDarkNetV8, SPPF


@pytest.fixture(autouse=True)
def eager_mode():
    tf.config.run_functions_eagerly(True)


# ---------------------------------------------------------------------------
# C2fBottleneck
# ---------------------------------------------------------------------------

class TestC2fBottleneck:
    def test_output_shape_with_shortcut(self):
        layer = C2fBottleneck(64, shortcut=True)
        x = tf.zeros([2, 8, 8, 64])
        out = layer(x, training=False)
        assert out.shape == (2, 8, 8, 64)

    def test_output_shape_no_shortcut(self):
        layer = C2fBottleneck(32, shortcut=False)
        x = tf.zeros([2, 4, 4, 32])
        out = layer(x, training=False)
        assert out.shape == (2, 4, 4, 32)

    def test_shortcut_adds_residual(self):
        layer = C2fBottleneck(16, shortcut=True)
        x = tf.ones([1, 2, 2, 16])
        out = layer(x, training=False)
        # Output differs from input (non-trivial transform)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# C2f
# ---------------------------------------------------------------------------

class TestC2f:
    def test_output_shape_n1(self):
        layer = C2f(128, n=1)
        x = tf.zeros([2, 8, 8, 64])
        out = layer(x, training=False)
        assert out.shape == (2, 8, 8, 128)

    def test_output_shape_n2(self):
        layer = C2f(256, n=2)
        x = tf.zeros([1, 4, 4, 128])
        out = layer(x, training=False)
        assert out.shape == (1, 4, 4, 256)

    def test_same_in_out_channels(self):
        layer = C2f(64, n=1)
        x = tf.zeros([2, 6, 6, 64])
        out = layer(x, training=False)
        assert out.shape == (2, 6, 6, 64)


# ---------------------------------------------------------------------------
# SPPF
# ---------------------------------------------------------------------------

class TestSPPF:
    def test_output_shape(self):
        layer = SPPF(512, kernel_size=5)
        x = tf.zeros([2, 7, 7, 512])
        out = layer(x, training=False)
        assert out.shape == (2, 7, 7, 512)

    def test_small_feature_map(self):
        layer = SPPF(256, kernel_size=5)
        x = tf.zeros([1, 3, 3, 256])
        out = layer(x, training=False)
        assert out.shape == (1, 3, 3, 256)


# ---------------------------------------------------------------------------
# CSPDarkNetV8
# ---------------------------------------------------------------------------

class TestCSPDarkNetV8:
    @pytest.fixture
    def backbone(self):
        return CSPDarkNetV8(model_id="cspdarknetv8s")

    def test_output_keys(self, backbone):
        x = tf.zeros([1, 672, 672, 3])
        out = backbone(x, training=False)
        assert set(out.keys()) == {"3", "4", "5"}

    def test_p3_shape(self, backbone):
        x = tf.zeros([2, 672, 672, 3])
        out = backbone(x, training=False)
        assert out["3"].shape == (2, 84, 84, 128)

    def test_p4_shape(self, backbone):
        x = tf.zeros([2, 672, 672, 3])
        out = backbone(x, training=False)
        assert out["4"].shape == (2, 42, 42, 256)

    def test_p5_shape(self, backbone):
        x = tf.zeros([2, 672, 672, 3])
        out = backbone(x, training=False)
        assert out["5"].shape == (2, 21, 21, 512)

    def test_output_specs_property(self, backbone):
        # Build by calling once
        _ = backbone(tf.zeros([1, 672, 672, 3]), training=False)
        specs = backbone.output_specs
        assert specs == {"3": 128, "4": 256, "5": 512}

    def test_no_nan_in_output(self, backbone):
        x = tf.random.normal([1, 672, 672, 3])
        out = backbone(x, training=False)
        for v in out.values():
            assert not tf.reduce_any(tf.math.is_nan(v)).numpy()
