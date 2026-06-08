"""Integration test: model forward pass → TAL loss, 2 training steps.

Verifies that the full stack (model build → forward → loss → gradient)
runs end-to-end with synthetic data.  No real TFDS required.

Intentionally uses a small image size (128×128) and few classes to keep
the test fast; architectural strides remain unchanged.
"""

import pytest
import numpy as np
import tensorflow as tf

from configs.model_config import ModelConfig
from models.yolo_v8 import build_yolov8
from losses.tal_loss import TaskAlignedLossExtended


_B  = 2
_H  = _W = 128   # small image; level-3 FPN will be 16×16
_M  = 4          # max GT slots per image (2 valid, 2 padding)
_NC = 4          # number of classes


@pytest.fixture(scope="module")
def small_model():
    cfg = ModelConfig(
        input_size=[_H, _W, 3],
        num_classes=_NC,
        with_polygons=True,
        with_distance=True,
        deploy=False,
    )
    model = build_yolov8(cfg)
    model.deploy = False
    model.build_and_init(cfg.input_size)
    return model


@pytest.fixture(scope="module")
def loss_fn():
    return TaskAlignedLossExtended(
        num_classes=_NC,
        with_polygons=True,
        with_distance=True,
    )


def _make_labels(b=_B, m=_M, rng=None):
    """Synthetic GT batch matching the label format that TAL loss expects."""
    if rng is None:
        rng = np.random.default_rng(42)

    # 2 valid GT per image; the rest are padding (masked by n_gt)
    n_gt = np.full([b], 2, dtype=np.int64)

    y1 = rng.uniform(0.05, 0.25, (b, m)).astype(np.float32)
    x1 = rng.uniform(0.05, 0.25, (b, m)).astype(np.float32)
    y2 = np.clip(y1 + rng.uniform(0.1, 0.3, (b, m)), 0.0, 1.0).astype(np.float32)
    x2 = np.clip(x1 + rng.uniform(0.1, 0.3, (b, m)), 0.0, 1.0).astype(np.float32)
    bboxes = np.stack([y1, x1, y2, x2], axis=-1)   # [B, M, 4] yxyx normalized

    classes = rng.integers(0, _NC, (b, m)).astype(np.int64)

    # PolyYOLO [B, M, 72] = [dx, dy, conf] × 24
    polygons = rng.uniform(0.0, 0.03, (b, m, 72)).astype(np.float32)
    polygons[:, :, 2::3] = 1.0   # vertex confidence = 1.0

    # log-scale distances; first 2 per image are valid, rest are sentinel -10.0
    log_dist = np.full((b, m), -10.0, dtype=np.float32)
    log_dist[:, :2] = rng.uniform(np.log(0.5), np.log(10.0), (b, 2)).astype(np.float32)

    return {
        'bbox':         tf.constant(bboxes),
        'classes':      tf.constant(classes),
        'n_gt':         tf.constant(n_gt),
        'polygons':     tf.constant(polygons),
        'log_distance': tf.constant(log_dist),
        'ignore_bg':    tf.zeros([b], dtype=tf.int64),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelLossIntegration:

    def test_loss_components_finite_two_steps(self, small_model, loss_fn):
        """Two forward+loss passes should produce finite scalars for all components."""
        images = tf.random.uniform([_B, _H, _W, 3], seed=0)
        labels = _make_labels()

        for step in range(2):
            feats = small_model(images, training=True)
            total, box, dfl, cls, dist, poly, poly_a, poly_d, poly_c = loss_fn(feats, labels)

            for name, val in [("total", total), ("box", box), ("dfl", dfl),
                               ("cls", cls), ("dist", dist), ("poly", poly),
                               ("poly_angle", poly_a), ("poly_dist", poly_d), ("poly_conf", poly_c)]:
                assert tf.math.is_finite(val), f"step {step}: {name}_loss is NaN/Inf ({val})"
            assert float(total) > 0.0, f"step {step}: total_loss should be positive"

    def test_gradients_all_finite_and_non_none(self, small_model, loss_fn):
        """Every trainable variable must receive a finite, non-None gradient."""
        images = tf.random.uniform([_B, _H, _W, 3], seed=1)
        labels = _make_labels()

        with tf.GradientTape() as tape:
            feats = small_model(images, training=True)
            total, *_ = loss_fn(feats, labels)

        grads = tape.gradient(total, small_model.trainable_variables)
        none_vars = [v.name for v, g in zip(small_model.trainable_variables, grads) if g is None]
        nan_vars  = [v.name for v, g in zip(small_model.trainable_variables, grads)
                     if g is not None and not bool(tf.reduce_all(tf.math.is_finite(g)))]

        assert not none_vars, f"None gradients for {len(none_vars)} vars: {none_vars[:5]}"
        assert not nan_vars,  f"NaN/Inf gradients for {len(nan_vars)} vars: {nan_vars[:5]}"

    def test_training_mode_output_keys(self, small_model):
        """deploy=False: model returns all 6 head branches."""
        small_model.deploy = False
        x = tf.zeros([1, _H, _W, 3])
        out = small_model(x, training=False)
        assert set(out.keys()) == {"box", "cls", "poly_angle", "poly_dist", "poly_conf", "dist"}

    def test_inference_mode_output_keys(self, small_model):
        """deploy=True: model returns decoded detection keys."""
        small_model.deploy = True
        x = tf.zeros([1, _H, _W, 3])
        out = small_model(x, training=False)
        small_model.deploy = False
        for key in ("bbox", "classes", "confidence", "num_detections"):
            assert key in out, f"Missing inference key: {key}"

    def test_optimizer_step_produces_finite_loss(self, small_model, loss_fn):
        """A single gradient step must not produce NaN/Inf loss.

        At random initialization the loss can spike on a single step before
        settling (non-convex landscape), so we only verify finiteness, not
        monotone decrease.
        """
        from optimizers.sgd_warmup import SGDTorch

        images = tf.random.uniform([_B, _H, _W, 3], seed=2)
        labels = _make_labels()

        sgd = SGDTorch(
            lr_fn=lambda step: tf.constant(1e-4),
            momentum=0.937,
            momentum_start=0.8,
            nesterov=True,
            weight_decay=0.0005,
            warmup_steps=0,
        )

        # One gradient step
        with tf.GradientTape() as tape:
            feats = small_model(images, training=True)
            total, *_ = loss_fn(feats, labels)
        grads = tape.gradient(total, small_model.trainable_variables)
        sgd.apply_gradients(zip(grads, small_model.trainable_variables))

        # Loss after update must be finite
        feats_after = small_model(images, training=False)
        loss_after, *_ = loss_fn(feats_after, labels)

        assert tf.math.is_finite(loss_after), \
            f"Loss is NaN/Inf after optimizer step: {float(loss_after)}"

    def test_bbox_only_model_no_polygon_keys(self):
        """bbox-only model does not produce polygon/distance head outputs."""
        cfg = ModelConfig(
            input_size=[_H, _W, 3],
            num_classes=_NC,
            with_polygons=False,
            with_distance=False,
            deploy=False,
        )
        model = build_yolov8(cfg)
        model.build_and_init(cfg.input_size)
        model.deploy = False
        out = model(tf.zeros([1, _H, _W, 3]), training=False)
        assert set(out.keys()) == {"box", "cls"}

    def test_loss_with_ignore_bg(self, small_model, loss_fn):
        """ignore_bg=1 for all images should not crash or produce NaN."""
        images = tf.random.uniform([_B, _H, _W, 3], seed=3)
        labels = _make_labels()
        labels = dict(labels)
        labels['ignore_bg'] = tf.ones([_B], dtype=tf.int64)

        feats = small_model(images, training=True)
        total, *_ = loss_fn(feats, labels)
        assert tf.math.is_finite(total), f"Loss with ignore_bg=1 is NaN/Inf: {total}"
