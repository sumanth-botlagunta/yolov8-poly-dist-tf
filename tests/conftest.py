"""Shared pytest fixtures for the YOLOv8 test suite.

All tests run in TensorFlow eager mode so shapes and values are immediately
available without tf.function compilation.

Fixtures:
    eager_mode:       session-scoped, forces eager execution
    tiny_model_cfg:   small ModelConfig for fast forward-pass tests
    synthetic_image:  float32 [1, 672, 672, 3] batch
    synthetic_labels: labels dict matching V8ParserExtended output schema
    mock_decoded_det: dict matching PolygonDecoder output (single example)
    mock_decoded_dist: dict matching ServingBotDetDecoder output (single example)
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf


# ---------------------------------------------------------------------------
# Session-scoped eager mode
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def eager_mode():
    """Force TF eager execution for the entire test session."""
    tf.config.run_functions_eagerly(True)
    yield
    tf.config.run_functions_eagerly(False)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def tiny_model_cfg():
    """Minimal ModelConfig for fast tests (3 classes, small boxes)."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from configs.model_config import (
        ModelConfig, BackboneConfig, DecoderConfig,
        HeadConfig, DetectionGeneratorConfig, NormActivationConfig,
    )
    return ModelConfig(
        input_size=[672, 672, 3],
        num_classes=3,
        angle_step=15,
        output_poly_size=24,
        output_dist_size=1,
        num_dist_block=1,
        with_polygons=True,
        with_distance=True,
        deploy=False,
        backbone=BackboneConfig(model_id='cspdarknetv8s', depth_scale=0.33, width_scale=0.5),
        decoder=DecoderConfig(),
        head=HeadConfig(smart_bias=True),
        detection_generator=DetectionGeneratorConfig(max_boxes=10),
        norm_activation=NormActivationConfig(),
    )


@pytest.fixture(scope='session')
def poly_dist_cfg():
    """Full ExperimentConfig loaded from yolov8_poly_dist.yaml."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from configs.yaml_loader import load_config
    cfg_path = os.path.join(
        os.path.dirname(__file__), '..',
        'configs/experiments/yolo/yolov8_poly_dist.yaml',
    )
    return load_config(cfg_path)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image():
    """Float32 image batch [1, 672, 672, 3] in [0, 1]."""
    tf.random.set_seed(42)
    return tf.random.uniform([1, 672, 672, 3], minval=0.0, maxval=1.0)


@pytest.fixture
def synthetic_labels():
    """Labels dict matching V8ParserExtended output schema.

    Returns a batch of B=1, max_instances=300, 39 classes.
    """
    B, MAX_I, N_CLS = 1, 300, 39
    POLY_DEPTH = 72  # 24 * 3

    return {
        'bbox': tf.random.uniform([B, MAX_I, 4], 0.0, 1.0),
        'classes': tf.zeros([B, MAX_I], dtype=tf.int64),
        'polygons': tf.random.uniform([B, MAX_I, POLY_DEPTH], -0.5, 0.5),
        'n_gt': tf.constant([5], dtype=tf.int64),
        'ignore_bg': tf.zeros([B], dtype=tf.int64),
        'log_distance': tf.fill([B, MAX_I], -10.0),  # all invalid
    }


@pytest.fixture
def mock_decoded_det():
    """Synthetic decoded dict matching PolygonDecoder output (single example)."""
    N, MAX_V = 5, 10938
    rng = np.random.RandomState(0)

    # Boxes as ymin/xmin/ymax/xmax normalized.
    boxes = rng.uniform(0.0, 0.5, (N, 4)).astype(np.float32)
    boxes[:, 2] = boxes[:, 0] + 0.1
    boxes[:, 3] = boxes[:, 1] + 0.1
    boxes = np.clip(boxes, 0.0, 1.0)

    polygons = np.full((N, MAX_V), -1.0, dtype=np.float32)
    # Write some valid xy points for first 3 objects.
    for i in range(3):
        pts = rng.uniform(0.2, 0.8, (20,)).astype(np.float32)
        polygons[i, :20] = pts

    return {
        'image': tf.constant(rng.randint(0, 256, (672, 672, 3), dtype=np.uint8)),
        'source_id': tf.constant('test_img_001'),
        'height': tf.constant(672, dtype=tf.int32),
        'width': tf.constant(672, dtype=tf.int32),
        'groundtruth_boxes': tf.constant(boxes),
        'groundtruth_classes': tf.constant(rng.randint(0, 39, (N,)), dtype=tf.int64),
        'groundtruth_polygons': tf.constant(polygons),
        'groundtruth_is_crowd': tf.constant([False] * N),
        'groundtruth_area': tf.constant(rng.uniform(100, 5000, (N,)).astype(np.float32)),
        'groundtruth_dontcare': tf.zeros([N], dtype=tf.int64),
    }


@pytest.fixture
def mock_decoded_dist(mock_decoded_det):
    """Like mock_decoded_det but includes groundtruth_dists."""
    result = dict(mock_decoded_det)
    N = result['groundtruth_classes'].shape[0]
    # Mix of valid distances and -1 (invalid).
    dists = np.array([1.5, 3.2, -1.0, 7.0, -1.0], dtype=np.float32)[:N]
    result['groundtruth_dists'] = tf.constant(dists)
    return result


@pytest.fixture
def mock_tfds_batch(mock_decoded_det):
    """Batch of 2 decoded examples (same content repeated) for pipeline tests."""
    def _repeat(v):
        return tf.stack([v, v], axis=0)

    return {k: _repeat(v) for k, v in mock_decoded_det.items()}
