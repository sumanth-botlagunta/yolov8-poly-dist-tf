"""Behavior/config contracts that live code depends on.

  - random_horizontal_flip keys polygon validity off the reserved -1.0 sentinel
    (`> -1.0`), so a legitimately-negative canvas coordinate is flipped as a real
    vertex, not skipped as padding.
  - SGDTorch uses bias_lr_scale as the absolute group-0/1 warmup-start LR.
  - Mosaic shear/translate are present in every tier YAML and parse into MosaicConfig.
  - validation parser random_flip is false, training parser random_flip is true, in
    every tier YAML.
"""

import os

import tensorflow as tf

from data_pipeline.augmentations import random_horizontal_flip
from configs.yaml_loader import load_config


_CFG_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'experiments', 'yolo'
)
_EXPERIMENTS = ['yolov8_poly_dist', 'yolov8_poly', 'yolov8_bbox']


def test_flip_transforms_legit_negative_vertex_not_sentinel():
    """A legit-negative polygon x (mosaic overflow) is flipped (x -> 1-x), while the
    -1.0 sentinel is left untouched."""
    tf.random.set_seed(4)  # forces do_flip = True (uniform() > 0.5)

    image = tf.zeros([8, 8, 3], tf.uint8)
    boxes = tf.constant([[0.4, 0.4, 0.6, 0.6]], tf.float32)
    # v0: x = -0.05 legit-negative; v1: 0.3 interior; v2: -1.0 true sentinel.
    polygons = tf.constant([[-0.05, 0.5, 0.3, 0.5, -1.0, -1.0]], tf.float32)

    _, _, out = random_horizontal_flip(image, boxes, polygons)
    pts = out.numpy().reshape(-1, 2)

    assert abs(pts[0, 0] - 1.05) < 1e-5, f"negative vertex not flipped: {pts[0]}"
    assert abs(pts[1, 0] - 0.7) < 1e-5, f"interior vertex flip wrong: {pts[1]}"
    assert pts[2, 0] == -1.0 and pts[2, 1] == -1.0, f"sentinel changed: {pts[2]}"


def test_bias_warmup_start_lr_is_absolute():
    """_effective_lr uses bias_lr_scale directly as the group-0/1 warmup-start LR
    (absolute), not bias_lr_scale * base_lr."""
    from optimizers.sgd_warmup import SGDTorch

    base_lr = 0.01
    bias_scale = 0.1
    lr_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr, decay_steps=10_000, alpha=0.01
    )
    opt = SGDTorch(lr_fn=lr_fn, bias_lr_scale=bias_scale, warmup_steps=1000)
    lr_bias_start = float(
        opt._effective_lr(base_lr=tf.constant(base_lr), t=tf.constant(0.0), group=1)
    )
    assert abs(lr_bias_start - bias_scale) < 1e-7, \
        f"bias warmup start LR should be {bias_scale} (absolute), got {lr_bias_start}"
    assert abs(lr_bias_start - bias_scale * base_lr) > 1e-7, \
        "bias warmup start must not be bias_lr_scale * base_lr"


def test_mosaic_affine_params_present_and_parsed():
    """Mosaic shear/translate are visible in every tier YAML and map onto MosaicConfig."""
    import yaml
    for name in _EXPERIMENTS:
        path = os.path.join(_CFG_DIR, f'{name}.yaml')
        with open(path) as f:
            raw = yaml.safe_load(f)
        mraw = raw['task']['train_data']['parser']['mosaic']
        # Mosaic rotation is hard-off in code, so degrees/rotate_prob are absent.
        for k in ('shear', 'translate'):
            assert k in mraw, f"{name}: mosaic.{k} missing from YAML"

        cfg = load_config(path)
        m = cfg.task.train_data.parser.mosaic
        assert m.shear == mraw['shear']
        assert m.translate == mraw['translate']


def test_eval_random_flip_false_train_flip_true():
    """The eval parser never flips; the train parser does, in every tier YAML."""
    for name in _EXPERIMENTS:
        cfg = load_config(os.path.join(_CFG_DIR, f'{name}.yaml'))
        assert cfg.task.validation_data.parser.random_flip is False, \
            f"{name}: validation_data.parser.random_flip should be false"
        assert cfg.task.train_data.parser.random_flip is True, \
            f"{name}: train_data.parser.random_flip should be true"
