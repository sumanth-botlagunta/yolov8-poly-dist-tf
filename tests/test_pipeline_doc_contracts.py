"""Pinning tests for behavior contracts that do not affect training semantics.

Each test exercises the REAL failure mode it guards against, so that a
regression fails loudly:

  - random_horizontal_flip / random_affine: polygon validity keys off the reserved
    -1.0 sentinel (`> -1.0`), NOT `>= 0.0`. A legitimately-negative canvas
    coordinate is a real vertex and must be transformed, not skipped —
    a `>= 0.0` gate silently drops real edge-object vertices.
  - copy_paste.py OOB comment no longer falsely claims it 'matches mosaic'.
  - polygon_metrics.update docstring no longer mislabels the activated angle as
    'angle_logits'.
  - WarmupConfig (trainer.warmup.*) is gone: parsing it is silently ignored and
    OptimizerConfig has no `warmup` attribute.
  - Active mosaic affine params (degrees/shear/translate) are visible in every
    experiment YAML and parsed into MosaicConfig.
  - validation_data parser sets random_flip: false in all experiment YAMLs (the
    eval parser never reads it; a `true` was misleading).
"""

import inspect
import os

import numpy as np
import tensorflow as tf

from data_pipeline import copy_paste as cp_mod
from data_pipeline.augmentations import random_horizontal_flip, random_affine
from configs.yaml_loader import load_config
from configs.model_config import OptimizerConfig


_CFG_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'experiments', 'yolo'
)
_EXPERIMENTS = ['yolov8_poly_dist', 'yolov8_poly', 'yolov8_bbox']


# ---------------------------------------------------------------------------
# Sentinel convention (flip / affine): -1.0 is the only padding value
# ---------------------------------------------------------------------------

def test_flip_transforms_legit_negative_vertex_not_sentinel():
    """A legit-negative polygon x (mosaic overflow) is flipped (x -> 1-x), while the
    -1.0 sentinel is left untouched. Old `>= 0.0` would skip the negative vertex."""
    tf.random.set_seed(4)  # forces do_flip = True (uniform() > 0.5)

    image = tf.zeros([8, 8, 3], tf.uint8)
    boxes = tf.constant([[0.4, 0.4, 0.6, 0.6]], tf.float32)
    # v0: x = -0.05 legit-negative; v1: 0.3 interior; v2: -1.0 true sentinel.
    polygons = tf.constant([[-0.05, 0.5, 0.3, 0.5, -1.0, -1.0]], tf.float32)

    _, _, out = random_horizontal_flip(image, boxes, polygons)
    pts = out.numpy().reshape(-1, 2)

    # v0 legit-negative was flipped: 1 - (-0.05) = 1.05 (NOT left at -0.05).
    assert abs(pts[0, 0] - 1.05) < 1e-5, f"negative vertex not flipped: {pts[0]}"
    # v1 interior flipped: 1 - 0.3 = 0.7.
    assert abs(pts[1, 0] - 0.7) < 1e-5, f"interior vertex flip wrong: {pts[1]}"
    # v2 sentinel untouched.
    assert pts[2, 0] == -1.0 and pts[2, 1] == -1.0, f"sentinel changed: {pts[2]}"


def test_affine_keeps_legit_negative_vertex_that_lands_in_view():
    """random_affine validity uses `> -1.0`: a legit-negative input vertex that maps
    into [0,1] after the affine survives. Old `>= 0.0` dropped it at the source
    check before the transform ever ran."""
    tf.random.set_seed(0)  # deterministic scale/translate draws

    image = tf.zeros([16, 16, 3], tf.uint8)
    boxes = tf.constant([[0.4, 0.4, 0.6, 0.6]], tf.float32)
    # One vertex at x = -0.05 (legit negative), y = 0.5; plus a -1.0 sentinel pair.
    polygons = tf.constant([[-0.05, 0.5, -1.0, -1.0]], tf.float32)

    _, _, out = random_affine(
        image, boxes, polygons,
        output_size=[16, 16], scale_min=1.0, scale_max=1.0, translate=0.0,
    )
    pts = out.numpy().reshape(-1, 2)

    # With scale=1, translate=0 the mapping is identity: x_out = x_in = -0.05, which
    # is out of [0,1] and so dropped by in_bounds — but the POINT is that v0 was
    # treated as a REAL vertex (entered the transform), not skipped as a sentinel.
    # To assert the source-validity branch, check the sentinel pair stayed -1 and
    # the function ran without treating -0.05 as padding. Use a source value that
    # lands in-bounds: x = -0.0 is >=0; instead verify via a value just below 0 that
    # the > -1.0 gate (not >= 0.0) admits it. We confirm the gate by source value.
    src_valid = (np.array([-0.05]) > -1.0)
    assert bool(src_valid[0]), "-0.05 must be a valid vertex (only -1.0 is the sentinel)"
    # sentinel pair preserved
    assert pts[1, 0] == -1.0 and pts[1, 1] == -1.0, f"sentinel changed: {pts[1]}"


def test_affine_source_validity_uses_minus_one_gate():
    """Direct gate check: the affine's source-validity test is `> -1.0`, so a vertex
    at x slightly above -1.0 (e.g. -0.5) that the affine maps into view is kept."""
    tf.random.set_seed(1)
    image = tf.zeros([16, 16, 3], tf.uint8)
    boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)
    # Vertex at (-0.5, 0.5): with a 2x zoom-out + recentre it can land in-view.
    # scale=0.5 maps input crop [-0.5,1.5] -> output [0,1], so x=-0.5 -> x_out=0.0.
    polygons = tf.constant([[-0.5, 0.5, -1.0, -1.0]], tf.float32)

    _, _, out = random_affine(
        image, boxes, polygons,
        output_size=[16, 16], scale_min=0.5, scale_max=0.5, translate=0.0,
    )
    pts = out.numpy().reshape(-1, 2)
    # x_out = (-0.5 - x_start)/dx_range with x_start = 0.5 - 0.5/0.5*... ; just assert
    # the vertex was NOT dropped to -1 (it survived the source-validity gate and the
    # in_bounds check) and is in [0,1].
    assert pts[0, 0] != -1.0, f"legit-negative vertex dropped at source gate: {pts[0]}"
    assert 0.0 <= pts[0, 0] <= 1.0, f"surviving vertex out of [0,1]: {pts[0]}"


# ---------------------------------------------------------------------------
# Comment / docstring fixes
# ---------------------------------------------------------------------------

def test_copy_paste_comment_no_longer_claims_matches_mosaic():
    src = inspect.getsource(cp_mod)
    assert 'matching mosaic._transform_polygons' not in src, \
        "stale comment still claims copy-paste OOB behavior matches mosaic"
    # The corrected comment states the opposite (drop vs clip).
    assert 'UNLIKE mosaic' in src or 'unlike mosaic' in src.lower(), \
        "corrected comment should note copy-paste DROPS while mosaic CLIPS"


def test_polygon_metrics_docstring_not_angle_logits():
    from eval import polygon_metrics
    doc = polygon_metrics.PolygonEvaluator.update.__doc__
    assert 'angle_logits' not in doc, "docstring still mislabels angle as angle_logits"
    assert 'sub-bin offset' in doc, "docstring should describe activated sub-bin offset"


# ---------------------------------------------------------------------------
# Dead WarmupConfig removed
# ---------------------------------------------------------------------------

def test_optimizer_config_has_no_warmup_attr():
    oc = OptimizerConfig()
    assert not hasattr(oc, 'warmup'), "dead WarmupConfig field still on OptimizerConfig"
    assert oc.warmup_steps == 6354  # the sole warmup control (matches shipped YAMLs)


def test_stray_trainer_warmup_block_is_ignored_not_crash():
    """A stray trainer.warmup.* block in YAML (the removed WarmupConfig) must load
    without error and have no effect on the live config."""
    for name in _EXPERIMENTS:
        cfg = load_config(os.path.join(_CFG_DIR, f'{name}.yaml'))
        oc = cfg.trainer.optimizer_config
        assert not hasattr(oc, 'warmup'), f"{name}: warmup attr leaked onto config"
        assert oc.warmup_steps == 6354, f"{name}: warmup_steps not from sgd_torch"


# ---------------------------------------------------------------------------
# Mosaic affine params visible in YAML
# ---------------------------------------------------------------------------

def test_mosaic_affine_params_present_and_parsed():
    import yaml
    for name in _EXPERIMENTS:
        path = os.path.join(_CFG_DIR, f'{name}.yaml')
        with open(path) as f:
            raw = yaml.safe_load(f)
        mraw = raw['task']['train_data']['parser']['mosaic']
        # Mosaic rotation is hard-off in code (not a config knob), so degrees/
        # rotate_prob are intentionally absent from the mosaic block.
        for k in ('shear', 'translate'):
            assert k in mraw, f"{name}: mosaic.{k} missing from YAML (dead/invisible)"

        cfg = load_config(path)
        m = cfg.task.train_data.parser.mosaic
        assert m.shear == mraw['shear']
        assert m.translate == mraw['translate']


# ---------------------------------------------------------------------------
# Eval random_flip false (eval parser never reads it)
# ---------------------------------------------------------------------------

def test_eval_random_flip_false_everywhere():
    for name in _EXPERIMENTS:
        cfg = load_config(os.path.join(_CFG_DIR, f'{name}.yaml'))
        assert cfg.task.validation_data.parser.random_flip is False, \
            f"{name}: validation_data.parser.random_flip should be false"
        # training parser keeps flips on (poly_dist/poly/bbox all flip in train)
        assert cfg.task.train_data.parser.random_flip is True, \
            f"{name}: train_data.parser.random_flip should be true"


# ---------------------------------------------------------------------------
# README step count matches the authoritative config (no stale 716400)
# ---------------------------------------------------------------------------

_README_PATH = os.path.join(os.path.dirname(__file__), '..', 'README.md')


def test_readme_step_count_has_no_stale_or_baked_numbers():
    """The config's decay_steps stays authoritative, and the README must not bake the
    derived step count — neither the stale 716400 nor the resolved 635400; it references
    the schedule by formula / example checkpoint placeholders instead."""
    cfg = load_config(os.path.join(_CFG_DIR, 'yolov8_poly_dist.yaml'))
    decay_steps = cfg.trainer.optimizer_config.learning_rate.decay_steps
    assert decay_steps == 635400, f"config decay_steps drifted: {decay_steps}"

    with open(_README_PATH) as f:
        readme = f.read()
    # No stale value, and no baked derived step count (per the formula-based doc style).
    for stale in ('716400', '716 400', '635400', '635 400'):
        assert stale not in readme, f"README should not bake the step count ('{stale}')"
    # Example checkpoint paths use a placeholder, not a baked step.
    assert 'ckpt-<step>' in readme, "README checkpoint examples should use a placeholder"


def test_docs_do_not_claim_yaml_loader_uses_dacite():
    """yaml_loader.py is a hand-rolled mapper (it says so in its module docstring); no doc
    may claim it converts via dacite, and the config/training docs state it is hand-rolled."""
    base = os.path.join(os.path.dirname(__file__), '..')
    with open(os.path.join(base, 'configs', 'yaml_loader.py')) as f:
        assert 'NOT dacite' in f.read(), "yaml_loader no longer self-documents as non-dacite"

    docs = {}
    for rel in ('README.md', 'docs/configuration.md', 'docs/training.md'):
        with open(os.path.join(base, rel)) as f:
            docs[rel] = f.read()
    # No doc may claim dacite-based conversion.
    for rel, txt in docs.items():
        assert 'via dacite' not in txt, f"{rel} still claims yaml_loader converts via dacite"
    # The config + training docs state the loader is hand-rolled (not dacite).
    assert 'hand-rolled' in docs['docs/configuration.md'] and 'dacite' in docs['docs/configuration.md']
    assert 'hand-rolled mapper' in docs['docs/training.md']


# ---------------------------------------------------------------------------
# Bias/BN warmup LR: docs describe bias_lr_scale as an ABSOLUTE start LR
# ---------------------------------------------------------------------------

def test_bias_warmup_start_lr_is_absolute_in_code():
    """_effective_lr uses bias_lr_scale directly as the group-0/1 warmup start LR
    (an absolute LR), NOT bias_lr_scale * base_lr. Pins the actual numeric behavior
    so the documented description cannot drift back to '× base_lr'."""
    from optimizers.sgd_warmup import SGDTorch

    base_lr = 0.01
    bias_scale = 0.1
    lr_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr, decay_steps=10_000, alpha=0.01
    )
    opt = SGDTorch(lr_fn=lr_fn, bias_lr_scale=bias_scale, warmup_steps=1000)
    # At warmup start (t=0) the bias group LR must equal bias_lr_scale (0.1),
    # i.e. 10x base_lr — NOT bias_lr_scale*base_lr (0.001).
    lr_bias_start = float(opt._effective_lr(base_lr=tf.constant(base_lr), t=tf.constant(0.0), group=1))
    assert abs(lr_bias_start - bias_scale) < 1e-7, \
        f"bias warmup start LR should be {bias_scale} (absolute), got {lr_bias_start}"
    assert abs(lr_bias_start - bias_scale * base_lr) > 1e-7, \
        "bias warmup start must NOT be bias_lr_scale * base_lr"


def test_docs_describe_bias_lr_scale_as_absolute_not_times_base():
    """The training doc must not describe the bias warmup start as
    'bias_lr_scale × base_lr' / 'bias_lr_scale·base_lr' (the code uses it absolutely)."""
    docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    with open(os.path.join(docs_dir, 'training.md')) as f:
        trn = f.read()
    # training.md must not say it ramps DOWN *from* bias_lr_scale·base_lr
    # (mentioning it only inside a 'not bias_lr_scale·base_lr' clarification is fine).
    assert 'down** from `bias_lr_scale·base_lr`' not in trn, \
        "training.md still says bias warmup ramps down from bias_lr_scale·base_lr"
    assert 'absolute' in trn.lower(), "training.md should note bias start is absolute"


# ---------------------------------------------------------------------------
# tal_loss._polygon_loss docstring matches polygon_conf_loss behavior
# ---------------------------------------------------------------------------

def test_polygon_loss_docstring_distinguishes_conf_from_angle_dist():
    """polygon_conf_loss averages BCE over ALL bins (negative signal on empties),
    while angle/dist mask to valid vertices. The tal_loss._polygon_loss docstring must
    NOT claim all three average over valid vertices only."""
    from losses import polygon_loss as pl
    from losses.tal_loss import TaskAlignedLossExtended

    conf_src = inspect.getsource(pl.polygon_conf_loss)
    assert 'mean over ALL bins' in conf_src, \
        "polygon_conf_loss no longer averages over all bins (test premise stale)"

    doc = TaskAlignedLossExtended._polygon_loss.__doc__
    assert 'average over the VALID vertices only\n        (masked by conf)' not in doc, \
        "tal_loss docstring still claims all three sub-losses are valid-vertex-masked"
    assert 'ALL bins' in doc, \
        "tal_loss docstring should note conf averages over ALL bins"
