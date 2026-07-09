"""Load experiment configuration from a YAML file into dataclass trees.

The YAML structure mirrors the experiment YAML (configs/experiments/yolo/*.yaml).
A hand-rolled mapper (NOT dacite) maps the parsed dict onto the typed dataclasses
defined in configs/model_config.py.

Usage:
    from configs.yaml_loader import load_config, load_config_from_dict

    cfg = load_config("configs/experiments/yolo/yolov8_poly_dist.yaml")
    print(cfg.task.model.num_classes)   # 39
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)


def _warn_unknown_keys(
    section: Dict[str, Any],
    recognized: set,
    section_name: str,
    ignored: frozenset = frozenset(),
) -> None:
    """Warn (never raise) about keys the loader does not consume in a section.

    The loader pulls keys with ``.get()``, so a typo (e.g. ``iou_gian`` for
    ``iou_gain``) silently falls back to the default with no signal. This surfaces
    such typos in the high-impact flat sections (loss gains, runtime). ``ignored``
    lists known-vestigial keys that are intentionally unparsed.
    """
    if not isinstance(section, dict):
        return
    unknown = set(section) - set(recognized) - set(ignored)
    if unknown:
        log.warning(
            "Unrecognized key(s) in '%s' config — ignored, default used: %s. "
            "Recognized: %s",
            section_name, sorted(unknown), sorted(recognized),
        )

from configs.model_config import (
    AcslConfig,
    BackboneConfig,
    DataConfig,
    DecoderConfig,
    DetectionGeneratorConfig,
    DistanceDataConfig,
    EmaConfig,
    ExperimentConfig,
    HeadConfig,
    LossConfig,
    LrScheduleConfig,
    ModelConfig,
    MosaicConfig,
    NormActivationConfig,
    OptimizerConfig,
    ParserConfig,
    RuntimeConfig,
    TaskConfig,
    TrainerConfig,
)


def load_config(yaml_path: str | Path) -> ExperimentConfig:
    """Parse *yaml_path* and return a fully-populated ExperimentConfig.

    A config may set a top-level ``base: <relative path>`` to inherit from another
    config; its own keys are then deep-merged on top (override wins). This lets a
    thin variant (e.g. a bf16 runtime override) reuse a full experiment without
    duplicating it. ``base`` may itself chain to another ``base``.
    """
    raw = _load_raw(Path(yaml_path))
    return load_config_from_dict(raw)


def _load_raw(yaml_path: Path) -> Dict[str, Any]:
    """Read a YAML file into a dict, resolving an optional ``base:`` include."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    base_ref = raw.pop("base", None)
    if base_ref is None:
        return raw
    base_raw = _load_raw((yaml_path.parent / base_ref).resolve())
    return _deep_merge(base_raw, raw)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins; dicts merge)."""
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    """Convert a raw YAML dict (experiment-YAML layout) to ExperimentConfig.

    The experiment YAML has a deeply-nested structure (runtime / task / trainer at
    the top level).  We extract the task, trainer, and runtime subtrees and map
    them onto the dataclasses.
    """
    task_cfg    = _build_task_config(raw.get("task", {}))
    trainer_cfg = _build_trainer_config(raw.get("trainer", {}))
    runtime_cfg = _build_runtime_config(raw.get("runtime", {}))

    config = ExperimentConfig(task=task_cfg, trainer=trainer_cfg, runtime=runtime_cfg)
    _fill_derived_fields(config)
    return config


def _fill_derived_fields(config: ExperimentConfig) -> None:
    """Compute steps_per_loop, train_steps, validation_steps from primary YAML values.

    Derived fields are overwritten every time so they stay consistent when
    batch size or epoch count changes without manually updating the YAML.
    Checkpoint_interval defaults to one epoch (steps_per_loop) if not set.
    """
    t  = config.trainer
    td = config.task.train_data
    vd = config.task.validation_data

    if t.train_total_examples > 0 and td.global_batch_size > 0:
        t.steps_per_loop = t.train_total_examples // td.global_batch_size

    if t.steps_per_loop > 0:
        t.train_steps = t.steps_per_loop * t.train_epochs

    if t.validation_total_examples > 0 and vd.global_batch_size > 0:
        t.validation_steps = t.validation_total_examples // vd.global_batch_size

    if t.checkpoint_interval == 0 and t.steps_per_loop > 0:
        t.checkpoint_interval = t.steps_per_loop


# ---------------------------------------------------------------------------
# Private helpers — one per major sub-tree
# ---------------------------------------------------------------------------

_RUNTIME_KEYS = frozenset({
    "distribution_strategy", "num_gpus", "mixed_precision_dtype", "run_eagerly",
    "enable_xla",
    "inter_op_threads", "intra_op_threads",
})

def _build_runtime_config(r: Dict[str, Any]) -> RuntimeConfig:
    _warn_unknown_keys(r, _RUNTIME_KEYS, "runtime")
    return RuntimeConfig(
        distribution_strategy=r.get("distribution_strategy", "mirrored"),
        num_gpus=r.get("num_gpus", -1),
        mixed_precision_dtype=r.get("mixed_precision_dtype", "float32"),
        run_eagerly=r.get("run_eagerly", False),
        enable_xla=r.get("enable_xla", False),
        inter_op_threads=r.get("inter_op_threads", 0),
        intra_op_threads=r.get("intra_op_threads", 0),
    )


def _build_task_config(t: Dict[str, Any]) -> TaskConfig:
    model_cfg      = _build_model_config(t.get("model", {}), t)
    loss_cfg       = _build_loss_config(t.get("losses", {}))
    train_data_cfg = _build_data_config(t.get("train_data", {}))
    val_data_cfg   = _build_data_config(t.get("validation_data", {}))

    return TaskConfig(
        model=model_cfg,
        losses=loss_cfg,
        train_data=train_data_cfg,
        validation_data=val_data_cfg,
        finetune_from=t.get("finetune_from"),
        freeze_modules=t.get("freeze_modules", []),
        freeze_backbone_layers=t.get("freeze_backbone_layers", 0),
        init_checkpoint=t.get("init_checkpoint"),
        init_checkpoint_modules=t.get("init_checkpoint_modules", ["backbone", "decoder"]),
        num_classes=t.get("num_classes", 39),
        with_polygons=t.get("with_polygons", True),
        with_distance=t.get("with_distance", True),
        min_distance=t.get("min_distance", 0.5),
        max_distance=t.get("max_distance", 10.0),
        ignore_dontcare=t.get("ignore_dontcare", True),
        ignore_iscrowds=t.get("ignore_iscrowds", False),
        iscrowds_labels=t.get("iscrowds_labels", [6, 13, 24, 36, 37]),
        per_category_metrics=t.get("per_category_metrics", False),
        gradient_clip_norm=t.get("gradient_clip_norm", 0.0),
        smart_bias_lr=t.get("smart_bias_lr", 0.1),
        find_best_score_thresh=t.get("find_best_score_thresh", True),
        summary_types=t.get("summary_types", "scalar,image"),
        summary_image_num=t.get("summary_image_num", 20),
        summary_image_draw_box=t.get("summary_image_draw_box", True),
        summary_image_draw_poly=t.get("summary_image_draw_poly", True),
    )


def _build_model_config(m: Dict[str, Any], task: Dict[str, Any]) -> ModelConfig:
    backbone_raw  = m.get("backbone", {}).get("darknet", m.get("backbone", {}))
    decoder_outer = m.get("decoder", {})
    decoder_raw   = decoder_outer.get("yolo_decoder", decoder_outer)
    head_raw      = m.get("head", {})
    det_gen_raw   = m.get("detection_generator", {})
    norm_act_raw  = task.get("norm_activation", {})

    backbone_cfg = BackboneConfig(
        model_id=backbone_raw.get("model_id", "cspdarknetv8s"),
        min_level=backbone_raw.get("min_level", 3),
        max_level=backbone_raw.get("max_level", 5),
        depth_scale=backbone_raw.get("depth_scale", 1.0),
        width_scale=backbone_raw.get("width_scale", 1.0),
        use_separable_conv=backbone_raw.get("use_separable_conv", False),
    )

    decoder_cfg = DecoderConfig(
        type=decoder_outer.get("type", "yolo_decoder"),
        version=decoder_raw.get("version", "v8"),
        model_type=decoder_raw.get("type", "s"),
        activation=decoder_raw.get("activation", "same"),
        use_separable_conv=decoder_raw.get("use_separable_conv", False),
    )

    head_cfg    = HeadConfig(smart_bias=head_raw.get("smart_bias", True))
    det_gen_cfg = DetectionGeneratorConfig(
        max_boxes=det_gen_raw.get("max_boxes", 300),
        nms_thresh=det_gen_raw.get("nms_thresh", 0.65),
        score_thresh=det_gen_raw.get("score_thresh", 0.05),
        nms_class_mode=det_gen_raw.get("nms_class_mode", "per_class"),
        # Inference distance clamp shares the task-level range (single source of
        # truth) so a custom range is honoured at inference/export, not just loss.
        min_distance=task.get("min_distance", 0.5),
        max_distance=task.get("max_distance", 10.0),
    )
    norm_act_cfg = NormActivationConfig(
        activation=norm_act_raw.get("activation", "relu"),
        norm_epsilon=norm_act_raw.get("norm_epsilon", 0.001),
        norm_momentum=norm_act_raw.get("norm_momentum", 0.97),
        use_sync_bn=norm_act_raw.get("use_sync_bn", False),
    )

    return ModelConfig(
        input_size=m.get("input_size", task.get("input_size", [672, 672, 3])),
        num_classes=task.get("num_classes", 39),
        angle_step=m.get("angle_step", 15),
        output_poly_size=task.get("output_poly_size", 24),
        output_dist_size=task.get("output_dist_size", 1),
        num_dist_block=task.get("num_dist_block", 1),
        with_polygons=task.get("with_polygons", True),
        with_distance=task.get("with_distance", True),
        deploy=m.get("deploy", True),
        backbone=backbone_cfg,
        decoder=decoder_cfg,
        head=head_cfg,
        detection_generator=det_gen_cfg,
        norm_activation=norm_act_cfg,
    )


_LOSS_KEYS = frozenset({
    "iou_gain", "cls_gain", "dfl_gain", "dist_gain", "poly_dist_gain",
    "poly_conf_gain", "poly_angle_gain", "poly_gain", "tal_alpha", "tal_beta",
    "topk", "acsl",
    "box_iou_type", "cls_loss_type", "label_smoothing", "focal_gamma", "focal_alpha",
})


def _build_loss_config(l: Dict[str, Any]) -> LossConfig:
    _warn_unknown_keys(l, _LOSS_KEYS, "losses")
    acsl_raw = l.get("acsl", {})
    acsl_cfg = AcslConfig(
        use_acsl=acsl_raw.get("use_acsl", False),
        bg_common_ratio=acsl_raw.get("bg_common_ratio", 0.38),
        bg_frequent_ratio=acsl_raw.get("bg_frequent_ratio", 1.0),
        bg_rare_ratio=acsl_raw.get("bg_rare_ratio", 0.17),
        common_cls=acsl_raw.get("common_cls", []),
        frequent_cls=acsl_raw.get("frequent_cls", []),
        rare_cls=acsl_raw.get("rare_cls", []),
        threshold=acsl_raw.get("threshold", 0.3),
    )
    return LossConfig(
        iou_gain=l.get("iou_gain", 7.5),
        cls_gain=l.get("cls_gain", 0.5),
        dfl_gain=l.get("dfl_gain", 1.5),
        dist_gain=l.get("dist_gain", 1.0),
        poly_dist_gain=l.get("poly_dist_gain", 0.45),
        poly_conf_gain=l.get("poly_conf_gain", 0.2),
        poly_angle_gain=l.get("poly_angle_gain", 0.4),
        poly_gain=l.get("poly_gain", 0.5),
        tal_alpha=l.get("tal_alpha", 0.5),
        tal_beta=l.get("tal_beta", 6.0),
        topk=l.get("topk", 10),
        box_iou_type=l.get("box_iou_type", "ciou"),
        cls_loss_type=l.get("cls_loss_type", "bce"),
        label_smoothing=l.get("label_smoothing", 0.0),
        focal_gamma=l.get("focal_gamma", 1.5),
        focal_alpha=l.get("focal_alpha", 0.25),
        acsl=acsl_cfg,
    )


_DATA_KEYS = frozenset({
    "tfds_name", "tfds_split", "tfds_data_dir", "global_batch_size", "is_training",
    "shuffle_buffer_size", "drop_remainder", "tfds_sampling_weights",
    "prob_copy_n_paste", "tfds_for_cnp", "tfds_for_cnp_split", "seed",
    "with_polygons", "with_distance", "class_remap_json_path",
    "private_threadpool_size", "parser", "distance_data",
})

def _build_data_config(d: Dict[str, Any]) -> DataConfig:
    _warn_unknown_keys(d, _DATA_KEYS, "data")
    parser_cfg    = _build_parser_config(d.get("parser", {}))
    dist_data_raw = d.get("distance_data")
    dist_data_cfg: Optional[DistanceDataConfig] = None
    if dist_data_raw:
        dist_data_cfg = _build_distance_data_config(dist_data_raw)

    return DataConfig(
        tfds_name=d.get("tfds_name", "cleaner_polygon2026:2.0.0"),
        tfds_split=d.get("tfds_split", "train"),
        tfds_data_dir=d.get("tfds_data_dir", "/home/user/tensorflow_datasets/"),
        global_batch_size=d.get("global_batch_size", 128),
        is_training=d.get("is_training", True),
        shuffle_buffer_size=d.get("shuffle_buffer_size", 1500),
        drop_remainder=d.get("drop_remainder", True),
        tfds_sampling_weights=d.get("tfds_sampling_weights"),
        prob_copy_n_paste=d.get("prob_copy_n_paste", 0.2),
        tfds_for_cnp=d.get("tfds_for_cnp"),
        tfds_for_cnp_split=d.get("tfds_for_cnp_split"),
        seed=d.get("seed"),
        with_polygons=d.get("with_polygons", True),
        with_distance=d.get("with_distance", False),
        class_remap_json_path=d.get("class_remap_json_path"),
        private_threadpool_size=d.get("private_threadpool_size", 0),
        parser=parser_cfg,
        distance_data=dist_data_cfg,
    )


def _build_distance_data_config(d: Dict[str, Any]) -> DistanceDataConfig:
    parser_cfg = _build_parser_config(d.get("parser", {}))
    return DistanceDataConfig(
        tfds_name=d.get("tfds_name", "servingbot_polygon:1.0.1"),
        tfds_split=d.get("tfds_split", "train"),
        tfds_data_dir=d.get("tfds_data_dir", "/home/user/tensorflow_datasets/"),
        global_batch_size=d.get("global_batch_size", 16),
        ignore_bg=d.get("ignore_bg", True),
        with_distance=d.get("with_distance", True),
        with_polygons=d.get("with_polygons", False),
        drop_remainder=d.get("drop_remainder", True),
        shuffle_buffer_size=d.get("shuffle_buffer_size", 200),
        parser=parser_cfg,
    )


def _build_parser_config(p: Dict[str, Any]) -> ParserConfig:
    mosaic_raw = p.get("mosaic", {})
    mosaic_cfg = MosaicConfig(
        mosaic_frequency=mosaic_raw.get("mosaic_frequency", 0.5),
        mixup_frequency=mosaic_raw.get("mixup_frequency", 0.0),
        mosaic_center=mosaic_raw.get("mosaic_center", 0.25),  # matches Mosaic.__init__
        aug_scale_min=mosaic_raw.get("aug_scale_min", 0.5),
        aug_scale_max=mosaic_raw.get("aug_scale_max", 1.5),
        tile_scale_min=mosaic_raw.get("tile_scale_min", 0.0),
        tile_scale_max=mosaic_raw.get("tile_scale_max", 0.0),
        mosaic_crop_mode=mosaic_raw.get("mosaic_crop_mode", "scale"),
        area_thresh=mosaic_raw.get("area_thresh", 0.5),
        jitter=mosaic_raw.get("jitter", 0.0),
        group_size=mosaic_raw.get("group_size", 32),
        decodes_per_output=mosaic_raw.get("decodes_per_output", 4),
        degrees=mosaic_raw.get("degrees", 10.0),
        rotate_prob=mosaic_raw.get("rotate_prob", 0.10),
        shear=mosaic_raw.get("shear", 0.0),
        perspective=mosaic_raw.get("perspective", 0.0),
        translate=mosaic_raw.get("translate", 0.1),
        close_mosaic_epochs=mosaic_raw.get("close_mosaic_epochs", 0),
    )
    return ParserConfig(
        angle_step=p.get("angle_step", 15),
        max_num_instances=p.get("max_num_instances", 300),
        max_vertices=p.get("max_vertices", 10938),
        resample_points=p.get("resample_points", 0),
        aug_rand_hue=p.get("aug_rand_hue", 0.015),
        aug_rand_saturation=p.get("aug_rand_saturation", 0.7),
        aug_rand_brightness=p.get("aug_rand_brightness", 0.4),
        aug_rand_translate=p.get("aug_rand_translate", 0.1),
        aug_scale_min=p.get("aug_scale_min", 1.0),
        aug_scale_max=p.get("aug_scale_max", 1.0),
        random_flip=p.get("random_flip", True),
        resize_with_random_method=p.get("resize_with_random_method", True),
        skip_crowd_during_training=p.get("skip_crowd_during_training", True),
        dummy_distance=p.get("dummy_distance", True),
        with_polygons=p.get("with_polygons", True),
        albumentations_frequency=p.get("albumentations_frequency", 1.0),
        # NOTE: aug_rand_angle / aug_rand_perspective were dead config — parsed but
        # never forwarded to V8ParserExtended nor applied (geometry lives in the
        # mosaic-stage random_perspective: degrees/shear/translate). Removed; a stray
        # key in an old YAML is silently ignored.
        jitter=p.get("jitter", 0.0),
        area_thresh=p.get("area_thresh", 0.1),
        eval_gray_border=p.get("eval_gray_border", False),
        min_meter=p.get("min_meter", 0.5),
        max_meter=p.get("max_meter", 10.0),
        mosaic=mosaic_cfg,
    )


_TRAINER_KEYS = frozenset({
    "train_epochs", "train_total_examples", "validation_total_examples",
    "checkpoint_interval", "best_checkpoint_eval_metric",
    "best_checkpoint_metric_comp", "max_to_keep", "optimizer_config",
    "grad_accum_steps", "mid_epoch_resume",
    # steps_per_loop / train_steps / validation_steps are auto-derived but may be
    # present in YAML as documentation; accept silently.
    "steps_per_loop", "train_steps", "validation_steps",
})

def _build_trainer_config(t: Dict[str, Any]) -> TrainerConfig:
    _warn_unknown_keys(t, _TRAINER_KEYS, "trainer")
    opt_raw    = t.get("optimizer_config", {})
    ema_raw    = opt_raw.get("ema", {})
    # The optimizer/schedule TYPE selects which nested param block to read, e.g.
    # `optimizer: {type: adamw, adamw: {...}}`; defaults are 'sgd' / 'cosine'.
    lr_block   = opt_raw.get("learning_rate", {})
    lr_type    = lr_block.get("type", "cosine")
    lr_raw     = lr_block.get(lr_type, lr_block.get("cosine", lr_block))
    opt_block  = opt_raw.get("optimizer", {})
    opt_type   = opt_block.get("type", "sgd")
    sgd_raw    = opt_block.get(opt_type, opt_block)
    # SGD momentum/bias warmup is driven by OptimizerConfig.warmup_steps; LR
    # warmup is LrScheduleConfig.warmup_steps.

    ema_cfg    = EmaConfig(
        average_decay=ema_raw.get("average_decay", 0.9999),
        dynamic_decay=ema_raw.get("dynamic_decay", True),
    )
    lr_cfg     = LrScheduleConfig(
        type=lr_type,
        initial_learning_rate=lr_raw.get("initial_learning_rate", 0.01),
        decay_steps=lr_raw.get("decay_steps", 635400),
        alpha=lr_raw.get("alpha", 0.01),
        step_size=lr_raw.get("step_size", 30000),
        gamma=lr_raw.get("gamma", 0.1),
        power=lr_raw.get("power", 1.0),
        warmup_steps=lr_raw.get("warmup_steps", 0),
        warmup_init_lr=lr_raw.get("warmup_init_lr", 0.0),
    )
    opt_cfg    = OptimizerConfig(
        type=opt_type,
        momentum=sgd_raw.get("momentum", 0.937),
        momentum_start=sgd_raw.get("momentum_start", 0.8),
        nesterov=sgd_raw.get("nesterov", True),
        weight_decay=sgd_raw.get("weight_decay", 0.0005),
        beta_1=sgd_raw.get("beta_1", 0.9),
        beta_2=sgd_raw.get("beta_2", 0.999),
        warmup_steps=sgd_raw.get("warmup_steps", 7164),
        ema=ema_cfg,
        learning_rate=lr_cfg,
    )

    return TrainerConfig(
        train_epochs=t.get("train_epochs", 300),
        train_total_examples=t.get("train_total_examples", 0),
        validation_total_examples=t.get("validation_total_examples", 0),
        checkpoint_interval=t.get("checkpoint_interval", 0),
        best_checkpoint_eval_metric=t.get("best_checkpoint_eval_metric", "F1score50"),
        best_checkpoint_metric_comp=t.get("best_checkpoint_metric_comp", "higher"),
        max_to_keep=t.get("max_to_keep", 300),
        grad_accum_steps=t.get("grad_accum_steps", 1),
        mid_epoch_resume=t.get("mid_epoch_resume", False),
        optimizer_config=opt_cfg,
    )


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    """Recursively convert a dataclass tree to a plain dict (for logging)."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return {
            f.name: config_to_dict(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)
        }
    if isinstance(cfg, list):
        return [config_to_dict(v) for v in cfg]
    return cfg
