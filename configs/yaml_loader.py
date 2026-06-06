"""Load experiment configuration from a YAML file into dataclass trees.

The YAML structure mirrors docs/experiment_config.yaml.  We use dacite to map
the parsed dict onto the typed dataclasses defined in configs/model_config.py.

Usage:
    from configs.yaml_loader import load_config, load_config_from_dict

    cfg = load_config("configs/experiments/yolo/yolov8_poly_dist.yaml")
    print(cfg.task.model.num_classes)   # 39
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import dacite
import yaml

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
    TaskConfig,
    TrainerConfig,
    WarmupConfig,
)

_DACITE_CONFIG = dacite.Config(
    strict=False,
    cast=[int, float, bool, str],
)


def load_config(yaml_path: str | Path) -> ExperimentConfig:
    """Parse *yaml_path* and return a fully-populated ExperimentConfig."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    return load_config_from_dict(raw)


def load_config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    """Convert a raw YAML dict (from docs/experiment_config.yaml layout) to
    ExperimentConfig.

    The YAML produced by docs/experiment_config.yaml has a deeply-nested
    structure (runtime / task / trainer at the top level).  We extract the
    task and trainer subtrees and map them onto the dataclasses.
    """
    task_raw = raw.get("task", {})
    trainer_raw = raw.get("trainer", {})

    task_cfg = _build_task_config(task_raw)
    trainer_cfg = _build_trainer_config(trainer_raw)

    return ExperimentConfig(task=task_cfg, trainer=trainer_cfg)


# ---------------------------------------------------------------------------
# Private helpers — one per major sub-tree
# ---------------------------------------------------------------------------

def _build_task_config(t: Dict[str, Any]) -> TaskConfig:
    model_cfg = _build_model_config(t.get("model", {}), t)
    loss_cfg = _build_loss_config(t.get("losses", {}))
    train_data_cfg = _build_data_config(t.get("train_data", {}))
    val_data_cfg = _build_data_config(t.get("validation_data", {}))

    return TaskConfig(
        model=model_cfg,
        losses=loss_cfg,
        train_data=train_data_cfg,
        validation_data=val_data_cfg,
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
    )


def _build_model_config(m: Dict[str, Any], task: Dict[str, Any]) -> ModelConfig:
    backbone_raw = m.get("backbone", {}).get("darknet", m.get("backbone", {}))
    decoder_outer = m.get("decoder", {})
    decoder_raw = decoder_outer.get("yolo_decoder", decoder_outer)
    head_raw = m.get("head", {})
    det_gen_raw = m.get("detection_generator", {})
    norm_act_raw = task.get("norm_activation", {})

    backbone_cfg = BackboneConfig(
        model_id=backbone_raw.get("model_id", "cspdarknetv8s"),
        min_level=backbone_raw.get("min_level", 3),
        max_level=backbone_raw.get("max_level", 5),
        depth_scale=backbone_raw.get("depth_scale", 1.0),
        width_scale=backbone_raw.get("width_scale", 1.0),
        dilate=backbone_raw.get("dilate", False),
        use_reorg_input=backbone_raw.get("use_reorg_input", False),
        use_separable_conv=backbone_raw.get("use_separable_conv", False),
    )

    decoder_cfg = DecoderConfig(
        type=decoder_outer.get("type", "yolo_decoder"),
        version=decoder_raw.get("version", "v8"),
        model_type=decoder_raw.get("type", "s"),
        activation=decoder_raw.get("activation", "same"),
        use_separable_conv=decoder_raw.get("use_separable_conv", False),
    )

    head_cfg = HeadConfig(smart_bias=head_raw.get("smart_bias", True))

    det_gen_cfg = DetectionGeneratorConfig(
        max_boxes=det_gen_raw.get("max_boxes", 300),
        nms_thresh=det_gen_raw.get("nms_thresh", 0.65),
        iou_thresh=det_gen_raw.get("iou_thresh", 0.001),
        nms_type=det_gen_raw.get("nms_type", "greedy"),
        pre_nms_points=det_gen_raw.get("pre_nms_points", 30000),
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


def _build_loss_config(l: Dict[str, Any]) -> LossConfig:
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
        acsl=acsl_cfg,
    )


def _build_data_config(d: Dict[str, Any]) -> DataConfig:
    parser_cfg = _build_parser_config(d.get("parser", {}))
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
        tfds_sampling_weights=d.get("tfds_sampling_weights"),
        prob_copy_n_paste=d.get("prob_copy_n_paste", 0.2),
        tfds_for_cnp=d.get("tfds_for_cnp"),
        tfds_for_cnp_split=d.get("tfds_for_cnp_split"),
        seed=d.get("seed"),
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
        parser=parser_cfg,
    )


def _build_parser_config(p: Dict[str, Any]) -> ParserConfig:
    mosaic_raw = p.get("mosaic", {})
    mosaic_cfg = MosaicConfig(
        mosaic_frequency=mosaic_raw.get("mosaic_frequency", 0.5),
        mixup_frequency=mosaic_raw.get("mixup_frequency", 0.0),
        mosaic_center=mosaic_raw.get("mosaic_center", 0.2),
        aug_scale_min=mosaic_raw.get("aug_scale_min", 0.4),
        aug_scale_max=mosaic_raw.get("aug_scale_max", 1.9),
        mosaic_crop_mode=mosaic_raw.get("mosaic_crop_mode", "scale"),
        area_thresh=mosaic_raw.get("area_thresh", 0.5),
        jitter=mosaic_raw.get("jitter", 0.0),
    )
    return ParserConfig(
        angle_step=p.get("angle_step", 15),
        max_num_instances=p.get("max_num_instances", 300),
        max_vertices=p.get("max_vertices", 10938),
        aug_rand_hue=p.get("aug_rand_hue", 0.015),
        aug_rand_saturation=p.get("aug_rand_saturation", 0.7),
        aug_rand_brightness=p.get("aug_rand_brightness", 0.4),
        aug_rand_translate=p.get("aug_rand_translate", 0.1),
        aug_scale_min=p.get("aug_scale_min", 1.0),
        aug_scale_max=p.get("aug_scale_max", 1.0),
        random_flip=p.get("random_flip", True),
        letter_box=p.get("letter_box", True),
        resize_with_random_method=p.get("resize_with_random_method", True),
        skip_crowd_during_training=p.get("skip_crowd_during_training", True),
        dummy_distance=p.get("dummy_distance", True),
        with_polygons=p.get("with_polygons", True),
        albumentations_frequency=p.get("albumentations_frequency", 1.0),
        mosaic=mosaic_cfg,
    )


def _build_trainer_config(t: Dict[str, Any]) -> TrainerConfig:
    opt_raw = t.get("optimizer_config", {})
    ema_raw = opt_raw.get("ema", {})
    lr_raw = opt_raw.get("learning_rate", {}).get(
        "cosine", opt_raw.get("learning_rate", {})
    )
    sgd_raw = opt_raw.get("optimizer", {}).get(
        "sgd_torch", opt_raw.get("optimizer", {})
    )
    warmup_raw = t.get("warmup", {}).get("linear", t.get("warmup", {}))

    ema_cfg = EmaConfig(
        average_decay=ema_raw.get("average_decay", 0.9999),
        dynamic_decay=ema_raw.get("dynamic_decay", True),
    )
    lr_cfg = LrScheduleConfig(
        initial_learning_rate=lr_raw.get("initial_learning_rate", 0.01),
        decay_steps=lr_raw.get("decay_steps", 716400),
        alpha=lr_raw.get("alpha", 0.01),
    )
    warmup_cfg = WarmupConfig(
        warmup_steps=warmup_raw.get("warmup_steps", 7164),
        warmup_learning_rate=warmup_raw.get("warmup_learning_rate", 0.0),
    )
    opt_cfg = OptimizerConfig(
        momentum=sgd_raw.get("momentum", 0.937),
        momentum_start=sgd_raw.get("momentum_start", 0.8),
        nesterov=sgd_raw.get("nesterov", True),
        weight_decay=sgd_raw.get("weight_decay", 0.0005),
        warmup_steps=sgd_raw.get("warmup_steps", 7164),
        ema=ema_cfg,
        learning_rate=lr_cfg,
        warmup=warmup_cfg,
    )

    return TrainerConfig(
        train_epochs=t.get("train_epochs", 300),
        train_steps=t.get("train_steps", 716400),
        steps_per_loop=t.get("steps_per_loop", 2388),
        checkpoint_interval=t.get("checkpoint_interval", 2388),
        validation_interval=t.get("validation_interval", 2388),
        best_checkpoint_eval_metric=t.get("best_checkpoint_eval_metric", "F1score50"),
        best_checkpoint_metric_comp=t.get("best_checkpoint_metric_comp", "higher"),
        max_to_keep=t.get("max_to_keep", 300),
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
