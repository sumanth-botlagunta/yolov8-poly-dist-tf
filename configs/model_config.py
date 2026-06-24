"""Configuration dataclasses for the YOLOv8 polygon + distance model.

These mirror the YAML structure in docs/experiment_config.yaml and can be
populated by parsing the YAML or constructed programmatically.

Dataclasses:
    RuntimeConfig
    NormActivationConfig
    BackboneConfig
    DecoderConfig
    HeadConfig
    DetectionGeneratorConfig
    ModelConfig
    LossConfig
    AcslConfig
    MosaicConfig
    ParserConfig
    DataConfig
    DistanceDataConfig
    OptimizerConfig
    EmaConfig
    LrScheduleConfig
    TrainerConfig
    TaskConfig
    ExperimentConfig
"""

import dataclasses
from typing import Dict, List, Optional


@dataclasses.dataclass
class RuntimeConfig:
    distribution_strategy: str = "mirrored"
    num_gpus: int = -1
    mixed_precision_dtype: str = "float32"
    run_eagerly: bool = False
    enable_xla: bool = False
    # CPU thread-pool caps (0 = leave TF defaults). On machines where the
    # process is cgroup-capped to fewer cores than are visible (e.g. 13 of 128),
    # TF's default pools oversubscribe massively and thrash; cap them to the
    # actual quota. Applied in scripts/run_train.py before any TF op runs.
    inter_op_threads: int = 0
    intra_op_threads: int = 0


@dataclasses.dataclass
class NormActivationConfig:
    activation: str = "relu"
    norm_epsilon: float = 0.001
    norm_momentum: float = 0.97
    use_sync_bn: bool = False


@dataclasses.dataclass
class BackboneConfig:
    model_id: str = "cspdarknetv8s"
    min_level: int = 3
    max_level: int = 5
    depth_scale: float = 1.0
    width_scale: float = 1.0
    use_separable_conv: bool = False


@dataclasses.dataclass
class DecoderConfig:
    type: str = "yolo_decoder"
    version: str = "v8"
    model_type: str = "s"
    activation: str = "same"
    use_separable_conv: bool = False


@dataclasses.dataclass
class HeadConfig:
    smart_bias: bool = True


@dataclasses.dataclass
class DetectionGeneratorConfig:
    max_boxes: int = 300
    nms_thresh: float = 0.65
    score_thresh: float = 0.05
    nms_type: str = "greedy"
    pre_nms_points: int = 30000
    min_distance: float = 0.5
    max_distance: float = 10.0


@dataclasses.dataclass
class ModelConfig:
    input_size: List[int] = dataclasses.field(default_factory=lambda: [672, 672, 3])
    num_classes: int = 39
    angle_step: int = 15
    output_poly_size: int = 24
    output_dist_size: int = 1
    num_dist_block: int = 1
    with_polygons: bool = True
    with_distance: bool = True
    deploy: bool = True
    backbone: BackboneConfig = dataclasses.field(default_factory=BackboneConfig)
    decoder: DecoderConfig = dataclasses.field(default_factory=DecoderConfig)
    head: HeadConfig = dataclasses.field(default_factory=HeadConfig)
    detection_generator: DetectionGeneratorConfig = dataclasses.field(
        default_factory=DetectionGeneratorConfig
    )
    norm_activation: NormActivationConfig = dataclasses.field(
        default_factory=NormActivationConfig
    )


@dataclasses.dataclass
class AcslConfig:
    use_acsl: bool = False
    bg_common_ratio: float = 0.38
    bg_frequent_ratio: float = 1.0
    bg_rare_ratio: float = 0.17
    common_cls: List[int] = dataclasses.field(default_factory=list)
    frequent_cls: List[int] = dataclasses.field(default_factory=list)
    rare_cls: List[int] = dataclasses.field(default_factory=list)
    threshold: float = 0.3


@dataclasses.dataclass
class LossConfig:
    iou_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    dist_gain: float = 1.0
    poly_dist_gain: float = 0.45
    poly_conf_gain: float = 0.2
    poly_angle_gain: float = 0.4
    poly_gain: float = 0.5
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    topk: int = 10
    # Box IoU loss variant: ciou (default) | giou | diou | eiou | siou.
    box_iou_type: str = "ciou"
    # Cls loss variant: bce (default) | focal | varifocal; label_smoothing 0 = off.
    cls_loss_type: str = "bce"
    label_smoothing: float = 0.0
    focal_gamma: float = 1.5
    focal_alpha: float = 0.25
    acsl: AcslConfig = dataclasses.field(default_factory=AcslConfig)


@dataclasses.dataclass
class MosaicConfig:
    mosaic_frequency: float = 0.5
    mixup_frequency: float = 0.0
    # 0.25 = half-range of the mosaic split point as a fraction of canvas size.
    # Matches Mosaic.__init__ default + its docstring + the poly_dist tier YAML; the
    # dataclass previously defaulted to 0.2, silently disagreeing with the runtime
    # default whenever a tier YAML omitted mosaic_center.
    mosaic_center: float = 0.25
    # Canvas->output warp scale-gain bounds (stock YOLO [0.5, 1.5]). This is the
    # ONLY source of per-sample size variety now — per-image placement scale is
    # fixed (consistent upright tiles), see data_pipeline/mosaic.py.
    aug_scale_min: float = 0.5
    aug_scale_max: float = 1.5
    mosaic_crop_mode: str = "scale"
    area_thresh: float = 0.5
    jitter: float = 0.0
    # Mosaic image diversity (see data_pipeline/mosaic.py). A group of `group_size`
    # decoded images is mapped to `group_size // decodes_per_output` outputs; each
    # mosaic draws 4 source images from the group. `decodes_per_output` (R) is the
    # number of freshly-decoded images each output consumes AND the data-pipeline
    # decode multiplier: R=4 is stock-YOLO (4 distinct images per mosaic, no reuse);
    # R=1 reuses each image 4× (throughput-neutral). Larger group_size = more varied
    # combinations at the same R. group_size must be a multiple of decodes_per_output
    # and >= 4 (validated in scripts/run_train.py:_validate_config).
    group_size: int = 32
    decodes_per_output: int = 4
    # Full-affine (random_perspective) params applied after mosaic assembly and to
    # non-mosaic single images. degrees/shear in degrees; translate as a fraction of
    # output size; perspective coefficient (0 disables). scale gain uses
    # aug_scale_min/aug_scale_max (scale ∈ [aug_scale_min, aug_scale_max]).
    # Rotation is gated by rotate_prob: a fraction (1 - rotate_prob) of outputs stay
    # upright (angle forced to 0); only rotate_prob of them rotate by ±degrees. shear
    # defaults to 0 (no shear) so the corrected mosaic shows upright panels.
    degrees: float = 10.0
    rotate_prob: float = 0.10
    shear: float = 0.0
    perspective: float = 0.0
    translate: float = 0.1
    # Disable mosaic + mixup for the final N epochs (Ultralytics close_mosaic). 0 = off.
    # The trainer rebuilds the train stream with mosaic_frequency/mixup_frequency=0 once
    # the run reaches (total_epochs - close_mosaic_epochs).
    close_mosaic_epochs: int = 0


@dataclasses.dataclass
class ParserConfig:
    angle_step: int = 15
    max_num_instances: int = 300
    max_vertices: int = 10938
    # If > 0, resample every polygon to this many vertices at decode time so the
    # augmentation pipeline carries [N, 2*resample_points] instead of the raw
    # (huge) stored width. 0 = off. The 24-bin radial target is preserved.
    resample_points: int = 0
    aug_rand_hue: float = 0.015
    aug_rand_saturation: float = 0.7
    aug_rand_brightness: float = 0.4
    aug_rand_translate: float = 0.1
    aug_scale_min: float = 1.0
    aug_scale_max: float = 1.0
    random_flip: bool = True
    letter_box: bool = True
    resize_with_random_method: bool = True
    skip_crowd_during_training: bool = True
    dummy_distance: bool = True
    with_polygons: bool = True
    albumentations_frequency: float = 1.0
    # (Removed: aug_rand_angle / aug_rand_perspective were dead — never forwarded to
    #  V8ParserExtended nor applied. Geometric aug is the mosaic-stage
    #  random_perspective, configured via MosaicConfig.degrees/shear/translate.)
    jitter: float = 0.0
    area_thresh: float = 0.1
    eval_gray_border: bool = False
    # Distance range (for distance parser only)
    min_meter: float = 0.5
    max_meter: float = 10.0
    mosaic: MosaicConfig = dataclasses.field(default_factory=MosaicConfig)


@dataclasses.dataclass
class DistanceDataConfig:
    tfds_name: str = "servingbot_polygon:1.0.1"
    tfds_split: str = "train"
    tfds_data_dir: str = "/home/user/tensorflow_datasets/"
    global_batch_size: int = 16
    ignore_bg: bool = True
    with_distance: bool = True
    with_polygons: bool = False
    drop_remainder: bool = True
    shuffle_buffer_size: int = 200
    parser: ParserConfig = dataclasses.field(default_factory=ParserConfig)


@dataclasses.dataclass
class DataConfig:
    tfds_name: str = "cleaner_polygon2026:2.0.0"
    tfds_split: str = "train"
    tfds_data_dir: str = "/home/user/tensorflow_datasets/"
    global_batch_size: int = 128
    is_training: bool = True
    shuffle_buffer_size: int = 1500
    drop_remainder: bool = True
    tfds_sampling_weights: Optional[List[float]] = None
    prob_copy_n_paste: float = 0.2
    # Defaults are None to match yaml_loader._build_data_config({}) — a bare
    # DataConfig() must behave identically to an empty YAML. Non-None defaults
    # here previously made direct construction silently enable copy-paste
    # (truthy tfds_for_cnp) and seeded shuffling (seed=1000) while the YAML path
    # left them off. The shipped experiment YAMLs set these explicitly.
    tfds_for_cnp: Optional[str] = None
    tfds_for_cnp_split: Optional[str] = None
    seed: Optional[int] = None
    with_polygons: bool = True
    with_distance: bool = False
    class_remap_json_path: Optional[str] = None
    # tf.data private threadpool size for the training pipeline (0 = TF default,
    # which sizes to all VISIBLE cores — set this to the real core quota on
    # cgroup-capped machines).
    private_threadpool_size: int = 0
    parser: ParserConfig = dataclasses.field(default_factory=ParserConfig)
    distance_data: Optional[DistanceDataConfig] = None


@dataclasses.dataclass
class EmaConfig:
    average_decay: float = 0.9999
    dynamic_decay: bool = True


@dataclasses.dataclass
class LrScheduleConfig:
    # type selects the schedule builder (optimizers/factory.py:LR_SCHEDULES). Default
    # 'cosine' = the current tf.keras CosineDecay; alternatives: linear, step,
    # polynomial, constant. New types are additive — the cosine path is unchanged.
    type: str = "cosine"
    initial_learning_rate: float = 0.01
    decay_steps: int = 635400
    alpha: float = 0.01
    # step decay: lr *= gamma every step_size steps. polynomial: decay power.
    step_size: int = 30000
    gamma: float = 0.1
    power: float = 1.0
    # Optional linear LR warmup wrapped around the base schedule (0 = OFF, the default,
    # so cosine/SGD keep using SGDTorch's own momentum+bias warmup unchanged).
    warmup_steps: int = 0
    warmup_init_lr: float = 0.0


@dataclasses.dataclass
class OptimizerConfig:
    # type selects the optimizer builder (optimizers/factory.py:OPTIMIZERS). Default
    # 'sgd' = the current SGDTorch (3 param groups + momentum/bias warmup); alternatives:
    # adamw, adam. The sgd path is unchanged.
    type: str = "sgd"
    momentum: float = 0.937
    momentum_start: float = 0.8
    nesterov: bool = True
    weight_decay: float = 0.0005
    # Adam/AdamW moment coefficients (ignored by SGD).
    beta_1: float = 0.9
    beta_2: float = 0.999
    # SGD momentum/bias warmup length (the legacy nested WarmupConfig was never read and
    # has been removed). NOT the LR warmup — that lives on LrScheduleConfig.warmup_steps.
    warmup_steps: int = 7164
    ema: EmaConfig = dataclasses.field(default_factory=EmaConfig)
    learning_rate: LrScheduleConfig = dataclasses.field(default_factory=LrScheduleConfig)


@dataclasses.dataclass
class TrainerConfig:
    train_epochs: int = 300
    train_total_examples: int = 0
    validation_total_examples: int = 0
    # Derived — computed by _fill_derived_fields in yaml_loader from the above
    train_steps: int = 0
    steps_per_loop: int = 0
    validation_steps: int = 0
    checkpoint_interval: int = 0
    best_checkpoint_eval_metric: str = "F1score50"
    best_checkpoint_metric_comp: str = "higher"
    max_to_keep: int = 300
    # Gradient accumulation: apply the optimizer once every N micro-batches (effective
    # batch = global_batch_size × N). 1 = OFF (default, byte-identical). The LR schedule
    # advances per OPTIMIZER UPDATE (every N steps), so with N>1 set decay_steps in terms
    # of effective steps. Epoch accounting (data passes) is unaffected.
    grad_accum_steps: int = 1
    optimizer_config: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)


@dataclasses.dataclass
class TaskConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    losses: LossConfig = dataclasses.field(default_factory=LossConfig)
    train_data: DataConfig = dataclasses.field(default_factory=DataConfig)
    validation_data: DataConfig = dataclasses.field(default_factory=DataConfig)
    # Fine-tuning: load the FULL model (EMA/deployed weights) from a trained checkpoint
    # into a fresh optimizer/EMA/step (new LR schedule). Takes precedence over
    # init_checkpoint; both are no-ops once the run has its own checkpoints (resume wins).
    finetune_from: Optional[str] = None
    # Freeze whole modules (set trainable=False) — names from {backbone, decoder, head}.
    # Their weights stop updating and their BatchNorm runs in inference mode (frozen
    # running stats). Common with finetune_from: freeze [backbone] (or [backbone, decoder])
    # to adapt only the head. Empty = nothing frozen (default).
    freeze_modules: List[str] = dataclasses.field(default_factory=list)
    init_checkpoint: Optional[str] = None
    init_checkpoint_modules: List[str] = dataclasses.field(
        default_factory=lambda: ["backbone", "decoder"]
    )
    num_classes: int = 39
    with_polygons: bool = True
    with_distance: bool = True
    min_distance: float = 0.5
    max_distance: float = 10.0
    ignore_dontcare: bool = True
    ignore_iscrowds: bool = False
    iscrowds_labels: List[int] = dataclasses.field(
        default_factory=lambda: [6, 13, 24, 36, 37]
    )
    per_category_metrics: bool = False
    gradient_clip_norm: float = 0.0
    smart_bias_lr: float = 0.1
    find_best_score_thresh: bool = True
    summary_types: str = "scalar,image"
    summary_image_num: int = 20
    summary_image_draw_box: bool = True
    summary_image_draw_poly: bool = True


@dataclasses.dataclass
class ExperimentConfig:
    task: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
