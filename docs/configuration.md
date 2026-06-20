# Configuration Reference

Training is driven entirely by an **experiment YAML** (e.g.
`configs/experiments/yolo/yolov8_poly_dist.yaml`). It is the authoritative hyperparameter
source. This page explains how configs load, the section/field layout, and the invariants
checked before training.

## How configs load

`configs/yaml_loader.py:load_config(path)` reads the YAML and maps it onto the dataclasses in
`configs/model_config.py` with a **hand-rolled** mapper (not `dacite`, despite the dependency):

- **Inheritance:** a config may set a top-level `base: <relative-or-abs path>` to inherit from
  another config; its own keys are then deep-merged on top (override wins). The `bf16` variant
  uses this to layer precision settings over the base poly_dist config.
- **Unknown keys** outside the `runtime` / `losses` sections are **silently ignored** — a typo
  in a key name will NOT error; it just won't take effect. Check the dataclass field names here
  when a setting seems to have no effect.
- **Derived fields** (`train_steps`, `steps_per_loop`, `validation_steps`,
  `checkpoint_interval`, `learning_rate.decay_steps`) are computed by `_fill_derived_fields`
  from `train_epochs` and `train_total_examples` — see [training.md](training.md).

## Top-level structure

```
ExperimentConfig
├── runtime            RuntimeConfig          strategy / precision / XLA / thread caps
└── task               TaskConfig
    ├── model          ModelConfig            architecture, heads, detection generator
    ├── losses         LossConfig             TAL gains + ACSL
    ├── train_data     DataConfig             training stream (+ parser, + distance_data)
    ├── validation_data DataConfig            eval stream
    └── trainer        TrainerConfig          epochs, checkpoints, optimizer
```

## `runtime` — `RuntimeConfig`

Applied in `scripts/run_train.py:_apply_runtime_config` before any TF op runs.

| Field | Default | Notes |
|-------|---------|-------|
| `distribution_strategy` | `mirrored` | `one_device` for single GPU (poly_dist uses one_device). |
| `mixed_precision_dtype` | `float32` | `mixed_bfloat16` for the bf16 tier (heads pinned float32, no loss scaling). |
| `enable_xla` | `false` | `tf.config.optimizer.set_jit`. |
| `inter_op_threads` / `intra_op_threads` | `0` | CPU thread-pool caps (0 = TF default). Set to the real core quota on cgroup-capped hosts to avoid oversubscription. |

## `task.model` — `ModelConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `input_size` | `[672, 672, 3]` | Live model input. Smart-bias init assumes this. |
| `num_classes` | `39` | Drives head width AND the checkpoint-migration module rule. |
| `angle_step` | `15` | Polygon angular bin size; `output_poly_size` must equal `360 // angle_step`. |
| `output_poly_size` | `24` | Polygon vertices/bins. **Invariant:** `== 360 // angle_step`. |
| `with_polygons` / `with_distance` | `true` | Toggle the polygon / distance heads (the three tiers). |
| `deploy` | `true` | `true` bakes NMS into the forward pass (eval/export); the trainer sets it `false` for raw head outputs. |
| `backbone` | `cspdarknetv8s` | **model_id takes precedence** over `depth_scale`/`width_scale` (both 1.0 in YAML but the model is `-S`). |
| `detection_generator` | — | `max_boxes=300`, `nms_thresh=0.65`, `score_thresh=0.05`, distance range `[0.5, 10.0]`. |

## `task.losses` — `LossConfig`

Gains and the Task-Aligned assignment parameters. See [losses.md](losses.md) for the
normalization conventions.

| Field | Default | Notes |
|-------|---------|-------|
| `iou_gain` / `cls_gain` / `dfl_gain` | 7.5 / 0.5 / 1.5 | Detection gains (Ultralytics defaults). |
| `dist_gain` | 1.0 | Distance L1 (÷ `num_objs`). |
| `poly_dist_gain` / `poly_angle_gain` / `poly_conf_gain` | 0.45 / 0.4 / 0.2 | Polygon sub-losses. |
| `poly_gain` | 0.5 | Overall multiplier on the summed polygon loss. |
| `tal_alpha` / `tal_beta` / `topk` | 0.5 / 6.0 / 10 | Alignment metric `score^α × IoU^β`, top-k. |
| `acsl.use_acsl` | `false` | Parsed but **not implemented** — `use_acsl: true` raises (design_register entry 11). |

## `task.train_data` / `validation_data` — `DataConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `tfds_name` / `tfds_split` / `tfds_data_dir` | — | The source TFDS dataset. |
| `global_batch_size` | `128` | Detection batch; the distance stream's batch is merged on top, and throughput/logs count the merged total. |
| `tfds_sampling_weights` | `None` | Per-source weights for multi-TFDS weighted sampling. |
| `prob_copy_n_paste` | `0.2` | Copy-paste probability; `tfds_for_cnp` sets the RGBA source dataset. |
| `seed` | `None` | Base seed; the three shuffle stages use `seed`, `seed+1`, `seed+2`. |
| `private_threadpool_size` | `0` | tf.data threadpool (0 = all visible cores). Cap on cgroup-limited hosts. |
| `parser` | `ParserConfig` | Augmentation + polygon settings (below). |
| `distance_data` | `None` | The merged distance stream (training only). |

### `task.train_data.parser` — `ParserConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `resample_points` | `0` | Arc-length resample polygons to this many vertices at decode (poly_dist sets 64). Makes the 24-bin radial target track shapes. |
| `aug_rand_hue` / `_saturation` / `_brightness` | 0.015 / 0.7 / 0.4 | HSV jitter (brightness is **additive** — design_register entry 3). |
| `random_flip` / `letter_box` | `true` | Horizontal flip; aspect-preserving resize. |
| `skip_crowd_during_training` | `true` | Drop `is_crowd` GT at parse time (design_register entry 1). |
| `albumentations_frequency` | `1.0` | Albumentations applied to detection rows only. |
| `mosaic` | `MosaicConfig` | Mosaic + the post-mosaic `random_perspective` affine (below). |

### `task.train_data.parser.mosaic` — `MosaicConfig`

Geometric augmentation lives here; the `random_perspective` affine runs in the mosaic stage
for both mosaic and single images (the parser no longer applies a separate affine).

| Field | Default | Notes |
|-------|---------|-------|
| `mosaic_frequency` | `0.5` | Per-output probability of building a mosaic (vs a single-image warp). |
| `mosaic_center` | `0.25` | Half-range of the 2× canvas split point. |
| `aug_scale_min` / `aug_scale_max` | 0.4 / 1.9 | Per-image scale AND the warp scale-gain bounds (explicit, not symmetric-magnitude). |
| `degrees` / `shear` / `translate` / `perspective` | 10 / 2 / 0.1 / 0 | `random_perspective` strength (degrees; translate as a fraction; perspective 0 disables). |
| `group_size` | `32` | Mosaic source pool per group. **Invariant:** multiple of `decodes_per_output`, ≥ 4. |
| `decodes_per_output` | `4` | **R** — decodes per emitted sample = diversity/throughput knob. **4 = stock-YOLO** (4 distinct images per mosaic, no reuse, ~4× decode); lower = more reuse, less decode (1 = throughput-neutral). See [data_pipeline.md](data_pipeline.md). |

## `task.trainer` — `TrainerConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `train_epochs` | `300` | One epoch = one nominal pass of training samples. |
| `train_total_examples` | `0` | Used to derive `steps_per_loop = train_total_examples // batch`. |
| `best_checkpoint_eval_metric` | `F1score50` | Metric for the `best_*` checkpoint (`_comp: higher`). |
| `max_to_keep` | `300` | Epoch-boundary checkpoints retained. |
| `optimizer_config` | `OptimizerConfig` | SGD + warmup + cosine + EMA (below). |

### `task.trainer.optimizer_config` — `OptimizerConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `momentum` / `momentum_start` | 0.937 / 0.8 | Nesterov momentum (warms from start → momentum). |
| `weight_decay` | `0.0005` | Applied to `weight_keys` (kernel/weight), not biases/BN. |
| `warmup_steps` | `7164` | Linear LR warmup; BN/bias groups ramp DOWN from `smart_bias_lr` (design_register entry 2). |
| `learning_rate.initial_learning_rate` / `decay_steps` / `alpha` | 0.01 / 635400 / 0.01 | Cosine decay. `decay_steps` should equal `steps_per_loop × epochs` (`run_train` warns otherwise). |
| `ema.average_decay` / `dynamic_decay` | 0.9999 / `true` | EMA `min(0.9999, (1+step)/(10+step))`. EMA weights are swapped in for eval. |

## `task` checkpoint fields

| Field | Default | Notes |
|-------|---------|-------|
| `init_checkpoint` | `None` | Warm-start source. Auto-detected: legacy → frozen/structural; **this codebase's own checkpoint → native** (loads complete EMA weights). See [checkpoint_migration](../tools/checkpoint_migration.py). |
| `init_checkpoint_modules` | `[backbone, decoder]` | Which modules to warm-start. Add `head` to also transfer the (same-class) head from a same-codebase checkpoint. |

## Validated invariants (`scripts/run_train.py:_validate_config`)

Training refuses to start if any of these fail:

- `task.init_checkpoint` (if set) exists (`.index` present).
- `model.output_poly_size == 360 // model.angle_step`.
- `mosaic.group_size >= 4` and `group_size % decodes_per_output == 0`.
- `output_dir` is creatable.
- Each referenced TFDS data dir exists (incl. `distance_data`).

`run_train` additionally **warns** (not fatal) if `learning_rate.decay_steps != train_steps`.
