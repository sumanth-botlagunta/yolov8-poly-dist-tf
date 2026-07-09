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
  `checkpoint_interval`) are computed by `_fill_derived_fields` from `train_epochs` and
  `train_total_examples` — see [training.md](training.md). `learning_rate.decay_steps` is
  an **explicit YAML value** (not derived); `run_train` warns when it diverges from
  `train_steps`.

## Top-level structure

```
ExperimentConfig
├── runtime            RuntimeConfig          strategy / precision / XLA / thread caps
└── task               TaskConfig
    ├── model          ModelConfig            architecture, heads, detection generator
    ├── losses         LossConfig             TAL gains + ACSL
    ├── train_data     DataConfig             training stream (+ parser, + distance_data)
    └── validation_data DataConfig            eval stream
├── trainer            TrainerConfig          epochs, checkpoints, optimizer
```

## `runtime` — `RuntimeConfig`

Applied in `train/run_train.py:_apply_runtime_config` before any TF op runs.

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
| `num_classes` | `39` | Drives the cls head width and the smart-bias init. |
| `angle_step` | `15` | Polygon angular bin size; `output_poly_size` must equal `360 // angle_step`. |
| `output_poly_size` | `24` | Polygon vertices/bins. **Invariant:** `== 360 // angle_step`. |
| `with_polygons` / `with_distance` | `true` | Toggle the polygon / distance heads (the three tiers). |
| `deploy` | `true` | `true` bakes NMS into the forward pass (eval/export); the trainer sets it `false` for raw head outputs. |
| `backbone` | `cspdarknetv8s` | **model_id takes precedence** over `depth_scale`/`width_scale` (both 1.0 in YAML but the model is `-S`). |
| `detection_generator` | — | `max_boxes=300`, `nms_thresh=0.65`, `score_thresh=0.05`, distance range `[0.5, 10.0]`. |
| `detection_generator.nms_class_mode` | `per_class` | NMS suppression scope: `per_class` runs NMS independently per class; `agnostic` runs ONE NMS over all boxes (suppresses cross-class duplicates at the same location). Eval-time post-processing only. |

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
| `box_iou_type` | `ciou` | Box regression loss: `ciou` (default) · `giou` · `diou` · `eiou` · `siou`. |
| `cls_loss_type` | `bce` | Classification loss: `bce` (default) · `focal` · `varifocal`. |
| `label_smoothing` | `0.0` | Softens BCE targets (0 = off). |
| `focal_gamma` / `focal_alpha` | 1.5 / 0.25 | Focal/varifocal parameters (ignored for `bce`). |
| `acsl.use_acsl` | `false` | Parsed but **not implemented** — `use_acsl: true` raises at startup rather than silently no-op. |

## `task.train_data` / `validation_data` — `DataConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `tfds_name` / `tfds_split` / `tfds_data_dir` | — | The source TFDS dataset. |
| `global_batch_size` | `128` | Detection batch; the distance stream's batch is merged on top, and throughput/logs count the merged total. |
| `tfds_sampling_weights` | `None` | Per-source weights for multi-TFDS weighted sampling. |
| `prob_copy_n_paste` | `0.2` | Copy-paste probability; `tfds_for_cnp` sets the RGBA source dataset. |
| `seed` | `None` | Base seed; the three shuffle stages use `seed`, `seed+1`, `seed+2`. |
| `private_threadpool_size` | `0` | tf.data threadpool (0 = all visible cores). Cap on cgroup-limited hosts. |
| `drop_remainder` | train `true` / val `false` | Validation must keep the final partial batch so every image is scored; `run_train` rejects `true` on the val stream. |
| `parser` | `ParserConfig` | Augmentation + polygon settings (below). |
| `distance_data` | `None` | The merged distance stream (training only). |

### `task.train_data.parser` — `ParserConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `aug_rand_hue` / `_saturation` / `_brightness` | 0.015 / 0.7 / 0.4 | HSV jitter (brightness is **additive**, not multiplicative). |
| `random_flip` | `true` | Horizontal flip. |
| `rotate` / `rotate_degrees` | `false` / `null` | Pre-warp rotation for non-mosaic singles only; the mosaic warp never rotates. |
| `skip_crowd_during_training` | `true` | Drop `is_crowd` GT at parse time. |
| `albumentations_frequency` | `1.0` | Albumentations applied to detection rows only. |
| `mosaic` | `MosaicConfig` | Mosaic + the post-mosaic `random_perspective` affine (below). |

### `task.train_data.parser.mosaic` — `MosaicConfig`

Geometric augmentation lives here; the `random_perspective` affine runs in the mosaic stage
for both mosaic and single images (the parser no longer applies a separate affine).

| Field | Default | Notes |
|-------|---------|-------|
| `mosaic_frequency` | `0.5` | Per-output probability of building a mosaic (vs a single-image warp). |
| `mosaic_center` | `0.25` | Half-range of the 2× canvas split point. |
| `aug_scale_min` / `aug_scale_max` | 0.5 / 1.5 | Canvas→output warp scale-gain bounds (stock YOLO). The ONLY source of per-sample size variety — per-image placement scale is fixed (upright tiles). |
| `shear` / `translate` / `perspective` | 0 / 0.1 / 0 | `random_perspective` strength (degrees; translate as a fraction; perspective 0 disables). |
| `close_mosaic_epochs` | `0` | Disable mosaic + mixup for the final N epochs (Ultralytics close_mosaic; 0 = off). |
| `group_size` | `32` | Mosaic source pool per group. **Invariant:** multiple of `decodes_per_output`, ≥ 4. |
| `decodes_per_output` | `4` | **R** — decodes per emitted sample. 4 = stock YOLO: each mosaic draws 4 distinct images with no cross-output reuse (~4× decode work). R<4 trades diversity for throughput (each image recurs in 4/R outputs) and hurts accuracy; `run_train` warns. See [data_pipeline.md](data_pipeline.md). |

## `trainer` — `TrainerConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `train_epochs` | `300` | One epoch = one nominal pass of training samples. |
| `train_total_examples` | `0` | Used to derive `steps_per_loop = train_total_examples // batch`. |
| `best_checkpoint_eval_metric` | `F1score50` | Metric for the `best_*` checkpoint (`_comp: higher`). |
| `max_to_keep` | `300` | Epoch-boundary checkpoints retained. |
| `grad_accum_steps` | `1` | Gradient accumulation: apply the optimizer once every N micro-batches (**effective batch = `global_batch_size × N`**). `1` = off (byte-identical). With `N>1` the LR schedule advances per *optimizer update* (every N steps), so set `decay_steps` in **effective** steps; epoch accounting (data passes) is unaffected. |
| `optimizer_config` | `OptimizerConfig` | SGD + warmup + cosine + EMA (below). |

### `trainer.optimizer_config` — `OptimizerConfig`

The optimizer and LR schedule are config-selectable via `type` keys (registry in
`optimizers/factory.py`); the `type` selects which nested parameter block is read
(e.g. `optimizer: {type: sgd, sgd: {…}}`). Defaults are `sgd` / `cosine`.

| Field | Default | Notes |
|-------|---------|-------|
| `optimizer.type` | `sgd` | Optimizer: `sgd` (default) · `adamw` · `adam`. Parameters live in the matching nested block (`sgd:` / `adamw:` / `adam:`). |
| `beta_1` / `beta_2` | 0.9 / 0.999 | Adam/AdamW moment coefficients (ignored by SGD). |
| `momentum` / `momentum_start` | 0.937 / 0.8 | Nesterov momentum (warms from start → momentum). |
| `weight_decay` | `0.0005` | Applied to weight tensors (`kernel`) only, not biases/BN — per SGDTorch's three param groups. |
| `warmup_steps` | `≈3 × steps_per_loop` | SGD momentum/bias warmup (≈3 epochs of steps); the weight group's LR ramps UP from 0 while the BN/bias groups ramp DOWN from `smart_bias_lr`. Set explicitly in the experiment YAML for the run's steps/epoch. |
| `learning_rate.type` | `cosine` | Schedule: `cosine` (default) · `linear` · `step` · `polynomial` · `constant`. |
| `learning_rate.initial_learning_rate` / `decay_steps` / `alpha` | 0.01 / `steps_per_loop × epochs` / 0.01 | `decay_steps` should equal `steps_per_loop × epochs`, i.e. `train_steps // grad_accum_steps` (`run_train` warns otherwise). |
| `learning_rate.step_size` / `gamma` / `power` | 30000 / 0.1 / 1.0 | Used by `step` (gamma every step_size) / `polynomial` (power) schedules. |
| `learning_rate.warmup_steps` / `warmup_init_lr` | 0 / 0.0 | Optional linear LR-warmup wrapper (0 = off; SGD keeps its own momentum/bias warmup). |
| `ema.average_decay` / `dynamic_decay` | 0.9999 / `true` | EMA decay `0.9999 × (1 − exp(−step/2000))` (YOLOv5/v8 ModelEMA ramp). EMA weights are swapped in for eval. |

Gradient clipping is `task.gradient_clip_norm` (default `0.0` = off): SGDTorch clips per-call,
keras optimizers (adam/adamw) set `global_clipnorm`. Activation is `task.model.norm_activation.activation`
(`relu` default; `silu`/`swish`, `gelu`, `leaky_relu`, `mish`, `hardswish` all supported).

## `task` checkpoint fields

| Field | Default | Notes |
|-------|---------|-------|
| `finetune_from` | `None` | **Fine-tuning** (same task, new data): load the FULL model from a trained `ckpt-N`, preferring its EMA/deployed weights, into a **fresh optimizer/EMA/step** (new LR schedule). Overrides via `--finetune_from`. Mutually exclusive with `init_checkpoint`. Fresh-start-only (resume ignores it). See [guides/finetuning](guides/finetuning.md). |
| `freeze_modules` | `[]` | Freeze whole modules (`trainable=False`; BN in inference mode) — subset of `{backbone, decoder, head}`. Excluded from grads/optimizer/EMA. At least one module must stay trainable. |
| `freeze_backbone_layers` | `0` | Partial freezing: freeze the **first N** top-level backbone layers (`stem_conv1 … sppf`, 10 total) — the "freeze early layers, fine-tune the rest" recipe. `0` = off. The startup log lists what was frozen. |
| `init_checkpoint` | `None` | **Transfer-init** (new task/head): warm-start source for the selected modules, from a checkpoint **produced by this codebase**. Loaded via the EMA-aware full-model loader (`common/ckpt_loading.py:restore_eval_weights` — EMA shadows preferred); non-selected modules keep their fresh init. Fresh-start-only (resume ignores it). |
| `init_checkpoint_modules` | `[backbone, decoder]` | Which modules `init_checkpoint` warm-starts (head randomly initialized otherwise). For same-task continuation use `finetune_from` instead. |

## Validated invariants (`train/run_train.py:_validate_config`)

Training refuses to start if any of these fail:

- `task.init_checkpoint` (if set) exists (`.index` present).
- `model.output_poly_size == 360 // model.angle_step`.
- `mosaic.group_size >= 4` and `group_size % decodes_per_output == 0`.
- `output_dir` is creatable.
- Each referenced TFDS data dir exists (incl. `distance_data`).

`run_train` additionally **warns** (not fatal) if `learning_rate.decay_steps != train_steps`.
