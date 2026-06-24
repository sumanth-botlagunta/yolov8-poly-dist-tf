# Training

Entry point: `scripts/run_train.py` (for long runs, prefer `tools/train_supervisor.sh` — see
[scripts.md](scripts.md)). Configs are dataclasses (`configs/model_config.py`) loaded from YAML
by the hand-rolled mapper in `configs/yaml_loader.py` (not dacite); see
[configuration.md](configuration.md). Run:

```bash
python -m scripts.run_train \
    --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output_dir /path/to/output
```

## Configs

Three experiment tiers under `configs/experiments/yolo/` share the same code:

| Config | Heads |
|--------|-------|
| `yolov8_bbox.yaml` | box + cls |
| `yolov8_poly.yaml` | box + cls + polygon |
| `yolov8_poly_dist.yaml` | all 6 heads |

### Config inheritance (`base:`)
A config may set a top-level `base: <relative path>` to inherit from another and deep-merge its
own keys on top (override wins; dicts merge). Example — `yolov8_poly_dist_bf16.yaml` is now just
an XLA A/B overlay (the base config already runs `mixed_bfloat16`):

```yaml
base: yolov8_poly_dist.yaml
runtime:
  enable_xla: true
```

Validation at startup (`run_train.py:_validate_config`) checks invariants such as
`output_poly_size == 360 // angle_step` and that `output_dir` is writable.

## Optimizer & schedule

The optimizer and LR schedule are **config-selectable** (`optimizer.type` / `learning_rate.type`,
registry in `optimizers/factory.py`); the defaults below (`sgd` / `cosine`; `sgd_torch` is an
accepted alias for `sgd`) are the historical path and are reproduced byte-identically.
Alternatives: optimizers `adamw` / `adam`;
schedules `linear` / `step` / `polynomial` / `constant`; an optional linear LR-warmup wrapper.
See [configuration.md](configuration.md) for the fields.

- `optimizers/sgd_warmup.py:SGDTorch` — SGD + Nesterov momentum (0.937), decoupled weight decay,
  **3 param groups** (BN / bias / weights) with momentum warmup. During warmup the weight group's
  LR ramps **up** from 0 while bias/BN ramp **down** from `bias_lr_scale` (an absolute LR,
  default `0.1` = 10× the initial weight LR, not `bias_lr_scale·base_lr`); after warmup all
  groups use the schedule LR.
- LR: cosine decay, initial 0.01, α=0.01, over `decay_steps` (= `steps_per_loop × train_epochs`);
  linear warmup over `warmup_steps`.
- `optimizers/ema.py:ExponentialMovingAverage` — dynamic decay `min(0.9999, (1+step)/(10+step))`,
  incremented before the decay is read (matches Ultralytics ModelEMA). EMA weights are swapped in
  for evaluation (`swap_in`) and swapped back after (`swap_out`). It asserts the model is fully built when
  constructed (shadow/variable counts must match).

## Training loop — `train/`
- `task.py:YoloV8Task` — builds the model, computes loss, runs train/validation steps, owns the
  COCO/distance/polygon evaluators.
- `trainer.py:YoloV8Trainer` — custom loop (not Orbit) so it can: swap EMA weights around
  validation, merge the zipped detection+distance stream, save checkpoints at epoch end, handle
  SIGTERM for preemption, and auto-resume from the newest checkpoint across both `output_dir/`
  (epoch-boundary saves) and `output_dir/resume/` (mid-epoch interruption saves, rotated, max 2);
  whichever has the higher global step wins. It also drives a live progress bar
  (`tools/shared/progress.py`), appends each validation to `<run>/val_history.jsonl`, and — when
  `mosaic.close_mosaic_epochs > 0` — rebuilds a mosaic-free training stream for the final N epochs
  (`_maybe_close_mosaic`, Ultralytics close_mosaic).
- `viz_utils.py` — renders box/polygon overlays for TensorBoard image summaries.

**Epoch accounting**: when `steps_per_loop > 0` (the normal case — computed as
`train_total_examples // batch_size`), every epoch runs exactly that many steps from one
**persistent iterator** over the infinite training stream. After a mid-epoch resume
(`YoloV8Trainer._steps_for_epoch`) only the remainder to the next multiple is run, keeping epoch
boundaries at exact multiples of `steps_per_loop`. The derived fields are consistent by
construction: `decay_steps = steps_per_loop × train_epochs`, `checkpoint_interval = steps_per_loop`
(one epoch), and warmup is a small multiple of `steps_per_loop`.
`run_train.py:_validate_config` warns at startup if `decay_steps` in the YAML diverges from
`steps_per_loop × train_epochs`. When `steps_per_loop == 0`
(synthetic/test configs with no example count configured) the loop falls back to data-driven epochs.

Checkpoints are written to `output_dir/` every `checkpoint_interval` steps (defaults to one epoch);
TensorBoard events to `output_dir/tb_events/`.

## Mixed precision & XLA
Applied in `run_train.py:_apply_runtime_config` from `RuntimeConfig`: XLA via
`tf.config.optimizer.set_jit`, mixed precision via the global Keras policy, and
`inter_op_threads`/`intra_op_threads` (applied before any op). **`yolov8_poly_dist.yaml` now
defaults to `mixed_bfloat16`** (prediction heads are pinned float32 in `models/head.py`; loss
runs float32; no loss scaling needed). `yolov8_poly_dist_bf16.yaml` is now just an
`enable_xla: true` A/B overlay on top of the base config. Prefer `bfloat16` over `float16`.
Validate on a few hundred steps (loss finite, curves tracking a float32 baseline) before a full
run; benchmark with `/benchmark`.

## Distributed training (multi-GPU)
`MirroredStrategy` data-parallel training is supported and **numerically identical** to
single-device. `yolov8_poly_dist.yaml` defaults to `distribution_strategy: one_device` with
`num_gpus: 1` (avoids MirroredStrategy variable-wrapping overhead on a single GPU). To use
multiple GPUs switch to `MirroredStrategy`; pass `--debug` or a custom strategy to override.

How it stays correct:
- The model + optimizer are built inside `strategy.scope()`, and optimizer momentum slots are
  pre-created there (`optimizer.build(...)`) — variables cannot be created inside `strategy.run`.
- The merged stream is built at the **global** batch size and split per-replica via
  `experimental_distribute_dataset` (true data parallelism — each global batch is sliced across
  replicas). Keep `global_batch_size` divisible by the replica count.
- The train step is dispatched with `strategy.run`; per-replica losses are `SUM`-reduced for
  logging.
- The loss normalizers (`num_objs`, `target_scores_sum`) are **all-reduced to global counts**
  (`losses/tal_loss.py:_replica_sum`) and `SGDTorch` **all-reduces gradients** across replicas
  (`_all_reduce_gradients`). Both are no-ops under a single replica, so single-GPU runs are
  byte-for-byte unchanged.

Validation runs on a single replica (reads the primary mirror) — it is not the throughput
bottleneck and this keeps the COCO/distance/polygon aggregation simple. The 2-replica path is
covered by `tests/integration/test_multigpu.py` (run in a fresh process; it splits a CPU into two
logical devices).

## Derived fields
`steps_per_loop`, `train_steps`, and `validation_steps` are computed from
`train_total_examples` / `validation_total_examples` and batch sizes — don't hand-edit them:
`steps_per_loop = train_total_examples // global_batch_size`,
`train_steps = steps_per_loop × train_epochs`, `checkpoint_interval = steps_per_loop` (one
epoch), and warmup is a small multiple of `steps_per_loop`. The resolved values for your config
appear in the startup banner logged by `YoloV8Trainer._log_startup_info`.
`decay_steps` is an explicit YAML field (not derived); `run_train.py` warns if it
diverges from `train_steps` so schedule drift is caught at startup.

## File logging
`train.log` is written automatically to `output_dir/train.log`. It captures the same output
as stdout (absl-logging). If a run is killed and resumed the log is appended to, not overwritten.

## Resume from a specific checkpoint
By default the trainer auto-resumes from the newest checkpoint across both `output_dir/`
(epoch-boundary saves) and `output_dir/resume/` (mid-epoch interruption saves, rotated, max 2);
whichever has the higher global step wins. To start from a specific step instead (e.g. after
manually selecting the best non-final checkpoint):

```bash
python -m scripts.run_train \
    --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output_dir /path/to/output \
    --resume_from /path/to/output/ckpt-STEP
```

## Augmentation TensorBoard samples
Augmented training images are logged every epoch under the tag `train/augmentations` in
TensorBoard. Each panel shows a mosaic of the first batch with ground-truth boxes and polygon
overlays rendered by `train/viz_utils.py`. Images are captured **before** the GPU colour
augmentation pass (`batch_color_aug.py`), so they show the geometric/mosaic result in uint8
without HSV jitter or Albumentations applied.

## Polygon sub-loss metrics
The three polygon loss components are logged separately:
- `train/poly_angle_loss` — sub-bin angle-offset BCE (mean over the **valid** vertices per anchor)
- `train/poly_dist_loss`  — radial distance L2 `(softplus(pred) − target)²` (mean over valid vertices)
- `train/poly_conf_loss`  — vertex-validity BCE (mean over **ALL 24 bins**; empty bins get the negative signal)

These are useful for diagnosing which polygon component is not converging.

## TensorBoard structure & tag names
Every scalar is written with a markdown `description` (full name + formula) shown in the
TensorBoard tooltip — the registry lives in `train/metric_meta.py`. Scalars are grouped into
clearly-separated top-level sections so the headline metrics aren't buried under the per-class flood:

| Group | Contents |
|-------|----------|
| `train/` | per-step loss components, `lr`, `momentum`, `ema_decay`, throughput/timing (`step_time_ms`, `data_wait_ms`, `throughput_img_per_s`), and the **debug scalars** below. `train/mean/<k>` = epoch means. |
| `val/` | the **headline** validation metrics only: `mAP`, `mAP50`, `AR100`, `F1score50`, `precision50`, `recall50`, polygon + distance metrics. |
| `per_class/<metric>/<NN_name>` | per-category detection metrics, grouped **by metric** so all classes of one metric (e.g. `per_class/ap50/*`) sit together — out of the `val/` group. |
| `epoch/` | per-epoch timing (`time_s`, `val_time_s`, `eta_s`), `throughput`, `best_<metric>`. |
| `system/` | `gpu_mem_gb`, `gpu_mem_peak_gb`. |

The per-class tag's zero-padded index (`<NN>_<name>`, from `configs/class_map.py:DETECTION_CLASSES`)
keeps TensorBoard's ordering numeric while the class name makes it readable; fill in real names in
`DETECTION_CLASSES` and they propagate to the tags and the image-overlay labels. Image summaries
are `train/augmentations`, `val/predictions`, `val/ground_truth`.

### Debugging scalars (`train/`)

| Scalar | What it tells you |
|--------|-------------------|
| `grad_norm` | Global L2 norm of gradients **before** clipping. Spikes = unstable step / bad batch; compare vs `gradient_clip_norm` to see if clipping is engaging. |
| `weight_norm` | Global L2 norm of trainable weights. Steady climb vs plateau shows whether weight decay is balancing growth. |
| `update_ratio` | `lr·‖grad‖ / ‖weights‖` — the per-step relative update size (Karpathy). Healthy ≈ **1e-3**; ≫ that = LR too high, ≪ = too low / stuck. **The single best "is my LR right" signal.** |
| `lr_bias` / `lr_weight` | Per-param-group effective LR (SGDTorch). During warmup `lr_bias` ramps **down** from `bias_lr_scale` and `lr_weight` ramps **up** from 0 — so you can SEE the warmup happening; both flatten to the schedule LR after. |
| `data_wait_ms` | Time the loop blocked waiting for the next batch. High = input-bound (GPU starved); see [data_pipeline.md](../data_pipeline.md). |
`train/throughput_img_per_s` uses wall-clock time (compute + data wait) over the merged
batch size (144). Together these allow diagnosing whether the bottleneck is in tf.data or on
the GPU.

## Continuous evaluation
`tools/eval.py --watch` polls an `output_dir` for new checkpoints and evaluates each one,
appending results to `eval_log.jsonl`. Useful for monitoring a long training run without manual
intervention:

```bash
python -m tools.eval --watch \
    --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
    --watch_dir /path/to/output \
    --interval 300
```

To evaluate every checkpoint already in a run directory once (instead of polling), use
`--all` in place of `--watch`.
