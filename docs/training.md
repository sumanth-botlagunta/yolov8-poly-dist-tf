# Training

Entry point: `scripts/run_train.py`. Configs are dataclasses (`configs/model_config.py`) loaded
from YAML via `dacite` (`configs/yaml_loader.py`). Use the `/train` skill or:

```bash
python scripts/run_train.py \
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
own keys on top (override wins; dicts merge). Example — `yolov8_poly_dist_bf16.yaml` is just:

```yaml
base: yolov8_poly_dist.yaml
runtime:
  mixed_precision_dtype: bfloat16
  enable_xla: true
```

Validation at startup (`run_train.py:_validate_config`) checks invariants such as
`output_poly_size == 360 // angle_step` and that `output_dir` is writable.

## Optimizer & schedule
- `optimizers/sgd_warmup.py:SGDTorch` — SGD + Nesterov momentum (0.937), decoupled weight decay,
  **3 param groups** (BN / bias / weights) with momentum warmup. During warmup the weight group's
  LR ramps **up** from 0 while bias/BN ramp **down** from `bias_lr_scale·base_lr`; after warmup all
  groups use the schedule LR.
- LR: cosine decay, initial 0.01, α=0.01, 716,400 steps; linear warmup (7164 steps).
- `optimizers/ema.py:ExponentialMovingAverage` — dynamic decay `min(0.9999, (1+step)/(10+step))`,
  incremented before the decay is read (matches Ultralytics ModelEMA). EMA weights are swapped in
  for evaluation and swapped back after (`swap_weights`). It asserts the model is fully built when
  constructed (shadow/variable counts must match).

## Training loop — `train/`
- `task.py:YoloV8Task` — builds the model, computes loss, runs train/validation steps, owns the
  COCO/distance/polygon evaluators.
- `trainer.py:YoloV8Trainer` — custom loop (not Orbit) so it can: swap EMA weights around
  validation, merge the zipped detection+distance stream, save checkpoints at epoch end, handle
  SIGTERM for preemption, and auto-resume from the latest checkpoint in `output_dir`.
- `viz_utils.py` — renders box/polygon overlays for TensorBoard image summaries.

Checkpoints are written to `output_dir/` every `checkpoint_interval` steps (defaults to one epoch);
TensorBoard events to `output_dir/tb_events/`.

## Mixed precision & XLA
Applied in `run_train.py:_apply_runtime_config` from `RuntimeConfig`: XLA via
`tf.config.optimizer.set_jit`, mixed precision via the global Keras policy. Default is `float32`.
Prefer `bfloat16` (no loss scaling) over `float16`. Validate on a few hundred steps (loss finite,
curves tracking the float32 baseline) before a full run; benchmark with `/benchmark`.

## Distributed training
`distribution_strategy: mirrored` shards the global batch across replicas (per-replica batch =
`global_batch_size / num_replicas`). The model is built inside `strategy.scope()`.

## Derived fields
`steps_per_loop`, `train_steps`, and `validation_steps` are computed from
`train_total_examples` / `validation_total_examples` and batch sizes — don't hand-edit them.

## File logging
`train.log` is written automatically to `output_dir/train.log`. It captures the same output
as stdout (absl-logging). If a run is killed and resumed the log is appended to, not overwritten.

## Resume from a specific checkpoint
By default the trainer auto-resumes from the latest checkpoint in `output_dir`. To start from
a specific step (e.g. after manually selecting the best non-final checkpoint):

```bash
python scripts/run_train.py \
    --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output_dir /path/to/output \
    --resume_from /path/to/output/ckpt-STEP
```

## Augmentation TensorBoard samples
Augmented training images are logged every epoch under the tag `train/augmentations` in
TensorBoard. Each panel shows a mosaic of the first batch after augmentation with ground-truth
boxes and polygon overlays rendered by `train/viz_utils.py`.

## Polygon sub-loss metrics
The three polygon loss components are logged separately:
- `train/poly_angle_loss` — angle bin BCE (averaged over 24 bins per anchor)
- `train/poly_dist_loss`  — radial distance L1
- `train/poly_conf_loss`  — vertex confidence BCE

These are useful for diagnosing which polygon component is not converging.

## Continuous evaluation
`tools/continuous_eval.py` watches an `output_dir` for new checkpoints and evaluates each one,
appending results to `eval_log.jsonl`. Useful for monitoring a long training run without manual
intervention:

```bash
python tools/continuous_eval.py \
    --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
    --watch_dir /path/to/output \
    --interval 300
```
