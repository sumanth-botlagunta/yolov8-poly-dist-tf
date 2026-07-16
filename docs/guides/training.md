# Guide: Training a model end to end

A step-by-step walkthrough of launching, monitoring, and resuming a training run. For *what*
each config field / loss / optimizer is, see the reference docs (linked inline).

## 0. Prerequisites

- Environment set up (`pip install -r requirements.txt`, TF 2.16). See the top-level
  [README](../../README.md#setup).
- The TFDS datasets the config references are built and visible to TFDS — see
  [datasets.md](../datasets.md). For the full model these are `cleaner_polygon2026`,
  `field_misrecog2026`, `station_misrecog`, the copy-paste source `cleaner_copy_paste`, and the
  distance stream `servingbot_polygon`.
- (Optional, recommended) the init checkpoint for warm-starting backbone+decoder
  (`task.init_checkpoint` in the YAML). Without it the model trains from scratch.

## 1. Pick the tier (config)

Three configs share one code path — pick by which heads you want:

| Config (`configs/experiments/yolo/`) | Heads | Use |
|---|---|---|
| `yolov8_bbox.yaml` | box + cls | detection only |
| `yolov8_poly.yaml` | + polygon | detection + segmentation |
| `yolov8_poly_dist.yaml` | + distance | all 6 heads (the full model) |

The config is the single source of truth for hyperparameters. Skim
[configuration.md](../configuration.md) for the fields you may want to change (batch size,
epochs, LR, augmentation, optimizer/loss variants).

## 2. Launch

**Recommended — under the supervisor** (auto-restarts across crashes/OOM, resumes from the newest
checkpoint, stops cleanly on a `STOP` file):

```bash
nohup bash train/train_supervisor.sh \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output_dir /path/to/run_dir \
    >> /path/to/run_dir/supervisor.log 2>&1 &
```

**Foreground** (short tests / debugging — `--debug` runs eager + verbose):

```bash
python -m train.run_train \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output_dir /path/to/run_dir [--debug]
```

`train/run_train.py:_validate_config` checks invariants before training starts (e.g.
`output_poly_size == 360 // angle_step`, mosaic `group_size % decodes_per_output == 0`,
and — when `parser.rotate` is on — `parser.rotate_degrees` is a positive number) and **warns**
if `learning_rate.decay_steps != train_steps`.

## 3. What an epoch is

The training stream is infinite; the trainer runs **exactly `steps_per_loop` steps per epoch**,
where `steps_per_loop = train_total_examples // global_batch_size` (derived from the config). So
one epoch = one nominal data pass, and the startup banner / LR schedule / checkpoint interval are
all true by construction. See [training.md](../training.md#epoch-semantics).

## 4. Monitor

- **Live progress bar** — on a terminal you get an Ultralytics-style bar per epoch (loss
  components, img/s, ETA) and a per-batch bar during validation. When stdout is redirected to a
  log file (the supervisor case), it prints clean periodic one-line summaries instead (no
  carriage-return spam).
- **TensorBoard** — `tensorboard --logdir /path/to/run_dir/tb_events`. Scalars include the loss
  components, `train/lr`, `train/step_time_ms`, `train/data_wait_ms` (watch this — high values
  mean input-bound), `train/throughput_img_per_s`, per-category val metrics, and image summaries.
- **Validation history** — each validation appends one line to `<run>/val_history.jsonl`. Inspect
  the trend without TensorBoard:
  ```bash
  python -m utils.reports.val_history /path/to/run_dir --list
  ```
  See the [validation guide](validation.md).

## 5. Checkpoints, resume, and stopping

- Epoch-boundary checkpoints are written to `output_dir/` (kept up to `max_to_keep`);
  mid-epoch interruption saves go to `output_dir/resume/` (rotated, max 2).
- The **best** checkpoint (highest `F1score50` by default) is saved to `output_dir/best_ckpt/`.
- **Resume is automatic** — on restart the trainer restores from the newest checkpoint across both
  locations (higher global step wins). To resume from a specific one: `--resume_from <prefix>`.
- **Stop cleanly** under the supervisor: `touch /path/to/run_dir/STOP`. The current run finishes
  its step/epoch and exits without a restart. (Remove the `STOP` file before relaunching.)
- **Provenance** — at startup the run dir gets `params.yaml` (the full resolved config) and
  `run_metadata.json` (the git commit + dirty flag, the command line, the
  resolved TFDS dataset versions, and the environment). Together they make every checkpoint
  reproducible: `git checkout <commit>`, use `params.yaml`, those dataset versions.

## 6. Close-mosaic (optional knob)

`parser.mosaic.close_mosaic_epochs: N` disables mosaic + mixup for the final N epochs — the
trainer rebuilds a mosaic-free stream at epoch `total - N` (the Ultralytics close_mosaic knob).
Default 0 (off). It is a training-semantics change, so treat it as an experiment on the tier
config, not a default to turn on.

## 7. Common pitfalls

- **`decay_steps` warning at startup** — set `learning_rate.decay_steps` to `steps_per_loop ×
  epochs` so cosine decay reaches its floor exactly at the end.
- **High `train/data_wait_ms`** — the pipeline is input-bound; the GPU is waiting. See
  [data_pipeline.md](../data_pipeline.md) (the `decodes_per_output` / thread-pool knobs).
- **OOM** — lower `global_batch_size`; the supervisor auto-restarts and resumes, but fix the cause.
- **Want a bigger effective batch than fits?** Set `trainer.grad_accum_steps: N` — the optimizer
  applies once every N micro-batches, so the **effective batch = `global_batch_size × N`** at the
  memory cost of one micro-batch. Default `1` (off). Note: with `N>1` the LR schedule advances per
  optimizer update, so set `learning_rate.decay_steps` in *effective* steps. See
  [configuration.md](../configuration.md).
- **bf16** — `mixed_bfloat16` only helps on Ampere+ Tensor-Core GPUs; on older GPUs it can be slower.

## Related
- Reference: [training.md](../training.md) · [configuration.md](../configuration.md) ·
  [losses.md](../losses.md) · [data_pipeline.md](../data_pipeline.md)
- Next: [validation & best checkpoint](validation.md) · [fine-tuning](finetuning.md)
