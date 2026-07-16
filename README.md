# YOLOv8 Polygon + Distance - TensorFlow

TensorFlow 2.16 reimplementation of YOLOv8 extended with two additional output heads:
- **Polygon segmentation**: PolyYOLO radial format, 24 vertices at 15° intervals
- **Per-object distance estimation**: separate labeled dataset merged at the batch level

The codebase supports three experiment tiers (config-driven, shared code):

| Tier | Heads | Use case |
|------|-------|----------|
| `yolov8_bbox` | box + cls | Detection only |
| `yolov8_poly` | box + cls + polygon | Detection + segmentation |
| `yolov8_poly_dist` | all 6 heads | Detection + segmentation + distance |

Full documentation is in [`docs/`](docs/); see the [index](#documentation) at the bottom.

---

## Architecture

| Property | Value |
|----------|-------|
| Input | 672 × 672 × 3 |
| Backbone | CSPDarkNetV8-S (depth=0.33, width=0.5) |
| FPN levels | P3 / P4 / P5 (strides 8, 16, 32) |
| Heads | box (DFL 64-ch), cls (39-ch), poly_angle / poly_dist / poly_conf (24-ch each), dist (1-ch) |
| Classes | 39 |
| Activation | ReLU (default; configurable: silu/gelu/leaky_relu/mish/hardswish) |
| Optimizer | SGDTorch (default): coupled WD (torch semantics, through momentum), Nesterov, per-param-group, linear momentum warmup; selectable adamw/adam |
| LR schedule | Cosine decay (default), initial 0.01, alpha 0.01, over the full `train_steps` (300 epochs); selectable linear/step/polynomial/constant |
| EMA | Dynamic decay `0.9999 × (1 − exp(−step/2000))` (YOLOv5/v8 ModelEMA ramp) |

---

## Setup

Use the provided Docker image (it bundles CUDA/cuDNN and all dependencies). Inside the
container, clone the repo and install it editable:

```bash
git clone <repo-url>
cd <cloned-dir>
pip install -e .
```

Verify the GPU is visible:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

> All commands below are run from the repo root and use `python -m <module>` so they resolve
> imports whether or not the package is installed.

---

## Training

Training is almost always a long (multi-hour/day) run, so launch it detached through the
supervisor; it keeps training alive across crashes/OOM-kills, auto-resumes from the newest
checkpoint, and survives an SSH disconnect:

```bash
# Start (full model, all 6 heads). Swap the config for the yolov8_poly / yolov8_bbox tiers.
nohup bash train/train_supervisor.sh \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output_dir /path/to/run_dir \
    >> /path/to/run_dir/supervisor.log 2>&1 &

# Watch the live logs
tail -f /path/to/run_dir/supervisor.log
```

- **Stop on purpose:** `touch /path/to/run_dir/STOP`; the supervisor exits after the current
  attempt instead of restarting (or send SIGTERM/Ctrl-C to write a resume checkpoint first).
- **Crash-loop guard:** 5 consecutive exits within 120s abort the supervisor (a real bug, not
  an OOM blip; check `train.log`).
- On a fresh host, run `bash utils/pipeline/cloud_diagnose.sh <config.yaml>` first to measure pipeline
  throughput / CPU throttling before committing to a full run.

To run training in the foreground (short tests / debugging) call the entry point directly:

```bash
python -m train.run_train \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output_dir /path/to/output \
    [--debug]      # eager mode + verbose logging
```

Checkpoints are saved to `output_dir/` every `checkpoint_interval` steps. By default this is
one epoch, derived from your config as `train_total_examples // global_batch_size`, so
checkpoints land on epoch boundaries. Set `checkpoint_interval` in your YAML to override.  
TensorBoard events are written to `output_dir/tb_events/`:

```bash
tensorboard --logdir /path/to/run_dir/tb_events
```

Auto-resume on preemption is automatic: at startup the trainer restores from the newest
checkpoint across `output_dir/` (epoch-boundary saves) and `output_dir/resume/` (mid-epoch
interruption saves, rotated, max 2); whichever has the higher global step wins.

### Mixed precision & XLA

All tiers run in `float32` with XLA off. To opt into `bfloat16` (heads stay pinned to
float32) plus XLA on Tensor-Core GPUs, use the performance variant:

```bash
nohup bash train/train_supervisor.sh \
    --config configs/experiments/yolo/yolov8_poly_dist_bf16.yaml \
    --output_dir /path/to/run_dir >> /path/to/run_dir/supervisor.log 2>&1 &
```

That config is a thin override: it inherits everything from `yolov8_poly_dist.yaml` via a
top-level `base:` key and sets only `mixed_precision_dtype: bfloat16` and
`runtime.enable_xla: true`. `bfloat16` needs no loss scaling (unlike `float16`).
Validate on a few hundred steps (loss finite) before a
full run, and use `python -m utils.pipeline.benchmark_pipeline` to record the throughput delta. Any config
can inherit from another with `base: <relative-path.yaml>`; see
[docs/configuration.md](docs/configuration.md).

---

## Evaluation

```bash
python -m utils.eval \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint /path/to/run_dir/ckpt-<step> \
    --split val --per_category
```

Reports mAP / mAP50 / AR100 / F1score50, polygon and distance metrics, and (with
`--per_category`) a per-class table. During training, each validation appends one full report
to `<run_dir>/val_history.jsonl`; pull any epoch (or the best) back into the ckpt-format
txt/json/csv with `python -m utils.reports.val_history <run_dir> --epoch N` (or `--best`). See
[docs/metrics.md](docs/metrics.md) for what each metric means.

For a per-class confusion matrix (where a class's detections leak into other classes or
background), run `python -m utils.confusion_matrix --config <yaml> --checkpoint <ckpt> --split val`.

---

## Notebooks

`notebooks/` holds three self-contained analysis notebooks (run from the repo root):

- `01_data_pipeline_walkthrough.ipynb` - build the training input pipeline and inspect it stage by stage.
- `02_tensorboard_analysis.ipynb` - post-run analysis of a run directory's TensorBoard scalars and `val_history.jsonl`.
- `03_checkpoint_inspection.ipynb` - load a checkpoint/SavedModel, run inference, and inspect raw head statistics.

---

## Export

Export produces the on-device Qualcomm SNPE/DLC SavedModel, a drop-in replacement for
the deployed device DLC (raw per-head outputs, `[0,255]` input, DFL-decoded boxes, no
in-graph NMS):

```bash
python -m utils.export.export_saved_model \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint /path/to/run_dir/ckpt-<step> \
    --output_dir /path/to/export \
    --input_size 672,416
```

Then convert with the unchanged `snpe-tensorflow-to-dlc → snpe-dlc-quantize → snpe-net-run`
pipeline. The full workflow and the box channel-order contract are in
[docs/device_export.md](docs/device_export.md).

To run the exported SavedModel on a folder of images (it reconstructs deploy-style boxes,
polygons, and distance from the flat heads), use `utils/export/inference_saved_model.py`
(see [docs/guides/inference.md](docs/guides/inference.md)).

---

## Project layout

```
configs/        experiment YAMLs + config dataclasses (configs/model_config.py) + loader
data_pipeline/  multi-TFDS sampling, copy-paste, mosaic, parsers, distance-stream merge
models/         CSPDarkNetV8 backbone, FPN-PAN decoder, 6-head, detection generator
losses/         TAL assigner + box / cls / dfl / polygon / distance losses
optimizers/     SGDTorch (momentum warmup) + EMA
eval/           COCO / polygon / distance evaluators + per-category report + metric metadata
train/          task, custom trainer loop, run_train entry point + supervisor
common/         shared library: viz, ckpt loading, runtime setup, progress, run metadata
utils/          CLIs: eval, confusion_matrix; export/ (savedmodel, decode, inference); reports/; pipeline/
notebooks/      data-pipeline walkthrough, TensorBoard/run analysis, checkpoint inspection
tests/          unit / integration / smoke
```

---

## Documentation

| Topic | Doc |
|-------|-----|
| Architecture - backbone, decoder, heads, polygon formats | [docs/architecture.md](docs/architecture.md) |
| Datasets - required TFDS datasets, schemas, init checkpoint | [docs/datasets.md](docs/datasets.md) |
| Data pipeline - sampling, mosaic, augmentation, polygon encoding | [docs/data_pipeline.md](docs/data_pipeline.md) |
| Metrics - glossary of eval metrics | [docs/metrics.md](docs/metrics.md) |
| Losses - TAL assignment, gains, normalization conventions | [docs/losses.md](docs/losses.md) |
| Training - loop, EMA, epoch accounting, distributed | [docs/training.md](docs/training.md) |
| Configuration - every YAML section/field + invariants | [docs/configuration.md](docs/configuration.md) |
| Scripts & tools - every command, with inputs explained | [docs/scripts.md](docs/scripts.md) |
| On-device SNPE/DLC export | [docs/device_export.md](docs/device_export.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Testing | [docs/testing.md](docs/testing.md) |
