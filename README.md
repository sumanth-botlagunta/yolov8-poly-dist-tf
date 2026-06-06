# YOLOv8 Polygon + Distance — TensorFlow

TensorFlow 2.16 reimplementation of YOLOv8 extended with two additional output heads:
- **Polygon segmentation** — PolyYOLO radial format, 24 vertices at 15° intervals
- **Per-object distance estimation** — separate labeled dataset merged at the batch level

The codebase supports three experiment tiers (config-driven, shared code):

| Tier | Heads | Use case |
|------|-------|----------|
| `yolov8_bbox` | box + cls | Detection only |
| `yolov8_poly` | box + cls + polygon | Detection + segmentation |
| `yolov8_poly_dist` | all 6 heads | Detection + segmentation + distance |

---

## Architecture

| Property | Value |
|----------|-------|
| Input | 672 × 672 × 3 |
| Backbone | CSPDarkNetV8-S (depth=0.33, width=0.5) |
| FPN levels | P3 / P4 / P5 (strides 8, 16, 32) |
| Heads | box (DFL 64-ch), cls (39-ch), poly_angle / poly_dist / poly_conf (24-ch each), dist (1-ch) |
| Classes | 39 |
| Activation | ReLU throughout |
| Optimizer | SGDTorch — decoupled WD, Nesterov, per-param-group, linear momentum warmup |
| LR schedule | Cosine decay, initial 0.01, α=0.01, 716 400 steps |
| EMA | Dynamic decay `min(0.9999, (1+step)/(10+step))` |

---

## Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate yolov8-tf
pip install -e .
```

### pip

```bash
pip install -r requirements.txt
pip install -e .
```

Requires **CUDA 12.5** and **cuDNN 9.1**. Verify GPU visibility:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Training

```bash
# Full model (all 6 heads)
python scripts/run_train.py \
    --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output_dir /path/to/output

# Detection + segmentation only
python scripts/run_train.py \
    --config  configs/experiments/yolo/yolov8_poly.yaml \
    --output_dir /path/to/output

# Detection only (fastest)
python scripts/run_train.py \
    --config  configs/experiments/yolo/yolov8_bbox.yaml \
    --output_dir /path/to/output

# Debug (eager mode, verbose)
python scripts/run_train.py \
    --config  configs/experiments/yolo/yolov8_bbox.yaml \
    --output_dir /tmp/debug_run \
    --debug
```

Checkpoints are saved to `output_dir/` every `checkpoint_interval` steps (default 2388).  
TensorBoard events are written to `output_dir/tb_events/`.

```bash
tensorboard --logdir /path/to/output/tb_events
```

Auto-resume on preemption: the trainer restores from the latest checkpoint in `output_dir` at startup with no extra flags.

---

## Checkpoint Migration

Migrate an old checkpoint (backbone + decoder only) to the new model:

```bash
# Step 1 — inspect variables in the old checkpoint
python tools/checkpoint_migration.py list \
    --ckpt initial_checkpoint_folder/ckpt-920304

# Step 2 — dry-run: show which variables matched / missed
python tools/checkpoint_migration.py map \
    --ckpt initial_checkpoint_folder/ckpt-920304 \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml

# Step 3 — migrate and save
python tools/checkpoint_migration.py migrate \
    --ckpt   initial_checkpoint_folder/ckpt-920304 \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --output /tmp/migrated_ckpt/ckpt \
    --modules backbone decoder
```

Set `task.init_checkpoint` in your YAML to the migrated checkpoint path so `train.py` loads it automatically.

---

## Evaluation

```bash
python tools/eval.py \
    --config      configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint  /path/to/output/ckpt-716400 \
    --split       val \
    --output_json /tmp/results.json
```

Metrics reported: **mAP** (0.50:0.95), **mAP50**, **AR100**, **F1@50**, **dist_MAE**, **dist_RMSE** (meters), **poly_mIoU**, **poly_AP50**.

---

## Export

```bash
# SavedModel (deploy=True, NMS baked in)
python tools/export_saved_model.py \
    --config      configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint  /path/to/output/ckpt-716400 \
    --output_dir  /tmp/saved_model

# Also produce a .tflite file
python tools/export_saved_model.py \
    --config      configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint  /path/to/output/ckpt-716400 \
    --output_dir  /tmp/saved_model \
    --tflite
```

---

## Testing

```bash
# Unit tests only (fast, no TFDS required)
pytest tests/unit/ -v

# Integration tests (no TFDS required, uses synthetic data)
pytest tests/integration/ -v

# Dry smoke tests (no TFDS, validates EMA + optimizer + checkpoint loop)
pytest tests/smoke/test_train_10_steps.py::TestDrySmoke -v

# Real-data smoke test (requires TFDS_DATA_DIR and the TFDS dataset)
TFDS_DATA_DIR=/path/to/tfds pytest -m smoke tests/smoke/ -v

# All non-smoke tests
pytest tests/unit/ tests/integration/ -v

# With coverage
pytest tests/unit/ tests/integration/ --cov=. --cov-report=term-missing
```

---

## Project Structure

```
configs/
  experiments/yolo/        Experiment YAMLs (yolov8_bbox, yolov8_poly, yolov8_poly_dist)
  model_config.py          All config dataclasses
  yaml_loader.py           YAML → ExperimentConfig via dacite
  registry.py              Registry for backbone / decoder / head classes

data_pipeline/
  tfds_decoders.py         PolygonDecoder, ServingBotDetDecoder, CopyPasteDecoder
  input_reader.py          Multi-TFDS weighted sampling + distance stream merge
  copy_paste.py            CopyAndPasteModule (prob=0.2, before Mosaic)
  mosaic.py                4-image Mosaic stitch + MixUp
  augmentations.py         Albumentations via tf.py_function
  yolo_parser.py           V8ParserExtended — polygon → PolyYOLO format
  distance_parser.py       V8DistanceParser

models/
  backbone.py              CSPDarkNetV8 (C2f blocks, SPPF)
  decoder.py               FPN-PAN decoder with C2f stacks
  head.py                  YoloV8Head — 6 branches, smart bias init
  detection_generator.py   YoloV8Layer — DFL decode + class-agnostic NMS
  yolo_v8.py               YoloV8 model + build_yolov8() factory

losses/
  tal_assigner.py          TaskAlignedAssigner (score^0.5 × IoU^6, top-k=10)
  tal_loss.py              TaskAlignedLossExtended (CIoU + DFL + BCE + polygon + distance)
  polygon_loss.py          Angle CE + radial L1 + vertex BCE
  distance_loss.py         L1 on log-scale, sentinel-masked

optimizers/
  sgd_warmup.py            SGDTorch — 3 param groups, decoupled WD, Nesterov, momentum warmup
  ema.py                   ExponentialMovingAverage — dynamic decay, swap_weights toggle

eval/
  coco_metrics.py          COCOEvaluator (mAP, mAP50, AR100, F1@50)
  distance_metrics.py      DistanceEvaluator (MAE, RMSE in meters)
  polygon_metrics.py       PolygonEvaluator (mask IoU via cv2.fillPoly, poly_AP50)

train/
  task.py                  YoloV8Task — build, train_step, validation_step, evaluators
  trainer.py               YoloV8Trainer — custom loop, EMA swap, CheckpointManager, TensorBoard

scripts/
  train.py                 Training entry point (absl-py flags)

tools/
  checkpoint_migration.py  List / map / migrate old checkpoints (fuzzy name matching)
  eval.py                  Standalone evaluation script
  export_saved_model.py    Export SavedModel + optional TFLite

tests/
  unit/                    Per-component unit tests (89 passing)
  integration/             End-to-end pipeline tests (14 passing)
  smoke/                   10-step training loop + @pytest.mark.smoke real-data tests
```

---

## Configuration

All hyperparameters live in the experiment YAML. Override per-run by editing a copy — no code changes needed.

Key sections in `configs/experiments/yolo/yolov8_poly_dist.yaml`:

```yaml
task:
  init_checkpoint: initial_checkpoint_folder/ckpt-920304   # backbone+decoder init
  num_classes: 39

  model:
    input_size: [672, 672, 3]

  losses:
    iou_gain: 7.5
    cls_gain: 0.5
    dfl_gain: 1.5
    dist_gain: 1.0
    poly_dist_gain: 0.45
    poly_angle_gain: 0.4
    poly_conf_gain: 0.2

  train_data:
    tfds_name: "cleaner_polygon2026:2.0.0"
    global_batch_size: 128
    tfds_data_dir: /path/to/tensorflow_datasets

trainer:
  train_epochs: 300
  optimizer_config:
    learning_rate:
      initial_learning_rate: 0.01
      decay_steps: 716400
      alpha: 0.01
    momentum: 0.937
    weight_decay: 0.0005
```

---

## Polygon Format Reference

| Stage | Format | Notes |
|-------|--------|-------|
| TFDS input | `[N, max_vertices+2]` xy normalized, padded with -1 | Raw dataset format |
| PolyYOLO (training GT) | `[N, 72]` = `[dx, dy, conf] × 24` | Origin implicit (cx, cy of box) |
| Prediction output | `[B, max_det, 24, 3]` = `(conf, dist, angle_logit)` | From detection_generator |
| Cartesian (eval) | `[N, max_vertices, 2]` yx denormalized | For mask IoU |

Vertex angles are fixed: `θᵢ = i × 2π/24` for i = 0…23. Distance regressor predicts the radial distance at each angle; angle head selects the dominant bin via softmax.
