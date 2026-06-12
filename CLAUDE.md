# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Implemented and actively trained.** This is a working TensorFlow 2.16 codebase, not a
plan. Source lives under `data_pipeline/`, `models/`, `losses/`, `optimizers/`, `eval/`,
`train/`, `configs/`, `scripts/`, and `tools/`, with a `tests/` suite and notebooks.

The authoritative hyperparameter reference is the experiment YAML you train with, e.g.
`configs/experiments/yolo/yolov8_poly_dist.yaml`. Developer docs live in `docs/`
(architecture, data pipeline, losses, training, testing). The older
`docs/implementation_plan.md` / top-level `MASTER_PLAN.md` references are obsolete and gone.

**Before flagging a surprising behavior as a bug, read `docs/design_register.md`** — the
intentional design register listing deliberate decisions (crowd policy, additive HSV,
warmup ramp direction, polygon conf-over-all-bins, mosaic canvas formulation, `-1.0`
polygon sentinel, etc.) that must not be "fixed" without a deliberate re-train/team
decision.

## What This System Does

A TensorFlow reimplementation of YOLOv8 with two extensions beyond standard detection:
1. **Polygon segmentation** using PolyYOLO radial format (24 vertices, 15° angle steps)
2. **Distance estimation** from a separate dataset merged at the batch level

Input: 672×672×3, Backbone: CSPDarkNetV8-S, 39 classes, 6 output heads (box, cls, poly_angle, poly_dist, poly_conf, dist).

Three config-driven tiers share the same code:

| Tier (`configs/experiments/yolo/`) | Heads | Use case |
|------|-------|----------|
| `yolov8_bbox.yaml` | box + cls | Detection only |
| `yolov8_poly.yaml` | box + cls + polygon | Detection + segmentation |
| `yolov8_poly_dist.yaml` | all 6 heads | Detection + segmentation + distance |

## Architecture Overview

### Data Pipeline

Multi-TFDS weighted sampling (each source `.repeat()`ed → stationary weights, infinite stream; images stay **encoded** through shuffle via `SkipDecoding`) → decode → pre-resize to 672² → Copy-Paste (composites at 672²; object scaled by `current/original` dims so relative size matches full-res compositing exactly) → Mosaic (**4-in/4-out**, **canvas formulation**: per-image resize at the drawn scale → `_place_in_cell` into the 2× canvas → ONE `random_perspective` warp to the output; the composed-affine variant was reverted 2026-06-11 — ~3× slower on the production CPU) → small post-unbatch shuffle → Flip → Polygon preprocessing → **uint8 out**. Color augmentation (normalize → HSV jitter → albumentations) runs per-BATCH on the GPU inside `train_step` (`data_pipeline/batch_color_aug.py`) with exactly the per-image randomness the parsers used to apply; albumentations applies only to detection rows (`ignore_bg == 0`).

The geometric transform (`random_perspective`) is applied inside the mosaic stage for **both** the 4-image mosaic and non-mosaic single images; the parser no longer applies a separate affine.

**Epoch semantics**: the training stream is infinite; the trainer runs **exactly
`steps_per_loop` steps per epoch** (= `train_total_examples // batch` = **2118** for the
poly_dist config; 271,166 examples verified against the TFDS builders 2026-06-10) from one
persistent iterator, so one epoch = one nominal data pass and the startup banner / LR
schedule (`decay_steps = steps_per_loop × epochs` = 635,400) / checkpoint interval are all
true by construction. After a mid-epoch resume only the remainder to the next epoch
boundary is run (`YoloV8Trainer._steps_for_epoch`). Eval datasets do not repeat.
`run_train` warns if `decay_steps != train_steps`. Step logs report compute, data-wait,
and wall-clock throughput separately (`train/data_wait_ms`).

The distance dataset (`servingbot_polygon:1.0.1`) is a **separate stream** merged via `tf.data.Dataset.zip()` and concatenated on the batch dimension. Distance-only samples carry `ignore_bg=1` to suppress class loss on background. The distance stream is **training-only** (no validation merge path).

Training batch sizes: 128 (detection) + 16 (distance) — throughput/logs count the merged 144.

### Polygon Representation

Polygons go through three formats across the pipeline:

| Stage | Format |
|-------|--------|
| TFDS input | `[N, max_vertices+2]` flat xy normalized, padded with -1 |
| PolyYOLO target (training/loss) | `[N, 72]` = `[dist, angle, conf] × 24` (interleaved; `angle` = sub-bin offset in `[0,1)`; see `losses/tal_loss.py:_polygon_loss`) |
| Cartesian (decode/eval/viz) | `[24, 2]` pixel `(x, y)` vertices reconstructed from the radial format (`(i + angle)·angle_step`, conf-gated ≥ 0.4) — eval GT labels themselves stay in the radial `[N, 72]` format |

### Model Heads

All heads operate per-pixel across 3 FPN levels (strides 8, 16, 32):
- `box`: DFL distribution, 64 channels (4 × 16 bins)
- `cls`: 39 channels
- `poly_angle`, `poly_dist`, `poly_conf`: 24 channels each
- `dist`: 1 channel (log-scale distance)

### Loss (TAL)

Task-Aligned assignment (`losses/tal_assigner.py`): alignment metric = `score^0.5 × IoU^6.0`,
top-k=10, spatial (anchor-center-in-box) constraint, max-IoU duplicate resolution. Soft
classification targets follow the Ultralytics recipe: `one_hot × (align_norm × pos_overlaps)`
where `pos_overlaps` is the per-GT max IoU.

Loss gains from config: iou=7.5, cls=0.5, dfl=1.5, dist=1.0, poly_dist=0.45, poly_angle=0.4,
poly_conf=0.2, with an overall `poly_gain` multiplier (default 0.5) applied to the summed
polygon loss.

**Loss normalization conventions** (`losses/tal_loss.py`, `losses/polygon_loss.py`):
- Box CIoU, DFL, and cls divide by `target_scores_sum = max(sum(target_scores), 1)`; box and
  DFL are additionally weighted per-anchor by `sum(target_scores, -1)`.
- Distance L1 divides by `num_objs` (total GT object count in the batch, both detection and
  distance streams). The valid-sentinel mask (`gt_distance > -10.0`) is applied to the
  numerator inside `distance_l1_loss`; detection-stream GTs contribute zero to the numerator.
- Polygon **angle** target is the per-bin **sub-bin offset** `(vertex_angle − bin_start)/angle_step ∈ [0,1)`
  (not a one-hot). Loss = BCE on `sigmoid(pred)`, averaged over the **valid vertices only**
  (masked by the conf channel), normalized by `num_objs`. Decode: `vertex_angle = (i + sigmoid(pred))·angle_step`.
- Polygon **dist** uses L2+softplus: `(target − softplus(pred))²`, averaged over the **valid
  vertices only** (masked by the conf channel), normalized by `num_objs`.
- Polygon **conf** uses BCE on per-bin validity, averaged over **ALL 24 bins** (occupied → 1,
  empty → 0), normalized by `num_objs`. Conf is the decode gate and must see negatives —
  the earlier masked form (2026-06, valid-bins-only) gave empty bins zero gradient ever, so
  their conf drifted above the 0.4 decode/viz threshold while their dist stayed untrained →
  the star/spiky polygon artifacts seen in val overlays. Changed 2026-06-11; the masked form
  is preserved in `polygon_conf_loss`'s docstring as a one-line swap.
- All three polygon sub-losses (`poly_angle_loss`, `poly_dist_loss`, `poly_conf_loss`) are
  logged separately to TensorBoard; the combined `poly_loss` is their gain-weighted sum
  multiplied by `poly_gain`.

Distance loss: L1 on log-scale, masked to samples where `gt_distance > -10.0` (invalid sentinel = -10.0). Valid distance range: [0.5, 10.0] meters.

### Optimizer

SGD with Nesterov momentum (0.937), cosine LR decay (initial=0.01, alpha=0.01), 300 epochs / 635,400 steps (`optimizers/sgd_warmup.py`). EMA with dynamic decay: `min(0.9999, (1+step)/(10+step))` (`optimizers/ema.py`). EMA weights are swapped in for evaluation and swapped back afterward.

## Actual File Layout

```
data_pipeline/
  tfds_decoders.py     # PolygonDecoder, ServingBot decoders
  input_reader.py      # Multi-TFDS weighted sampling + distance-stream merge
  copy_paste.py        # CopyAndPaste (prob=0.2, applied before Mosaic)
  mosaic.py            # Mosaic (freq=0.5): 2× canvas + random_perspective; MixUp (freq=0.0)
  augmentations.py     # random_perspective (full affine) / flip / HSV / Albumentations
  parser.py            # Base parser interface
  yolo_parser.py       # V8ParserExtended (detection + polygon)
  distance_parser.py   # V8DistanceParser
models/
  backbone.py          # CSPDarkNetV8 (depth_scale=0.33, width_scale=0.5 for -S)
  decoder.py           # FPN-PAN with C2f stacks
  head.py              # YoloV8Head (6 branches)
  detection_generator.py  # NMS post-processing (max_boxes=300, nms=0.65)
  yolo_v8.py           # build_yolov8 assembly
losses/
  tal_assigner.py      # TaskAlignedAssigner (stop-gradient)
  tal_loss.py          # TaskAlignedLossExtended (box/cls/dfl/dist/poly)
  polygon_loss.py      # angle / dist / conf components
  distance_loss.py     # log-scale L1
optimizers/
  ema.py               # ExponentialMovingAverage wrapper
  sgd_warmup.py        # SGD + Nesterov + momentum warmup + cosine decay
eval/
  coco_metrics.py      # COCO mAP (is_crowd / is_dontcare handling)
  polygon_metrics.py   # polygon IoU metrics
  distance_metrics.py  # distance error metrics
train/
  task.py              # YoloV8Task (build, loss, metrics, train/val steps)
  trainer.py           # YoloV8Trainer (custom loop, EMA swap, checkpoints, signals)
  viz_utils.py         # box/polygon overlay rendering for TensorBoard image summaries
configs/
  model_config.py      # config dataclasses
  yaml_loader.py       # YAML → dataclasses (hand-rolled mapping; no dacite)
  registry.py
  class_map.py         # DETECTION_CLASSES list (39 class names, index = category_id)
  data/ model/ optimizer/ experiments/yolo/   # composable YAML fragments
scripts/
  run_train.py         # entry point (config load, validation, strategy, runtime flags)
tools/
  benchmark_pipeline.py diagnose_pipeline.py cloud_diagnose.sh
  reencode_tfds_672.py  # one-time 672² pre-resized dataset variants (+orig_height/width)
  checkpoint_migration.py compare_checkpoints.py
  eval.py export_saved_model.py trace_shapes.py continuous_eval.py
tests/                 # unit/ integration/ smoke/ + component tests
```

## Configs & running

- Configs are plain dataclasses (`configs/model_config.py`) loaded from composable YAML by a
  hand-rolled mapper (`configs/yaml_loader.py` — NOT dacite, despite the dependency in
  `requirements.txt`; unknown keys outside the `runtime`/`losses` sections are silently
  ignored). `scripts/run_train.py:_validate_config` checks invariants
  (e.g. `output_poly_size == 360 // angle_step`) before training.
- Common workflows are wrapped as Claude Code skills (`.claude/skills/`): `/train`, `/eval`,
  `/export`, `/benchmark`, `/test`, `/check-env`, `/migrate-ckpt`, `/visualize-aug`.
- Runtime flags (XLA via `tf.config.optimizer.set_jit`, mixed precision via the global Keras
  policy, distribution strategy) are applied in `scripts/run_train.py:_apply_runtime_config`
  from `RuntimeConfig`. Default precision is `float32`.

## Testing

`pytest` suite under `tests/`:
- `tests/unit/` — backbone, decoders, model forward, EMA, sgd_warmup, tal_assigner, and the
  coco/distance/polygon evaluators
- `tests/integration/` — full pipeline, checkpoint migration
- `tests/smoke/` — 10-step end-to-end training
- top-level — decoders, parser, mosaic, copy_paste, losses, polygon preprocessing, batch shapes

Run with `/test` or `pytest tests/unit tests/smoke -v`.

## Key Implementation Notes

- **Copy-Paste order**: applied on decoded data *before* Mosaic
- **Copy-Paste source**: separate TFDS `cleaner_copy_paste:1.0.0` with RGBA images (4-channel alpha mask)
- **Crowd handling**: `skip_crowd_during_training=True` filters at parse time; `ignore_bg` flag masks class loss at loss time
- **Smart bias init**: class bias = `log(5 / num_classes / (input_size/stride)^2)`; box bias = 1.0
  (`input_size` is the live model input, 672 — matches all checkpoints, which are 672×672)
- **Init checkpoint**: loads only backbone + decoder weights; head is randomly initialized
- **Backbone config**: despite `depth_scale: 1.0` / `width_scale: 1.0` in the YAML, model_id is `cspdarknetv8s` (small) — the model_id takes precedence
- **Polygon conf in predictions**: `predictions['polygons'][:, :, :, 0]` values are already sigmoid-activated by the detection generator — they are not raw logits. Apply your threshold directly.
- **Polygon angle is a sub-bin offset**: the `poly_angle` channel is the offset within a bin (`(vertex_angle − bin_start)/angle_step ∈ [0,1)`), **not** a one-hot of the dominant bin. Decode the vertex angle as `(i + sigmoid(pred))·angle_step`; this is consumed in `detection_generator` / `polygon_metrics` / `viz_utils`. (See `losses/polygon_loss.py` for the masked-mean conventions.)
- **Geometry lives in the mosaic stage**: `random_perspective` (rotation/scale/shear/translate, clip-to-edge) runs in `mosaic.py` for both the 4-image mosaic and single images; the parser does not apply a separate affine.
- **Polygon validity is the `-1.0` sentinel, not non-negativity**: `transform_boxes_polygons` keys vertex validity off `pts[:, :, 0] > -1.0`, **not** `>= 0.0`. Mosaic-canvas overflow can place an in-view object's vertex at a slightly-negative input-normalized coordinate; that is a real vertex and is transformed + **clipped-to-edge** (like the box GT for the same overflow), not dropped as padding. Only the reserved `-1.0` (see design register entry 10) is treated as "no vertex". This keeps polygon GT consistent with box GT.
- **Mosaic is 4-in/4-out**: each `padded_batch(4)` group emits 4 samples (one per decoded image) — one group coin flip; mosaic branch builds 4 mosaics of the same 4 images via rotated quadrant permutations (independent random draws), single branch warps each image. No decoded image is discarded; per-sample mosaic probability is exactly `mosaic_frequency`.
- **`padded_batch(4)` uses explicit `padding_values`**: `input_reader` pins a per-key padding dict over the decoder element spec. Critically `groundtruth_polygons` pads with **`-1.0`** (the sentinel) — the default `0.0` is a valid top-left vertex coordinate, so 0-padded rows would read as real vertices and corrupt the radial target. Every other key gets its natural empty (image 0, strings `''`, ints 0, boxes/area/dists 0.0, `is_crowd` False). Keyed by name so it is robust to spec reordering.
- **Three shuffle stages, distinct seeds**: detection source shuffle = `self._seed`, copy-paste (cnp) source shuffle = `self._seed + 1`, post-unbatch shuffle = `self._seed + 2`. Distinct streams so the stages don't share an RNG (a shared seed correlates the permutations and partially undoes each stage's decorrelation; the cnp/detection zip would also pair the same indices every epoch).
- **`mosaic_center` default is 0.25 everywhere**: `MosaicConfig` dataclass default, `Mosaic.__init__` default, and all tier YAMLs agree on 0.25 (half-range of the split point). The dataclass/loader previously defaulted to 0.2, silently disagreeing with the runtime default when a tier YAML omitted the key; the bbox/poly YAMLs now set it explicitly.
- **Mosaic image path is the canvas formulation**: per-image `tf.image.resize` at the drawn scale → `_place_in_cell` into the 2× canvas → ONE `apply_perspective_image` warp to the output (`make_perspective_matrix` / `transform_boxes_polygons` in `augmentations.py`). A composed-affine variant (fold each quadrant's affine into `M`, warp each source full-frame) was tried and **measured ~3× slower on the production CPU** (`ImageProjectiveTransformV3` is several times costlier per pixel than `tf.image.resize`; 4 full warps per mosaic vs 4 resizes + 1 warp) — don't reintroduce it without a cloud measurement. Both forms are geometrically identical; the label path never changed.
- **Parsers emit uint8**: color aug (normalize/HSV/albumentations) happens per-batch on GPU in `train_step` via `data_pipeline/batch_color_aug.py` (exact per-image randomness; equivalence-tested). `validation_step` just casts `/255`. TensorBoard `train/augmentations` images are therefore pre-color-aug (geometry + labels only).
- **Copy-paste runs on the pre-resized 672² background**: `CopyAndPasteModule` scales the object by `(current/original)` per axis (original dims from the `height`/`width` fields), so the relative-size distribution is exactly the full-resolution one. Without those fields the correction is 1 (backward compatible).
- **Warp scale gain uses explicit bounds**: `make_perspective_matrix(scale_min=, scale_max=)` draws from the configured `[aug_scale_min, aug_scale_max]` (poly_dist: [0.4, 1.9]). The symmetric-magnitude form is kept for back-compat but widened [0.4, 1.9] to [0.1, 1.9] — the mostly-gray-frame bug (fixed 2026-06-11).
- **Polygon vertex resampling**: `parser.resample_points: 64` is set in the poly_dist YAML for BOTH the detection and distance streams (dataclass default stays 0); decode-time resample to `[N, 128]`, radial target exact for ≤64-vertex polygons.
- **Copy-paste polygon fit = even resample, not truncation**: the cnp source decoder does NOT resample, so a pasted object can carry far more polygon columns than the (resampled) background. When `cur_cols >= n_poly_cols`, copy-paste **evenly resamples** the valid vertices to the column budget (`resample_polygons`) instead of slicing the first N — slicing kept only a leading contour arc (~3% of the loop) and corrupted the radial target. `resample_polygons` itself stable-argsorts valid-first to **compact scattered sentinels** before sampling, because copy-paste invalidates out-of-bounds vertices in place (interleaved `-1`s); on decode-time prefix input the sort is a no-op (byte-identical). Both are **train-semantics** changes: they alter the polygon GT for copy-pasted objects, so do not merge into a live run mid-flight.
- **Pre-resized dataset variants**: `tools/reencode_tfds_672.py` builds `<name>_672` TFDS copies (672² JPEG + `orig_height`/`orig_width`, which `PolygonDecoder` prefers). Detection sets only — the distance parser letterboxes (aspect-preserving) so servingbot must stay full-resolution. The YAML carries commented switch-over lines.
- **Polygon binning is a segment formulation**: `_preprocess_polygons_v2` uses `unsorted_segment_max` + first-winner `unsorted_segment_min` instead of a `[N, P, 24]` one-hot — exactly output-equivalent including argmax-first tie behavior (tests assert equality).
- **Runtime defaults (poly_dist YAML)**: `one_device` on 1 GPU, `mixed_bfloat16` (heads pinned float32 in `models/head.py`, no loss scaling needed), thread-pool caps for cgroup-capped hosts (`runtime.inter_op_threads`/`intra_op_threads`, `train_data.private_threadpool_size`). The training stream sets `tf.data` `deterministic=False` (sample order is not seed-reproducible; augmentation randomness unaffected).
- **Polygon sub-losses are logged separately**: TensorBoard tags `train/poly_angle_loss`, `train/poly_dist_loss`, and `train/poly_conf_loss` allow diagnosing which polygon component is not converging.
- **TensorBoard scalars carry names + formulae**: every scalar is written with a markdown `description` (`train/metric_meta.py`); per-category metrics are tagged by class name (`val/cls/<NN>_<name>/<metric>`), not bare index.

## Dependencies

See `requirements.txt`. Core: `tensorflow==2.16.1`, `tensorflow-datasets>=4.9.0`,
`albumentations`, `opencv-python-headless` (no libGL system dep — works on CI/servers),
`pycocotools`, `scikit-image`, `dacite` (listed but currently unused — `yaml_loader.py` is
hand-rolled), `PyYAML`, `absl-py`, `numpy`;
`pytest` / `pytest-cov` for tests.
