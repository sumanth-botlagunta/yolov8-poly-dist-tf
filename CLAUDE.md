# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Implemented and actively trained.** This is a working TensorFlow 2.16 codebase, not a
plan. Source lives under `data_pipeline/`, `models/`, `losses/`, `optimizers/`, `eval/`,
`train/`, `configs/`, `common/`, and `utils/`, with a `tests/` suite.

The authoritative hyperparameter reference is the experiment YAML you train with, e.g.
`configs/experiments/yolo/yolov8_poly_dist.yaml`. Developer docs live in `docs/`
(architecture, data pipeline, losses, training, testing). The older
`docs/implementation_plan.md` / top-level `MASTER_PLAN.md` references are obsolete and gone.

`.claude/design_register.md` documents non-obvious design choices (crowd policy, additive HSV,
warmup ramp direction, polygon conf-over-all-bins, mosaic canvas formulation, `-1.0` polygon
sentinel, etc.) and the reasoning behind them. Several differ from stock YOLOv8 for a measured
or historical reason, and several affect training — so changing them means re-training.

## What This System Does

A TensorFlow reimplementation of YOLOv8 with two extensions beyond standard detection:
1. **Polygon segmentation** using PolyYOLO radial format (24 vertices, 15° angle steps)
2. **Distance estimation** from a separate dataset merged at the batch level

Input: 672×672×3, Backbone: CSPDarkNetV8-S, 39 classes, 6 output heads (box, cls, poly_angle, poly_dist, poly_conf, dist).
All convolutions use relu (`norm_activation.activation`); activation layers carry no
weights, so checkpoints load across activation settings — but weights are only
meaningful under the activation they were trained with.

Three config-driven tiers share the same code:

| Tier (`configs/experiments/yolo/`) | Heads | Use case |
|------|-------|----------|
| `yolov8_bbox.yaml` | box + cls | Detection only |
| `yolov8_poly.yaml` | box + cls + polygon | Detection + segmentation |
| `yolov8_poly_dist.yaml` | all 6 heads | Detection + segmentation + distance |

## Architecture Overview

### Data Pipeline

Multi-TFDS weighted sampling (each source `.repeat()`ed → stationary weights, infinite stream; images stay **encoded** through shuffle via `SkipDecoding`) → decode → pre-resize to 672² (aspect-preserving **letterbox**: long side = 672, gray-114 pad, the SAME math the eval path uses) → Mosaic (**G-in/(G//R)-out**, default `group_size`=32 / `decodes_per_output`=4 → 8 outputs; each output picks 4 source images via a windowed slice of one per-group permutation, R=4 = stock-YOLO no-reuse; **canvas formulation**, per-tile order: content-slice (the letterbox content region of the tile) → Copy-Paste (applied PER TILE inside the mosaic branch only; prob per tile) → per-TILE flip (independent 0.5 each; the canvas is never mirrored whole) → optional per-tile random-window crop (`tile_crop_min/max`; poly_dist has it OFF) → `_place_in_cell` into the 2× canvas → ONE `random_perspective` warp to the output (poly_dist: scale `[0.4, 1.9]`, rotation **hard-OFF** in the mosaic warp, translate 0; non-mosaic singles instead take the letterboxed image + one warp with the PARSER-level `aug_scale_min/max`=1.0 + `aug_rand_translate`=0.1 + optional `parser.rotate`/`rotate_degrees` pre-warp rotation (default off) — the mosaic module owns flip during training, the parser flip is disabled for the train stream; singles never copy-paste); the composed-affine variant was reverted — ~3× slower on the production CPU) → small post-unbatch shuffle → Polygon preprocessing → **uint8 out**. Color augmentation (normalize → HSV jitter → albumentations) runs per-BATCH on the GPU inside `train_step` (`data_pipeline/batch_color_aug.py`) with exactly the per-image randomness the parsers used to apply; albumentations applies only to detection rows (`ignore_bg == 0`).

The geometric transform (`random_perspective`) is applied inside the mosaic stage for **both** the 4-image mosaic and non-mosaic single images; the parser no longer applies a separate affine.

**Epoch semantics**: the training stream is infinite; the trainer runs **exactly
`steps_per_loop` steps per epoch** (= `train_total_examples // batch`, derived from the
config) from one persistent iterator, so one epoch = one nominal data pass and the startup
banner / LR schedule (`decay_steps = steps_per_loop × epochs`) / checkpoint interval are all
true by construction. After a mid-epoch resume only the remainder to the next epoch
boundary is run (`YoloV8Trainer._steps_for_epoch`). Eval datasets do not repeat.
`run_train` warns if `decay_steps != train_steps`. Step logs report compute, data-wait,
and wall-clock throughput separately (`train/data_wait_ms`).

The distance dataset (`servingbot_polygon:1.0.1`) is a **separate stream** merged via `tf.data.Dataset.zip()` and concatenated on the batch dimension. Distance-only samples carry `ignore_bg=1` to suppress class loss on background. The distance stream is **training-only** (no validation merge path).

Training batch = the detection `global_batch_size` + the distance stream's batch (merged on
the batch dim); throughput/logs count the merged total.

### Polygon Representation

Polygons go through three formats across the pipeline:

| Stage | Format |
|-------|--------|
| TFDS input | `[N, max_vertices+2]` flat xy normalized, padded with -1 |
| PolyYOLO target (training/loss) | `[N, 72]` = `[dist, angle, conf] × 24` (interleaved; `angle` = sub-bin offset in `[0,1)`; see `losses/tal_loss.py:_polygon_loss`) |
| Cartesian (decode/eval/viz) | `[M, 2]` pixel `(x, y)` vertices after conf-gate ≥ 0.4 (M ≤ 24 occupied bins) reconstructed from the radial format (`(i + angle)·angle_step`); the pre-gate intermediate is `[24, 2]` — eval GT labels themselves stay in the radial `[N, 72]` format |

### Model Heads

All heads operate per-pixel across 3 FPN levels (strides 8, 16, 32):
- `box`: DFL distribution, 64 channels (4 × 16 bins)
- `cls`: 39 channels
- `poly_angle`, `poly_dist`, `poly_conf`: 24 channels each
- `dist`: 1 channel (log-scale distance)

### Loss (TAL)

Task-Aligned assignment (`losses/tal_assigner.py`): alignment metric = `score^0.5 × CIoU^6.0` (Complete IoU clamped at 0 — the reference recipe's `bbox_iou(..., CIoU=True).clamp_(0)`, mirrored byte-exactly from `losses/tal_loss._bbox_iou_loss("ciou")` via `_pairwise_ciou`),
top-k=10, spatial (anchor-center-in-box) constraint, max-IoU duplicate resolution. Soft
classification targets follow the Ultralytics recipe: `one_hot × (align_norm × pos_overlaps)`
where `pos_overlaps` is the per-GT max IoU.

Loss gains from config: iou=7.5, cls=0.5, dfl=1.5, dist=1.0, poly_dist=0.45, poly_angle=0.4,
poly_conf=0.2, with an overall `poly_gain` multiplier (default 0.5) applied to the summed
polygon loss.

The box and cls losses are **config-selectable** (`losses/tal_loss.py`): `losses.box_iou_type`
= `ciou` (default) / `giou` / `diou` / `eiou` / `siou`; `losses.cls_loss_type` = `bce` (default) /
`focal` / `varifocal`; `losses.label_smoothing` (default 0). Defaults reproduce the CIoU+BCE path
byte-identically. Polygon/distance losses are unchanged.

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
  the earlier masked form (valid-bins-only) gave empty bins zero gradient ever, so
  their conf drifted above the 0.4 decode/viz threshold while their dist stayed untrained →
  the star/spiky polygon artifacts seen in val overlays. The masked form
  is preserved in `polygon_conf_loss`'s docstring as a one-line swap.
- All three polygon sub-losses (`poly_angle_loss`, `poly_dist_loss`, `poly_conf_loss`) are
  logged separately to TensorBoard; the combined `poly_loss` is their gain-weighted sum
  multiplied by `poly_gain`.

Distance loss: L1 on log-scale, masked to samples where `gt_distance > -10.0` (invalid sentinel = -10.0). Valid distance range: [0.5, 10.0] meters.

### Optimizer

SGD with Nesterov momentum (0.937), cosine LR decay (initial=0.01, alpha=0.01) over the full `train_steps` (`optimizers/sgd_warmup.py`). EMA with dynamic decay: `0.9999 × (1 − exp(−step/2000))` — the standard YOLOv5/YOLOv8 ModelEMA ramp (`optimizers/ema.py`). EMA weights are swapped in for evaluation and swapped back afterward.

The optimizer and LR schedule are **config-selectable** via a registry (`optimizers/factory.py`): `optimizer.type` = `sgd` (default) / `adamw` / `adam`; `learning_rate.type` = `cosine` (default) / `linear` / `step` / `polynomial` / `constant`, plus an optional linear LR-warmup. Defaults reproduce the SGD+cosine path byte-identically. Gradient clipping is `task.gradient_clip_norm` (SGD clips per-call; keras optimizers set `global_clipnorm`).

## Actual File Layout

```
data_pipeline/
  tfds_decoders.py     # PolygonDecoder, ServingBot decoders
  input_reader.py      # Multi-TFDS weighted sampling + distance-stream merge
  copy_paste.py        # CopyAndPaste (prob per tile, applied per-tile inside Mosaic)
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
  coco_eval_custom.py  # COCOevalCustom: F1score50 confidence sweep + dontcare absorption
  polygon_metrics.py   # polygon IoU metrics
  distance_metrics.py  # distance error metrics
  metrics_report.py    # ckpt-format report writer (best-conf + all-conf tables)
  val_history.py       # val_history.jsonl store (append/load/select/best)
  metric_meta.py       # TensorBoard scalar names + formula descriptions
  failure_mining.py    # FailureCollector (worst-K fp/fn/lowiou per class)
train/
  task.py              # YoloV8Task (build, loss, metrics, train/val steps)
  trainer.py           # YoloV8Trainer (custom loop, EMA swap, checkpoints, signals)
  run_train.py         # entry point (config load, validation, strategy, runtime flags)
  train_supervisor.sh  # supervised launcher (nohup + auto-restart)
common/                # shared library imported by core code and CLIs
  viz_utils.py         # box/polygon overlay rendering for TensorBoard image summaries
  run_metadata.py      # run provenance (git/env/datasets) -> run_metadata.json
  ckpt_loading.py runtime_setup.py progress.py
configs/
  model_config.py      # config dataclasses
  yaml_loader.py       # YAML → dataclasses (hand-rolled mapping; no dacite)
  registry.py
  class_map.py         # DETECTION_CLASSES list (39 class names, index = category_id)
  experiments/yolo/    # self-contained experiment YAMLs (bbox / poly / poly_dist / poly_dist_bf16)
utils/                 # runnable CLIs
  eval.py              # standalone evaluation over one or many checkpoints
  export/              # model export + inference
    export_saved_model.py         # host/server SavedModel (NMS baked in)
    export_device_savedmodel.py   # on-device SNPE-shaped SavedModel
    inference_saved_model.py      # folder inference: predictions JSON + visuals
  reports/             # validation-history extraction + reporting
    val_history.py       # inspect/extract/export val_history.jsonl (txt/json/csv/xlsx/parquet)
  pipeline/            # data-pipeline diagnostics
    benchmark_pipeline.py diagnose_pipeline.py cloud_diagnose.sh
  confusion_matrix.py  # per-class detection confusion matrix (ckpt or SavedModel)
notebooks/             # 01 data-pipeline walkthrough / 02 TensorBoard analysis / 03 checkpoint inspection
tests/                 # unit/ integration/ smoke/ + component tests
```

## Configs & running

- Configs are plain dataclasses (`configs/model_config.py`) loaded from composable YAML by a
  hand-rolled mapper (`configs/yaml_loader.py` — NOT dacite, despite the dependency in
  `requirements.txt`; unknown keys outside the `runtime`/`losses` sections are silently
  ignored). `train/run_train.py:_validate_config` checks invariants
  (e.g. `output_poly_size == 360 // angle_step`) before training.
- Common workflows are documented as copy-paste commands in `docs/scripts.md`
  (training/eval/export/benchmark/infer).
- Runtime flags (XLA via `tf.config.optimizer.set_jit`, mixed precision via the global Keras
  policy, distribution strategy) are applied in `train/run_train.py:_apply_runtime_config`
  from `RuntimeConfig`. Default precision is `float32`.

## Testing

`pytest` suite under `tests/`:
- `tests/unit/` — backbone, decoders, model forward, EMA, sgd_warmup, tal_assigner, and the
  coco/distance/polygon evaluators
- `tests/integration/` — full pipeline, native warm-start loading
- `tests/smoke/` — 10-step end-to-end training
- top-level — decoders, parser, mosaic, copy_paste, losses, polygon preprocessing, batch shapes

Run with `/test` or `pytest tests/unit tests/smoke -v`.

## Key Implementation Notes

- **Copy-Paste order**: applied inside the mosaic tile path (per tile), mosaic-only — non-mosaic singles never copy-paste
- **Copy-Paste source**: separate TFDS `cleaner_copy_paste:1.0.0` with RGBA images (4-channel alpha mask)
- **Crowd handling**: `skip_crowd_during_training=True` filters at parse time; `ignore_bg` flag masks class loss at loss time
- **NMS suppression scope is config-selectable** (`detection_generator.nms_class_mode`): `per_class`
  (default) vs `agnostic` (one NMS over all boxes regardless of class). Eval-time only. Measured
  head-to-head: agnostic loses recall on region classes that
  legitimately contain other objects (bathroom/entrance/doorway) — more than its precision gain —
  so `per_class` stays the default.
- **Smart bias init**: class bias = `log(5 / num_classes / (input_size/stride)^2)`; box bias = 1.0
  (`input_size` is the live model input, 672 — matches all checkpoints, which are 672×672)
- **Seed-init (fresh runs only)**: two mutually-exclusive, resume-skipped paths in `task.initialize` (`train/task.py`). `task.finetune_from` = **fine-tuning** (same task): loads the FULL model from a trained `ckpt-N`'s EMA/deployed weights (`restore_eval_weights`) into a fresh optimizer/EMA/step. `task.init_checkpoint` = **transfer-init**: full-model restore via `restore_eval_weights` (handles trainer/EMA checkpoint layouts completely — a bare `model/` object-graph restore misses list-tracked C2f variables), then non-selected modules (default: head) are put back to their fresh random init. The trainer skips both when a resumable checkpoint exists (`_will_resume`) — so a dropped fine-tune just resumes normally. `--finetune_from` CLI overrides the config field.
- **Backbone config**: despite `depth_scale: 1.0` / `width_scale: 1.0` in the YAML, model_id is `cspdarknetv8s` (small) — the model_id takes precedence
- **Polygon conf in predictions**: `predictions['polygons'][:, :, :, 0]` values are already sigmoid-activated by the detection generator — they are not raw logits. Apply your threshold directly.
- **Polygon angle is a sub-bin offset**: the `poly_angle` channel is the offset within a bin (`(vertex_angle − bin_start)/angle_step ∈ [0,1)`), **not** a one-hot of the dominant bin. Decode the vertex angle as `(i + sigmoid(pred))·angle_step`; this is consumed in `detection_generator` / `polygon_metrics` / `viz_utils`. (See `losses/polygon_loss.py` for the masked-mean conventions.)
- **Geometry lives in the mosaic stage**: `random_perspective` (rotation/scale/shear/translate, clip-to-edge) runs in `mosaic.py` for both the 4-image mosaic and single images; the parser does not apply a separate affine.
- **Polygon validity is the `-1.0` sentinel, not non-negativity**: `transform_boxes_polygons` keys vertex validity off `pts[:, :, 0] > -1.0`, **not** `>= 0.0`. Mosaic-canvas overflow can place an in-view object's vertex at a slightly-negative input-normalized coordinate; that is a real vertex and is transformed + **clipped-to-edge** (like the box GT for the same overflow), not dropped as padding. Only the reserved `-1.0` (see design register entry 10) is treated as "no vertex". This keeps polygon GT consistent with box GT.
- **Mosaic is G-in/(G//R)-out** (`group_size`=32, `decodes_per_output`=R=4 → 8 outputs): each `padded_batch(group_size)` group emits `group_size // R` samples. Each output independently flips `mosaic_frequency` (per-output, not per-group → exact per-sample frequency, no batch clustering); a mosaic draws 4 source images from one per-group `tf.random.shuffle` at **Sidon-set shifts** (`_SIDON_SHIFTS` in `mosaic.py`): R=4 uses the contiguous window {0,1,2,3}, which tiles the permutation (4 distinct images, zero cross-output reuse = stock YOLO); at R<4 each image recurs in exactly 4/R outputs but any two outputs of a group share **at most one** source image (the earlier contiguous window SLID at R<4 — adjacent outputs shared 3/4 sources at R=1, ~82 near-duplicate pairs per 128-batch measured). R<4's remaining trade-off is volume (an epoch consumes R/4 as many distinct images), not correlation. Epoch step count is unchanged by R. See `data_pipeline/mosaic.py`.
- **`padded_batch(group_size)` uses explicit `padding_values`**: `input_reader` pins a per-key padding dict over the decoder element spec. Critically `groundtruth_polygons` pads with **`-1.0`** (the sentinel) — the default `0.0` is a valid top-left vertex coordinate, so 0-padded rows would read as real vertices and corrupt the radial target. Every other key gets its natural empty (image 0, strings `''`, ints 0, boxes/area/dists 0.0, `is_crowd` False). Keyed by name so it is robust to spec reordering.
- **Three shuffle stages, distinct seeds**: detection source shuffle = `self._seed`, copy-paste (cnp) source shuffle = `self._seed + 1`, post-unbatch shuffle = `self._seed + 2`. Distinct streams so the stages don't share an RNG (a shared seed correlates the permutations and partially undoes each stage's decorrelation; the cnp/detection zip would also pair the same indices every epoch).
- **`mosaic_center` default is 0.25 everywhere**: `MosaicConfig` dataclass default, `Mosaic.__init__` default, and all tier YAMLs agree on 0.25 (half-range of the split point). The dataclass/loader previously defaulted to 0.2, silently disagreeing with the runtime default when a tier YAML omitted the key; the bbox/poly YAMLs now set it explicitly.
- **Mosaic image path is the canvas formulation**: per-image `tf.image.resize` at the drawn scale → `_place_in_cell` into the 2× canvas → ONE `apply_perspective_image` warp to the output (`make_perspective_matrix` / `transform_boxes_polygons` in `augmentations.py`). A composed-affine variant (fold each quadrant's affine into `M`, warp each source full-frame) was tried and **measured ~3× slower on the production CPU** (`ImageProjectiveTransformV3` is several times costlier per pixel than `tf.image.resize`; 4 full warps per mosaic vs 4 resizes + 1 warp) — don't reintroduce it without a cloud measurement. Both forms are geometrically identical; the label path never changed.
- **Parsers emit uint8**: color aug (normalize/HSV/albumentations) happens per-batch on GPU in `train_step` via `data_pipeline/batch_color_aug.py` (exact per-image randomness; equivalence-tested). `validation_step` just casts `/255`. TensorBoard `train/augmentations` images are therefore pre-color-aug (geometry + labels only).
- **Copy-paste runs on the letterbox content region of a mosaic tile**: `CopyAndPasteModule` scales the object by `(current/original)` per axis (original dims from the `height`/`width` fields), so the relative-size distribution is exactly the full-resolution one. Without those fields the correction is 1 (backward compatible).
- **Per-tile random-window crop is config-gated** (`mosaic.tile_crop_min/max`): when enabled, each mosaic tile crops a random WINDOW of its content — side fraction `s ~ U[tile_crop_min, tile_crop_max]` at a random position within bounds — then scales the crop to fill its quadrant (a zoom/translate scale-invariance signal). Default `0/0` = OFF (the content region fills its quadrant unchanged); the poly_dist YAML keeps it off, so size variety comes only from the whole-canvas warp gain `aug_scale_min/max` = `[0.4, 1.9]`. Bounds are validated `0 < min <= max <= 1` by `_validate_config`. Rotation is **hard-OFF** in the mosaic warp (not a config knob); single-image rotation is the separate parser-level `rotate`/`rotate_degrees` (default off). **Pre-resized-dataset caveat**: the tile content region is derived from the letterbox geometry, so any stored-resized dataset variant MUST be **letterbox-encoded** — a squash-encoded variant carrying `orig_height`/`width` fields would be mis-sliced by the content-region derivation.
- **Warp scale gain uses explicit bounds**: `make_perspective_matrix(scale_min=, scale_max=)` draws from the configured `[aug_scale_min, aug_scale_max]` (poly_dist: [0.5, 1.5]). The symmetric-magnitude form is kept for back-compat but widened the configured bounds — the mostly-gray-frame bug (since fixed).
- **Polygon vertex resampling**: `resample_polygons` (`data_pipeline/augmentations.py`) samples uniformly along the closed contour, filling every radial bin the boundary crosses. It exists only for copy-paste, which uses it to fit a pasted object's polygon into the background's column budget. Polygons otherwise flow through the pipeline at their raw stored width (up to `[N, 10940]`).
- **Copy-paste polygon fit = even resample, not truncation**: the cnp source decoder does NOT resample, so a pasted object can carry far more polygon columns than the (resampled) background. When `cur_cols >= n_poly_cols`, copy-paste **evenly resamples** the valid vertices to the column budget (`resample_polygons`) instead of slicing the first N — slicing kept only a leading contour arc (~3% of the loop) and corrupted the radial target. `resample_polygons` itself stable-argsorts valid-first to **compact scattered sentinels** before sampling, because copy-paste invalidates out-of-bounds vertices in place (interleaved `-1`s); on decode-time prefix input the sort is a no-op (byte-identical). Both are **train-semantics** changes: they alter the polygon GT for copy-pasted objects, so do not merge into a live run mid-flight.
- **Polygon binning is a segment formulation**: `_preprocess_polygons_v2` uses `unsorted_segment_max` + first-winner `unsorted_segment_min` instead of a `[N, P, 24]` one-hot — exactly output-equivalent including argmax-first tie behavior (tests assert equality).
- **Runtime defaults (poly_dist YAML)**: `one_device` on 1 GPU, `mixed_bfloat16` (heads pinned float32 in `models/head.py`, no loss scaling needed), thread-pool caps for cgroup-capped hosts (`runtime.inter_op_threads`/`intra_op_threads`, `train_data.private_threadpool_size`). The training stream sets `tf.data` `deterministic=False` (sample order is not seed-reproducible; augmentation randomness unaffected).
- **Polygon sub-losses are logged separately**: TensorBoard tags `train/poly_angle_loss`, `train/poly_dist_loss`, and `train/poly_conf_loss` allow diagnosing which polygon component is not converging.
- **TensorBoard scalars carry names + formulae**: every scalar is written with a markdown `description` (`eval/metric_meta.py`). Scalars are grouped into separate top-level sections so the headline metrics aren't buried: `train/` (losses, `lr`, `grad_norm` pre-clip global gradient norm, throughput/data-wait), `val/` (headline detection + polygon + distance metrics only), **`per_class/<metric>/<NN_name>`** (per-category, grouped BY metric so all classes of one metric sit together — out of `val/`), `epoch/`, `system/`. The per-class index is zero-padded + class-named (from `configs/class_map.py`), not a bare index.

## Dependencies

See `requirements.txt`. Core: `tensorflow==2.16.1`, `tensorflow-datasets>=4.9.0`,
`albumentations`, `opencv-python-headless` (no libGL system dep — works on CI/servers),
`pycocotools`, `scikit-image`, `dacite` (listed but currently unused — `yaml_loader.py` is
hand-rolled), `PyYAML`, `absl-py`, `numpy`;
`pytest` / `pytest-cov` for tests.
