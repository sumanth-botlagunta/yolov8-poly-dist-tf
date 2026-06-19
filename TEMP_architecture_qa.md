# TEMP — Architecture / Training Q&A

> Temporary notes file. Delete when no longer needed (you asked me to keep it here for now).
> Code references are to this repo (`yolov8-poly-dist-tf`) at the current branch.

---

## Q1. We changed the "core model architecture" and the init checkpoint comes from the OLD architecture. Does that affect accuracy? How is it handled?

**Two separate things are being conflated — let's split them.**

### (a) The changes WE made on this branch are NOT architecture changes
- `models/backbone.py`: stride-2 convs now use explicit `ZeroPadding2D` + `valid` instead of `padding="same"`. **Numerically identical**, adds **no trainable variables** (ZeroPadding has none). Same 336 variables, same names/shapes.
- `models/decoder.py`: `static_resize` flag for SNPE-clean export. Dynamic at train/eval; **identical math**.
- The box reorder is **export-only** (in `tools/export_device_dlc.py`), not in the model.

➡️ **An existing checkpoint loads unchanged and training resumes identically.** These do not affect accuracy at all (byte-identical forward pass; we verified full-model diff ≈ 2.86e-6 = float noise).

### (b) The init checkpoint from the old codebase = a WARM START, not the final model
Config (`configs/experiments/yolo/yolov8_poly_dist.yaml`):
```yaml
init_checkpoint: initial_checkpoint_folder/ckpt-920304
init_checkpoint_modules: [backbone, decoder]
```
- The init checkpoint is loaded by `tools/checkpoint_migration.py` (called from `train/task.py::initialize`).
- It loads **backbone + decoder only**. The **head is always randomly initialized** (smart-bias init in `models/head.py`) and trained from scratch.
- Matching is **structural** — by variable *role* (kernel/bias/gamma/beta/moving_mean/moving_variance) + *shape*, in traversal order — **not by name** (`align_structures`). 
- **Any old variable with no structural match is reported and SKIPPED, never silently copied** (`align_structures` → `unmatched_old`, `clean=False` warning). So if the old and new backbone/decoder genuinely differ in structure, the non-matching parts simply start from random init — you are warned, nothing is corrupted.

**How it affects accuracy:** the init checkpoint only provides a *starting point* for backbone+decoder features. The model is then **fully trained (300 epochs / 635,400 steps)**, so the final accuracy is determined by training, not by the old architecture. A good warm start → faster convergence and usually a better optimum; a poor/partial match → slower convergence, but training still reaches a valid model. The new model does **not** "inherit" the old model's accuracy or its bugs — it learns its own weights. The head, in particular, is 100% trained fresh.

**Bottom line:** initializing from an old-architecture checkpoint is safe and normal (transfer learning of the feature extractor). It influences convergence speed / final quality through how many features transfer, but it cannot make the model "wrong" — unmatched weights are skipped with a warning, and everything is retrained.

---

## Q2. Can we change the architecture for better results?

**Yes.** The codebase is config-driven and the migration tool supports partial transfer, so architecture changes are practical. Where to change things:

| Change | Where | Notes / trade-offs |
|---|---|---|
| Backbone width/depth | `depth_scale` / `width_scale` (model config) / `model_id` | Bigger = more accuracy, **more compute & memory** (critical for your embedded target). Note: current `model_id=cspdarknetv8s` (small) takes precedence over the YAML's 1.0/1.0 — see CLAUDE.md. |
| Activation | backbone/decoder/head `activation` (default `relu`) | SiLU/Swish often improves accuracy slightly; relu converts most cleanly to DLC. |
| Input resolution | `model.input_size` | Higher res → small-object recall up, compute up. (Also interacts with the device decode — keep it consistent.) |
| Extra/!modified heads | `models/head.py` | New head = new variables = retrained; loss gains in config. |
| Loss gains / TAL params | `losses` section | Cheapest "architecture-free" way to chase accuracy (iou/cls/dfl/poly gains, topk, tal_alpha/beta). |

**What's required when you change architecture:**
1. The init checkpoint will only transfer the variables that still match (role+shape) — `tools/checkpoint_migration.py report` shows exactly what maps; unmatched parts train from scratch.
2. **You must retrain** (fully or fine-tune) — an architecture change is not checkpoint-compatible for the changed parts.
3. Re-export + re-convert the DLC; re-check the on-device decode contract (box order, anchor count) still matches.

**Recommendation order (accuracy per unit risk):** loss-gain / augmentation tuning → activation → input resolution → width/depth. Always weigh against the device compute/memory budget you flagged earlier.

---

## Q3. Which input is "better" — normalized or non-normalized?

**This model is trained on `[0,1]` (plain `÷255`). That is what you must feed it. There is no choice — train-time and inference-time normalization MUST match.**

- Training: parsers emit `uint8`; `÷255` + color aug happen per-batch on GPU (`data_pipeline/batch_color_aug.py`), and any direct `model()` call goes through `train/task.py::normalize_images` (`uint8 → /255`). **No mean/std standardization is applied.**
- Device: the raw generator sets `IMAGE_NROM_FLAG=False` (writes raw `[0,255]`), and the DLC has `÷255` **baked in** (`tools/export_device_dlc.py --normalize`). So the device feeds raw `[0,255]` and the graph divides by 255 → exactly the `[0,1]` the model trained on. ✅ Correct and consistent.
- The raw generator also offers `IMAGE_NORM_TYPE=1` (ImageNet mean/std standardization). **Do NOT use it with this model** — it was trained with `÷255` only, so mean/std standardization would feed a distribution the model never saw → accuracy drop.

**General principle (if you were choosing for a *new* training run):**
- `÷255` (`[0,1]`): simplest, fully sufficient for YOLO-style nets — what's used here.
- mean/std standardization: can slightly help optimization for some backbones, but only ever helps if **train and inference use the identical statistics**. It is not "better" unless you retrain for it and bake the same transform into the DLC.

**Verdict for you today:** keep `÷255` / `[0,1]` (the baked path). Don't switch to standardization unless you retrain end-to-end and bake the same transform into the export.

---

## Q4. The label flags (`is_crowd`, `is_dontcare`, `ignore_bg`, …) — what are they, and what checks use them in training vs validation?

There are **two kinds**: real dataset annotations (from TFDS) and a pipeline-set flag.

### Dataset annotations (per ground-truth object)
| Flag | Source | Meaning |
|---|---|---|
| `is_crowd` | TFDS `objects/is_crowd` (`data_pipeline/tfds_decoders.py`) | A crowd/group region (many instances), not a single clean object. |
| `is_dontcare` | TFDS `objects/is_dontcare` | A "don't-care" region — detections here should be neither rewarded nor penalized. |

### Pipeline flag (per image / per row)
| Flag | Source | Meaning |
|---|---|---|
| `ignore_bg` | set by the parsers, **not** a dataset label | `0` = detection data (full supervision). `1` = distance-stream rows (`servingbot`, merged via `input_reader`) that carry **no class/polygon GT** for background. |

### How they're used in TRAINING
- **`is_crowd`** → `skip_crowd_during_training: true` (config). In `data_pipeline/yolo_parser.py` (~line 111) crowd objects are **filtered out at parse time** (`valid = logical_not(is_crowd)`), so they never become training targets.
- **`is_dontcare`** → not used as a training signal in the parser (it's an eval concept here; carried through for eval).
- **`ignore_bg`** → used in the **loss** (`losses/tal_loss.py`):
  - `_class_loss`: when `ignore_bg=1`, the class BCE is **masked to foreground anchors only** (distance images have no background-class labels, so we don't push background everywhere). `mask = (1-ignore_bg) + ignore_bg*fg`.
  - polygon loss: rows with `ignore_bg=1` have the **polygon loss zeroed entirely** (they have no polygon GT; otherwise the conf head would be wrongly trained to emit 0 on real objects).
  - Also in `data_pipeline/batch_color_aug.py`: albumentations runs **only on detection rows** (`ignore_bg == 0`).

### How they're used in VALIDATION (`eval/coco_metrics.py`, `eval/polygon_metrics.py`)
Config: `ignore_dontcare: true`, `ignore_iscrowds: false`, `iscrowds_labels: [6, 13, 24, 36, 37]`.
- **`is_crowd` + class in `iscrowds_labels`** → GT is **skipped entirely** (only when `ignore_iscrowds=true`; currently **false**, so this path is off unless you enable it). When skipped, it is *not* counted as a missed detection.
- **`is_dontcare`** (with `ignore_dontcare=true`) → mapped to **`iscrowd=1` in pycocotools**: it **absorbs** any overlapping detection (IoU>0.5) so a detection landing on a don't-care region is neither a TP nor an FP.
- In the **polygon** evaluator, both crowd and dontcare GT are **dropped from the recall denominator and from matching** (`_eval_gt_mask`), so they can't inflate or deflate polygon recall.

**Summary of checks:** training removes crowd objects (`skip_crowd`) and masks class/polygon loss on distance rows (`ignore_bg`); validation excludes crowd (optionally) and absorbs dontcare so they don't distort precision/recall/mAP.

---

## Q5. Does the validation data affect any training parameter (i.e., is there leakage)?

**No. Validation data never changes a trainable parameter.** Verified in `train/task.py` and `train/trainer.py`:

1. `validation_step` runs `model(images, training=False)` — **no `GradientTape`, no `optimizer.apply_gradients`**. `training=False` means **BatchNorm uses its stored moving statistics and does NOT update them**. So no weights, no BN running stats, nothing learnable is touched by validation data.
2. The validation stream is **separate** from training: the distance/servingbot merge is **training-only** (`input_reader`), and eval datasets **do not `.repeat()`**. No validation example enters the training iterator.
3. Around validation the trainer **swaps in EMA (shadow) weights**, runs eval, then **swaps them back** in a `try/finally` (`trainer.py` ~line 230–239) so the live training weights are restored exactly even if eval fails.
4. The **only** coupling: validation **metrics** are used to pick the *best* checkpoint (`best_checkpoint_eval_metric`). That selects which already-trained checkpoint to keep — it **does not feed gradients back** or alter optimization. (If you tune hyper-parameters by hand based on val metrics, that's indirect human-in-the-loop selection, not in-training leakage.)

**Conclusion:** training parameters are a pure function of the training data + optimizer. Validation is read-only and isolated.
