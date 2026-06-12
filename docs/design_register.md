# Intentional Design Register

This file records **deliberate** design decisions in this codebase that a future
reviewer (human or automated) might otherwise re-flag as a bug. Each entry states the
behavior, where it lives, and a one-paragraph rationale. If you are about to "fix" one
of these, stop and read the rationale first: the current behavior is intentional.

Adding to this register: when a review settles that a surprising-looking behavior is
correct-by-design (not just deferred), record it here with its rationale and code
location so the conclusion is not re-litigated.

---

## 1. Crowd policy: train filters crowds, COCO eval counts them

Training drops `is_crowd` annotations at parse time
(`parser.skip_crowd_during_training=True`,
`data_pipeline/yolo_parser.py` / `distance_parser.py`), while COCO evaluation
(`eval/coco_metrics.py`) follows the standard COCO protocol and *includes* crowd
regions in its `iscrowd` matching. This is a **known train/eval mismatch**: the model
never sees crowd GT during training but is scored against a metric that accounts for
crowds. It is left as-is — a **team decision is pending** on whether to keep crowds out
of training (current PolyYOLO-parity behavior) or to teach the model on crowd regions.
Do not silently "align" the two paths; the divergence is recorded and owned.

## 2. BN/bias warmup LR ramps DOWN from `bias_lr_scale = 0.1`

During warmup, parameter groups 0 (BatchNorm) and 1 (bias) start at
`bias_lr_scale` (default `0.1`, an absolute LR — 10× the initial weight LR of 0.01
in the provided config) and **ramp down** to `base_lr`, while the
weight group ramps up from `0` to `base_lr`
(`optimizers/sgd_warmup.py:_effective_lr` uses `bias_lr_scale` directly as the
group-0/1 start, not multiplied by `base_lr`). Ultralytics warms biases up from a higher
`warmup_bias_lr` rather than down. This **ramp-down direction is legacy parity** with
the original TF2-Vision codebase the model was migrated from, and the live checkpoints
were trained under it. Differing from Ultralytics here is intentional; do not invert the
ramp to "match upstream" without a deliberate re-train decision.

## 3. HSV brightness is ADDITIVE

The value/brightness channel of HSV augmentation uses additive jitter
(`tf.image.random_brightness(image, val)`, `data_pipeline/augmentations.py`), i.e. a
±`val` *offset*, not a multiplicative gain. Standard YOLO HSV-V is a multiplicative
gain. The additive form is **legacy parity** with the original codebase and matches the
augmentation distribution the live checkpoints were trained under. Intentional; do not
convert to multiplicative without a re-train decision.

## 4. Distance head has no validation-time evaluation

The distance dataset (`servingbot_polygon`) is a **training-only** stream merged via
`tf.data.Dataset.zip()` on the batch dimension; there is no validation merge path, so
the distance head is never scored during validation (`eval/distance_metrics.py` exists
but is not wired into the val loop). This is a **known gap**: distance regression is
trained but not validated, so distance-head regressions would not surface in val
metrics. **Recommendation for a future change:** add a distance validation stream and
wire `distance_metrics` into the val step. Recorded here so the absence is not mistaken
for a wiring bug.

## 5. `num_objs` and `target_scores_sum` normalizers include distance rows

The loss normalizers count GT objects across **both** the detection and distance streams
of the merged batch: `num_objs` is the total GT object count over both streams, and
`target_scores_sum` is computed over the full assembled batch (`losses/tal_loss.py`,
`losses/polygon_loss.py`). Distance-only samples (carrying `ignore_bg=1`) therefore
contribute to these denominators. This is a **settled normalizer convention** — the
distance rows are real batch elements and counting them keeps per-object loss scaling
consistent across the merged batch. Do not re-scope the normalizers to detection-only.

## 6. Polygon **conf** loss = BCE over all 24 bins (as of 2026-06-11)

`polygon_conf_loss` (`losses/polygon_loss.py`) averages binary cross-entropy over **all**
24 angular bins (occupied → 1, empty → 0), not only the valid/occupied bins. Rationale:
conf is the **decode gate** — at inference it decides which bins emit a vertex — so it
must be trained on negatives (empty bins) to learn to suppress them; a valid-bin-only
mask would never teach it to output low confidence on empty bins. The earlier
masked-to-valid-bins form is **preserved in the `polygon_conf_loss` docstring** for
provenance. This over-all-bins form is intentional and current.

## 7. Polygon **angle** and **dist** losses are masked to valid bins

By contrast with conf (entry 6), `polygon_angle_loss` and `polygon_dist_loss`
(`losses/polygon_loss.py`) average **only over valid (occupied) bins**. Rationale: angle
(sub-bin offset) and dist (radial distance) are regression targets that are **undefined
on empty bins** — there is no vertex there to regress toward — so including empty bins in
their mean would inject meaningless targets. The asymmetry with conf (all bins) vs
angle/dist (valid bins) is deliberate and correct.

## 8. Mosaic image path = canvas formulation (not composed affine)

The 4-image mosaic assembles a 2× canvas and then applies a single `random_perspective`
(`data_pipeline/mosaic.py`), rather than composing a per-image affine and warping each
source directly. A composed-affine formulation was **measured ~3× slower on the
production CPU** data pipeline. Do **not** reintroduce the composed-affine path on the
basis of a code-cleanliness argument without first re-measuring throughput on the target
(cloud/CPU) hardware — the canvas path is a deliberate, measured performance choice.

## 9. Clip-to-edge polygon convention

Geometric transforms (`random_perspective` in `data_pipeline/augmentations.py` /
`mosaic.py`) clip polygon vertices to the image edge rather than dropping polygons that
partially exit the frame. This keeps a partially-visible object's polygon GT consistent
with its (also-clipped) box GT, so the polygon and box heads see a coherent target for
the same object. Intentional; clipping rather than dropping is the chosen convention.

## 10. `-1.0` is the reserved polygon sentinel coordinate

Padded/invalid polygon vertices are stored as the coordinate value `-1.0`
(TFDS input is padded with `-1`; targets reserve `-1.0` for absent vertices). Any value
`> -1.0` is treated as a real (possibly out-of-[0,1], e.g. mosaic-canvas-overflow)
vertex; `== -1.0` means "no vertex here." Code that scans for valid vertices must use the
`> -1.0` test, **not** `>= 0.0` — a legitimately-negative canvas coordinate is a valid
vertex, not a sentinel. The `-1.0` value is reserved and must not be produced by any
transform as a real coordinate.

## 11. ACSL config knob is parsed but not implemented (fails loud)

`AcslConfig` (`configs/model_config.py`) and its YAML block (`acsl: { use_acsl, bg_*_ratio,
common_cls, frequent_cls, rare_cls, threshold }`) describe an Adaptive Class Suppression
Loss weighting scheme. The config is fully parsed (`configs/yaml_loader.py`), but the
weighting math is **not implemented** in `TaskAlignedLossExtended._class_loss`. Choosing a
specific ACSL formulation and re-calibrating `cls_gain` against it is a training-semantics
decision, not a bug fix, so it is intentionally left unimplemented.

To prevent the knob from silently lying (its previous behavior: `use_acsl: true` trained
identically to `false`), `TaskAlignedLossExtended.__init__` now **raises
NotImplementedError when `use_acsl=True`**, and `train/task.py:build_losses()` passes the
config value through so the guard is actually reached. All shipped experiment YAMLs set
`use_acsl: false`, so this changes no current training run. When ACSL is implemented,
replace the raise with the weighting and update this entry.
