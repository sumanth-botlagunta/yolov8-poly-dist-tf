# Design Decisions

Non-obvious design choices in this codebase and the reasoning behind them. Several differ from
stock Ultralytics YOLOv8 — usually for parity with the original TF2-Vision codebase the model was
migrated from, or for a measured performance reason. The items that affect **training** (augmentation,
loss, normalization) define the data and optimization the current checkpoints were trained under, so
changing them means re-training.

---

## 1. Crowd policy: training filters crowds, COCO eval counts them

Training drops `is_crowd` annotations at parse time (`parser.skip_crowd_during_training=True`,
`data_pipeline/yolo_parser.py` / `distance_parser.py`), while COCO evaluation
(`eval/coco_metrics.py`, `ignore_iscrowds: false`) **counts crowd GT in the recall denominator** —
which is *stricter than the standard COCO protocol* (stock pycocotools sets `ignore = iscrowd`,
excluding crowds from recall). Additionally, every GT of the categories in `iscrowds_labels`
([6,13,24,36,37]) is force-flagged `iscrowd=1` at eval, giving those classes the lenient
intersection-over-detection-area IoU and multi-match absorption. Net effect: the model never sees
crowd GT during training but is scored against required crowd GT at eval — an undetected crowd
region is an unrecoverable false negative that structurally caps those classes' best recall (and
the macro F1 they carry). This matches the original PolyYOLO behavior the checkpoints were trained
under. Training on crowd regions instead would change the training data and require a re-train.

## 2. BN/bias warmup LR ramps DOWN from `bias_lr_scale = 0.1`

During warmup, parameter groups 0 (BatchNorm) and 1 (bias) start at `bias_lr_scale` (default `0.1`,
an absolute LR — 10× the initial weight LR of 0.01) and **ramp down** to `base_lr`, while the weight
group ramps up from `0` to `base_lr` (`optimizers/sgd_warmup.py:_effective_lr` uses `bias_lr_scale`
directly as the group-0/1 start, not multiplied by `base_lr`). Ultralytics warms biases *up* from a
higher `warmup_bias_lr`. The ramp-down direction matches the original TF2-Vision codebase the model
was migrated from, and the checkpoints were trained under it.

## 3. HSV brightness is additive

The value/brightness channel of HSV augmentation uses additive jitter
(`tf.image.random_brightness(image, val)`, `data_pipeline/augmentations.py`) — a ±`val` offset,
not the multiplicative gain stock YOLO uses. The additive form matches the original codebase and the
augmentation distribution the checkpoints were trained under.

## 4. Distance head is not evaluated at validation time

The distance dataset (`servingbot_polygon`) is a training-only stream merged via
`tf.data.Dataset.zip()` on the batch dimension; there is no validation merge path, so the distance
head is never scored during validation (`eval/distance_metrics.py` exists but is not wired into the
val loop). Distance regression is therefore trained but not validated — a distance-head regression
would not surface in val metrics. A future change could add a distance validation stream and wire
`distance_metrics` into the val step.

## 5. `num_objs` / `target_scores_sum` normalizers include distance rows

The loss normalizers count GT objects across **both** the detection and distance streams of the
merged batch: `num_objs` is the total GT object count over both streams, and `target_scores_sum` is
computed over the full assembled batch (`losses/tal_loss.py`, `losses/polygon_loss.py`).
Distance-only samples (carrying `ignore_bg=1`) contribute to these denominators. The distance rows
are real batch elements, so counting them keeps per-object loss scaling consistent across the merged
batch. Under `losses.weighting: legacy_hard` (selectable; the tier YAMLs run `soft`) the detection
losses normalize by `num_objs` as well, sharing one merged-batch object-count denominator.

## 6. Polygon **conf** loss = BCE over all 24 bins

`polygon_conf_loss` (`losses/polygon_loss.py`) averages binary cross-entropy over **all** 24 angular
bins (occupied → 1, empty → 0), not only the occupied bins. Conf is the decode gate — at inference it
decides which bins emit a vertex — so it has to be trained on negatives (empty bins) to learn to
suppress them; a valid-bin-only mask never teaches it to output low confidence on empty bins. The
earlier masked-to-valid-bins form is kept in the `polygon_conf_loss` docstring as a one-line swap.

## 7. Polygon **angle** and **dist** losses are masked to valid bins

By contrast with conf, `polygon_angle_loss` and `polygon_dist_loss` average **only over occupied
bins**. Angle (sub-bin offset) and dist (radial distance) are regression targets that are undefined on
empty bins — there is no vertex to regress toward — so including empty bins would inject meaningless
targets. Hence the asymmetry: conf over all bins, angle/dist over occupied bins only.

## 8. Mosaic image path = canvas formulation (not composed affine)

The mosaic assembles a 2× canvas and applies a single `random_perspective` (`data_pipeline/mosaic.py`)
rather than composing a per-image affine and warping each source directly. The composed-affine
formulation was **measured ~3× slower on the production CPU** data pipeline
(`ImageProjectiveTransformV3` costs several times more per output pixel than `tf.image.resize`, and
the composed form does 4 full warps per mosaic vs 4 cheap resizes + 1 warp). The two forms are
geometrically identical; the canvas form is the faster one on the target hardware. Re-measure
throughput on the target hardware before changing it.

## 9. Clip-to-edge polygon convention

Geometric transforms (`random_perspective` in `data_pipeline/augmentations.py` / `mosaic.py`) clip
polygon vertices to the image edge rather than dropping polygons that partially exit the frame. This
keeps a partially-visible object's polygon GT consistent with its (also-clipped) box GT, so the
polygon and box heads see a coherent target for the same object.

## 10. `-1.0` is the reserved polygon sentinel coordinate

Padded/invalid polygon vertices are stored as the coordinate value `-1.0` (TFDS input is padded with
`-1`; targets reserve `-1.0` for absent vertices). Any value `> -1.0` is a real vertex (possibly
outside `[0,1]`, e.g. from mosaic-canvas overflow); `== -1.0` means "no vertex here." Code that scans
for valid vertices uses the `> -1.0` test, **not** `>= 0.0` — a legitimately-negative canvas
coordinate is a valid vertex. No transform should produce `-1.0` as a real coordinate.

## 11. ACSL config knob is parsed but not implemented (fails loud)

`AcslConfig` (`configs/model_config.py`) and its YAML block (`acsl: { use_acsl, bg_*_ratio,
common_cls, frequent_cls, rare_cls, threshold }`) describe an Adaptive Class Suppression Loss
weighting scheme. The config is fully parsed (`configs/yaml_loader.py`), but the weighting math is not
implemented in `TaskAlignedLossExtended._class_loss` — choosing a formulation and re-calibrating
`cls_gain` against it is a training-semantics choice, not a mechanical addition.

So the knob doesn't silently lie (its earlier behavior: `use_acsl: true` trained identically to
`false`), `TaskAlignedLossExtended.__init__` raises `NotImplementedError` when `use_acsl=True`, and
`train/task.py:build_losses()` passes the config value through so the guard is reached. All shipped
YAMLs set `use_acsl: false`, so this affects no current run. When ACSL is implemented, replace the
raise with the weighting and update this entry.

## 12. The single exporter bakes `/255` for the on-device DLC contract

`utils/export/export_saved_model.py` is the only exporter, and it reproduces the Qualcomm SNPE DLC
contract (see [device_export.md](device_export.md)). The on-device raw-image generator feeds raw
`[0,255]` float32 (`IMAGE_NROM_FLAG=False`), so the exported graph divides by 255 internally
(`--normalize`, default on) to reach the `[0,1]` the model trained on — the model itself has no
internal `/255` layer (`models/yolo_v8.py`); in-repo call paths (`validation_step`, `utils/eval.py`)
normalize via `train.task.normalize_images`. The export emits raw head logits as concatenated nodes
(`box/cls/poly_angle/poly_dist/poly_conf/dist`, `[1, N, C]`, levels 3→4→5) instead of a deploy/NMS
dict, and runs in float32 (not the training `mixed_bfloat16`) for a clean SNPE graph. Host tools that
want deploy-style detections from that SavedModel rebuild them from the flat heads via
`utils/export/device_decode.py`.

## 13. `jitter` config knob is reserved (parsed, defaulted to 0.0, not wired)

`MosaicConfig.jitter` and `ParserConfig.jitter` (`configs/model_config.py`, parsed by
`configs/yaml_loader.py`) are stored but not forwarded to `Mosaic` or the parser; every shipped tier
sets `jitter: 0.0`, so today it is behavior-neutral. It is kept (rather than removed) because it
mirrors an upstream Ultralytics hyperparameter, preserving YAML compatibility and the option to wire
it later. Wiring it would perturb box/polygon coordinates per sample in the mosaic stage — extra
CPU-pipeline work and a change to the augmentation distribution the checkpoints trained under, so it
is an augmentation change with throughput and re-train implications, not a free toggle. To remove it
instead, drop the field from both dataclasses, the two `yaml_loader` parse sites, and the `jitter: 0.0`
lines in all three tier YAMLs together.

## 14. Representational ceilings of the radial polygon format

These are limits of the PolyYOLO radial format itself (identical in the legacy codebase), not
implementation defects:

- **Radial center = box center.** For concave shapes whose box center lies outside the polygon, a bin
  ray can cross the boundary twice; the per-bin MAX keeps only the farther crossing. Centroid-centering
  would not fix deep concavities and would break checkpoint compatibility.
- **Per-bin MAX distance = outer boundary only.** Holes and interior concavities are unrepresentable
  (they would need a second radius channel — a format change). Arc-length resampling already removes
  the worst practical symptom (empty bins along long edges).
- **Polygon conf gate 0.4 is a single module constant** (`eval/polygon_metrics.DEFAULT_POLY_CONF_THRESH`),
  shared by decode-viz and the eval metric so they cannot drift — but tuning eval recall also moves the
  TensorBoard overlays. If the conf operating point is ever retuned, consider promoting it to a config
  field.

## 15. Activation is relu everywhere (a swish parity claim was tested and rejected)

A cross-codebase audit claimed the reference model ran a swish backbone/decoder with a relu
head, which briefly landed here as a swish-trunk config. The claim failed verification: a
training run with the swish trunk tracked the relu runs' val curves with no separation, and the
maintainer confirmed relu is correct — the audit's swish extraction was wrong. All tier YAMLs
use `norm_activation.activation: relu` for the whole network. General caveat that remains true:
activation settings are train-semantics — activation layers hold no variables, so checkpoints
*load* across activation settings but weights are only meaningful under the activation they
were trained with. Do not flip activations on a run in flight.

## 16. EMA dynamic decay is the exponential ramp

`decay = average_decay × (1 − exp(−step/2000))` (`optimizers/ema.py`), the standard
YOLOv5/YOLOv8 ModelEMA ramp. Decay starts at 0 (shadow = live weights), passes ~0.63×average_decay at
one time constant (2000 steps), and is within 1% of `average_decay` by ~10k steps. The earlier
hyperbolic form (`min(average_decay, (1+step)/(10+step))`) saturated much later — the eval
weights averaged over a shorter horizon through the mid-training epochs, which skews mid-run
val-curve comparisons against reference runs even though the final converged numbers barely
differ. Eval-side only (the EMA never feeds back into training), but the swap changes which
weights are checkpointed as eval weights — treat as fresh-run.

## 17. Per-class NMS kept over class-agnostic (measured, not assumed)

The original codebase ran class-agnostic NMS (one suppression pass over all classes). Ours is
per-class (after top-1 masking), selectable via `detection_generator.nms_class_mode`. A
head-to-head on one checkpoint (same raw predictions through both
modes) showed agnostic **worse** overall on this data: the precision gain from removing
cross-class duplicates is smaller than the recall loss on region classes that legitimately
contain other objects (bathroom / entrance / doorway boxes get suppressed by the higher-scored
objects inside them). Indoor scenes are full of nested different-class objects, so per-class
suppression is the correct scope here; `per_class` remains the default and `agnostic` stays
available for diagnostics.
