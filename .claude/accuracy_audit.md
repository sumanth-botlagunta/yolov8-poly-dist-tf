# Detection Accuracy Audit

Branch: `claude/detection-accuracy-audit-sxrjtl` - Date: 2026-07-24

Scope: training recipe + codebase audit for detection accuracy (recall, precision, F1).
Six parallel domain audits (losses/assignment, data pipeline, sampling/class balance,
model/heads, optimization, config sanity), each finding re-verified against the code
before action. Evaluation logic, metrics, and test data were NOT modified - accuracy
stays measured on the same ruler.

---

## 1. Executive summary

**Single highest-impact change:** the copy-paste occlusion gate (`517fcd3`). Copy-paste
composited objects over the tile's real, labeled GT without ever touching those labels
(`data_pipeline/copy_paste.py` - the annotation update only *appended* the new object).
With placement deliberately biased to the lower image band, where this floor-facing
dataset's own objects concentrate, every heavily-occluding paste (prob 0.2/tile, on 50%
mosaic samples, ~300 epochs) trained the model on a box/class pointing at pixels that
are no longer that object. That is direct wrong-label supervision against both
precision and recall; it is now skipped when a paste would cover >50% of any existing
GT box.

**Top 3 ranked:**

1. **Copy-paste buries existing GT** - HIGH, IMPLEMENTED (`517fcd3`).
2. **Gradient clipping disabled** (`gradient_clip_norm: 0.0` in every tier YAML while
   the reference recipe clips at 10.0 and the clip machinery sat implemented but
   unarmed) - MEDIUM, IMPLEMENTED (`4191dce`).
3. **Sampling weights front-load the misrecog sources** ([95, 2, 3] vs true proportions
   ~ [0.99, 0.0015, 0.009]: both small sources exhausted in the first ~8%/~29% of every
   epoch, leaving the epoch tail - and every epoch-boundary checkpoint - pure cleaner
   data) - MEDIUM, IMPLEMENTED (`b7b0ced`).

Also notable: `close_mosaic` was implemented and tested in the trainer but switched off
in every tier; it is now enabled for the final 10 epochs (`024129a`), matching the
Ultralytics default.

The loss/assignment stack and the model trunk audited **clean** - the accuracy levers in
this codebase are recipe/config-level, not loss-math bugs (see section 4).

---

## 2. Findings (ranked)

### F1 - Copy-paste never updates GT it occludes - HIGH - IMPLEMENTED `517fcd3`

- **Where:** `data_pipeline/copy_paste.py` - hard-mask composite
  (`hard_mask = alpha_canvas > 0.5; blended = tf.where(...)`) followed by an
  "update annotations" block that only appends the pasted object's box/class/polygon;
  nothing examined the tile's pre-existing `groundtruth_boxes`/`groundtruth_polygons`.
- **Mechanism:** an existing object fully/mostly covered by a paste keeps its stale
  label. TAL then assigns positive anchors to a box whose pixels are a different
  object: the cls head receives contradictory class supervision at that location
  (precision down), and the occluded class accrues no clean positives there (recall
  down). Placement bias (`height_limit=0.6` lower band) maximizes collision with real GT.
- **Fix:** after drawing the placement, compute the fraction of each existing GT box
  covered by the pasted object's box; skip the paste entirely when any fraction exceeds
  `max_occlusion_frac` (default 0.5 - mirroring the candidate-filter
  keep-if->=50%-visible convention). Sub-threshold overlaps still paste (natural
  occlusion is a useful signal). `None` restores old behavior.
- **Notes:** not covered by any `.claude/design_register.md` entry, so a gap rather than
  a documented trade-off. Train-semantics change -> fresh runs only.

### F2 - Gradient clipping disabled - MEDIUM - IMPLEMENTED `4191dce`

- **Where:** `gradient_clip_norm: 0.0` in all three tier YAMLs;
  `optimizers/sgd_warmup.py:140` no-ops when `clip_norm <= 0`. The clip path itself is
  correctly placed (post cross-replica sum, pre weight-decay coupling -
  PyTorch-equivalent).
- **Mechanism:** Ultralytics clips at `max_norm=10.0` every step. Without it, one
  pathological batch (extreme 0.4x mosaic warp, distance-stream outlier) injects an
  unbounded update that Nesterov momentum (0.937) replays for ~16 steps; late in the
  cosine schedule the trajectory can't recover, and EMA averages over the damage.
- **Fix:** `gradient_clip_norm: 10.0` in all tiers. Healthy steps are untouched;
  `train/grad_norm` (pre-clip) is already logged, so the actual spike frequency is
  checkable in TensorBoard history.

### F3 - Sampling weights front-load small sources - MEDIUM - IMPLEMENTED `b7b0ced`

- **Where:** `tfds_sampling_weights: [95, 2, 3]` in all tiers;
  `data_pipeline/input_reader.py:192-206` (`sample_from_datasets(...,
  stop_on_empty_dataset=False)` then `.repeat()` on the merged stream). Source sizes:
  268,384 / 416 / 2,366 (`trainer.train_total_examples` comment).
- **Mechanism:** weights bias interleave ORDER only (each image still seen once per
  lap - the documented semantics), but [95, 2, 3] over-draws field_misrecog ~13x and
  station_misrecog ~3.4x their true rate, exhausting them ~8%/~29% into each lap. The
  30k shuffle buffer (~11% of a lap) cannot undo it: the last ~60% of every epoch is
  pure cleaner data, and epoch-boundary checkpoints follow ~1,300 cleaner-only steps.
  The misrecog sources exist to fix specific FP/FN modes; front-loading causes
  within-epoch recency forgetting of exactly those modes (EMA partially absorbs it).
- **Fix:** weights proportional to source sizes `[268384, 416, 2366]` - all sources
  exhaust together, uniform spread, per-image frequency unchanged.

### F4 - close_mosaic implemented but off - MEDIUM - IMPLEMENTED `024129a`

- **Where:** `MosaicConfig.close_mosaic_epochs` default 0; no tier YAML set it. Trainer
  machinery complete (`train/trainer.py:342-371` `_maybe_close_mosaic`, covered by
  `tests/unit/test_close_mosaic.py`).
- **Mechanism:** mosaic's stitched canvases/warped scale statistics never occur at eval.
  Ultralytics disables mosaic for the final 10 epochs by default so BN stats, box
  regression, and score calibration settle on eval-time input statistics at the LR
  floor - a standard final-mAP/F1 gain.
- **Fix:** `close_mosaic_epochs: 10` in all tiers (copy-paste rides the mosaic branch
  and stops in the same window; step/epoch/LR accounting unchanged).

### F5 - Distance-stream shuffle window 200 - LOW/MEDIUM - IMPLEMENTED `4300ae1`

- **Where:** `distance_data.shuffle_buffer_size: 200` (poly_dist YAML), applied
  shuffle-before-repeat at `data_pipeline/input_reader.py:406`.
- **Mechanism:** each lap's order ~ shard read order beyond a 200-element window -> the
  16 distance rows merged into every batch are temporally correlated (same
  scene/session), i.e. noisy correlated gradients into the distance head and the shared
  box/cls losses those rows legitimately train.
- **Fix:** 2000 (records stay encoded through shuffle via `SkipDecoding`; cheap).

### F6 - Eval NMS top-1 class masking caps recall on confusable classes - MEDIUM - PROPOSED

- **Where:** `models/detection_generator.py:159-162` masks every anchor to its argmax
  class before NMS; the per-class branch then consumes `scores_masked[:, c]` (line 187).
- **Mechanism:** Ultralytics validates `multi_label=True` - an anchor may emit a
  candidate per class above threshold, so a correct-but-second-ranked class can still
  match GT. Here that detection is structurally impossible: a confusable pair costs an
  unrecoverable FN regardless of the confidence sweep. Design register #17 covers
  per-class-vs-agnostic scope, not top-1-vs-multi-label - this is not documented intent.
- **Why proposed, not implemented:** the change alters eval-time predictions with zero
  training change, i.e. it moves measured numbers without model improvement. Per the
  audit ground rules ("same ruler before and after") this belongs to the maintainer:
  if adopted, feed the unmasked `scores[:, c]` to each class's NMS (a one-line change in
  the `per_class` branch), ideally behind a `detection_generator` config knob with the
  current behavior as default for deploy parity, and A/B one checkpoint through both
  modes exactly like the register-#17 agnostic-NMS experiment.
- **Expected effect:** recall/mAP tail up on visually confusable class pairs.

### F7 - No class-balancing mechanism for 39 classes - MEDIUM - PROPOSED

- **Where:** sampling is source-weighted only (`input_reader.py`); mosaic tile choice is
  class-blind; ACSL is a documented fail-loud stub (`losses/tal_loss.py:229-235`, design
  register #11). The dead ACSL YAML block itself declares 16 of 39 classes "rare".
- **Mechanism:** cls BCE gradient mass is proportional to positive-anchor count, so
  rare-class logits stay under-calibrated -> low per-class recall at the F1score50
  operating point, dragging macro F1. (TAL assignment itself is per-GT and fair.)
- **Options (ascending invasiveness):** (a) a constant `[39]` per-class weight vector
  on the cls BCE in `TaskAlignedLossExtended._class_loss` (`losses/tal_loss.py:384-412`)
  - smallest change, easy A/B; (b) implement ACSL (config already plumbed; requires
  re-calibrating `cls_gain`, per the register); (c) rare-class-rich source oversampling
  via `.repeat(k)` *before* `sample_from_datasets` (the code comment at
  `input_reader.py:186-191` documents exactly this mechanism, deliberately unused).
  All are train-semantics; need per-class GT counts + a per_class/ TB baseline to
  design the weights (see section 5).

### F8 - Weight decay not batch-scaled - LOW/MEDIUM - PROPOSED

- **Where:** `weight_decay: 0.0005` (poly_dist YAML), consumed at
  `optimizers/sgd_warmup.py:157-171`, correctly kernels-only, coupled like PyTorch SGD.
- **Mechanism:** Ultralytics scales `wd *= batch x accumulate / 64`; at batch 128 the
  reference-equivalent decay is 0.001. At 5e-4 with half the steps/epoch of nbs-64, the
  cumulative per-epoch shrink is ~half the reference -> slightly weaker regularization
  over 300 epochs (typically a small late-training val-precision gap).
- **Why proposed:** the literal reference value (5e-4 on kernels) is satisfied and the
  legacy checkpoints trained under it - this is a hyperparameter judgment call.
  Experiment: one run at `weight_decay: 0.001`, compare best-F1score50 and the late
  val-precision curve.

### F9 - `mosaic.area_thresh` had three contradicting defaults - LOW (latent) - IMPLEMENTED `9323580`

- **Where:** dataclass default 0.1 + comment claiming "reference/legacy value"
  (`configs/model_config.py`) vs loader fallback 0.5 (`configs/yaml_loader.py`) vs
  `Mosaic.__init__` 0.5 (`data_pipeline/mosaic.py:252`).
- **Mechanism:** a YAML omitting the key would silently train at 0.5 while a
  programmatic `MosaicConfig()` said 0.1 - the exact silent-divergence class the earlier
  `mosaic_center` fix addressed; a wrong candidate-filter threshold silently drops
  partially-visible GT (recall down). All shipped YAMLs set it explicitly, so no live
  run was affected. `tests/test_mosaic.py::test_candidate_filter_legacy_parity` pins the
  intended per-tier values (poly_dist 0.5 legacy, bbox/poly 0.1 deliberate deviation).
- **Fix:** dataclass aligned to 0.5 with a comment naming the per-tier convention.
  (Remaining latent cousin: `Mosaic.__init__` defaults `decodes_per_output=4` vs config
  default 1 - always passed explicitly by `input_reader`, left as-is.)

### F10 - Six config sections had no unknown-key warning - LOW (preventive) - IMPLEMENTED `765c262`

- **Where:** `configs/yaml_loader.py` - top-level `task:`, `norm_activation`,
  `backbone`/`darknet`, `decoder`/`yolo_decoder`, `head`, and the `distance_data` body
  were all pulled with `.get()` and no `_warn_unknown_keys` coverage.
- **Mechanism:** a typo in the sections holding `num_classes`, `activation`,
  `gradient_clip_norm`, or `ignore_bg` silently trains a different model than the YAML
  claims. Now warned; all shipped YAMLs load warning-free (verified).

### F11 - CLAUDE.md mis-documented the single-path candidate filter - LOW (doc) - IMPLEMENTED `d7623de`

- **Where:** CLAUDE.md claimed the non-mosaic path has "no min_side floor"; the code
  applies the ~2px floor uniformly (`data_pipeline/mosaic.py:596`, `:893`, `:1069`) and
  `test_candidate_filter_legacy_parity` pins that as intended legacy parity.
- **Resolution:** initially flagged as a possible small-object recall regression on
  singles; the test resolves the intent in favor of the code, so the doc was fixed, not
  the filter. (If sub-2px labels are ever wanted back, that is a deliberate
  train-semantics experiment, not a bug fix.)

### F12 - Minor items - LOW - PROPOSED (cleanup only)

- `distance_data.parser.aug_rand_hue/saturation/brightness` are dead knobs: batch color
  aug reads `train_data.parser` for all merged rows (`train/task.py:317-326`); the
  distance-block values are stored but never applied (`distance_parser.py:56-58`).
  Remove them or warn when they differ from the detection stream's.
- `parser.dummy_distance` accepted and never read (`yolo_parser.py:61,82`);
  `distance_data.parser.with_polygons: true` in the YAML is ignored (data-level `false`
  wins, `input_reader.py:667`). Cosmetic; remove.
- `task.min_distance/max_distance` (detection-generator clamp) and
  `distance_data.parser.min_meter/max_meter` (training clip) are independent - nothing
  validates they agree. Add an equality check in `_validate_config`.
- bbox/poly tiers keep `shuffle_buffer_size: 1500` for the same ~271k-image stream the
  poly_dist tier shuffles with 30000 - looks stale; raise if those tiers are trained
  again.
- `sample_from_datasets(seed=...)` before `.repeat()` may replay the same source-choice
  sequence each lap (cosmetic - file order and the 30k buffer still reshuffle content).
- TAL top-k uses a per-anchor `> eps` guard vs the reference's per-GT keep-all-k
  (`losses/tal_assigner.py:164-173`) - immaterial under the shipped `weighting: soft`
  (dropped anchors carry near-zero target_scores); would matter under `legacy_hard`.

---

## 3. Implemented changes (per commit)

| Commit | What | Why | Validation | Confirming run |
|---|---|---|---|---|
| `517fcd3` | Copy-paste occlusion gate (`max_occlusion_frac=0.5`) in `data_pipeline/copy_paste.py` | Stop wrong-label supervision from pastes burying real GT | 3 new deterministic gate tests; full copy-paste suite 11/11; mosaic suite green | Fresh poly_dist run; watch precision + per-class recall of lower-band classes |
| `4191dce` | `gradient_clip_norm: 10.0` (all tiers) | Reference-recipe clip; bound momentum-amplified spike damage | Config loads; clip path covered by `tests/unit/test_sgd_warmup.py` | Same run; compare `train/grad_norm` spike epochs vs val dips on history |
| `b7b0ced` | `tfds_sampling_weights: [268384, 416, 2366]` (all tiers) | Uniform lap spread of misrecog sources; per-image frequency unchanged | Config loads | Same run; check within-epoch stability of misrecog-mode FP/FN |
| `4300ae1` | `distance_data.shuffle_buffer_size: 2000` | Decorrelate the 16 merged distance rows | Config loads | Same run; distance-loss variance in TB |
| `024129a` | `close_mosaic_epochs: 10` (all tiers) | Standard Ultralytics endgame on clean image statistics | Config loads; `tests/unit/test_close_mosaic.py` 3/3 | Same run; expect a val bump across the final 10 epochs |
| `9323580` | `MosaicConfig.area_thresh` default 0.1 -> 0.5 + corrected comment | Kill silent default divergence (dataclass vs loader vs module) | Legacy-parity + config tests green | None needed (no live-config behavior change) |
| `765c262` | Unknown-key warnings for 6 uncovered config sections | Typos in task/model/distance sections no longer silent | All 4 shipped YAMLs warning-free; injected typos warn; config tests 22/22 | None needed |
| `d7623de` | CLAUDE.md candidate-filter correction | Doc/code mismatch resolved per the pinning test | Doc-only | None needed |

All train-semantics changes (`517fcd3`, `4191dce`, `b7b0ced`, `4300ae1`, `024129a`)
apply to **fresh runs only** per repo convention - do not merge into a run in flight.
Recommended confirmation: one full poly_dist run vs the last baseline on the unchanged
eval ruler (`F1score50` best-checkpoint + `per_class/` TB sections). The changes are
independent; if attribution matters, stage `517fcd3` (data hygiene) separately from the
recipe trio (`4191dce`+`b7b0ced`+`024129a`).

Full validation state on the branch: `tests/unit` 370 passed / 3 failed - the 3
(`test_freezing.py` x3) fail identically on `main` (pre-existing Keras attribute
issues, unrelated); `tests/smoke` 5 passed; touched top-level suites
(`test_copy_paste`, `test_mosaic`, `test_parser`) 73+11 passed.

---

## 4. Verified sound

**Losses & assignment** (`losses/tal_loss.py`, `losses/tal_assigner.py`): alignment
metric `score^0.5 * CIoU^6` with clamped CIoU and padded-GT masking; soft target scores
*including* the `pos_overlaps` factor; duplicate resolution equivalent to reference
`select_highest_overlaps`; background NaN-safety; CIoU+DFL weighting/normalization
byte-matches reference (DFL clamp `reg_max-1.01`, floor/ceil CE); CIoU `alpha_v`
stop-gradient present; `ignore_bg` masks cls to fg-only on distance rows; normalizers
floored at 1; distance-stream GTs are correctly remapped (`SERVINGBOT_CLASS_REMAP`) and
legitimately train box/cls/DFL - no label-space pollution; merged-batch normalizer
pooling matches design register #5.

**Data pipeline** (fuzz-tested with synthetic eager runs at production parameters, by
the pipeline audit agent): mosaic scale/crop geometry box/polygon<->pixel exact (800+
trials); letterbox math is one shared implementation across train pre-resize, mosaic
content slicing, and eval - no train/eval divergence; copy-paste placement geometry
exact (300 trials); `-1.0` polygon sentinel consistent (`> -1.0` tests everywhere,
`padded_batch` pads polygons with -1); distance-stream batch merge schema-checked;
color-aug order train `/255 -> HSV -> albumentations(det rows)` vs eval `/255` only.

**Sampling/epochs:** once-per-lap coverage with `stop_on_empty_dataset=False` pinned;
three shuffle stages with disjoint seeds; `steps_per_loop = 271166//128 = 2118`,
`decay_steps = 635400 = 2118*300` exactly, drift warned; validation iterates to
exhaustion (`drop_remainder: false`) - every val image scored.

**Model/heads:** backbone/decoder faithful v8s (channels, C2f semantics, SPPF); BN
momentum 0.97 equivalent to PyTorch 0.03, eps 1e-3; DFL decode
(softmax -> projection -> stride) and anchor `(i+0.5)*stride` consistent across
head/loss/generator; smart bias init formula correct at 672 and applied before restore;
single sigmoid on cls; no pre-NMS truncation; `deploy: true` correctly forced off for
training. Known documented capacity limits (not bugs, retrain-scale): cls stem fixed at
128 ch at all levels (reference 128/256/512), box stem shared with the three polygon
branches, no P2 level.

**Optimization:** weight decay kernels-only, correctly coupled (no double decay via
regularizers); Nesterov math byte-matches PyTorch; momentum warmup 0.8 -> 0.937 over
exactly 3 epochs; cosine LR to 1e-4 ending exactly at train end; EMA =
`0.9999*(1-exp(-step/2000))` tracking all variables incl. BN stats, swap-in/out
try/finally-protected; checkpoint/resume preserves optimizer state, EMA shadows, and
step counters (documented intentional deviations: BN-group warmup ramp-down, register
#2; EMA ramp, register #16).

**Config plumbing:** `num_classes=39` = class map length = head channels;
`output_poly_size == 360//angle_step` enforced; every `losses.*`/`sgd`/`ema`/`trainer`
key traced to a consumer; eval letterbox = train pre-resize letterbox (shared
function); NMS params wired YAML -> generator.

---

## 5. Open questions

1. **Do gradient spikes actually occur?** `train/grad_norm` history on the last
   production run would confirm F2's practical impact (the fix is safe either way).
2. **Class-frequency table.** F7 needs per-class GT counts over the merged train stream
   (and per-class F1 from the `per_class/` TB section of the last run) to design either
   the BCE weight vector or ACSL tiers. The ACSL YAML's common/frequent/rare split
   suggests this data existed once - is it current?
3. **Occlusion-gate threshold (0.5)** mirrors the poly_dist candidate-filter convention
   but wasn't tuned; if the paste rate drops visibly, 0.6-0.7 would trade a little
   hygiene for volume. Also: the gate uses the pasted object's box as a proxy for its
   alpha mask - conservative for non-rectangular objects.
4. **Multi-label eval NMS (F6):** adopting it changes measured recall without model
   change. Decide whether the deployed decoder would also go multi-label (device parity
   is the constraint - see design register #12/#17), then A/B one checkpoint.
5. **servingbot_polygon train-split size** - assumed >> 200 (hence F5); if it is only a
   few thousand, 2000 effectively full-shuffles a lap, which is strictly fine.
6. **`close_mosaic_epochs: 10` vs EMA horizon:** the EMA time constant (~2k steps ~ 1
   epoch) is well inside the 10-epoch window, so eval weights fully settle on clean
   statistics - but if the window is ever shortened below ~3 epochs the EMA will still
   be blending mosaic-era weights at the final checkpoint.
7. **Dataset counts** (`train_total_examples: 271166`) were re-used from the YAML's
   builder-verified comment; `assert_cardinality=False` means a rebuilt/resharded
   dataset would silently shorten laps - re-verify on the next dataset rebuild
   (affects F3's proportional weights too).
