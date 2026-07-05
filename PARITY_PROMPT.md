# Cross-codebase training-parity audit — full extraction instructions

You have both repositories in context, with the conventions already established:
**old codebase** = `tf2-vision-yolo`, **new codebase** = `rvc-vision-model`.

Goal: an exhaustive, purely factual inventory of what the OLD codebase's TRAINING
path actually computes, item by item, against the new one. This is an EVIDENCE
EXTRACTION task, not an analysis task. Someone else will judge impact — you must
not.

## Hard rules — read twice, follow exactly

1. **Verbatim code only.** Every claim must be backed by a quoted code block with
   the file path and line range. Quote whole functions when they are ≤ 80 lines;
   otherwise quote the relevant block plus 5 lines of context on each side.
   NEVER paraphrase, summarize, or reconstruct old-codebase code from memory.
2. **Never trust comments or docstrings for tensor shapes or behavior.** Comments
   may be stale or wrong. Derive every shape from the constructing operations
   (the `tf.reduce_*` axis, the `tf.stack`/`reshape` arguments, the shape of the
   inputs at the call site) and show the derivation chain: "X is built at
   <file:line> as tf.stack([...], axis=...) over inputs of shape [...] → X is
   [B, M, A]".
3. **Resolve every config value end to end.** For each parameter report three
   things: (a) the default in code (quote the signature line — the FULL
   parameter list, never elide with `...`), (b) the value in the exact YAML/params
   file the production training used (quote the YAML lines), (c) the final
   resolved value. Name the config file path you used. If multiple config files
   exist and you cannot determine which one the 0.83 training used, list all
   candidates and mark the item UNVERIFIABLE — do not pick one silently.
4. **No impact estimates.** No percentages, no "this could explain X%", no
   severity ranking, no recommendations, no fixes. Verdicts are limited to
   exactly three words: `IDENTICAL`, `DIFFERENT`, `UNVERIFIABLE` — each followed
   by the evidence pair (old quote, new quote).
5. **`NOT FOUND` is a valid and welcome answer.** Guessing is not. If a function,
   file, or config key does not exist in the old codebase, say `NOT FOUND` and
   show the search you did (which directories/files you checked).
6. **The old codebase is the primary deliverable.** Quote the new codebase only
   as the comparison anchor. If output length forces cuts, cut new-side quotes,
   never old-side quotes.
7. **One item = one verdict.** Do not split a single code block into multiple
   difference entries, and do not report the same lines under two sections. If a
   difference spans sections, report it once and cross-reference its ID.
8. If you hit output limits, end with `[CONTINUED]` and resume exactly where you
   stopped when prompted "continue". Never silently truncate a quote.

## Output format

Markdown. Use the section IDs below (A0–A12). Within sections, give every
difference a stable ID `D-<n>` with a one-line factual summary, then the evidence
pair. End the whole report with:
- a coverage table: section → items checked → verdict counts,
- the list of every file you quoted (path + which sections),
- the list of every UNVERIFIABLE / NOT FOUND item in one place.

## Sections

### A0. Inventory and configs
- Directory listing (file names only) of the old repo's training-relevant dirs:
  losses/, dataloaders or ops/, model/, model/optimization/, configs or params/.
- The complete, verbatim content of the exact params/config file used by the
  production training run (the one that reached the accepted accuracy). If it
  chains/includes other files, quote those too.
- Any training launch script / trainer entry point: quote the argument defaults.

### A1. Image input contract
Trace one training image from decode to the tensor that enters the model:
every cast, divide, normalize, resize, pad/letterbox op — quoted in order with
file:line. State the final pixel value range ([0,1]? [0,255]? mean/std
subtracted?) and the final dtype. Do the same for the validation/eval path if it
differs.

### A2. Augmentation pipeline — ordered list with parameters
The full ORDERED sequence of train-time augmentations in the old codebase, each
with its resolved parameters (rule 3):
- mosaic: probability; canvas size; center/split-point distribution (quote the
  random draw); PER-TILE scale handling (quote the resize/jitter call for each
  tile and the random range); how tiles are cropped/placed; how labels are
  shifted/clipped;
- the post-mosaic affine/perspective warp: rotation (range AND whether it is
  gated by a probability), scale range, translate, shear, perspective; whether it
  applies to non-mosaic images too;
- flip (probability, label handling);
- HSV/color: quote the exact ops — is brightness additive or multiplicative? the
  three ranges;
- copy-paste: source dataset, probability, min-size gates, alpha blending, how
  pasted labels (box + polygon) are constructed;
- mixup: probability and formulation;
- letterbox/resize policy outside mosaic;
- any schedule that changes augmentation over training (e.g. disabling mosaic in
  final epochs).

### A3. Ground-truth construction
- Box format at each stage (y-first or x-first; normalized or pixel; when
  converted) — quote the conversion sites.
- Train-time filters: min area/side, crowd handling (is `is_crowd` dropped at
  parse time? quote), dontcare handling, max instances (value + truncation
  policy).
- Polygon radial target: number of bins; how the per-bin distance is chosen
  (max? first? mean?); what the ANGLE target is (one-hot bin? continuous
  offset within bin? quote the construction); what the CONF target is; the
  padding/sentinel value; whether stored vertices are resampled before binning
  (and how many points).
- Distance (depth) target: valid range, sentinel, log-scale or linear.

### A4. Anchor/grid generation
Quote the grid construction: offsets (0.5?), stride set, level order, (x,y) vs
(y,x) ordering of the anchor points tensor, units (pixels vs normalized).

### A5. Assigner (task-aligned or other)
- Quote the ENTIRE assigner call chain (all functions), old codebase.
- Derive (rule 2) the axis layout of: the alignment metric, the overlaps tensor,
  the candidate mask, `pos_align_metrics`, `pos_overlaps`, the normalized metric,
  and the final target_scores multiplier. State explicitly, with derivation:
  is each reduction per-GT (over anchors) or per-anchor (over GTs)?
- alpha, beta, topk, eps values (rule 3).
- The spatial candidate constraint (center-in-box? radius?), quoted.
- Duplicate-anchor resolution (max-IoU? order?), quoted.
- Whether any part is wrapped in stop_gradient.

### A6. Losses — every component
For each of: classification, box/IoU, DFL, polygon-angle, polygon-distance,
polygon-confidence, depth/distance:
- the exact formula as implemented (quote), including sigmoid/BCE-with-logits
  choices and any label smoothing;
- the exact masking (which anchors/vertices/bins contribute);
- the exact NORMALIZER (sum of target scores? object count? mask sum? quote its
  computation — this is high priority);
- the loss gain/weight and its resolved value (rule 3);
- for DFL: reg_max/bin count and the integral/expectation decode;
- for IoU loss: which variant (quote the full IoU function);
- how ignore_bg / background-only samples are excluded from each loss.
- The final total-loss assembly line (quote) including any overall multipliers
  and the batch-size or replica scaling.

### A7. Batch composition and the depth/distance stream
- Does the old training merge a separate distance/depth dataset into each batch?
  If yes: quote the merge (zip? concat? interleave?), the two batch sizes, and
  how detection losses are masked for those rows (and vice versa).
- If no: how was the distance head trained?

### A8. Optimizer and schedules
- The FULL optimizer class `__init__` signature (every parameter, no `...`) and
  the apply/update method — quoted whole.
- Parameter grouping: quote the code that decides which variables get weight
  decay and/or different LR (the key-matching logic and the key lists from
  config). State exactly: how many distinct LR trajectories exist during warmup,
  and for each group: start LR → end LR.
- Weight decay: where it is added (into gradient before momentum? applied to
  weights directly?), quoted.
- Momentum: value, nesterov flag, momentum warmup (start value, steps).
- LR schedule: base LR, decay type, decay steps, and the warmup wrapper — quote
  the warmup class and its resolved init/final LR and step count.
- Gradient clipping: type (global norm? value?), threshold, where applied.
- EMA: decay formula (constant? ramp?), update frequency, whether eval and/or
  checkpoints use EMA weights, any warm-up on the EMA.

### A9. Training loop mechanics
- global batch size, gradient accumulation (if any), steps per epoch, total
  steps/epochs for the production run (from config or logs).
- Mixed precision policy and loss scaling.
- Multi-GPU strategy and how gradients/losses are reduced across replicas.
- Evidence of STAGED training for the production checkpoint: init_checkpoint
  chains, restart configs, LR-restart values — quote any configs/logs found
  (the goal is the lineage: what did the accepted run initialize FROM, and was
  its LR schedule started fresh or continued).

### A10. Initialization / warm start
- The production config's init/warm-start fields (quote): which checkpoint,
  which modules (backbone only? full model?), any frozen variables, and the code
  that performs the load (partial restore? name mapping?).
- Head initialization: bias init values (quote the smart-bias math if present),
  kernel initializers.

### A11. Eval protocol (secondary priority)
- Confidence threshold(s), NMS type/threshold, max detections, maxDets used for
  the F1/AP computation, the F1 sweep grid, crowd/iscrowd handling at eval,
  dontcare handling, any per-class forced-crowd list.

### A12. Smells
- Any TODO/FIXME/HACK/workaround comments inside the old training path (quote).
- Any code conditioned on specific dataset names, class ids, or class counts.
- Any dead/disabled code paths in the old training path that the config
  nevertheless references.

Produce the report now, starting with A0.
