# Cross-codebase parity audit — round 2: MODEL GRAPH + unresolved items

You have both repositories in context, with the conventions already established:
**old codebase** = `tf2-vision-yolo`, **new codebase** = `rvc-vision-model`.

Round 1 audited the TRAINING path (config, augmentation, GT, assigner, losses,
optimizer) and found near-total parity. This round audits the two things round 1
did not cover: (I) the MODEL ARCHITECTURE and its output/decode contract, and
(II) the specific items round 1 marked NOT FOUND. Same discipline: this is
EVIDENCE EXTRACTION, not analysis. Someone else judges impact — you must not.

## Hard rules — identical to round 1, follow exactly

1. **Verbatim code only.** Every claim backed by a quoted block with file path
   and line range. Whole functions when ≤ 80 lines; otherwise the relevant block
   plus 5 lines of context. NEVER paraphrase or reconstruct old-codebase code.
2. **Never trust comments/docstrings for shapes, channel meanings, or behavior.**
   Derive from the constructing operations and show the derivation chain.
3. **Resolve every config value end to end**: (a) code default (quote the FULL
   signature — never elide with `...`), (b) the value in the exact production
   params file (quote the YAML lines), (c) final resolved value. If the config
   file is ambiguous, list candidates and mark UNVERIFIABLE.
4. **No impact estimates, no percentages, no recommendations.** Verdicts limited
   to `IDENTICAL` / `DIFFERENT` / `UNVERIFIABLE`, each with the evidence pair.
5. **`NOT FOUND` is a valid answer** (show the search you did). Guessing is not.
6. **The old codebase is the primary deliverable.** Cut new-side quotes first if
   output length forces cuts; never cut old-side quotes.
7. **One item = one verdict.** No double-reporting of the same lines.
8. If you hit output limits, end with `[CONTINUED]` and resume exactly where you
   stopped when prompted "continue". Never silently truncate a quote.

## Output format

Markdown, section IDs B0–B9 below. Differences get stable IDs `D-<n>` with a
one-line factual summary plus the evidence pair. End with: a coverage table
(section → items → verdict counts), the list of every file quoted, and all
UNVERIFIABLE / NOT FOUND items in one place.

## Sections

### B0. Model inventory and build entry
- File listing of the old repo's `model/` tree (backbones/, decoders/, heads/,
  layers/).
- The model-build entry point (`yolo_factory.py` / `yolo_model.py`): quote how
  the production config selects backbone/decoder/head classes and their
  constructor arguments (model id, depth/width multipliers, levels).

### B1. Backbone architecture (as built by the production config)
- Stem: op sequence, kernel sizes, strides, channels.
- Every stage in order: block class, number of repeats, in/out channels, stride,
  and how depth_scale/width_scale (or model-id presets) apply — quote the
  preset table/dict and the scaling arithmetic.
- Block internals (the CSP/C2f/Bottleneck equivalents): quote the full block
  `__init__`/`call` — conv counts, kernel sizes, split/concat topology,
  shortcut conditions.
- SPP/SPPF: present? params (pool sizes), position.
- Total: a layer-by-layer table (op, k, s, c_in, c_out) for the production
  backbone, derived from code.

### B2. Normalization and activation
- BatchNorm epsilon and momentum values (code default + config resolution),
  sync-BN or not.
- The activation function actually used in backbone / decoder / heads (quote the
  activation construction and the config value — e.g. relu vs leaky vs silu),
  including any places that hardcode a different activation.
- Any L2/weight regularizers, dropout, or DropBlock attached at layer level.

### B3. Decoder / neck
- The FPN-PAN wiring as built: upsample method (nearest/bilinear), lateral conv
  specs, concat order (which tensor first), block type + repeat count + channels
  per merge stage, downsample path convs. Quote the graph-building code.
- Level order convention (3→5 or 5→3) at input and output.

### B4. Heads — every branch
- For each output branch (box/DFL, cls, and the polygon angle/dist/conf and
  depth/distance branches): the conv stack (count, kernels, channels, norm,
  activation), the final projection layer (channels), whether stems are shared
  across branches or levels, per-level heads or shared weights.
- Output tensor layout: channel meaning and ORDER within the last axis (e.g.
  DFL: 4 groups × 16 bins — which coordinate order? l,t,r,b or t,l,b,r? x-first
  or y-first?), and the level order in the flattened output.
- Bias/kernel initializers for the final layers (quote the smart-bias math and
  any prior-prob init).

### B5. Output decode contract (where a migrated checkpoint would break)
- `dist2bbox` (or equivalent): quote it. State the coordinate convention at
  input (anchor points (x,y) or (y,x)) and output (xyxy/yxyx), and where the
  stride multiplication happens.
- DFL decode: softmax-expectation code, reg_max, applied before or after NMS
  assembly.
- Which head outputs get sigmoid and WHERE (in the model, in the loss, in the
  detection generator) — cls, poly_conf, poly_angle.
- Polygon decode to cartesian: the vertex angle formula (bin index + offset?
  one-hot argmax?), distance activation (softplus? exp?), conf gate threshold,
  the polygon center used (box center? anchor point?), and the units/scale of
  the radial distances (pixels at input res? normalized? relative to box size?).
- Distance (depth) head decode: activation and units (log-scale? meters?).
- NMS input assembly: score computation (cls sigmoid × objectness?), per-class
  or class-agnostic NMS, top-k pre-NMS.

### B6. Polygon radial TARGET construction (round-1 NOT FOUND: polygon_ops.py)
- The complete GT-side radial encoding: quote the function(s). Number of bins,
  the center the radii are measured FROM, per-bin reduction when multiple
  vertices fall in one bin (max/min/first?), the angle target value (bin
  one-hot? continuous offset within bin — formula?), the conf target, distance
  units/normalization, padding/sentinel handling, and any vertex resampling
  before binning (count, arc-length or index subsampling).

### B7. Round-1 leftovers
- **EMA** (`model/optimization/moving_average.py`): the full class — decay
  formula, what `dynamic_decay: true` does (quote the ramp math), which
  variables are shadowed, how/when EMA weights are swapped for eval and which
  weights go into checkpoints.
- **Loss aggregation scale**: quote `post_path_aggregation` and
  `cross_replica_aggregation` (and the `loss *= tf.cast(batch_size, ...)` call
  site context) — derive the exact end-to-end scale of the total loss w.r.t.
  per-example loss, batch size 128, and 2 replicas.
- **Mosaic per-tile jitter**: `resize_and_jitter_image` — full signature and
  body; what `jitter` mathematically does to the tile scale (formula + range);
  the resolved value of the Mosaic's `random_crop` (constructor default + the
  production config value if present).
- **Mosaic input feeding**: how the old input pipeline supplies the 4 images per
  mosaic — quote the dataset-side code (batching? windowing? fresh samples per
  mosaic?). State whether a source image can appear in more than one mosaic per
  epoch and how partner images are chosen.
- **`distance_data.with_distance: false`**: quote what `with_distance` gates in
  the old parser/task for the distance stream, and where the distance LABELS
  for training actually come from.

### B8. Model-level miscellany
- Any preprocessing baked into the model graph (rescale/normalize layers).
- `deploy` / inference-mode flags that change the graph between train and eval.
- Fused or custom ops; anything conditioned on dtype/precision.
- TODO/FIXME/HACK comments inside model/ (quote).

### B9. Coverage table, files quoted, UNVERIFIABLE/NOT FOUND list.

Produce the report now, starting with B0.
