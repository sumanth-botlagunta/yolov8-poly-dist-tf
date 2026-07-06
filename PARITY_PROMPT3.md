# Cross-codebase parity audit — round 3: four targeted extractions

You have both repositories in context, same conventions as rounds 1–2:
**old codebase** = `tf2-vision-yolo`, **new codebase** = `rvc-vision-model`.

Rounds 1–2 are done. This round is FOUR narrow items only. Same discipline:
EVIDENCE EXTRACTION, not analysis. Someone else judges impact — you must not.

## Hard rules — identical to rounds 1–2

1. **Verbatim code only.** Every claim backed by a quoted block with file path and
   line range. Whole functions when ≤ 80 lines. NEVER paraphrase old-codebase code.
2. **Never trust comments/docstrings** for shapes or behavior — derive from the ops.
3. **Resolve config values end to end**: code default (full signature, no `...`
   elision), the production params.yaml lines, final resolved value.
4. **No impact estimates, no recommendations.** Verdicts limited to
   `IDENTICAL` / `DIFFERENT` / `UNVERIFIABLE` with the evidence pair.
5. **`NOT FOUND` is a valid answer** (show the search you did). Guessing is not.
6. Old-codebase quotes are the primary deliverable; cut new-side quotes first if
   output length forces cuts.
7. If you hit output limits, end with `[CONTINUED]` and resume when prompted.

## C1. Loss aggregation — the exact end-to-end scale

Round 2 quoted `loss *= tf.cast(batch_size, ...)` and a
`cross_replica_aggregation(loss, num_replicas_in_sync)` call but not the bodies.
Quote, in full:

- The FULL body of `post_path_aggregation` (wherever it is defined).
- The FULL body of `cross_replica_aggregation` (wherever it is defined).
- The **origin of the `batch_size` variable** used at the `loss *= tf.cast(...)`
  call site: quote its assignment. Is it the config's `global_batch_size` (what
  number does the production params.yaml resolve it to, detection + distance?),
  or a `tf.shape(...)[0]` of a per-replica tensor?
- The base/orbit trainer's gradient section for this task: from
  `tape.gradient(...)` through `optimizer.apply_gradients(...)`, including ANY
  loss or gradient rescaling between them (division by num_replicas,
  `compute_average_loss`, loss-scale wrappers, anything).
- For ONE component loss (the classification loss): quote its normalizer — what
  exactly is summed to build the denominator (per-replica batch or global)?

## C2. Polygon distance term — what the LOSS trains, not what decode does

Round 2 showed decode-side `pred_dist = tf.exp(...)`. Now the training side:

- Quote the polygon **distance** loss term in the old loss file: the lines that
  transform the raw head output before comparing with the target. Is the
  prediction passed through `exp`, `softplus`, `sigmoid`, or nothing in TRAINING?
- Same for the polygon **angle** and **conf** loss terms (the activation applied
  to the raw prediction inside each term).
- If the loss trains one activation and the decode applies another, quote both
  side by side and mark DIFFERENT-WITHIN-OLD.

## C3. NMS score assembly — what is "objectness"?

Round 2 quoted score = `sigmoid(cls) × sigmoid(objectness)` in the old detection
generator. YOLOv8-style heads have no objectness branch, so:

- Quote the full score-assembly block in the old detection generator, with 10
  lines of context above (so the origin of each input tensor is visible).
- Trace the "objectness" tensor to its source: which head output / branch in
  the old head file produces it? Quote that branch's construction (channel
  count, layer). If it is the polygon conf channel or a slice of another
  output, show the slicing.
- State whether THIS code path is the one used for the reported validation
  metrics (quote the eval/task wiring that calls it), or an alternate/legacy
  path that production eval bypasses.
- Per-class or class-agnostic NMS, and the pre-NMS top-k, while you are there.

## C4. `with_distance: false` — what it gates and where distance labels come from

- Quote EVERY place the old code reads `with_distance` (parser, task, loss,
  model build) — the surrounding branch for each read, so what it enables or
  disables is visible.
- Quote where the old distance-stream parser reads/attaches the ground-truth
  distance values to the training example (the feature key and the tensor it
  lands in).
- Quote the old distance loss masking: which samples contribute to the distance
  loss numerator (the mask construction).
- End state to resolve: which dataset's samples actually train the distance
  head in the old production run.

## Output

Markdown, sections C1–C4, each ending with its verdict line(s). Then one final
table: item → verdict, and any NOT FOUND with the searches attempted.

Produce the report now, starting with C1.
