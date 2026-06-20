# Data Pipeline

All under `data_pipeline/`. The pipeline is built in `input_reader.py`; the detection and
distance datasets are separate streams merged at the batch level.

## Stage order

```
tfds.load (SkipDecoding: images stay ENCODED bytes through shuffle)
   → repeat each source dataset → sample_from_datasets(weights=[95,2,3])
   → shuffle (encoded bytes — KB/element, not MB; seed = self._seed)
   → decode (tf.string branch decodes inside parallel map)
   → pre-resize to 672² (uint8; preserves 'height'/'width' fields)
   → zip(cnp_dataset)  (cnp source shuffle seed = self._seed+1) → Copy-Paste
                           (copy_paste.py, prob 0.2)  ← BEFORE mosaic
   → padded_batch(group_size, padding_values=…)  ← polygons pad with -1.0
                           (sentinel), not 0.0 (a valid vertex coord); every key explicit
   → Mosaic                (mosaic.py): G in → G // R out (G = group_size 32,
                           R = decodes_per_output 4 → 8 outputs). Each output
                           independently flips mosaic_frequency (per-output, not
                           per-group); a mosaic draws 4 source images via a
                           width-4 window (step R) of one per-group random
                           permutation — R=4 tiles the permutation (4 distinct
                           images, zero cross-output reuse = stock YOLO), R<4
                           overlaps (reuse 4/R, varied partners). Each output runs
                           one random_perspective warp.
   → unbatch → shuffle(max(256, 4·group_size), seed=self._seed+2)  (disperses a
                           group's outputs; distinct seed from the two source shuffles)
   → parser polygon preprocessing  (yolo_parser.py / distance_parser.py)
                           parsers emit uint8 images — colour aug moved to GPU
   → batch(global_batch_size) + prefetch(AUTOTUNE)
   [train_step] HSV + Albumentations + /255 (batch_color_aug.py, on GPU;
                           Albumentations skipped for ignore_bg==1 distance rows)
```

The order matters: copy-paste augments *within* an image and must run before mosaic stitches
four images together. The pre-resize to 672² runs after decode and before copy-paste; the
`CopyAndPasteModule` reads the preserved `height`/`width` fields to scale pasted objects by
`(new/orig)` per axis, reproducing the relative size/placement of full-resolution compositing.
The geometric affine lives in the mosaic stage (`random_perspective`), applied to **both** the
mosaic and single-image branches — the parser no longer does an affine.

The training stream is **infinite** (each source dataset is `.repeat()`ed before
`sample_from_datasets`). Epoch length is enforced by the trainer (`steps_per_loop`, derived
from the config), not by data exhaustion. `tf.data.Options` sets
`deterministic=False` (removes head-of-line blocking in parallel maps) and optionally
`private_threadpool_size` (e.g. 13 in `yolov8_poly_dist.yaml` for cgroup-capped machines).

## Decoding — `tfds_decoders.py`

Both `tfds.load` sites pass `decoders={'image': tfds.decode.SkipDecoding()}` so images remain
as encoded JPEG/PNG bytes through the shuffle buffer (KBs per element instead of MBs). Each
decoder has a `tf.string` branch that decodes the bytes inside the parallel `map` call.

- `PolygonDecoder` — detection + polygon datasets.
- `ServingBotDetDecoder` — the distance dataset (carries `groundtruth_dists`). The ServingBot
  dataset has a single foreground class (id=0) that maps to class 35 in the main 39-class
  taxonomy. This remap is **hardcoded** in `configs/class_map.py:SERVINGBOT_CLASS_REMAP` — there
  is no JSON file. The decoder builds a full identity table `[0..num_classes-1]` and applies the
  `{0: 35}` override.
- `CopyPasteDecoder` — the copy-paste source TFDS (`cleaner_copy_paste:1.0.0`), RGBA images
  with a 4-channel alpha mask. Also loaded with `SkipDecoding`; the decoder uses `channels=4`.

## Distance stream merge — `input_reader.py:_merge_streams`
The distance dataset (`servingbot_polygon:1.0.1`) is an **independent, training-only** stream.
It is combined with the detection stream via `tf.data.Dataset.zip(...)` and concatenated on the
**batch dimension** (detection `global_batch_size` + the distance stream's batch). Distance-only samples set `ignore_bg=1`
so their class loss is masked to foreground (they have no detection labels for background).

> There is **no validation merge path** — `validation_data.distance_data` is `null`. Distance
> metrics come from the training stream only.

The merged stream gets a terminal `.prefetch(AUTOTUNE)` so the batch-concat overlaps the
training step; each sub-stream also prefetches internally.

## Parsers
- `yolo_parser.py:V8ParserExtended` — detection + polygon parsing, including
  `_preprocess_polygons_v2` (raw vertices → PolyYOLO radial target). The method uses an
  `unsorted_segment_max` / `segment_min` formulation (replacing the old `[N, P, 24]` one-hot
  expansion) — output-equivalent including ties, and tested for exact equality.
  The 672→672 resize is skipped when the static shape already matches (mosaic path).
  **Parsers now emit uint8 images.** Normalisation (`/255`), HSV jitter, and Albumentations
  have moved to `data_pipeline/batch_color_aug.py` and run inside `YoloV8Task.train_step`
  (GPU). `validation_step` casts uint8 to float32 and divides by 255 directly. This cuts
  host→device memory traffic by 4× and frees CPU tf.data workers from colour ops.
- `distance_parser.py:V8DistanceParser` — distance samples; encodes log-distance and sets
  `ignore_bg`. Also emits uint8; colour aug applied in the same `train_step` batch pass
  (Albumentations rows are gated by `ignore_bg==0` so distance-only rows skip it).
- Crowd handling: `skip_crowd_during_training=True` filters crowd annotations at parse time.

## Polygon formats (the three representations)

| Stage | Format | Notes |
|-------|--------|-------|
| TFDS input | `[N, max_vertices+2]` flat xy, normalized, **-1 padded** | both x and y are -1 for an invalid/padded pair |
| PolyYOLO target (loss) | `[N, 72] = [dist, angle, conf] × 24` interleaved | `tal_loss.py:_polygon_loss` reads `0::3`=dist, `1::3`=sub-bin angle offset, `2::3`=conf |
| Cartesian (transient, per matched pair) | `[K, 2]` pixel `(x, y)` | reconstructed from the radial format only at IoU time (`eval/polygon_metrics.py:_radial_to_cartesian`), conf-gated to `K ≤ 24` occupied bins; never persisted |
| Eval GT | `[N, 72]` radial (same as training GT) | GT is **not** converted to Cartesian — it stays in the radial format through eval |

**Radial encoding** (`_preprocess_polygons_v2`): for each of 24 angle bins, find valid vertices
whose angle from the box center falls in the bin and take the max-radius one; `dist` = that
radius, `conf` = 1 if any vertex present, and `angle` = that vertex's **sub-bin offset**
`(vertex_angle − bin_start)/angle_step ∈ [0,1)` (so the exact vertex angle is recoverable, not
just the bin). **Absent bins encode `dist = 0`, `angle = 0`, `conf = 0`** — so the distance head
learns to collapse non-existent vertices (intended PolyYOLO behavior). Decode uses
`vertex_angle = (i + angle)·angle_step` in `detection_generator` / `polygon_metrics` / `viz_utils`.

## Coordinate conventions
- GT boxes from decoders/parsers: **`yxyx` normalized** `[0,1]`.
- The loss/assigner convert to **`xyxy` pixels**.
- Mosaic image path is the **canvas formulation**: per-image `tf.image.resize` at the
  drawn scale → `_place_in_cell` crop/pad into the 2× canvas → ONE
  `apply_perspective_image` warp canvas→output (`M` drawn once via
  `augmentations.make_perspective_matrix`). The label path maps labels through
  `_scale_box_poly_to_canvas` → `transform_boxes_polygons(M)`. A composed-affine variant
  (per-quadrant affine folded into `M`, each source warped full-frame to the output) was
  implemented and MEASURED SLOWER on the production CPU (~95 ms·core/img vs ~35:
  `ImageProjectiveTransformV3` costs several times more per output pixel than
  `tf.image.resize` there, and the composed form pays 4 full warps per mosaic). Both
  formulations are geometrically identical — the label math never changed.
- The warp's scale gain is drawn from the **explicit** `[aug_scale_min, aug_scale_max]`
  config bounds (`make_perspective_matrix(scale_min=, scale_max=)`). The earlier symmetric
  magnitude form widened the configured `[0.4, 1.9]` to `[0.1, 1.9]`, occasionally shrinking
  content to ~1% area — the "mostly-gray frame" bug, since fixed.

## Performance notes
- Every `.map` uses `num_parallel_calls=AUTOTUNE`.
- The final consumed dataset ends in `.prefetch(AUTOTUNE)`.
- `SkipDecoding` keeps the shuffle buffer cheap: encoded JPEG/PNG bytes (≈ KB) instead of
  decoded float32 images (≈ 1.4 MB at 672×672). Decoding happens inside the parallel map.
- `tf.data.Options(deterministic=False)` is applied to the training stream (removes
  head-of-line blocking). `private_threadpool_size` (DataConfig field, default 0 = all cores)
  caps tf.data's worker count on cgroup-capped machines; `yolov8_poly_dist.yaml` sets it to 13.
- The post-unbatch `shuffle(128)` breaks up the 4-sample mosaic-group correlation clusters
  before the final `batch(global_batch_size)`.
- **Three pipeline changes target the three dominant CPU bottlenecks** (measured on the
  13-core-capped cloud host): pre-resizing before copy-paste (~18 ms·core/img at full-res),
  composed-affine mosaic eliminating the intermediate 2× canvas resize (~54 ms·core/img),
  and moving colour aug to GPU (~20 ms·core/img in the parser). Together they shift
  the heavy colour and geometry work off the CPU-capped tf.data threadpool.
- Colour augmentation (`batch_color_aug.py`) runs inside `train_step`; the `train/data_wait_ms`
  TensorBoard scalar (written by `YoloV8Trainer`) separates data-wait time from compute time,
  making it easy to tell whether the bottleneck is in tf.data or on the GPU.
- `parser.resample_points: 64` (both the detection and distance streams in
  `yolov8_poly_dist.yaml`) resamples polygons to 64 vertices at decode, so every downstream
  stage carries `[N, 128]` instead of the raw stored width (up to `[N, 10940]`). Arc-length
  resampling interpolates 64 points uniformly along the closed contour (interpolating on edges,
  NOT subsampling original vertices), so for polygons with more than 64 original vertices the
  radial target is within sampling tolerance, not exact; it is exact only for ≤64-vertex polygons.

## Polygon-GT correctness notes (train-semantics)

These govern the polygon ground truth the loss sees. Changing them alters the targets, so
they affect training — changing one mid-run would shift the GT a run is training against. See
`docs/design_register.md` for the `-1.0` sentinel and clip-to-edge conventions they build on.

- **`-1.0` is the only polygon sentinel.** Vertex validity is tested as `x > -1.0`, not
  `x >= 0.0`. A mosaic-canvas-overflow vertex with a slightly-negative input-normalized
  coordinate that lands in-view is a *real* vertex: `transform_boxes_polygons` transforms and
  **clips it to the edge** (consistent with the box GT for the same overflow), rather than
  dropping it as padding.
- **`padded_batch(4)` pads polygons with `-1.0`.** `input_reader` installs an explicit
  per-key `padding_values` dict; `groundtruth_polygons` pads with `-1.0`, because the default
  `0.0` is a valid top-left vertex coordinate and 0-padded rows would read as real vertices.
  Every other key gets its natural empty (image 0, `''`, ints 0, boxes/area/dists 0.0,
  `is_crowd` False).
- **Copy-paste fits wide polygons by even resample, not truncation.** The cnp source decoder
  does not resample, so a pasted object can carry far more polygon columns than the resampled
  background. When `cur_cols >= n_poly_cols`, copy-paste evenly resamples the valid vertices to
  the column budget (`resample_polygons`) instead of keeping the first N (a leading contour arc
  that discards the far side and corrupts the radial target).
- **`resample_polygons` compacts scattered sentinels.** Copy-paste invalidates out-of-bounds
  vertices in place, producing `-1` sentinels *interleaved* with valid ones. `resample_polygons`
  stable-argsorts valid-first to compact the kept vertices to a prefix before evenly sampling;
  on decode-time prefix input the sort is a no-op and the output is byte-identical.

- If decode + pre-resize still dominate (see `tools/pipeline/diagnose_pipeline.py` stage table),
  build pre-resized dataset variants ONCE with `tools/pipeline/reencode_tfds_672.py`: stores 672²
  JPEG + `orig_height`/`orig_width` (which `PolygonDecoder` prefers, keeping the copy-paste
  resolution correction exact). Detection sets only — servingbot must stay full-resolution
  because the distance parser letterboxes (aspect-preserving), and copy_paste crops are RGBA.
  The pre-resize map skips already-672² images via `tf.cond`.
- Use `tools/benchmark_pipeline.py` for end-to-end throughput and
  `tools/pipeline/diagnose_pipeline.py` for stage-by-stage attribution (its stage order MUST mirror
  `InputReader._build_detection_dataset` — keep them in sync).
