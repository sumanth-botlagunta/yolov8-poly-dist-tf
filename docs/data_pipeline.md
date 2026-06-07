# Data Pipeline

All under `data_pipeline/`. The pipeline is built in `input_reader.py`; the detection and
distance datasets are separate streams merged at the batch level.

## Stage order

```
multi-TFDS weighted sampling
   → Copy-Paste            (copy_paste.py, prob 0.2)      ← BEFORE mosaic, on decoded data
   → Mosaic (4-image)      (mosaic.py, freq 0.5; MixUp freq 0.0)
   → Albumentations / flip / jitter / affine / HSV  (augmentations.py)
   → parser polygon preprocessing  (yolo_parser.py / distance_parser.py)
   → batch + prefetch
```

The order matters: copy-paste augments *within* an image and must run before mosaic stitches
four images together.

## Decoding — `tfds_decoders.py`
- `PolygonDecoder` — detection + polygon datasets.
- `ServingBotDetDecoder` — the distance dataset (carries `groundtruth_dists`).
- `CopyPasteDecoder` — the copy-paste source TFDS (`cleaner_copy_paste:1.0.0`), RGBA images
  with a 4-channel alpha mask.

## Distance stream merge — `input_reader.py:_merge_streams`
The distance dataset (`servingbot_polygon:1.0.1`) is an **independent, training-only** stream.
It is combined with the detection stream via `tf.data.Dataset.zip(...)` and concatenated on the
**batch dimension** (detection 128 + distance 16 = 144). Distance-only samples set `ignore_bg=1`
so their class loss is masked to foreground (they have no detection labels for background).

> There is **no validation merge path** — `validation_data.distance_data` is `null`. Distance
> metrics come from the training stream only.

The merged stream gets a terminal `.prefetch(AUTOTUNE)` so the batch-concat overlaps the
training step; each sub-stream also prefetches internally.

## Parsers
- `yolo_parser.py:V8ParserExtended` — detection + polygon parsing, including
  `_preprocess_polygons_v2` (raw vertices → PolyYOLO radial target).
- `distance_parser.py:V8DistanceParser` — distance samples; encodes log-distance and sets
  `ignore_bg`.
- Crowd handling: `skip_crowd_during_training=True` filters crowd annotations at parse time.

## Polygon formats (the three representations)

| Stage | Format | Notes |
|-------|--------|-------|
| TFDS input | `[N, max_vertices+2]` flat xy, normalized, **-1 padded** | both x and y are -1 for an invalid/padded pair |
| PolyYOLO target (loss) | `[N, 72] = [dist, angle_norm, conf] × 24` interleaved | `tal_loss.py:_polygon_loss` reads `0::3`=dist, `1::3`=angle one-hot, `2::3`=conf |
| Eval Cartesian GT | `[N, max_vertices, 2]` yx denormalized | for mask IoU |

**Radial encoding** (`_preprocess_polygons_v2`): for each of 24 angle bins, find valid vertices
whose angle from the box center falls in the bin and take the max-radius one; `dist` = that
radius, `conf` = 1 if any vertex present, and the single bin with the global max radius gets the
one-hot `angle_norm`. **Absent bins encode `dist = 0`, `conf = 0`** — so the distance head learns
to collapse non-existent vertices (intended PolyYOLO behavior).

## Coordinate conventions (read carefully)
- GT boxes from decoders/parsers: **`yxyx` normalized** `[0,1]`.
- The loss/assigner convert to **`xyxy` pixels**.
- Mosaic transforms map input-normalized → output-normalized using the **exact** scaled
  dimensions (`new_h/new_w`) returned by `_letterbox_resize_to` (not re-derived from padding).

## Performance notes
- Every `.map` uses `num_parallel_calls=AUTOTUNE`.
- The final consumed dataset ends in `.prefetch(AUTOTUNE)`.
- Use the `/benchmark` skill (`tools/benchmark_pipeline.py`) to measure throughput.
