# On-device export — Qualcomm SNPE DLC (drop-in replacement)

`utils/export/export_device_savedmodel.py` exports a trained checkpoint to a TensorFlow SavedModel
laid out as a **drop-in replacement for the deployed on-device DLC**. The existing SNPE
conversion → quantization → net-run → result-extraction pipeline keeps working
**unchanged**; only the SavedModel path changes.

This is distinct from `utils/export/export_saved_model.py`, which bakes NMS into the graph and
emits the post-processed deploy dict for `[0,1]`-normalized input (server/host serving).

## The device contract

Reverse-engineered from the on-device tooling (the `snpe-tensorflow-to-dlc` command and
the result-extraction script):

| | |
|---|---|
| Input node | `input_image`  float32  `[1, 672, 416, 3]`  pixels in **[0, 255]** |
| Output nodes | `box`, `cls`, `poly_angle`, `poly_dist`, `poly_conf`, `dist` |

Each output is **one tensor per head** (one `.raw` file each), FPN levels concatenated
**3→4→5**, each `[1,H,W,C]→[1,H*W,C]` row-major, channels-last, **batch dim dropped**
(`[N, C]`). `box` is **DFL-decoded** (the deployed DLC bakes it in); the others are **RAW**
(no sigmoid/softplus/exp, no NMS — the on-device `YoloV8LayerModified` applies those, plus
stride/anchor/NMS, including to `box`):

| node | shape @ 672×416 | floats | meaning |
|------|------|--------|---------|
| `box`        | `[5733, 4]`  |  22 932 | **DFL-decoded** LTRB distances, pre-stride |
| `cls`        | `[5733, 39]` | 223 587 | raw class logits (pre-sigmoid) |
| `poly_angle` | `[5733, 24]` | 137 592 | raw per-vertex angle (pre-sigmoid sub-bin offset) |
| `poly_dist`  | `[5733, 24]` | 137 592 | raw per-vertex radial dist (pre-softplus) |
| `poly_conf`  | `[5733, 24]` | 137 592 | raw per-vertex confidence (pre-sigmoid) |
| `dist`       | `[5733,  1]` |   5 733 | raw log-distance (pre-exp) |

`N = 5733 = 84·52 + 42·26 + 21·13` (strides 8/16/32 over 672×416).

### Box DFL decode (baked, matches the deployed DLC)

The deployed DLC does not emit raw box logits — it bakes the DFL "integral" decode, and so
does this exporter (op-for-op): `[N,64] → reshape [N,4,16] → softmax over the 16 bins →
Σ·[0,1,…,15]` (a 1×1 `conv2d`, weights `[1,1,16,1]`, bias 0) `→ [N,4]`. This is exactly
`distance = Σ softmax(logits)·bin`, identical to `models/detection_generator.py::_decode_dfl`.
The result is the per-side distance **in bin units (pre-stride)**; the on-device
`YoloV8LayerModified` applies stride + anchor + NMS.

### Box channel order: `--legacy_box_order` (default ON)

The deployed decoder stores anchors as **(y, x)** (`make_anchor_points` →
`tf.stack((sy, sx))`) and `box_ops.dist2bbox(ver=1)` computes `anchor − lt` with **no axis
reverse**, returning `yxyx`. So it requires the box channels in **y-first** order:
`[top, left, bottom, right]`. The model (and this repo's `detection_generator`) is the
standard Ultralytics **x-first** `[left, top, right, bottom]`. Feeding x-first boxes to the
deployed decoder applies the left/right (x) offsets to the **y**-axis → every box is
transposed → host 0.68 / device 0.19. The exporter therefore **reorders the box head
`[l,t,r,b] → [t,l,b,r]` (`tf.gather [1,0,3,2]`)** by default, so the unchanged on-device
decoder reads each offset on the correct axis. Set `--legacy_box_order=False` only if you
decode with this repo (it expects x-first).

## Two device-specific transforms vs the `[0,1]` host export

1. **`/255` is baked in** (`--normalize`, default on). The raw-image generator writes
   raw [0,255] float32 (`IMAGE_NROM_FLAG=False`), so the graph divides by 255 to feed the
   model the [0,1] it was trained on (`train.task.normalize_images`).
2. **float32 graph** (not the training `mixed_bfloat16`) so the GraphDef converts cleanly
   in SNPE. The same checkpoint restores into either policy.

The model is fully convolutional, so a 672×672-trained checkpoint runs at 672×416
unchanged — the same size the deployed DLC runs at.

## Why a top-level-named graph

The on-device extractor reads `box:0.raw`, `cls:0.raw`, … (`'%s:0.raw' % node_name`),
i.e. SNPE resolves `--out_node box` to tensor `box:0`. A plain `tf.saved_model.save`
buries the head ops inside a `StatefulPartitionedCall` and renames outputs to
`Identity:0…`. The exporter therefore freezes the graph (inline + variables→constants)
and promotes each head to a clean **top-level op literally named** `box`/`cls`/…
(`input_image` is already top-level), re-emitting a v1 SavedModel laid out the way the
on-device extractor expects.

## Usage

```bash
# 1. Export the SavedModel (prefers EMA weights)
python -m utils.export.export_device_savedmodel \
    --config     configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint /path/to/ckpts/epochN \
    --output_dir /path/to/epochN_export/saved_model \
    --input_size 672,416

# 2. Convert to DLC — the usual command, only --input_network changes
./snpe-tensorflow-to-dlc \
    --input_network /path/to/epochN_export/saved_model \
    --output_path   model_pre.dlc \
    --input_dim input_image 1,672,416,3 \
    --out_node cls --out_node box --out_node poly_angle \
    --out_node poly_conf --out_node poly_dist --out_node dist

# 3. Quantize (raw [0,255] calibration list — unchanged)
./snpe-dlc-quantize \
    --input_list raw_images_672x416_image_list_000000-000027.txt \
    --input_dlc  model_pre.dlc \
    --output_dlc model_quant.dlc

# 4. Net-run on device (unchanged)
snpe-net-run --container model_quant.dlc \
    --input_list <eval>_raw_images_672x416_image_list_000000-002999.txt \
    --perf_profile burst
```

### Float32-policy guard

The exporter forces and asserts a float32 policy before/after building the
model, so a leaked `mixed_bfloat16` policy (which would make conv stems compute bf16
while float32-pinned heads hide it) fails fast at the source.

### SNPE converter: "unsupported masks ellipsis mask and new axis mask"

`snpe-tensorflow-to-dlc` rejects `StridedSlice` ops that set `ellipsis_mask` or
`new_axis_mask`. Those came from the C2f channel split written as `y[..., :c]` (the
ellipsis emits `ellipsis_mask=1`), reused 8× across the backbone/FPN/PAN → 16 ops. The
split is now written with explicit per-axis slices `y[:, :, :, :c]` (plain StridedSlice,
begin/end masks only — SNPE-supported), byte-identical numerically. `_concat_levels`
also now emits a fully **static** reshape (the device input is fixed `1,H,W,3`), so the
graph no longer contains the dynamic `Shape→StridedSlice→Pack→Reshape` subgraph.

If the converter still fails in `StridedSliceLayerBuilder`, it is choking on the
StridedSlice op itself (SNPE's builder is fragile regardless of mask), so the export
now removes **every** StridedSlice: the C2f channel split uses `tf.split` (a `Split`
op) instead of `y[..., :c]`/`y[:, :, :, :c]`, and the FPN upsample size is made a
compile-time constant **for the export only** (`decoder.static_resize`, set by the
exporter) instead of `tf.shape(ref)[1:3]` (which emitted Shape→StridedSlice). Training
and eval keep the **dynamic** resize (`tf.image.resize` to the runtime size), which is
robust to any input size or build-vs-run mismatch — a static size baked into the model
would mismatch the concat when the model is built at one size and run at another. Both
are numerically byte-identical and do not change training or checkpoints.

`test_graph_is_snpe_compatible` guards the exported GraphDef: **no `StridedSlice` at
all**, and no `Pack`/`Shape`. The remaining ops are all
standard SNPE-supported: `Conv2D`, `BiasAdd`, `Relu`, `MaxPool`,
`ResizeNearestNeighbor`, `Mul`/`Sub`/`Rsqrt`/`AddV2` (folded BatchNorm constants),
`ConcatV2`, `Reshape`, `StridedSlice`, `Squeeze`, `RealDiv` (the baked `/255`).

Tests: `tests/test_export_device_savedmodel.py`.
