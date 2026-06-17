# On-device export — Qualcomm SNPE DLC (drop-in replacement)

`tools/export_device_dlc.py` exports a trained checkpoint to a TensorFlow SavedModel
laid out as a **drop-in replacement for the legacy on-device DLC**. The existing SNPE
conversion → quantization → net-run → result-extraction pipeline keeps working
**unchanged**; only the SavedModel path changes.

This is distinct from `tools/export_saved_model.py`, which bakes NMS into the graph and
emits the post-processed deploy dict for `[0,1]`-normalized input (server/host serving).

## The legacy device contract

Reverse-engineered from the on-device tooling (the `snpe-tensorflow-to-dlc` command and
the result-extraction script — see `prompts/dlc_conversion.txt`):

| | |
|---|---|
| Input node | `input_image`  float32  `[1, 672, 416, 3]`  pixels in **[0, 255]** |
| Output nodes | `box`, `cls`, `poly_angle`, `poly_dist`, `poly_conf`, `dist` |

Each output is **one tensor per head** (one `.raw` file each), **RAW logits** (no
sigmoid/softplus/exp, no DFL decode, no NMS — the on-device `YoloV8LayerModified` does all
of that), with the FPN levels concatenated **3→4→5**, each `[1,H,W,C]→[1,H*W,C]`
row-major, channels-last:

| node | shape @ 672×416 | floats | meaning |
|------|------|--------|---------|
| `box`        | `[1, 5733, 64]` | 366 912 | raw DFL logits (4 sides × 16 bins) |
| `cls`        | `[1, 5733, 39]` | 223 587 | raw class logits (pre-sigmoid) |
| `poly_angle` | `[1, 5733, 24]` | 137 592 | raw per-vertex angle (pre-sigmoid sub-bin offset) |
| `poly_dist`  | `[1, 5733, 24]` | 137 592 | raw per-vertex radial dist (pre-softplus) |
| `poly_conf`  | `[1, 5733, 24]` | 137 592 | raw per-vertex confidence (pre-sigmoid) |
| `dist`       | `[1, 5733,  1]` |   5 733 | raw log-distance (pre-exp) |

`N = 5733 = 84·52 + 42·26 + 21·13` (strides 8/16/32 over 672×416).

## Two device-specific transforms vs the `[0,1]` host export

1. **`/255` is baked in** (`--normalize`, default on). The raw-image generator writes
   raw [0,255] float32 (`IMAGE_NROM_FLAG=False`), so the graph divides by 255 to feed the
   model the [0,1] it was trained on (`train.task.normalize_images`). See
   `docs/design_register.md` entry 12.
2. **float32 graph** (not the training `mixed_bfloat16`) so the GraphDef converts cleanly
   in SNPE. The same checkpoint restores into either policy.

The model is fully convolutional, so a 672×672-trained checkpoint runs at 672×416
unchanged — identical to the legacy export, which also ran 672×416.

## Why a top-level-named graph

The on-device extractor reads `box:0.raw`, `cls:0.raw`, … (`'%s:0.raw' % node_name`),
i.e. SNPE resolves `--out_node box` to tensor `box:0`. A plain `tf.saved_model.save`
buries the head ops inside a `StatefulPartitionedCall` and renames outputs to
`Identity:0…`. The exporter therefore freezes the graph (inline + variables→constants)
and promotes each head to a clean **top-level op literally named** `box`/`cls`/…
(`input_image` is already top-level), re-emitting a v1 SavedModel that mirrors the legacy
graph. `--verify` asserts these names exist in the GraphDef.

## Usage

```bash
# 1. Export the SavedModel (prefers EMA weights; --verify runs all contract checks)
python tools/export_device_dlc.py \
    --config     configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint /path/to/ckpts/epochN \
    --output_dir /path/to/epochN_export/saved_model \
    --input_size 672,416 \
    --verify

# 2. (optional) sanity-check the SavedModel on host images before converting
python -m utils.object_detection.inference_saved_model_yolov8 \
    --model /path/to/epochN_export/saved_model \
    --data  /path/to/eval_images --image_size 672,416 \
    --num_classes 39 --with_polygons --with_distance

# 3. Convert to DLC — IDENTICAL to the legacy command, only --input_network changes
./snpe-tensorflow-to-dlc \
    --input_network /path/to/epochN_export/saved_model \
    --output_path   model_pre.dlc \
    --input_dim input_image 1,672,416,3 \
    --out_node cls --out_node box --out_node poly_angle \
    --out_node poly_conf --out_node poly_dist --out_node dist

# 4. Quantize (raw [0,255] calibration list — unchanged)
./snpe-dlc-quantize \
    --input_list raw_images_672x416_image_list_000000-000027.txt \
    --input_dlc  model_pre.dlc \
    --output_dlc model_quant.dlc

# 5. Net-run on device (unchanged)
snpe-net-run --container model_quant.dlc \
    --input_list <eval>_raw_images_672x416_image_list_000000-002999.txt \
    --perf_profile burst
```

`--verify` checks, against a built model: top-level op names present (SNPE), signature
output shapes, that baked-in `/255` reproduces the raw model exactly, and that splitting
the concatenated nodes back to per-level and decoding with the in-repo
`YoloV8Layer` (the faithful port of the on-device `YoloV8LayerModified`) reproduces the
deploy path — i.e. the concatenation is the lossless layout the device decoder expects.

### Troubleshooting `--verify`

**`cls` ... `Not equal to tolerance`, ~60–80% mismatched elements (matching shapes/dtypes).**
This is a **precision asymmetry**, not a layout bug. The SavedModel is frozen in
float32, but the in-memory reference model was built under a leaked `mixed_bfloat16`
global policy (the base `yolov8_poly_dist.yaml` trains in bfloat16, and
`tools/runtime_setup.py` or an earlier import in a long-lived session can set that
policy). The prediction heads are pinned float32, so their conv **stems** compute in
bf16 while the head outputs still *report* float32 dtype — hence the mismatch with no
dtype clue. The exporter now calls `set_global_policy('float32')` and **asserts** it
stuck both before and after building the model, raising a clear error if a bf16 policy
leaked. Fix: run the export in a clean process / before any bfloat16 policy is set.

Tests: `tests/test_export_device_dlc.py`.
