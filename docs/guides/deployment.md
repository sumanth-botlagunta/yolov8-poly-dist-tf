# Guide: Deploying to device — SavedModel → Qualcomm SNPE DLC

End-to-end procedure to take a trained checkpoint to a quantized `.dlc` running on device, and to
**verify the on-device numbers match the host**. For the *why* behind the device contract (box
channel order, baked `/255`, float32 graph, BatchNorm fold), see
[device_export.md](../device_export.md).

## Overview of the pipeline

```
checkpoint ──export_device_savedmodel──▶ SavedModel ──snpe-tensorflow-to-dlc──▶ .dlc
   ──snpe-dlc-quantize──▶ quantized .dlc ──snpe-net-run──▶ raw outputs
```

## 1. Export the device SavedModel

```bash
python -m utils.export.export_device_savedmodel \
    --config     configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint /path/to/run_dir/ckpt-<step> \
    --output_dir /path/to/export/saved_model \
    --input_size 672,416
```
This prefers EMA weights, bakes in `/255`, emits a **float32** top-level-named graph, **folds
BatchNorm into the preceding conv** (so the DLC quantizes correctly), and keeps the on-device box
channel order (`--legacy_box_order`, default ON — the device decoder is y-first `[t,l,b,r]`).

## 2. Convert + quantize + run (SNPE — unchanged commands)

```bash
# convert
./snpe-tensorflow-to-dlc --input_network /path/to/export/saved_model --output_path model_pre.dlc \
    --input_dim input_image 1,672,416,3 \
    --out_node cls --out_node box --out_node poly_angle --out_node poly_conf --out_node poly_dist --out_node dist

# quantize (int8; raw [0,255] calibration list)
./snpe-dlc-quantize --input_list calibration_list.txt --input_dlc model_pre.dlc --output_dlc model_quant.dlc \
    --use_per_channel_quantization --adjust_bias_encoding --algorithms cle bc

# run on device
snpe-net-run --container model_quant.dlc --input_list eval_raw_list.txt --perf_profile burst
```
Build a representative calibration `.raw` set + list (use diverse images, e.g. a COCO subset — not
your eval set, to avoid bias). Per-channel weights + CLE + bias-correction materially improve int8
accuracy.

## 3. Score the host reference

Run the exported SavedModel on the eval images (host twin) for an apples-to-apples reference:
```bash
python -m utils.export.inference_saved_model --saved_model /path/.../saved_model --images <same_images> --emit json ...
```
A healthy export: SavedModel-JSON ≈ checkpoint eval; CPU `.dlc` ≈ SavedModel; quantized `.dlc`
slightly below (int8). A large gap means something upstream broke — go to step 4.

## 4. When device ≠ host — localize the divergence

The usual culprits, in order: **box channel order** (transposed boxes — `yfirst` is the
on-device/DLC order and the export default),
**un-folded BatchNorm** (quantizes badly), and **the calibration set**
(too small/biased → poor int8). See [device_export.md](../device_export.md) for the contract.

## Related
- Reference: [device_export.md](../device_export.md) · [scripts.md](../scripts.md) (the `utils.export.*` table)
- For host/server serving instead of device, see `utils/export/export_saved_model.py` ([scripts.md](../scripts.md)).
