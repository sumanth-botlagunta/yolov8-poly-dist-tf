# Guide: Deploying to device — SavedModel → Qualcomm SNPE DLC

End-to-end procedure to take a trained checkpoint to a quantized `.dlc` running on device, and to
**verify the on-device numbers match the host**. For the *why* behind the device contract (box
channel order, baked `/255`, float32 graph, BatchNorm fold), see
[device_export.md](../device_export.md).

## Overview of the pipeline

```
checkpoint ──export_device_dlc──▶ SavedModel ──snpe-tensorflow-to-dlc──▶ .dlc
   ──snpe-dlc-quantize──▶ quantized .dlc ──snpe-net-run──▶ raw outputs ──gen_pred_json──▶ predictions.json
```

## 1. Export the device SavedModel

```bash
python -m tools.device.export_device_dlc \
    --config     configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint /path/to/run_dir/ckpt-<step> \
    --output_dir /path/to/export/saved_model \
    --input_size 672,416 \
    --verify
```
This prefers EMA weights, bakes in `/255`, emits a **float32** top-level-named graph, **folds
BatchNorm into the preceding conv** (so the DLC quantizes correctly), and keeps the on-device box
channel order (`--legacy_box_order`, default ON — the device decoder is y-first `[t,l,b,r]`).
`--verify` runs all contract checks (see [device_export.md](../device_export.md#troubleshooting--verify);
it judges by **relative magnitude**, not element count — benign fused/unfused accumulation passes).

## 2. (Optional) sanity-check the SavedModel before converting

```bash
python -m tools.device.validate_device_export \
    --config /path/.../yolov8_poly_dist.yaml \
    --checkpoint /path/to/run_dir/ckpt-<step> \
    --saved_model /path/to/export/saved_model
```
Compares the in-repo model against the device SavedModel on val images — a quick "are the boxes
sane" check before SNPE.

## 3. Convert + quantize + run (SNPE — unchanged commands)

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
Build a representative calibration `.raw` set + list with
`tools/device/make_calibration_raws.py` (use diverse images, e.g. a COCO subset — not your
eval set, to avoid bias). Per-channel weights + CLE + bias-correction materially improve int8
accuracy.

## 4. Score the device output and compare to host

Turn the device raw outputs into a COCO predictions JSON, then score it the same way as host eval:

```bash
# edit the SPLITS list in-file (printed at startup), or pass --splits
python -m tools.device.gen_pred_json_from_dlc \
    --raw_root /path/to/netrun/output --transform_pkl /path/to/letterbox_transform.pkl \
    --output_json device_predictions.json --box_order yfirst
```
Run the exported SavedModel on the **same** images (host twin) for an apples-to-apples reference:
```bash
python -m tools.infer --saved_model /path/.../saved_model --images <same_images> --emit json ...
```
A healthy export: SavedModel-JSON ≈ checkpoint eval; CPU `.dlc` ≈ SavedModel; quantized `.dlc`
slightly below (int8). A large gap means something upstream broke — go to step 5.

## 5. When device ≠ host — localize the divergence

```bash
# is the SavedModel even SNPE-clean (no un-foldable ops, no leftover BatchNorm)?
python -m tools.device.check_snpe_ready /path/to/export/saved_model

# in-repo model vs the device SavedModel, on val images
python -m tools.device.validate_device_export --config <cfg> --checkpoint <ckpt> --saved_model <sm>
```
The usual culprits, in order: **box channel order** (transposed boxes → match `--box_order` to the
export; `yfirst` is the on-device/DLC order and the default),
**un-folded BatchNorm** (quantizes badly — `check_snpe_ready` flags it), and **the calibration set**
(too small/biased → poor int8). See [device_export.md](../device_export.md) for the contract.

## Related
- Reference: [device_export.md](../device_export.md) · [scripts.md](../scripts.md) (the `tools.device.*` table)
- For host/server serving instead of device, see `tools/export_saved_model.py` ([scripts.md](../scripts.md)).
