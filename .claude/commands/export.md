# /export — Export model to the SNPE-DLC SavedModel

Loads a trained checkpoint and exports the SavedModel used for on-device SNPE DLC
conversion. It emits per-head raw tensors on the device contract (no in-graph NMS)
and bakes `/255` so the device feeds raw `[0, 255]` pixels; the forward pass runs in
float32. This is a drop-in replacement for the deployed on-device Qualcomm SNPE DLC.

## Usage

```
/export --ckpt runs/poly_dist/best_F1score50/ckpt-1
```

## What to run

```bash
python -m utils.export.export_saved_model \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --checkpoint $CKPT_PATH \
  --output_dir exported_model/ \
  --input_size 672,416
```

Key flags:
- `--input_size H,W` — device input size (default `672,416`); anchors/box decode trace at it.
- `--normalize` (default true) — bake `/255` so the device feeds raw `[0, 255]` pixels.
- `--legacy_box_order` (default true) — emit `box` as `[top,left,bottom,right]` (y-first)
  to match the on-device `dist2bbox` + `(y,x)` anchors.
- `--debug_taps` — also emit intermediate tensors for SavedModel-vs-DLC bisection.

## What to report

- SavedModel output path
- Device-contract nodes present in the signature (`box`, `cls`, `poly_*`, `dist`) and their shapes
- BatchNorm-fold summary (folded/skipped counts) and any surviving `FusedBatchNorm*`
- Whether polygon and distance heads are present in the export
