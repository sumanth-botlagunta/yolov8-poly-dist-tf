# Guide: Running inference on images

Run a trained model on a folder of arbitrary images and get **predictions (JSON)** and/or
**annotated images**, at the model size or mapped back to the original image. Tool:
`utils/export/inference_saved_model.py`.

## Source: checkpoint or SavedModel

```bash
# from a training checkpoint (+ its config; EMA weights are preferred):
python -m utils.export.inference_saved_model \
    --config configs/experiments/yolo/yolov8_poly_dist.yaml \
    --checkpoint /path/to/run_dir/ckpt-100000 \
    --images /path/to/images_dir --output_dir /tmp/infer_out

# from an exported SavedModel (input size auto-read from the signature):
python -m utils.export.inference_saved_model \
    --saved_model /path/to/export/saved_model \
    --images /path/to/images_dir --output_dir /tmp/infer_out
```
`--images` accepts a single file or a directory (jpg/jpeg/png/bmp/webp). A progress bar shows
throughput.

The exported SavedModel is the device-contract artifact (raw per-head outputs, `[0,255]` input);
this tool reconstructs deploy-style boxes, polygons, and distance from those flat heads
(`utils/export/device_decode.py`), so the SavedModel and checkpoint paths produce the same outputs.

## What to emit — `--emit`

| `--emit` | Produces |
|---|---|
| `both` (default) | annotated `*_pred.png` per image **and** a `predictions.json` |
| `visual` | annotated images only |
| `json` | `predictions.json` only (COCO-style: `image_id`, `file_name`, `category_id/name`, `bbox` xywh, `score`, and `distance_m` if the model has a distance head) |

`predictions.json` goes to `<output_dir>/predictions.json` (override with `--predictions_json`).

## Output coordinate space — `--draw_on`

| `--draw_on` | Boxes/polygons/JSON are in… |
|---|---|
| `original` (default) | the **source image's** pixels (inverse-letterbox mapped — what you usually want) |
| `model` | the **model input** size (the exported 672 or 416 the network sees) |

The inverse-letterbox mapping is exact (unit-tested round-trip), so detections line up on the
original full-resolution image.

## Other flags

- `--score` — minimum confidence to keep/draw (default 0.25).
- `--no_poly` — boxes only (skip polygon contours).
- `--input_size` — override the square input size (0 = read from config/SavedModel).
- `--device_box_order` — box-channel order of a device-contract SavedModel: `yfirst` (the export
  default) or `xfirst` (a `--legacy_box_order=False` export). Mismatch transposes every box.

## Example: deployable predictions on original-size images

```bash
python -m utils.export.inference_saved_model --saved_model /path/to/export/saved_model \
    --images /path/to/photos --output_dir /tmp/out \
    --emit both --draw_on original --score 0.3
# -> /tmp/out/<name>_pred.png (annotated) + /tmp/out/predictions.json (original-pixel boxes)
```

## Related
- For the on-device equivalent (run the exported model the way the device does, score vs host),
  see the [deployment guide](deployment.md).
- Reference: [scripts.md](../scripts.md) (the `utils.export.inference_saved_model` entry).
