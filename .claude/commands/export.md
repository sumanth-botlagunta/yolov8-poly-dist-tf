# /export — Export model to SavedModel (and optionally TFLite)

Loads a trained checkpoint and exports a deployment-ready SavedModel with NMS
baked into the forward pass (`deploy=True`).

## Usage

```
/export --ckpt runs/poly_dist/best_F1score50/ckpt-1
/export --ckpt runs/poly_dist/best_F1score50/ckpt-1 --tflite
```

## What to run

```bash
python tools/export_saved_model.py \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --checkpoint $CKPT_PATH \
  --output exported_model/ \
  [--tflite]
```

## What to report

- SavedModel output path
- Model signature (input/output tensor specs)
- Inference latency for one 672×672 image (CPU and GPU)
- TFLite model size (if --tflite)
- Whether polygon and distance outputs are present in the signature
