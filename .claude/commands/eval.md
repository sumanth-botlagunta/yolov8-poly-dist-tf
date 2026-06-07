# /eval — Run standalone evaluation on a checkpoint

Evaluates a saved checkpoint on the validation or test split and prints a metrics table.

## Usage

```
/eval --ckpt runs/poly_dist_20240101/ckpt-2388
/eval --ckpt runs/poly_dist_20240101/best_F1score50/ckpt-1 --split test
/eval --ckpt runs/poly_dist_20240101/ckpt-2388 --output /tmp/results.json
```

## What to run

```bash
python tools/eval.py \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --checkpoint $CKPT_PATH \
  --split val \
  [--output results.json]
```

## What to report

- mAP@50, mAP@50:95, F1@50
- Polygon mIoU (if with_polygons)
- Distance MAE/RMSE (if with_distance)
- Per-class AP table (top 10 worst classes)
- Whether EMA weights were used (should always be yes)
