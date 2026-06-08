# /eval — Run standalone evaluation on a checkpoint

Evaluates a saved checkpoint on the validation or test split and prints a metrics table.

## Usage

```
/eval --ckpt runs/poly_dist_20240101/ckpt-2388
/eval --ckpt runs/poly_dist_20240101/best_F1score50/ckpt-1 --split test
/eval --ckpt runs/poly_dist_20240101/ckpt-2388 --output_dir /tmp/results/
/eval --ckpt runs/poly_dist/best_F1score50/ckpt-1 --per_category
```

## What to run

```bash
python tools/eval.py \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --checkpoint $CKPT_PATH \
  --split val \
  [--output_dir /tmp/results/] \
  [--per_category]
```

## What to report

- mAP@50, mAP@50:95, F1@50, best_conf_thresh
- Polygon mIoU (if with_polygons)
- Distance MAE, RMSE, AbsRel (near/far splits) (if with_distance)
- Per-class AP50 table with AP75/AR50 when `--per_category` is set
- Whether EMA weights were used (should always be yes)
- With `--output_dir`: results JSON + per-category text table written to disk
