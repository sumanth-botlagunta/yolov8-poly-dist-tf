# /eval — Run evaluation on one or many checkpoints

Evaluates saved checkpoints on the validation or test split and prints (or logs) a metrics table.
One eval code path with three modes:

- **single** (default): evaluate one checkpoint and print the metrics table.
- **all**: evaluate every checkpoint already in a run directory once, appending each result to
  `{watch_dir}/eval_log.jsonl`.
- **watch**: poll a run directory and evaluate each new checkpoint as it appears, appending to
  `{watch_dir}/eval_log.jsonl`.

## Usage

```
/eval --ckpt runs/poly_dist_20240101/ckpt-2388
/eval --ckpt runs/poly_dist_20240101/best_F1score50/ckpt-1 --split test
/eval --ckpt runs/poly_dist_20240101/ckpt-2388 --output_dir /tmp/results/
/eval --ckpt runs/poly_dist/best_F1score50/ckpt-1 --per_category
/eval --all   --watch_dir runs/poly_dist_20240101/
/eval --watch --watch_dir runs/poly_dist_20240101/ --interval 600
```

## What to run

```bash
# single checkpoint
python -m utils.eval \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --checkpoint $CKPT_PATH \
  --split val \
  [--output_dir /tmp/results/] \
  [--per_category]

# every existing checkpoint in a run directory, once
python -m utils.eval \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --all --watch_dir runs/{run_name}/

# keep polling for new checkpoints (Ctrl-C to stop; safe to restart)
python -m utils.eval \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --watch --watch_dir runs/{run_name}/ \
  --interval 300
```

`--interval` (seconds) controls how often `--watch` polls for a new checkpoint. Default: 300s.
`--max_evals` stops `--watch` after N evaluations (0 = unlimited).

## What it does

- Builds the model, restores EMA weights (preferred over raw), runs full inference over the split.
- single: prints mAP@50, mAP@50:95, F1@50, best_conf_thresh, polygon mIoU (if with_polygons),
  distance MAE/RMSE/AbsRel (if with_distance), and a per-class AP/AR table with `--per_category`;
  with `--output_dir`, writes the metrics + per-category JSON; with `--output_json`, writes
  COCO-format detection results.
- all / watch: evaluate each checkpoint and append `{metrics, checkpoint, timestamp}` JSON lines
  to `{watch_dir}/eval_log.jsonl`; already-evaluated checkpoints are skipped in `--watch`.

## What to report

- mAP@50, mAP@50:95, F1@50, best_conf_thresh
- Polygon mIoU (if with_polygons)
- Distance MAE, RMSE, AbsRel (near/far splits) (if with_distance)
- Per-class AP50 table with AP75/AR50 when `--per_category` is set
- Whether EMA weights were used (should always be yes)
- For all / watch: track `eval_log.jsonl` for the mAP / F1 trend over training
