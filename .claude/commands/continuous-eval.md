# /continuous-eval — Watch a run directory and auto-evaluate new checkpoints

Polls an output directory for new checkpoints and evaluates each one automatically.
Results are appended to `{watch_dir}/eval_log.jsonl` so you can track mAP over time
without manual eval runs.

## Usage

```
/continuous-eval --watch runs/poly_dist_20240101/
/continuous-eval --watch runs/poly_dist_20240101/ --interval 600
```

## What to run

```bash
python tools/continuous_eval.py \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --watch_dir runs/{run_name}/ \
  --interval 300
```

`--interval` (seconds) controls how often to poll for a new checkpoint. Default: 300s.

## What it does

- Polls `{watch_dir}/ckpt/` using `tf.train.latest_checkpoint()` every `--interval` seconds
- Skips checkpoints already in `eval_log.jsonl`
- On a new checkpoint: runs full val evaluation (EMA weights, all metrics)
- Appends `{step, checkpoint, metrics}` JSON line to `{watch_dir}/eval_log.jsonl`
- Prints a one-line summary per checkpoint to stdout

## What to monitor

- `eval_log.jsonl` for mAP trend over training
- Stop the process with Ctrl-C; it is safe to restart (already-evaluated checkpoints are skipped)
