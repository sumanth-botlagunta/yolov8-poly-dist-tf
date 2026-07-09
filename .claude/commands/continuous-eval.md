# /continuous-eval — use /eval --watch

Continuous evaluation has been folded into `/eval`. Use the `--watch` (or `--all`) mode there.

```
/eval --watch --watch_dir runs/poly_dist_20240101/ --interval 600
/eval --all   --watch_dir runs/poly_dist_20240101/
```

## What to run

```bash
python -m utils.eval \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --watch --watch_dir runs/{run_name}/ \
  --interval 300
```

`--watch` polls `{watch_dir}` with `tf.train.latest_checkpoint()` every `--interval` seconds and
evaluates each new checkpoint (skipping already-seen ones), appending `{metrics, checkpoint,
timestamp}` JSON lines to `{watch_dir}/eval_log.jsonl`. `--all` instead evaluates every checkpoint
already present once. See `/eval` for the full flag list.
