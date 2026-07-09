# Scripts & Tools

Every runnable script: what it does, **all** its inputs, and a copy-paste command. Run from the
repo root. Python tools use `python -m <module>` so imports resolve whether or not the package is
installed editable; shell scripts use `bash <path>`.

Most tools take `--config <experiment.yaml>` (from `configs/experiments/yolo/`) and a checkpoint
path **prefix** (e.g. `/run/ckpt-100000`, no extension). See [configuration.md](configuration.md)
for the config fields and [guides/finetuning.md](guides/finetuning.md) for warm-starting.

## Core workflow

### `python -m train.run_train` — launch training
Runs the training loop. For long runs prefer the supervisor (next entry).
- `--config` (req) — experiment YAML.
- `--output_dir` (req) — where checkpoints, `tb_events/`, and `val_history.jsonl` are written.
- `--debug` — eager execution + verbose logging (slow; for debugging only).
- `--resume_from` — resume from a specific checkpoint prefix (overrides the auto-latest).
- `--finetune_from` — **fine-tune**: seed a fresh run from a trained checkpoint's EMA weights with
  a fresh optimizer/LR (overrides `task.finetune_from`). Distinct from `--resume_from` (same run,
  continues). See [guides/finetuning.md](guides/finetuning.md).
```bash
python -m train.run_train --config configs/experiments/yolo/yolov8_poly_dist.yaml --output_dir /run
# fine-tune:
python -m train.run_train --config <finetune.yaml> --output_dir /finetune_run --finetune_from /src_run/ckpt-100000
```

### `bash train/train_supervisor.sh` — supervised training (recommended for long runs)
Keeps training alive across crashes/OOM, auto-resumes, detaches from SSH.
- `--config` (req) — experiment YAML.
- `--output_dir` (req) — run directory. `touch <output_dir>/STOP` to stop without restart.
```bash
nohup bash train/train_supervisor.sh --config configs/experiments/yolo/yolov8_poly_dist.yaml --output_dir /run >> /run/supervisor.log 2>&1 &
```

### `python -m utils.eval` — evaluate one or many checkpoints
COCO mAP/F1, polygon, and distance metrics on val/test (EMA weights preferred). One eval code
path with three modes:
- **single** (default, `--checkpoint <ckpt>`): evaluate one checkpoint and print the metric table.
  - `--config` (req), `--checkpoint` — YAML and checkpoint prefix.
  - `--split` — `val` (default) / `test` / `train`.
  - `--per_category` — also print the per-class AP/AR table.
  - `--output_json` — write COCO-format detection results to this path.
  - `--output_dir` — write `metrics.json` (+ `per_category_metrics.json`) here.
  - `--dump_failures` — mine the worst predictions per class (false positives, missed GT,
    low-IoU matches) and write them as annotated images to `<output_dir>/failures/<NN_name>/`
    (`--failures_dir` / `--failures_per_class` to tune). Pairs with the `per_class/` TB metrics:
    see a weak class, then look at *why*.
- **all** (`--all --watch_dir <dir>`): evaluate every checkpoint already in `<dir>` once,
  appending each result to `<dir>/eval_log.jsonl`.
- **watch** (`--watch --watch_dir <dir>`): poll `<dir>` and evaluate each new checkpoint as it
  appears, appending to `<dir>/eval_log.jsonl`. `--interval` — seconds between polls.
  `--max_evals` — stop after N evaluations (0 = unlimited).
- `--limit_batches N` (any mode) — stop after N eval batches (0 = full split): a fast **sampled
  probe**, not a substitute for the full-split numbers.
```bash
python -m utils.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --split val --per_category
python -m utils.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --all   --watch_dir /run
python -m utils.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --watch --watch_dir /run --interval 300
```

### `python -m utils.export.inference_saved_model` — folder inference: predictions JSON + visuals
Loads a checkpoint **or** a SavedModel and runs over a folder of images, emitting a COCO-style
predictions JSON and/or annotated images, in the model-input or original-image coordinate space.
- `--config` + `--checkpoint`, **or** `--saved_model` (one source required).
- `--images` (req) — an image file or a directory of images.
- `--emit` — `visual` | `json` | `both` (default `both`).
- `--draw_on` — `original` (map detections back to source pixels, default) | `model` (the exported 672/416 size).
- `--output_dir` — where annotated PNGs + `predictions.json` are written. `--predictions_json` overrides the JSON path.
- `--score` — min confidence to keep/draw (default 0.25). `--no_poly` — boxes only.
- `--input_size` — override the square input size (0 = read from config/SavedModel).
```bash
python -m utils.export.inference_saved_model --saved_model /export/saved_model --images /imgs \
    --output_dir /tmp/out --emit both --draw_on original
```

## Export

### `python -m utils.export.export_device_savedmodel` — on-device SNPE/DLC export (most common)
SavedModel that drop-in-replaces the deployed device DLC. See [device_export.md](device_export.md).
- `--config` (req), `--checkpoint` (req), `--output_dir` (req).
- `--input_size` — `H,W` for the device (e.g. `672,416`).
- `--normalize` (default on) — bake `/255` so the graph accepts raw `[0,255]` input.
- `--legacy_box_order` (default on) — reorder box channels `[l,t,r,b]→[t,l,b,r]` to match the
  on-device decoder (y-first); set `False` only if you decode with this repo.
- `--debug_taps` — emit intermediate tap nodes for SavedModel-vs-DLC bisection.
```bash
python -m utils.export.export_device_savedmodel --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --output_dir /export --input_size 672,416
```

### `python -m utils.export.export_saved_model` — host/server SavedModel
Deploy SavedModel with NMS baked in; expects `[0,1]` input.
- `--config` (req), `--checkpoint` (req), `--output_dir` (req).
- `--tflite` — also run the TFLite converter and write `model.tflite`.
```bash
python -m utils.export.export_saved_model --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --output_dir /export
```

## Pipeline

### `python -m utils.pipeline.benchmark_pipeline` — throughput benchmark
- `--config` (req). `--steps` — steps to time (default 100). `--profile` — save a TF profiler trace.
```bash
python -m utils.pipeline.benchmark_pipeline --config configs/experiments/yolo/yolov8_poly_dist.yaml --steps 100
```

### `python -m utils.pipeline.diagnose_pipeline` — stage-by-stage attribution
- `--config` (req). `--samples`, `--batches` — workload size. `--threadpool-sweep` — sweep
  `private_threadpool_size` values (comma list).
```bash
python -m utils.pipeline.diagnose_pipeline --config configs/experiments/yolo/yolov8_poly_dist.yaml --samples 768 --batches 10
```

### `bash utils/pipeline/cloud_diagnose.sh` — one-shot cloud bring-up check
- Takes the experiment YAML as a **positional** argument (defaults to the poly_dist tier if
  omitted). Runs the diagnose + benchmark (cold & warm) and measures CPU throttle.
```bash
bash utils/pipeline/cloud_diagnose.sh configs/experiments/yolo/yolov8_poly_dist.yaml
```

### `python -m utils.reports.val_history` — inspect / extract / export the validation history
Reads `<run>/val_history.jsonl` (one report appended per epoch), or a single report JSON. No SQL, no DB.
- positional `path` — the run dir, the `val_history.jsonl` file, or a single report JSON (a `<ckpt>_val.json` from `utils.eval --output_dir`).
- (no selector) / `--list` — trend table: epoch / step / F1score50 / mAP / mAP50 / AR100.
- `--epoch N` / `--step N` / `--checkpoint SUBSTR` / `--best` — select one record.
- `--format txt|json|csv|xlsx|parquet` (default txt, the exact ckpt format) — `--best-only`, `-o OUT`.
  `xlsx`/`parquet` export per class×threshold tables (require `-o`; the selected record, else the whole run).
- `--export-csv PATH` — whole history → one flat headline CSV (pandas if installed).
```bash
python -m utils.reports.val_history /run --list
python -m utils.reports.val_history /run --best --format txt -o best.txt
python -m utils.reports.val_history /run --epoch 42 --format json
python -m utils.reports.val_history /run/ckpt-99000_val.json --best-only         # render a single report JSON
python -m utils.reports.val_history /run --best --format xlsx -o best.xlsx        # best epoch -> workbook
python -m utils.reports.val_history /run --format parquet -o /tmp/run.parquet     # whole run -> parquet
```

## Future / recommended additions

- **Distance validation.** The distance head is trained but never scored at validation time
  (the shipped distance dataset is training-only). A future change would add a distance
  validation stream and wire `eval/distance_metrics.py` into the val loop. This is a
  training-semantics change (needs a held-out distance split), not a pure tooling add.
- **Metrics dashboard.** `val_history --format parquet` already emits parquet; a small notebook over the
  aggregated parquet would give per-category F1 trends across runs.
