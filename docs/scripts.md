# Scripts & Tools

Every runnable script: what it does, **all** its inputs, and a copy-paste command. Run from the
repo root. Python tools use `python -m <module>` so imports resolve whether or not the package is
installed editable; shell scripts use `bash <path>`.

Most tools take `--config <experiment.yaml>` (from `configs/experiments/yolo/`) and a checkpoint
path **prefix** (e.g. `/run/ckpt-100000`, no extension). See [configuration.md](configuration.md)
for the config fields and [guides/finetuning.md](guides/finetuning.md) for warm-starting.

## Core workflow

### `python -m scripts.run_train` — launch training
Runs the training loop. For long runs prefer the supervisor (next entry).
- `--config` (req) — experiment YAML.
- `--output_dir` (req) — where checkpoints, `tb_events/`, and `val_history.jsonl` are written.
- `--debug` — eager execution + verbose logging (slow; for debugging only).
- `--resume_from` — resume from a specific checkpoint prefix (overrides the auto-latest).
- `--finetune_from` — **fine-tune**: seed a fresh run from a trained checkpoint's EMA weights with
  a fresh optimizer/LR (overrides `task.finetune_from`). Distinct from `--resume_from` (same run,
  continues). See [guides/finetuning.md](guides/finetuning.md).
```bash
python -m scripts.run_train --config configs/experiments/yolo/yolov8_poly_dist.yaml --output_dir /run
# fine-tune:
python -m scripts.run_train --config <finetune.yaml> --output_dir /finetune_run --finetune_from /src_run/ckpt-100000
```

### `bash tools/train_supervisor.sh` — supervised training (recommended for long runs)
Keeps training alive across crashes/OOM, auto-resumes, detaches from SSH.
- `--config` (req) — experiment YAML.
- `--output_dir` (req) — run directory. `touch <output_dir>/STOP` to stop without restart.
```bash
nohup bash tools/train_supervisor.sh --config configs/experiments/yolo/yolov8_poly_dist.yaml --output_dir /run >> /run/supervisor.log 2>&1 &
```

### `python -m tools.eval` — evaluate one or many checkpoints
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
python -m tools.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --split val --per_category
python -m tools.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --all   --watch_dir /run
python -m tools.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --watch --watch_dir /run --interval 300
```

### `python -m tools.compare_nms_modes` — per-class vs class-agnostic NMS, side by side
Runs the network once per batch and scores the same raw outputs through two detection generators
that differ only in `nms_class_mode` (`per_class` vs `agnostic`), so both modes see byte-identical
predictions. Prints a headline-metric comparison plus the per-class best-F1 movers, and writes
both full ckpt-format reports. NMS is pure post-processing — valid for any existing checkpoint,
no retraining.
- `--config` (req), `--checkpoint` (req). `--split` — `val` (default) / `test`.
- `--output_dir` — where the two reports go (default `<ckpt_dir>/nms_compare`).
```bash
python -m tools.compare_nms_modes --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000
```

### `python -m tools.infer` — folder inference: predictions JSON + visuals
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
python -m tools.infer --saved_model /export/saved_model --images /imgs \
    --output_dir /tmp/out --emit both --draw_on original
```

## Export

### `python -m tools.device.export_device_dlc` — on-device SNPE/DLC export (most common)
SavedModel that drop-in-replaces the deployed device DLC. See [device_export.md](device_export.md).
- `--config` (req), `--checkpoint` (req), `--output_dir` (req).
- `--input_size` — `H,W` for the device (e.g. `672,416`).
- `--verify` — run all contract checks (op names, baked `/255`, decode parity).
- `--normalize` (default on) — bake `/255` so the graph accepts raw `[0,255]` input.
- `--legacy_box_order` (default on) — reorder box channels `[l,t,r,b]→[t,l,b,r]` to match the
  on-device decoder (y-first); set `False` only if you decode with this repo / `gen_pred_json`.
- `--debug_taps` — emit intermediate tap nodes for SavedModel-vs-DLC bisection.
```bash
python -m tools.device.export_device_dlc --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --output_dir /export --input_size 672,416 --verify
```

### `python -m tools.export_saved_model` — host/server SavedModel
Deploy SavedModel with NMS baked in; expects `[0,1]` input.
- `--config` (req), `--checkpoint` (req), `--output_dir` (req).
- `--tflite` — also run the TFLite converter and write `model.tflite`.
```bash
python -m tools.export_saved_model --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --output_dir /export
```

## Pipeline

### `python -m tools.benchmark_pipeline` — throughput benchmark
- `--config` (req). `--steps` — steps to time (default 100). `--profile` — save a TF profiler trace.
```bash
python -m tools.benchmark_pipeline --config configs/experiments/yolo/yolov8_poly_dist.yaml --steps 100
```

### `python -m tools.pipeline.diagnose_pipeline` — stage-by-stage attribution
- `--config` (req). `--samples`, `--batches` — workload size. `--threadpool-sweep` — sweep
  `private_threadpool_size` values (comma list).
```bash
python -m tools.pipeline.diagnose_pipeline --config configs/experiments/yolo/yolov8_poly_dist.yaml --samples 768 --batches 10
```

### `bash tools/cloud_diagnose.sh` — one-shot cloud bring-up check
- Takes the experiment YAML as a **positional** argument (defaults to the poly_dist tier if
  omitted). Runs the diagnose + benchmark (cold & warm) and measures CPU throttle.
```bash
bash tools/cloud_diagnose.sh configs/experiments/yolo/yolov8_poly_dist.yaml
```

### `python -m tools.val_history` — inspect / extract the validation history
Reads `<run>/val_history.jsonl` (one report appended per epoch). No SQL, no DB.
- positional `path` — the run dir or the `val_history.jsonl` file.
- (no selector) / `--list` — trend table: epoch / step / F1score50 / mAP / mAP50 / AR100.
- `--epoch N` / `--step N` / `--checkpoint SUBSTR` / `--best` — select one record.
- `--format txt|json|csv` (default txt, the exact ckpt format) — `--best-only`, `-o OUT`.
- `--export-csv PATH` — whole history → one flat CSV (pandas if installed).
```bash
python -m tools.val_history /run --list
python -m tools.val_history /run --best --format txt -o best.txt
python -m tools.val_history /run --epoch 42 --format json
```

### `python -m tools.val_report_txt` — render a single report JSON to the ckpt-format txt
Standalone sibling of `val_history`: renders one validation report JSON (a `<ckpt>_val.json`
from `tools.eval --output_dir`, or one extracted via `val_history --format json`) into the exact
ckpt-format `.txt` (best-conf-per-category table + mean + all-conf sweep). `--best-only` keeps just
the best table.
```bash
python -m tools.val_report_txt /run/ckpt-99000_val.json --best-only
```

### `python -m tools.pipeline.export_val_metrics` — export validation metrics to xlsx/parquet
Reads `val_history.jsonl` (or a single report JSON) and writes xlsx/csv/parquet for trend analysis.
- `--input` (req) — a `val_history.jsonl`, a run dir containing it, or a single report JSON. `--out_dir` (req), `--basename`.
- `--formats` — comma list (`xlsx,csv,parquet`). `--aggregate` — combine all epochs into one table.
```bash
python -m tools.pipeline.export_val_metrics --input /run/val_history.jsonl --aggregate --formats xlsx,parquet
```

## Device diagnostics

These localize an on-device accuracy gap (DLC vs host). See [device_export.md](device_export.md).

| Command | What it does |
|---------|--------------|
| `python -m tools.device.gen_pred_json_from_dlc` | Build a COCO prediction JSON from DLC/SavedModel raw outputs (edit the `SPLITS` list in-file; `--splits` overrides). `--raw_root --transform_pkl --output_json`. |
| `python -m tools.device.validate_device_export` | Compare the in-repo model vs the device SavedModel on val images. `--config --checkpoint --saved_model`. |
| `python -m tools.device.check_snpe_ready` | Scan an exported SavedModel for SNPE-incompatible ops. `<saved_model_dir>`. |
| `python -m tools.device.make_calibration_raws` | Build a calibration `.raw` set + input list for `snpe-dlc-quantize`. |

## Future / recommended additions

- **Distance validation.** The distance head is trained but never scored at validation time
  (the shipped distance dataset is training-only). A future change would add a distance
  validation stream and wire `eval/distance_metrics.py` into the val loop. This is a
  training-semantics change (needs a held-out distance split), not a pure tooling add.
- **Metrics dashboard.** `export_val_metrics` already emits parquet; a small notebook over the
  aggregated parquet would give per-category F1 trends across runs.
