# Scripts & Tools

Every runnable script: what it does, **all** its inputs, and a copy-paste command. Run from the
repo root. Python tools use `python -m <module>` so imports resolve whether or not the package is
installed editable; shell scripts use `bash <path>`.

Most tools take `--config <experiment.yaml>` (from `configs/experiments/yolo/`) and a checkpoint
path **prefix** (e.g. `/run/ckpt-100000`, no extension). See [configuration.md](configuration.md)
for the config fields and [checkpoint_migration.md](checkpoint_migration.md) for warm-starting.

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
- **all** (`--all --watch_dir <dir>`): evaluate every checkpoint already in `<dir>` once,
  appending each result to `<dir>/eval_log.jsonl`.
- **watch** (`--watch --watch_dir <dir>`): poll `<dir>` and evaluate each new checkpoint as it
  appears, appending to `<dir>/eval_log.jsonl`. `--interval` — seconds between polls.
  `--max_evals` — stop after N evaluations (0 = unlimited).
```bash
python -m tools.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --split val --per_category
python -m tools.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --all   --watch_dir /run
python -m tools.eval --config configs/experiments/yolo/yolov8_poly_dist.yaml --watch --watch_dir /run --interval 300
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
SavedModel that drop-in-replaces the legacy device DLC. See [device_export.md](device_export.md).
- `--config` (req), `--checkpoint` (req), `--output_dir` (req).
- `--input_size` — `H,W` for the device (e.g. `672,416`).
- `--verify` — run all contract checks (op names, baked `/255`, decode parity).
- `--normalize` (default on) — bake `/255` so the graph accepts raw `[0,255]` input.
- `--legacy_box_order` (default on) — reorder box channels `[l,t,r,b]→[t,l,b,r]` to match the
  legacy on-device decoder; set `False` only if you decode with this repo / `gen_pred_json`.
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

## Checkpoints

### `python -m tools.checkpoint_migration` — migrate / warm-start
Subcommands: `list`, `map`, `mapping`, `report`, `dump`, `migrate`. See
[checkpoint_migration.md](checkpoint_migration.md).
- `--ckpt` (req) — source checkpoint prefix. `--config` (req for map/migrate) — target YAML.
- `--output` — output checkpoint prefix (migrate). `--modules backbone decoder [head]` — which
  modules to transfer (default: 39-class rule). `--strategy auto|native|frozen|map|structural|name`.
```bash
python -m tools.checkpoint_migration migrate --ckpt /old/ckpt-N --config configs/experiments/yolo/yolov8_poly_dist.yaml --output /tmp/migrated/ckpt
```

### `python -m tools.trace_shapes` — compare two model sources by variable shape
- `--src1` (req), `--src2` (req) — each a checkpoint prefix or an experiment YAML.
- `--filter` — substring filter on variable names. `--only-mismatch` — show only shape mismatches.
- `--by-shape`, `--stats-only`, `--no-colour` — output modes.
```bash
python -m tools.trace_shapes --src1 /old/ckpt-N --src2 configs/experiments/yolo/yolov8_poly_dist.yaml --only-mismatch
```

### `python -m tools.shared.compare_checkpoints` — name-matched checkpoint diff
- `--ckpt1` (req), then `--ckpt2` **or** `--config`. `--modules`, `--grep`, `--no-colour`.
```bash
python -m tools.shared.compare_checkpoints --ckpt1 A --config configs/experiments/yolo/yolov8_poly_dist.yaml
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

### `python -m tools.pipeline.reencode_tfds_672` — build pre-resized 672² datasets
One-time: stores `<name>_672` TFDS copies (672² JPEG) to cut decode cost.
- `--datasets` (req) — TFDS name(s). `--data_dir` — TFDS root. `--size` (default 672). `--splits`.
```bash
python -m tools.pipeline.reencode_tfds_672 --datasets <tfds_name> --data_dir ~/tensorflow_datasets
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
- `--input` (req) — a `val_history.jsonl`, a run dir containing it, or a single report JSON. `--out_dir`, `--basename`.
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

On-device diagnostics live under `tools/device/debug/` (used as needed when a DLC export diverges):

| Command | What it does |
|---------|--------------|
| `python -m tools.device.debug.diagnose_device_export` | Localize where the device graph diverges (eager → tf.function → SavedModel). `--config --checkpoint`. |
| `python -m tools.device.debug.dump_savedmodel_raw` | Dump per-node `.raw` outputs in net-run layout. `--saved_model --raw_image`. |
| `python -m tools.device.debug.compare_dlc_raw` | Node-by-node SavedModel-vs-DLC raw diff (numpy only). `--a --b`. |
| `python -m tools.device.debug.compare_dlc_debug` | Recursive per-layer `--debug` tree comparison, ranked by divergence. |
| `python -m tools.device.debug.compare_dlc_raw_batch` | Aggregate node diffs across many Result folders. |
| `python -m tools.device.debug.savedmodel_on_device_raw` | Run the SavedModel on the device's `.raw` bytes, draw detections. `--saved_model --raw`. |
| `python -m tools.device.debug.visualize_device_export` | Visualize device SavedModel detections at device size. `--config --saved_model`. |
| `python -m tools.device.debug.gen_pred_json_from_savedmodel` | Host twin of gen_pred_json_from_dlc (SavedModel instead of DLC raw). |
| `python -m tools.device.debug.make_calibration_raws` | Build a calibration `.raw` set + input list for `snpe-dlc-quantize`. |

## Future / recommended additions

- **Distance validation.** The distance head is trained but never scored at validation time
  ([design_register.md](design_register.md) entry 4). A future change would add a distance
  validation stream and wire `eval/distance_metrics.py` into the val loop. This is a
  training-semantics change (needs a held-out distance split), not a pure tooling add.
- **Metrics dashboard.** `export_val_metrics` already emits parquet; a small notebook over the
  aggregated parquet would give per-category F1 trends across runs.
