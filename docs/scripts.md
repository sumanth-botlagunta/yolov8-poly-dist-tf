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
- `--output_dir` (req) — where checkpoints, `tb_events/`, and `val_metrics/` are written.
- `--debug` — eager execution + verbose logging (slow; for debugging only).
- `--resume_from` — resume from a specific checkpoint prefix (overrides the auto-latest).
```bash
python -m scripts.run_train --config configs/experiments/yolo/yolov8_poly_dist.yaml --output_dir /run
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

### `python -m tools.infer` — overlays on arbitrary images
Loads a checkpoint **or** a SavedModel, draws box+polygon overlays, prints class/score/distance.
- `--config` + `--checkpoint`, **or** `--saved_model` (one source required).
- `--images` (req) — an image file or a directory of images.
- `--output_dir` — where annotated PNGs are written (default `/tmp/infer_out`).
- `--score` — min confidence to draw (default 0.25). `--no_poly` — boxes only.
- `--input_size` — override the square input size (0 = read from config/SavedModel).
```bash
python -m tools.infer --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --images /imgs --output_dir /tmp/out
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

### `python -m tools.pipeline.export_val_metrics` — export saved validation metrics
Reads `val_metrics/*.json` and writes xlsx/csv/parquet for trend analysis.
- `--input` (req) — a metrics JSON or a `val_metrics/` dir. `--out_dir`, `--basename`.
- `--formats` — comma list (`xlsx,csv,parquet`). `--aggregate` — combine all epochs into one table.
```bash
python -m tools.pipeline.export_val_metrics --input /run/val_metrics --aggregate --formats xlsx,parquet
```

## Device diagnostics

These localize an on-device accuracy gap (DLC vs host). See [device_export.md](device_export.md).

| Command | What it does |
|---------|--------------|
| `python -m tools.device.gen_pred_json_from_dlc` | Build a COCO prediction JSON from DLC/SavedModel raw outputs (edit the `SPLITS` list in-file; `--splits` overrides). `--raw_root --transform_pkl --output_json`. |
| `python -m tools.device.validate_device_export` | Compare the in-repo model vs the device SavedModel on val images. `--config --checkpoint --saved_model`. |
| `python -m tools.device.diagnose_device_export` | Localize where the device graph diverges (eager → tf.function → SavedModel). `--config --checkpoint`. |
| `python -m tools.device.check_snpe_ready` | Scan an exported SavedModel for SNPE-incompatible ops. `<saved_model_dir>`. |
| `python -m tools.device.dump_savedmodel_raw` | Dump per-node `.raw` outputs in net-run layout. `--saved_model --raw_image`. |
| `python -m tools.device.compare_dlc_raw` | Node-by-node SavedModel-vs-DLC raw diff (numpy only). `--a --b`. |
| `python -m tools.device.savedmodel_on_device_raw` | Run the SavedModel on the device's `.raw` bytes, draw detections. `--saved_model --raw`. |
| `python -m tools.device.visualize_device_export` | Visualize device SavedModel detections at device size. `--config --saved_model`. |

## Future / recommended additions

- **Distance validation.** The distance head is trained but never scored at validation time
  ([design_register.md](design_register.md) entry 4). A future change would add a distance
  validation stream and wire `eval/distance_metrics.py` into the val loop. This is a
  training-semantics change (needs a held-out distance split), not a pure tooling add.
- **Metrics dashboard.** `export_val_metrics` already emits parquet; a small notebook over the
  aggregated parquet would give per-category F1 trends across runs.
