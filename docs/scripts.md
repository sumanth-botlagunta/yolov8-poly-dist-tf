# Scripts & Tools

Every runnable script in the repo, what it does, its key inputs, and a copy-paste command.
Scripts are grouped: **core workflow** (training/eval/export), **pipeline**, **device export
(SNPE/DLC)**, and **diagnostics**. Replace the angle-bracket paths with yours.

Conventions: most Python tools take `--config <experiment.yaml>` and a checkpoint path prefix
(e.g. `/run/ckpt-100000`, no extension). The experiment YAMLs live in
`configs/experiments/yolo/` — see [configuration.md](configuration.md).

## Core workflow

| Script | Purpose | Key inputs | One-liner |
|--------|---------|-----------|-----------|
| `scripts/run_train.py` | Launch a training run | `--config`, `--output_dir`, `--debug` | `python scripts/run_train.py --config configs/experiments/yolo/yolov8_poly_dist.yaml --output_dir /run` |
| `tools/train_supervisor.sh` | **Recommended** for long runs: keeps training alive across crashes/OOM, auto-resumes, detaches from SSH | `--config`, `--output_dir`; `touch <dir>/STOP` to stop | `nohup bash tools/train_supervisor.sh --config configs/experiments/yolo/yolov8_poly_dist.yaml --output_dir /run >> /run/supervisor.log 2>&1 &` |
| `tools/eval.py` | Evaluate a checkpoint on val/test: COCO mAP, F1, per-category, polygon & distance metrics | `--config`, `--checkpoint`, `--split`, `--per_category`, `--output_dir` | `python tools/eval.py --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --split val --per_category` |
| `tools/infer.py` | Run on arbitrary images, save box+polygon overlays (print distances) | `--config`+`--checkpoint` **or** `--saved_model`; `--images`, `--output_dir`, `--score` | `python tools/infer.py --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --images /imgs --output_dir /tmp/out` |
| `tools/export_saved_model.py` | Export to TF SavedModel (deploy=True, NMS baked) + optional TFLite (`--tflite`). Expects `[0,1]` input | `--config`, `--checkpoint`, `--output_dir`, `--tflite` | `python tools/export_saved_model.py --config configs/experiments/yolo/yolov8_poly_dist.yaml --checkpoint /run/ckpt-100000 --output_dir /export` |
| `tools/continuous_eval.py` | Watch a run dir, auto-evaluate each new checkpoint, append to `eval_log.jsonl` | `--config`, `--watch_dir`, `--interval`, `--max_evals` | `python tools/continuous_eval.py --config configs/experiments/yolo/yolov8_poly_dist.yaml --watch_dir /run` |

## Checkpoints

| Script | Purpose | Key inputs | One-liner |
|--------|---------|-----------|-----------|
| `tools/checkpoint_migration.py` | Load weights into the model from a legacy **or** same-codebase checkpoint (auto strategy) | `migrate --ckpt --config --output [--strategy auto\|native\|frozen\|structural]` | `python tools/checkpoint_migration.py migrate --ckpt /old/ckpt-N --config configs/experiments/yolo/yolov8_poly_dist.yaml --output /tmp/migrated/ckpt` |
| `tools/trace_shapes.py` | Compare two model sources by variable shape/position (pre-migration sanity) | `--src1`, `--src2` (ckpt or YAML), `--only-mismatch` | `python tools/trace_shapes.py --src1 /old/ckpt-N --src2 configs/experiments/yolo/yolov8_poly_dist.yaml` |
| `tools/shared/compare_checkpoints.py` | Name-matched side-by-side diff of two checkpoints (or ckpt vs model) | `--ckpt1`, `--ckpt2`/`--config`, `--grep` | `python tools/shared/compare_checkpoints.py --ckpt1 A --config configs/experiments/yolo/yolov8_poly_dist.yaml` |

## Pipeline

| Script | Purpose | Key inputs | One-liner |
|--------|---------|-----------|-----------|
| `tools/benchmark_pipeline.py` | Measure tf.data throughput (imgs/sec), detect the bottleneck stage | `--config`, `--steps`, `--profile` | `python tools/benchmark_pipeline.py --config configs/experiments/yolo/yolov8_poly_dist.yaml --steps 100` |
| `tools/pipeline/diagnose_pipeline.py` | Stage-by-stage throughput attribution (decode/copy-paste/mosaic/...) | `--config`, `--samples`, `--batches` | `python tools/pipeline/diagnose_pipeline.py --config configs/experiments/yolo/yolov8_poly_dist.yaml --samples 768 --batches 10` |
| `tools/cloud_diagnose.sh` | One-shot cloud bring-up check (cold+warm pipeline, CPU throttle) | `--config` | `bash tools/cloud_diagnose.sh --config configs/experiments/yolo/yolov8_poly_dist.yaml` |
| `tools/pipeline/reencode_tfds_672.py` | One-time: build `<name>_672` pre-resized (672²) TFDS copies to cut decode cost | `--datasets`, `--data_dir`, `--size`, `--splits` | `python tools/pipeline/reencode_tfds_672.py --datasets <tfds_name> --data_dir ~/tensorflow_datasets` |
| `tools/pipeline/export_val_metrics.py` | Export saved `val_metrics/*.json` to xlsx/csv/parquet for trend analysis | `--input`, `--formats`, `--aggregate` | `python tools/pipeline/export_val_metrics.py --input /run/val_metrics --aggregate --formats xlsx,parquet` |

## Device export (Qualcomm SNPE / DLC)

See [device_export.md](device_export.md) for the full on-device workflow and the box channel-order contract.

| Script | Purpose | One-liner |
|--------|---------|-----------|
| `tools/device/export_device_dlc.py` | Export a SavedModel matching the legacy SNPE DLC contract (raw heads, `[0,255]` in, DFL-decoded boxes, `--legacy_box_order`) | `python tools/device/export_device_dlc.py --config <yaml> --checkpoint /run/ckpt-N --output_dir /dlc_export` |
| `tools/device/gen_pred_json_from_dlc.py` | Build a COCO prediction JSON from DLC/SavedModel raw outputs (edit the `SPLITS` list in-file) | `python tools/device/gen_pred_json_from_dlc.py --raw_root /netrun --transform_pkl /x_transform.pkl --output_json /tmp/pred.json` |
| `tools/device/validate_device_export.py` | End-to-end compare the in-repo model vs the device SavedModel on val images | `python tools/device/validate_device_export.py --config <yaml> --checkpoint /run/ckpt-N --saved_model /export` |
| `tools/device/diagnose_device_export.py` | Localize where the device graph diverges (eager → tf.function → SavedModel, per-stage numerics) | `python tools/device/diagnose_device_export.py --config <yaml> --checkpoint /run/ckpt-N` |
| `tools/device/check_snpe_ready.py` | Scan an exported SavedModel for SNPE-incompatible ops (StridedSlice) | `python tools/device/check_snpe_ready.py /export/saved_model` |
| `tools/device/dump_savedmodel_raw.py` | Dump per-node `.raw` outputs in SNPE net-run layout (for DLC diffing) | `python tools/device/dump_savedmodel_raw.py --saved_model /export --raw_image /x.raw` |
| `tools/device/compare_dlc_raw.py` | Compare SavedModel vs DLC raw outputs node-by-node (numpy only) | `python tools/device/compare_dlc_raw.py --a /sm_raw --b /dlc_raw` |
| `tools/device/savedmodel_on_device_raw.py` | Run the SavedModel on the device's `.raw` bytes, draw detections (input-format check) | `python tools/device/savedmodel_on_device_raw.py --saved_model /export --raw /dev_raw` |
| `tools/device/visualize_device_export.py` | Visualize device SavedModel detections on random val images at device size | `python tools/device/visualize_device_export.py --config <yaml> --saved_model /export` |

## Future / recommended additions

- **Distance validation.** The distance head is trained but never scored at validation time
  (see [design_register.md](design_register.md) entry 4). A future change would add a distance
  validation stream and wire `eval/distance_metrics.py` into the val loop so distance
  regressions surface in metrics. This is a training-semantics change (needs a held-out
  distance split), not a pure tooling add.
- **Metrics dashboard.** `export_val_metrics.py` already emits parquet; a small notebook or
  Streamlit view over the aggregated parquet would give per-category F1 trends across runs.
