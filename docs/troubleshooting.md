# Troubleshooting

Common failures and what to check. For on-device export (`--verify` mismatches, box transpose),
see [device_export.md](device_export.md).

| Symptom | Likely cause / fix |
|---------|--------------------|
| `Failed to load TFDS dataset ...` at startup | The TFDS dataset/version isn't on disk or `tfds_data_dir` is wrong. Confirm the dataset is built under `TFDS_DATA_DIR` and that the path in the YAML matches. The distance dataset (`servingbot_polygon`) is **training-only** — don't reference it from a validation split. |
| `Config validation failed (...)` at startup | An invariant in `scripts/run_train.py:_validate_config` failed — a missing init checkpoint, `output_poly_size != 360 // angle_step`, or a bad mosaic `group_size`/`decodes_per_output` (group must be ≥ 4 and a multiple of R). The message lists each error. |
| Config key seems to have no effect | The hand-rolled loader **silently ignores unknown keys** outside the `runtime`/`losses` sections. Check the key name against the dataclass field in `configs/model_config.py` (see [configuration.md](configuration.md)). |
| OOM during training | Lower `train_data.global_batch_size` or the input size. Multi-GPU `MirroredStrategy` shards each global batch across replicas (per-replica = `global_batch_size / num_replicas`); keep the global batch divisible by the replica count. |
| `NaN` loss after a few steps | Usually too-high LR or unstable mixed precision. Use the default `float32` config to isolate; prefer `bfloat16` over `float16` if enabling mixed precision (no loss scaling needed); confirm GT boxes/polygons are valid (no degenerate zero-area boxes). |
| Eval metrics look identical to raw (worse) weights | EMA weights weren't swapped in. During training EMA is swapped before validation and back after (`optimizers/ema.py:swap_weights`). `tools/eval.py` and `tools/export_saved_model.py` use `tools/shared/ckpt_loading.restore_eval_weights`, which auto-detects EMA shadows in a periodic `ckpt-N` and swaps them in (a `best_*` checkpoint already holds EMA weights). |
| Data pipeline is the bottleneck (low `imgs/sec`, high `train/data_wait_ms`) | Run `python -m tools.benchmark_pipeline --config <yaml>` for throughput and `python -m tools.pipeline.diagnose_pipeline --config <yaml>` for stage-by-stage attribution. If decode/pre-resize dominate, build the 672² pre-resized dataset variants once with `python -m tools.pipeline.reencode_tfds_672`. Mosaic `decodes_per_output` (R) multiplies decode work — lower it (toward 1) if epochs are data-bound. |
| Warm-start didn't load the backbone fully | Warm-start from a periodic `ckpt-N` or `best_*` checkpoint (they carry EMA), not a model-only export — the complete weights live in the EMA shadows. See [checkpoint_migration.md](checkpoint_migration.md). |
| Slow first epoch / SSH disconnect kills training | Launch via `tools/train_supervisor.sh` under `nohup` (see the README Training section) so the run detaches and auto-restarts. |
