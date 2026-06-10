# Testing

`pytest` suite under `tests/`. All tests run in **eager mode** (`tests/conftest.py` autouses
`tf.config.run_functions_eagerly(True)`).

## Layout

| Location | Scope | Needs TFDS? |
|----------|-------|-------------|
| `tests/unit/` | pure component unit tests: backbone, decoders, model forward, EMA, sgd_warmup, tal_assigner, coco/distance/polygon evaluators (including crowd/dontcare handling), config loading, viz_utils | no |
| `tests/integration/` | end-to-end pipeline, checkpoint migration, weight-map migration | no |
| `tests/smoke/` | training-loop smoke (`TestDrySmoke` on synthetic data, 10 steps) + real-data smoke (`TestRealDataSmoke`, `@pytest.mark.smoke`) | only the marked real-data class |
| `tests/test_*.py` (top level) | component tests: decoders, parser, copy_paste, mosaic, losses (computation + reference parity + polygon conventions + distance loss), polygon preprocessing, batch shapes | no |

**Top-level test files (10 files):** `test_batch_shape_consistency.py`, `test_copy_paste.py`,
`test_decoders.py` (includes encoded-bytes / `SkipDecoding` decoder tests), `test_distance_loss.py`,
`test_loss_computation.py`, `test_loss_reference_parity.py`,
`test_mosaic.py` (includes 4-in/4-out mosaic assertions), `test_parser.py`,
`test_polygon_loss_conventions.py`,
`test_polygon_preprocessing.py` (includes segment-equivalence tests asserting exact output parity
of the `unsorted_segment_max` / `segment_min` formulation vs the old one-hot reference).

**Unit test files (15 files):** `test_backbone.py`, `test_bf16_policy.py` (bfloat16 Keras policy
applied correctly, heads remain float32), `test_coco_crowd_dontcare.py`,
`test_coco_evaluator.py`, `test_config_loading.py`, `test_decoders.py`,
`test_distance_evaluator.py`, `test_ema.py`, `test_model_forward.py`,
`test_polygon_evaluator.py`, `test_sgd_warmup.py`, `test_tal_assigner.py`,
`test_task_validation_streaming.py`,
`test_trainer_epoch_math.py` (verifies `YoloV8Trainer._steps_for_epoch` for fresh starts, full
epochs, and mid-epoch resume remainder),
`test_viz_utils.py`.

**Integration test files (5 files):** `test_full_pipeline.py`, `test_checkpoint_migration.py`,
`test_weight_map_migration.py`, `test_multigpu.py`, `test_ckpt_eval_loading.py`.

`test_multigpu.py` runs a real 2-replica `MirroredStrategy` on two **virtual CPU devices** to
validate the distributed-training machinery (global-count loss normalizers, cross-replica
gradient all-reduce, EMA + pre-built optimizer slots under `strategy.run`). It must run in a
**fresh process** — splitting the CPU into logical devices only works before TF's device context
is initialized, so in the shared suite run it self-skips. CI runs it as a separate step:
`pytest tests/integration/test_multigpu.py`.

## Running

```bash
# Fast, no datasets — what CI runs:
pytest tests/unit tests/integration -q

# A single file or test:
pytest tests/test_loss_reference_parity.py -q
pytest tests/unit/test_tal_assigner.py::TestTaskAlignedAssigner -q

# Skip the real-data smoke tests (they self-skip if TFDS_DATA_DIR is unset anyway):
pytest tests -m "not smoke" -q

# Real-data smoke (requires the TFDS datasets on disk):
TFDS_DATA_DIR=/path/to/tensorflow_datasets pytest tests/smoke -m smoke -q

# With coverage:
pytest tests/unit tests/integration --cov -q
```

Or use the `/test` skill.

## Conventions for new tests
- Unit/integration tests **must not** require TFDS — build synthetic tensors inline.
- Write **discriminating** assertions (pin a value/relationship that fails on regression), not
  just "runs without error". See `tests/test_loss_reference_parity.py` for the pattern: it fails
  against the buggy behavior and passes once fixed.
- Reuse `tests/conftest.py` fixtures (`tiny_model_cfg`, `synthetic_image`, `synthetic_labels`, …).
- Match coordinate conventions (`yxyx` normalized GT vs `xyxy` pixels in the loss) — mismatches are
  the most common test bug.
- The `tf-test-writer` subagent (`.claude/agents/`) knows these conventions.

## CI
`.github/workflows/test.yml` installs `requirements.txt` and runs `pytest -m "not smoke"`
on push/PR — the whole suite except the real-data smoke tests (the synthetic dry-smoke loop
is included). The real-data smoke suite is intentionally out of CI (needs the datasets); run it
locally before large changes.
