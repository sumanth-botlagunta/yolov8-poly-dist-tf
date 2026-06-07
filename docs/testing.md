# Testing

`pytest` suite under `tests/`. All tests run in **eager mode** (`tests/conftest.py` autouses
`tf.config.run_functions_eagerly(True)`).

## Layout

| Location | Scope | Needs TFDS? |
|----------|-------|-------------|
| `tests/unit/` | pure component unit tests (backbone, decoders, model forward, EMA, sgd_warmup, tal_assigner, the coco/distance/polygon evaluators, config loading, viz_utils) | no |
| `tests/integration/` | end-to-end pipeline, checkpoint migration | no |
| `tests/smoke/` | training-loop smoke (`TestDrySmoke` on synthetic data) + real-data smoke (`TestRealDataSmoke`, `@pytest.mark.smoke`) | only the marked real-data class |
| `tests/test_*.py` (top level) | component tests: decoders, parser, copy_paste, mosaic, losses, polygon preprocessing, batch shapes | no |

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
