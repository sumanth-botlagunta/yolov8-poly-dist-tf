# /test — Run the test suite

Runs the project test suite with pytest. Pass optional arguments after the command.

## Usage

```
/test                          # run all unit tests (fast, eager mode)
/test unit                     # run tests/unit/ only
/test integration              # run tests/integration/ only
/test -k polygon               # filter by keyword
/test --cov                    # run with coverage report
/test smoke                    # run @pytest.mark.smoke tests (needs real TFDS)
```

## What to run

```bash
cd /Users/sumanth/Documents/Developer/yolov8-poly-dist-tf\ 
```

For `$ARGUMENTS`:
- empty → `pytest tests/unit/ -v`
- `unit` → `pytest tests/unit/ -v`
- `integration` → `pytest tests/integration/ -v`
- `smoke` → `pytest -m smoke tests/smoke/ -v`
- `--cov` → `pytest tests/unit/ tests/integration/ --cov=. --cov-report=term-missing`
- anything else → pass through: `pytest $ARGUMENTS -v`

All tests run with TF eager mode enabled via `conftest.py` — no need to set it manually.

Report: number of passed/failed, any shape assertion errors, any NaN/Inf in loss outputs.
