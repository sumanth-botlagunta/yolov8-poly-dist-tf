# /benchmark — Profile the data pipeline throughput

Runs `utils/pipeline/benchmark_pipeline.py` to measure how fast the tf.data pipeline
feeds batches to the model. Use this to detect data starvation before a long run.

## Usage

```
/benchmark                     # benchmark with yolov8_poly_dist config, 100 steps
/benchmark bbox                # benchmark yolov8_bbox pipeline
/benchmark --steps 50          # run only 50 steps
```

## What to run

```bash
python -m utils.pipeline.benchmark_pipeline \
  --config configs/experiments/yolo/yolov8_{tier}.yaml \
  --steps 100
```

## What to report

- Steps/sec and images/sec (target: >500 imgs/sec on 2×GPU)
- Time breakdown: decode / augment / batch / prefetch
- Whether the pipeline is GPU-bound or data-bound
- Any bottleneck identified (e.g. Albumentations on CPU too slow)

If throughput < 300 imgs/sec, flag as a blocker — the model will be GPU-starved.
