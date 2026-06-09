"""Data pipeline throughput benchmark.

Builds the full tf.data pipeline (without a model) and measures how fast
batches are produced.  Run this before a long training run to detect
data-starvation bottlenecks.

Usage:
    python tools/benchmark_pipeline.py \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --steps 100 \
        [--profile]   # saves TF profiler trace to /tmp/pipeline_profile/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_pipeline(cfg):
    """Construct the FULL training data pipeline from config (no model).

    Uses the same builder as training (``build_input_reader_from_config``) so the
    benchmark always matches the real pipeline — decode → copy-paste → mosaic /
    random_perspective → parser, plus the distance-stream merge when configured —
    and can't drift from it.
    """
    from data_pipeline.input_reader import build_input_reader_from_config

    return build_input_reader_from_config(
        data_cfg=cfg.task.train_data,
        task_cfg=cfg.task,
        is_training=True,
    )()


def _images_of(batch):
    """Return the image tensor from a (images, labels) tuple or a raw dict."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch['image']


def _run_benchmark(ds, n_steps: int) -> dict:
    """Iterate over *n_steps* batches and return timing stats.

    Step time is the wall-clock *between* successive batches (i.e. how long the
    pipeline takes to produce a batch), and the per-batch image count is read
    from the actual batch — so the merged detection+distance batch (144) is
    measured correctly, not assumed to be the detection size (128).
    """
    import numpy as np

    step_times = []
    total_images = 0
    batch_size = 0

    log.info("Warming up (2 steps)...")
    for batch in ds.take(2):
        _ = _images_of(batch).numpy()[0, 0, 0, 0]  # force full materialization

    log.info("Benchmarking %d steps...", n_steps)
    start_total = time.perf_counter()
    prev = time.perf_counter()

    for i, batch in enumerate(ds.take(n_steps)):
        imgs = _images_of(batch)
        _ = imgs.numpy()[0, 0, 0, 0]              # force materialization
        now = time.perf_counter()
        step_times.append(now - prev)
        prev = now

        batch_size = int(imgs.shape[0])
        total_images += batch_size
        if (i + 1) % 10 == 0:
            recent = step_times[-10:]
            imgs_sec = batch_size / (sum(recent) / len(recent))
            log.info("Step %3d/%d — %.1f imgs/sec (recent avg)", i + 1, n_steps, imgs_sec)

    elapsed = time.perf_counter() - start_total
    step_arr = np.array(step_times)

    return {
        'total_elapsed_sec': elapsed,
        'total_images': total_images,
        'batch_size': batch_size,
        'imgs_per_sec_avg': total_images / elapsed,
        'steps_per_sec_avg': n_steps / elapsed,
        'step_time_p50_ms': float(np.percentile(step_arr, 50)) * 1000,
        'step_time_p95_ms': float(np.percentile(step_arr, 95)) * 1000,
        'step_time_p99_ms': float(np.percentile(step_arr, 99)) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 data pipeline benchmark")
    parser.add_argument(
        '--config', required=True,
        help='Path to experiment YAML config'
    )
    parser.add_argument(
        '--steps', type=int, default=100,
        help='Number of steps to benchmark (default: 100)'
    )
    parser.add_argument(
        '--profile', action='store_true',
        help='Save TF Profiler trace to /tmp/pipeline_profile/'
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.yaml_loader import load_config

    log.info("Loading config: %s", args.config)
    cfg = load_config(args.config)
    batch_size = cfg.task.train_data.global_batch_size

    log.info("Building pipeline...")
    ds = _build_pipeline(cfg)

    if args.profile:
        import tensorflow as tf
        profile_dir = '/tmp/pipeline_profile'
        log.info("Profiling enabled — output: %s", profile_dir)
        tf.profiler.experimental.start(profile_dir)

    stats = _run_benchmark(ds, n_steps=args.steps)

    if args.profile:
        tf.profiler.experimental.stop()
        log.info("Profile saved to %s — open with TensorBoard", profile_dir)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Config:              {args.config}")
    print(f"  Batch size (actual): {stats['batch_size']}  (detection {batch_size}"
          f"{' + distance merge' if stats['batch_size'] != batch_size else ''})")
    print(f"  Steps:               {args.steps}")
    print(f"  Total elapsed:       {stats['total_elapsed_sec']:.1f} sec")
    print(f"  Throughput:          {stats['imgs_per_sec_avg']:.0f} imgs/sec")
    print(f"  Steps/sec:           {stats['steps_per_sec_avg']:.2f}")
    print(f"  Step time p50:       {stats['step_time_p50_ms']:.1f} ms")
    print(f"  Step time p95:       {stats['step_time_p95_ms']:.1f} ms")
    print(f"  Step time p99:       {stats['step_time_p99_ms']:.1f} ms")
    print("=" * 60)

    target = 500
    actual = stats['imgs_per_sec_avg']
    if actual < target:
        print(f"\n  WARNING: Throughput {actual:.0f} imgs/sec is below target "
              f"{target} imgs/sec.")
        print("  The training loop will be data-starved. "
              "Check TFDS read speed and augmentation CPU usage.")
        print("  Consider: reduce albumentations_frequency, add more shuffle workers,")
        print("  or pre-cache the decoded dataset.")
    else:
        print(f"\n  OK: Throughput {actual:.0f} imgs/sec exceeds target {target} imgs/sec.")


if __name__ == '__main__':
    main()
