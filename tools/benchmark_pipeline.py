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
    """Construct the data pipeline from config without a model."""
    from data_pipeline.tfds_decoders import PolygonDecoder
    from data_pipeline.input_reader import InputReader

    data_cfg = cfg.task.train_data
    task_cfg = cfg.task

    names = [n.strip() for n in data_cfg.tfds_name.split(',')]
    splits = [s.strip() for s in data_cfg.tfds_split.split(',')]

    decoder = PolygonDecoder(
        max_vertices=data_cfg.parser.max_vertices,
        num_classes=task_cfg.num_classes,
        with_distance=False,
    )

    # NOTE: parser is None here — we only benchmark the decode + augmentation
    # stages up to (but not including) the polygon preprocessing, which
    # requires the full parser from Phase 2.
    reader = InputReader(
        tfds_names=names,
        tfds_split=splits,
        tfds_data_dir=data_cfg.tfds_data_dir,
        tfds_sampling_weights=data_cfg.tfds_sampling_weights,
        global_batch_size=data_cfg.global_batch_size,
        is_training=True,
        decoder=decoder,
        parser=None,
        seed=data_cfg.seed,
        shuffle_buffer_size=data_cfg.shuffle_buffer_size,
    )
    return reader()


def _run_benchmark(ds, n_steps: int, batch_size: int) -> dict:
    """Iterate over *n_steps* batches and return timing stats."""
    import tensorflow as tf
    import numpy as np

    step_times = []
    total_images = 0

    log.info("Warming up (2 steps)...")
    for i, batch in enumerate(ds.take(2)):
        pass  # warm-up

    log.info("Benchmarking %d steps (batch_size=%d)...", n_steps, batch_size)
    start_total = time.perf_counter()

    for i, batch in enumerate(ds.take(n_steps)):
        t0 = time.perf_counter()
        # Force execution by accessing batch shape.
        if isinstance(batch, tuple):
            _ = batch[0].shape
        else:
            _ = batch['image'].shape
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
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

    stats = _run_benchmark(ds, n_steps=args.steps, batch_size=batch_size)

    if args.profile:
        tf.profiler.experimental.stop()
        log.info("Profile saved to %s — open with TensorBoard", profile_dir)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Config:              {args.config}")
    print(f"  Batch size:          {batch_size}")
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
