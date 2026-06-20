"""THROWAWAY probe: CPU/cgroup throttle facts + input-pipeline thread A/B.

Run this on the TRAINING HOST. It touches no production code — it only reads the
kernel's /sys cgroup files and rebuilds the detection pipeline with different
thread settings to time data-only throughput. Use it to answer two questions:

  1. Is the 13-core limit a cgroup QUOTA (raisable infra knob) or physical cores?
     → the "CPU / cgroup facts" block prints the quota + throttle counters.
  2. What private_threadpool_size / op-thread setting is fastest here?
     → the A/B times data-only out/s for several private_threadpool_size values.

Notes:
  * private_threadpool_size is a per-dataset tf.data option, so it is A/B'd within
    ONE process run.
  * inter/intra op threads can only be set ONCE before TF initializes, so to compare
    them pass --intra-op / --inter-op and RE-RUN the script with different values.
  * Detection stream only (distance disabled) — mosaic is a detection-stream cost.

When you've picked the best settings, put them in the YAML
(runtime.intra_op_threads, train_data.private_threadpool_size) and DELETE this file.

    python -m tools.pipeline.probe_cpu_threads --steps 40 --warmup 10
    python -m tools.pipeline.probe_cpu_threads --intra-op 8 --steps 40   # then compare
"""

import argparse
import os
import time

import tensorflow as tf

from configs.yaml_loader import load_config

_DEFAULT_CONFIG = "configs/experiments/yolo/yolov8_poly_dist.yaml"


def _read(path: str):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def print_cpu_facts():
    print("=" * 70)
    print("CPU / cgroup facts")
    print("=" * 70)
    print(f"  os.cpu_count() (visible cores)     : {os.cpu_count()}")
    if hasattr(os, "sched_getaffinity"):
        print(f"  sched_getaffinity (usable cores)   : {len(os.sched_getaffinity(0))}")

    # cgroup v2
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    if cpu_max is not None:
        print(f"  cgroup v2 cpu.max (quota period)   : {cpu_max}")
        parts = cpu_max.split()
        if parts and parts[0] != "max":
            print(f"     → CPU QUOTA = {int(parts[0]) / int(parts[1]):.2f} cores "
                  f"(this is a RAISABLE infra limit, not physical cores)")
        else:
            print("     → no cgroup CPU quota (limited only by physical/affinity cores)")
        stat = _read("/sys/fs/cgroup/cpu.stat")
        if stat:
            print("  cgroup v2 cpu.stat:")
            for line in stat.splitlines():
                print(f"     {line}")
            _print_throttle_ratio(stat)
    else:
        # cgroup v1
        q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and p and int(q) > 0:
            print(f"  cgroup v1 quota/period = {q}/{p} "
                  f"→ QUOTA = {int(q) / int(p):.2f} cores (RAISABLE infra limit)")
        else:
            print("  cgroup v1: no CPU quota set (or unreadable)")
        stat = _read("/sys/fs/cgroup/cpu/cpu.stat")
        if stat:
            print("  cgroup v1 cpu.stat:")
            for line in stat.splitlines():
                print(f"     {line}")
            _print_throttle_ratio(stat)
    print()


def _print_throttle_ratio(stat: str):
    vals = {}
    for line in stat.splitlines():
        kv = line.split()
        if len(kv) == 2 and kv[1].isdigit():
            vals[kv[0]] = int(kv[1])
    periods = vals.get("nr_periods", 0)
    throttled = vals.get("nr_throttled", 0)
    if periods:
        pct = 100.0 * throttled / periods
        verdict = ("HEAVILY throttled — raising the CPU quota should help a lot"
                   if pct > 25 else
                   "lightly throttled" if pct > 5 else
                   "not meaningfully throttled")
        print(f"     → throttled {throttled}/{periods} periods = {pct:.1f}%  ({verdict})")


def _build_detection_ds(config, private_threadpool: int):
    """Detection-only training dataset with an overridden private_threadpool_size."""
    from data_pipeline.input_reader import build_input_reader_from_config

    config.task.train_data.private_threadpool_size = private_threadpool
    reader = build_input_reader_from_config(
        data_cfg=config.task.train_data,
        task_cfg=config.task,
        is_training=True,
    )
    reader._distance_reader = None  # detection-only
    return reader(None)


def _time_data_only(ds, n_steps: int, warmup: int) -> float:
    it = iter(ds)
    for _ in range(warmup):
        _ = next(it)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        b = next(it)
        _ = (b[0] if isinstance(b, (tuple, list)) else b["image"]).shape
    return n_steps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description="CPU/cgroup probe + pipeline thread A/B")
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--intra-op", type=int, default=0,
                    help="intra_op threads (set ONCE; re-run script to compare values)")
    ap.add_argument("--inter-op", type=int, default=0,
                    help="inter_op threads (set ONCE; re-run script to compare values)")
    ap.add_argument("--threadpools", type=int, nargs="+",
                    default=[13, 26, 0],
                    help="private_threadpool_size values to A/B (0 = all visible cores)")
    args = ap.parse_args()

    # Op-thread caps must be applied before any op runs.
    if args.intra_op > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.intra_op)
    if args.inter_op > 0:
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_op)

    print_cpu_facts()

    config = load_config(args.config)
    batch = config.task.train_data.global_batch_size
    spl = config.trainer.steps_per_loop or 1
    print("=" * 70)
    print(f"Pipeline thread A/B  (batch={batch}, steps={args.steps}, "
          f"intra_op={args.intra_op or 'default'}, inter_op={args.inter_op or 'default'})")
    print("=" * 70)
    print(f"{'private_threadpool':>20} | {'steps/s':>8} | {'img/s':>9} | {'epoch(min)':>10}")
    print("-" * 60)
    for tp in args.threadpools:
        try:
            ds = _build_detection_ds(config, tp)
            sps = _time_data_only(ds, args.steps, args.warmup)
            label = str(tp) if tp > 0 else "0 (all cores)"
            print(f"{label:>20} | {sps:8.2f} | {sps*batch:9.1f} | {spl/sps/60:10.1f}")
        except Exception as e:  # keep going so one bad value doesn't kill the sweep
            print(f"{tp:>20} | FAILED: {type(e).__name__}: {e}")
    print("-" * 60)
    print("Pick the fastest row; set runtime.intra_op_threads + "
          "train_data.private_threadpool_size in the YAML, then delete this file.")


if __name__ == "__main__":
    main()
