#!/usr/bin/env bash
# One-shot cloud diagnostics for the data-pipeline bottleneck.
#
#   bash tools/cloud_diagnose.sh [config]          (default: yolov8_poly_dist.yaml)
#
# Writes everything to diagnose_<timestamp>.log — send that file back.
# Runtime: roughly 10-15 minutes, CPU only (no training, no GPU memory used),
# safe to run while nothing else is training.

set -uo pipefail
CONFIG=${1:-configs/experiments/yolo/yolov8_poly_dist.yaml}
LOG="diagnose_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

section() { echo; echo "================ $1 ================"; }

section "HOST / QUOTA"
date
nproc
uname -a
free -g | head -2
# cgroup v2 then v1: the REAL CPU quota this process gets (e.g. "1300000 100000" = 13 cores)
echo "cpu.max (cgv2):      $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
echo "cfs_quota_us (cgv1): $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo n/a) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo n/a)"
nvidia-smi -L 2>/dev/null || echo "no GPU visible"

section "CPU THROTTLE COUNTERS (baseline)"
# nr_throttled / throttled_usec growing during the benchmark = we are hitting the quota
cat /sys/fs/cgroup/cpu.stat 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null || echo n/a

section "STAGE ATTRIBUTION (datasets, encodings, per-stage rates, threadpool sweep)"
python tools/pipeline/diagnose_pipeline.py --config "$CONFIG" --samples 768 --batches 10 \
    --threadpool-sweep 0,13,26

section "END-TO-END PIPELINE BENCHMARK run 1 (cold cache)"
python tools/benchmark_pipeline.py --config "$CONFIG" --steps 150

section "END-TO-END PIPELINE BENCHMARK run 2 (warm cache)"
python tools/benchmark_pipeline.py --config "$CONFIG" --steps 150

section "CPU THROTTLE COUNTERS (after)"
cat /sys/fs/cgroup/cpu.stat 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null || echo n/a

section "DONE"
echo "Send back: $LOG"
