#!/usr/bin/env bash
# Training supervisor: keeps run_train.py alive across crashes/OOM-kills and
# resumes automatically — no manual babysitting.
#
#   nohup bash tools/train_supervisor.sh \
#       --config configs/experiments/yolo/yolov8_poly_dist.yaml \
#       --output_dir /path/to/run_dir \
#       >> /path/to/run_dir/supervisor.log 2>&1 &
#
# (nohup/tmux matters: running training in a bare VS Code remote terminal ties
#  it to the SSH session — a disconnect can stall stdout and freeze the run for
#  hours, or kill it outright. Under nohup the process never sees the hangup.)
#
# Behavior:
#   * Restarts training whenever it exits abnormally (e.g. kernel OOM "Killed"
#     = exit 137). Resume is automatic: the trainer restores the newest of the
#     epoch-boundary checkpoints and resume/ interruption checkpoints, then
#     runs exactly the remaining steps of the interrupted epoch.
#   * STOPPING ON PURPOSE: `touch <output_dir>/STOP` — the supervisor exits
#     after the current attempt ends instead of restarting. (Also: Ctrl-C /
#     SIGTERM to the supervisor forwards the signal to training so it writes a
#     resume checkpoint, then exits without restarting.)
#   * Exit code 0 from training (run completed) ends the supervisor.
#   * Crash-loop guard: 5 consecutive exits within 120s abort the supervisor —
#     that is a real bug, not an OOM blip; check train.log.
set -u

CONFIG=""; OUTPUT_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config)     CONFIG="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
[ -n "$CONFIG" ] && [ -n "$OUTPUT_DIR" ] || { echo "usage: $0 --config <yaml> --output_dir <dir>"; exit 2; }
mkdir -p "$OUTPUT_DIR"
STOP_FILE="$OUTPUT_DIR/STOP"
[ -f "$STOP_FILE" ] && { echo "[supervisor] STOP file already present — refusing to start. Remove $STOP_FILE first."; exit 0; }

CHILD_PID=""
INTERRUPTED=0
forward() {  # forward signal to training so it writes a resume checkpoint
  INTERRUPTED=1
  echo "[supervisor] signal received — forwarding to training (pid $CHILD_PID) for graceful save"
  [ -n "$CHILD_PID" ] && kill -TERM "$CHILD_PID" 2>/dev/null
}
trap forward TERM INT

fast_fail_count=0
attempt=0
while true; do
  attempt=$((attempt + 1))
  start_ts=$(date +%s)
  echo "[supervisor] attempt $attempt starting at $(date '+%F %T')"
  python scripts/run_train.py --config "$CONFIG" --output_dir "$OUTPUT_DIR" &
  CHILD_PID=$!
  wait "$CHILD_PID"; code=$?
  dur=$(( $(date +%s) - start_ts ))
  echo "[supervisor] training exited code=$code after ${dur}s"

  [ "$INTERRUPTED" = "1" ] && { echo "[supervisor] exiting after forwarded signal (no restart)"; exit 0; }
  [ -f "$STOP_FILE" ] && { echo "[supervisor] STOP file found — exiting (no restart)"; exit 0; }
  [ "$code" = "0" ] && { echo "[supervisor] training completed normally — exiting"; exit 0; }

  if [ "$dur" -lt 120 ]; then
    fast_fail_count=$((fast_fail_count + 1))
    if [ "$fast_fail_count" -ge 5 ]; then
      echo "[supervisor] 5 consecutive failures under 120s — aborting (real bug, see train.log)"
      exit 1
    fi
  else
    fast_fail_count=0
  fi
  echo "[supervisor] restarting in 30s (touch $STOP_FILE to prevent)"
  sleep 30
done
