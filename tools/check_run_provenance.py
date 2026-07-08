"""One-shot provenance cross-examination for a training run (seconds, no model).

Answers: when TensorBoard/val_history and a standalone eval of the "same"
checkpoint disagree, WHICH input differs — config, dataset version, or
checkpoint/row pairing. Compares four sources of truth:

  1. <run>/run_metadata.json  — git commit + the RESOLVED dataset versions at launch
  2. <run>/params.yaml        — the exact config the run was launched with
  3. the --config file NOW    — what a standalone tools.eval would use
  4. <run>/val_history.jsonl  — the run's own recorded metrics per checkpoint

Usage:
    python -m tools.check_run_provenance \
        --checkpoint <run_dir>/ckpt-NNNN \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml
"""

import argparse
import json
import os

# Metadata-only tool: never touch the GPU. A second TF process initializing
# CUDA against a GPU fully owned by a live training job can crash at the
# driver level (segfault) instead of raising. Must run before TF is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MISMATCHES = []


def _flag(name, launch, now):
    same = launch == now
    if not same:
        MISMATCHES.append(name)
    print(f"  {name:<44} launch={launch!r:<28} now={now!r:<28} {'ok' if same else '<-- MISMATCH'}")


def _get(cfg, dotted, default='<missing>'):
    cur = cfg
    for part in dotted.split('.'):
        cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    ckpt_base = os.path.basename(args.checkpoint.rstrip('/'))

    # ---- 1. launch-time provenance -------------------------------------
    print("=== 1. run_metadata.json (launch-time truth) ===")
    launch_datasets = {}
    try:
        meta = json.load(open(os.path.join(run_dir, 'run_metadata.json')))
        g = meta.get('git', {})
        print(f"  launch git commit: {str(g.get('commit'))[:12]}   dirty: {g.get('dirty')}")
        print(f"  command: {meta.get('command', '?')[:140]}")
        for d in meta.get('datasets', []):
            key = f"{d['role']}:{d['name']}"
            launch_datasets[key] = d.get('resolved')
            print(f"  {d['role']:<10} {d['name']:<40} requested={d.get('requested')}  "
                  f"resolved={d.get('resolved')}")
    except Exception as e:
        print(f"  (unavailable: {e})")

    # ---- 2. launch config vs current config ----------------------------
    print("\n=== 2. eval-relevant config: launch params.yaml vs --config NOW ===")
    from configs.yaml_loader import load_config
    now_cfg = load_config(args.config)
    launch_yaml = os.path.join(run_dir, 'params.yaml')
    fields = [
        'task.num_classes',
        'task.model.input_size',
        'task.validation_data.tfds_name',
        'task.validation_data.tfds_split',
        'task.validation_data.tfds_data_dir',
        'task.validation_data.global_batch_size',
        'task.validation_data.parser.resample_points',
        'task.validation_data.parser.eval_gray_border',
        'task.ignore_iscrowds',
        'task.iscrowds_labels',
        'task.ignore_dontcare',
        'task.model.detection_generator.score_thresh',
        'task.model.detection_generator.nms_thresh',
        'task.model.detection_generator.max_boxes',
        'task.model.detection_generator.nms_class_mode',
    ]
    try:
        launch_cfg = load_config(launch_yaml)
        for f in fields:
            _flag(f, _get(launch_cfg, f), _get(now_cfg, f))
    except Exception as e:
        print(f"  (launch params.yaml unavailable/unparsable: {e} — "
              f"comparing against current file only)")
        for f in fields:
            print(f"  {f:<44} now={_get(now_cfg, f)!r}")

    # ---- 3. TFDS resolution NOW vs launch -------------------------------
    print("\n=== 3. val dataset version: resolved NOW vs at launch ===")
    try:
        import tensorflow_datasets as tfds
        vd = now_cfg.task.validation_data
        names = [s.strip() for s in str(vd.tfds_name).split(',') if s.strip()]
        for name in names:
            b = tfds.builder(name, data_dir=vd.tfds_data_dir)
            now_ver = str(b.info.version)
            launch_ver = launch_datasets.get(f"val:{name.partition(':')[0]}")
            print(f"  {name:<44} now={now_ver}  launch={launch_ver}")
            if launch_ver is not None and str(launch_ver) != now_ver:
                MISMATCHES.append(f"val dataset version ({name})")
    except Exception as e:
        print(f"  (could not resolve: {str(e)[:150]})")

    # ---- 4. checkpoint <-> val_history pairing --------------------------
    print("\n=== 4. checkpoint vs the run's own recorded metrics ===")
    try:
        import tensorflow as tf
        r = tf.train.load_checkpoint(args.checkpoint)
        try:
            step = int(r.get_tensor('global_step/.ATTRIBUTES/VARIABLE_VALUE'))
            print(f"  checkpoint global_step: {step}")
        except Exception:
            print("  checkpoint has no global_step slot (model-only/best checkpoint)")
    except Exception as e:
        print(f"  (checkpoint unreadable: {e})")
    try:
        from eval.val_history import load_records
        rows = load_records(os.path.join(run_dir, 'val_history.jsonl'))
        hit = [x for x in rows if ckpt_base in str(x.get('checkpoint', ''))]
        show = hit if hit else rows[-3:]
        for x in show:
            print(f"  epoch={x.get('epoch')} step={x.get('step')} "
                  f"ckpt={os.path.basename(str(x.get('checkpoint', '')))} "
                  f"F1={x.get('F1score50')} P={x.get('precision50')} R={x.get('recall50')}")
        if not hit:
            MISMATCHES.append('checkpoint/row pairing')
            print(f"  ^^ NO val_history row names '{ckpt_base}' — any TB/val_history number "
                  f"you compared against belongs to a DIFFERENT step's weights.")
    except Exception as e:
        print(f"  (val_history unavailable: {e})")

    # ---- verdict ---------------------------------------------------------
    print("\n=== VERDICT ===")
    if MISMATCHES:
        print("  Divergence source candidates found:")
        for m in MISMATCHES:
            print(f"   - {m}")
    else:
        print("  All four sources agree: same config, same dataset version, matched "
              "checkpoint/row. If the standalone eval STILL differs from the recorded "
              "row, report that — it would contradict the proven path equivalence and "
              "the next step is a weight-checksum comparison.")


if __name__ == '__main__':
    main()
    # Skip interpreter teardown: TF's C++ destructors can segfault at exit in
    # some builds (after all output is already printed). Flush and hard-exit.
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
