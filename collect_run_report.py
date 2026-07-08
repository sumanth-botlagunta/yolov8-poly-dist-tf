"""One-shot run report: every remaining open question in a single printout.

Pure stdlib — no TensorFlow, no imports that can crash.

Usage (from the repo root):
    python collect_run_report.py \
        --run_dir /path/to/run_dir \
        [--config configs/experiments/yolo/<the yaml you train with>] \
        [--checkpoint ckpt-NNNNN]
"""
import argparse, json, os, re, subprocess, sys

REPO = os.path.dirname(os.path.abspath(__file__))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=15, cwd=REPO).stdout.strip()
    except Exception as e:
        return f"(failed: {e})"


def show_yaml_blocks(path, title):
    print(f"\n--- {title}: {path} ---")
    if not os.path.exists(path):
        print("  (file not found)")
        return
    keys = re.compile(r'^\s*(losses:|optimizer_config:|optimizer:|learning_rate:|ema:|runtime:'
                      r'|type:|sgd|momentum|weight_decay|warmup|initial_learning_rate|decay_steps'
                      r'|.*_gain:|tfds_name|tfds_sampling_weights|global_batch_size|resample_points'
                      r'|mosaic_frequency|decodes_per_output|group_size|disable_onednn'
                      r'|mixed_precision|distribution_strategy|num_gpus|train_total_examples'
                      r'|train_epochs|skip_crowd|init_checkpoint|finetune_from)')
    for i, line in enumerate(open(path, errors='replace'), 1):
        if keys.match(line) and not line.strip().startswith('#'):
            print(f"  {i:>4} {line.rstrip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--checkpoint', default=None)
    a = ap.parse_args()

    print("=== A. working tree NOW ===")
    print("  branch :", sh("git rev-parse --abbrev-ref HEAD"))
    print("  head   :", sh("git log --oneline -2").replace("\n", "\n           "))
    st = sh("git status --short").splitlines()
    print(f"  dirty  : {len(st)} modified/untracked files")
    for line in st[:10]:
        print("          ", line)

    print("\n=== B. launch provenance (run_metadata.json) ===")
    launch_commit = None
    try:
        meta = json.load(open(os.path.join(a.run_dir, 'run_metadata.json')))
        g = meta.get('git', {})
        launch_commit = g.get('commit')
        print(f"  commit : {str(launch_commit)[:12]}   dirty: {g.get('dirty')}")
        print(f"  command: {meta.get('command','?')[:150]}")
    except Exception as e:
        print(f"  (unavailable: {e})")

    print("\n=== C. what that launch commit is ===")
    if launch_commit:
        print("  " + (sh(f"git show -s --format='%h %ad %s' {launch_commit}")
                      or "(commit not found in local history!)"))
        print("  contains disable_onednn support:",
              bool(sh(f"git grep -l disable_onednn {launch_commit} -- configs/ 2>/dev/null")))
        print("  contains sgd_legacy support   :",
              bool(sh(f"git grep -l sgd_legacy {launch_commit} -- optimizers/ 2>/dev/null")))

    print("\n=== D. run.diff (the run's uncommitted code at launch) ===")
    rd = os.path.join(a.run_dir, 'run.diff')
    if os.path.exists(rd):
        txt = open(rd, errors='replace').read()
        files = re.findall(r'^diff --git a/(\S+)', txt, re.M)
        print(f"  EXISTS: {len(txt)} bytes, {len(files)} files touched:")
        for f in files[:20]:
            print("   ", f)
        print("  --- first 60 lines ---")
        print("  " + "\n  ".join(txt.splitlines()[:60]))
    else:
        print("  (no run.diff in run dir)")

    show_yaml_blocks(os.path.join(a.run_dir, 'params.yaml'),
                     "E. LAUNCH config (key lines)")
    if a.config:
        show_yaml_blocks(os.path.join(REPO, a.config), "F. config file NOW (key lines)")

    print("\n=== G. checkpoint <-> val_history pairing ===")
    vh = os.path.join(a.run_dir, 'val_history.jsonl')
    try:
        rows = [json.loads(l) for l in open(vh) if l.strip()]
        tail = rows[-3:]
        hits = ([r for r in rows if a.checkpoint and a.checkpoint in str(r.get('checkpoint', ''))]
                or tail)
        label = "matching rows" if a.checkpoint and hits is not tail else "last rows"
        print(f"  ({label})")
        for r in hits[-5:]:
            print(f"  epoch={r.get('epoch')} step={r.get('step')} "
                  f"ckpt={os.path.basename(str(r.get('checkpoint','')))} "
                  f"F1={r.get('F1score50')} P={r.get('precision50')} R={r.get('recall50')}")
        if a.checkpoint and hits is tail:
            print(f"  ^^ NO row names '{a.checkpoint}' — TB numbers compared against it "
                  f"belong to a different step's weights.")
    except Exception as e:
        print(f"  (val_history unavailable: {e})")

    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
