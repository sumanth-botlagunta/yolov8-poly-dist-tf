"""TEMPORARY throwaway: read a checkpoint's head conv widths (esp. the cls stem).

Standalone (TensorFlow only, no repo imports) so it runs against ANY checkpoint,
legacy or ours. Purpose: settle whether the classification-head stem width differs
between the legacy detector and ours — ours pins it to 128 channels at every FPN
level; stock/likely-legacy scales it (128 / 256 / 512 at P3 / P4 / P5). A width
divergence is invisible to warm-start (the head is re-initialized on transfer), so
the only way to see it is to read the variable shapes off a checkpoint that carries
the head.

Read the output like this:
  - "distinct cls-stem conv output widths" == {128}         -> flat, no difference.
  - "distinct cls-stem conv output widths" == {128,256,512} -> scaled; ours (flat
    128) is a real P4/P5 classifier-capacity cut. Widen models/head.py `_CLS_HIDDEN`.

Run it on the ORIGINAL legacy checkpoint if you have it (that reveals legacy's true
width). Running on the migrated warm-start source (already in OUR variable shapes)
only confirms what warm-start loads, not legacy's native width.

Usage:
    python -m utils.inspect_ckpt_head /path/to/ckpt-920304
    python -m utils.inspect_ckpt_head --self_test

Delete this file once the cls-stem question is settled.
"""

import argparse
import re
import sys

import tensorflow as tf


def _cout(shape):
    """Output-channel count of a conv kernel [kh, kw, cin, cout]; None otherwise."""
    return int(shape[-1]) if len(shape) == 4 else None


def _guess_level(name):
    """Best-effort FPN level (3/4/5) from a variable name; '?' if unclear."""
    m = re.search(r'(?:^|[^0-9])([345])(?:[^0-9]|$)', name)
    return m.group(1) if m else '?'


def inspect(ckpt_path):
    """List head/cls/box variables + summarize cls-stem conv widths. Returns the
    set of distinct cls-stem conv output widths found."""
    try:
        variables = tf.train.list_variables(ckpt_path)   # [(name, shape), ...]
    except Exception as e:                                # noqa: BLE001
        print(f"ERROR: could not read checkpoint '{ckpt_path}': {e}")
        return None

    head_like = [(n, s) for n, s in variables
                 if any(k in n.lower() for k in ('cls', 'class', 'box', 'pred', 'head'))]
    cls_like = [(n, s) for n, s in head_like
                if ('cls' in n.lower() or 'class' in n.lower())]

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Total variables: {len(variables)}  |  head-like: {len(head_like)}  "
          f"|  cls-like: {len(cls_like)}\n")

    print("--- all cls/class variables (name : shape) ---")
    for n, s in sorted(cls_like):
        print(f"  {list(s)!s:20s}  {n}")

    # Split cls conv kernels (4-D) into STEM (hidden width, cout>=64, not a 'pred'
    # layer) vs the final PREDICTION conv (cout==num_classes, small). Only the stem
    # width is the F2 question; the prediction conv is always num_classes.
    cls_convs = [(n, s) for n, s in cls_like if _cout(s) is not None]
    stem_convs = [(n, s) for n, s in cls_convs
                  if 'pred' not in n.lower() and _cout(s) >= 64]
    pred_convs = [(n, s) for n, s in cls_convs if (n, s) not in stem_convs]

    widths = sorted({_cout(s) for _, s in stem_convs})
    per_level = {}
    for n, s in stem_convs:
        per_level.setdefault(_guess_level(n), set()).add(_cout(s))

    print("\n--- cls-stem summary ---")
    print(f"  cls conv kernels: {len(cls_convs)}  (stem: {len(stem_convs)}, "
          f"pred/other: {len(pred_convs)})")
    if pred_convs:
        print(f"  prediction-layer convs (cout=num_classes, ignored for width): "
              f"{sorted({_cout(s) for _, s in pred_convs})}")
    print(f"  distinct cls-stem conv output widths: {set(widths) or '(none — no 4-D cls stem kernels)'}")
    for lvl in sorted(per_level):
        print(f"    level {lvl}: widths {sorted(per_level[lvl])}")
    if widths == [128] or widths == []:
        print("  => flat/absent — no cls-stem width difference vs ours (F2 likely DEAD).")
    elif len(widths) > 1:
        print("  => SCALED widths — ours (flat 128) is a real classifier-capacity cut (F2 REAL).")
    print()
    return set(widths)


def _self_test():
    import os
    import tempfile

    class _Head(tf.Module):
        def __init__(self):
            # Mimic a SCALED cls stem (128/256/512) + a box stem, to prove the
            # inspector reads per-level widths and flags a scaled head.
            self.cls_s1_3_kernel = tf.Variable(tf.zeros([3, 3, 128, 128]))
            self.cls_s1_4_kernel = tf.Variable(tf.zeros([3, 3, 256, 256]))
            self.cls_s1_5_kernel = tf.Variable(tf.zeros([3, 3, 512, 512]))
            self.cls_pred_3_kernel = tf.Variable(tf.zeros([1, 1, 128, 39]))
            self.box_pred_3_kernel = tf.Variable(tf.zeros([1, 1, 136, 64]))

    tmp = tempfile.mkdtemp()
    prefix = os.path.join(tmp, 'ckpt')
    tf.train.Checkpoint(model=_Head()).write(prefix)

    widths = inspect(prefix)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    check("cls widths == {128,256,512} (per-level read correct)",
          widths == {128, 256, 512})
    # A flat head should read as {128}: rebuild flat and confirm.
    class _Flat(tf.Module):
        def __init__(self):
            self.cls_s1_3_kernel = tf.Variable(tf.zeros([3, 3, 128, 128]))
            self.cls_s1_4_kernel = tf.Variable(tf.zeros([3, 3, 128, 128]))
            self.cls_s1_5_kernel = tf.Variable(tf.zeros([3, 3, 128, 128]))
    prefix2 = os.path.join(tmp, 'flat')
    tf.train.Checkpoint(model=_Flat()).write(prefix2)
    check("flat head reads as {128}", inspect(prefix2) == {128})

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Inspect a checkpoint's head/cls-stem conv widths.")
    ap.add_argument('checkpoint', nargs='?', help='Checkpoint path prefix (e.g. /run/ckpt-920304).')
    ap.add_argument('--self_test', action='store_true', help='Run the built-in self-test and exit.')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if _self_test() else 1)
    if not args.checkpoint:
        ap.error("provide a checkpoint path (or --self_test)")
    inspect(args.checkpoint)


if __name__ == '__main__':
    main()
