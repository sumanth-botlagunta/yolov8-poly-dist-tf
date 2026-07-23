"""TEMPORARY throwaway: read a checkpoint's head conv widths (esp. the cls stem).

Standalone (TensorFlow only, no repo imports) so it runs against ANY checkpoint,
legacy or ours. Purpose: settle whether the classification-head stem width differs
between the legacy detector and ours — ours pins it to 128 channels at every FPN
level; stock/likely-legacy scales it (128 / 256 / 512 at P3 / P4 / P5). A width
divergence is invisible to warm-start (the head is re-initialized on transfer), so
the only way to see it is to read the variable shapes off a checkpoint that carries
the head.

Legacy names its head branches differently from ours, so this does NOT rely on the
substring "cls". It locates the classification branch by SHAPE — the final head
conv whose output channels == num_classes is unambiguously the classifier — then
reports the stem widths and dumps every head conv so nothing is hidden.

Read the output:
  - The `head 4-D conv output-width histogram` and the cls-branch stem widths.
    all 128  -> flat, same as ours (F2 DEAD).
    128/256/512 (or any scaled set) -> legacy has more P4/P5 classifier capacity;
    ours (flat 128) is a real cut -> widen models/head.py `_CLS_HIDDEN`.

Run it on the ORIGINAL legacy checkpoint (reveals legacy's true width). The migrated
warm-start source is already in OUR shapes and only confirms what warm-start loads.

Usage:
    python -m utils.inspect_ckpt_head /path/to/ckpt-XXXXX               # num_classes=39
    python -m utils.inspect_ckpt_head /path/to/ckpt-XXXXX --num_classes 39
    python -m utils.inspect_ckpt_head --self_test

Delete this file once the cls-stem question is settled.
"""

import argparse
import sys
from collections import Counter

import tensorflow as tf


def _cout(shape):
    """Output-channel count of a conv kernel [kh, kw, cin, cout]; None otherwise."""
    return int(shape[-1]) if len(shape) == 4 else None


def _base(name):
    """Object-graph var name -> readable base (drop the /.ATTRIBUTES/... tail)."""
    return name.split('/.ATTRIBUTES')[0]


# Substrings that mark a conv as backbone/decoder (not head), used to keep the
# head dump readable when head convs have generic names.
_NON_HEAD = ('backbone', 'stem', 'stage', 'csp', 'c2f', 'sppf', 'darknet',
             'fpn', 'pan', 'decoder', 'upsample', 'downsample', 'neck')

_HEAD_TERMS = ('cls', 'class', 'box', 'pred', 'head', 'conf', 'score', 'obj',
               'logit', 'reg', 'detect', 'dist', 'poly')


def inspect(ckpt_path, num_classes=39):
    """Locate the cls branch by shape, report stem widths, dump head convs.

    Returns the sorted list of distinct cls-branch STEM conv widths (excluding the
    num_classes prediction conv), or None if the cls branch could not be located.
    """
    try:
        variables = tf.train.list_variables(ckpt_path)      # [(name, shape), ...]
    except Exception as e:                                   # noqa: BLE001
        print(f"ERROR: could not read checkpoint '{ckpt_path}': {e}")
        return None

    kernels = [(n, s) for n, s in variables if _cout(s) is not None]   # 4-D convs

    # The classifier is located by SHAPE: the head conv with cout == num_classes.
    cls_pred = [(n, s) for n, s in kernels if _cout(s) == num_classes]

    # Head convs for the dump: name-matched to head terms, excluding backbone/decoder.
    head_kernels = [(n, s) for n, s in kernels
                    if any(t in n.lower() for t in _HEAD_TERMS)
                    and not any(t in n.lower() for t in _NON_HEAD)]

    print(f"\nCheckpoint: {ckpt_path}   (num_classes assumed = {num_classes})")
    print(f"Total variables: {len(variables)}  |  4-D conv kernels: {len(kernels)}  "
          f"|  head-like convs: {len(head_kernels)}\n")

    # Compact, usually-decisive: the output-width histogram of head convs.
    hist = Counter(_cout(s) for _, s in head_kernels)
    print("--- head 4-D conv output-width histogram (width: count) ---")
    for w in sorted(hist):
        tag = ''
        if w == num_classes: tag = '  <- cls prediction (num_classes)'
        elif w == 64:        tag = '  <- box DFL (4*reg_max) likely'
        print(f"  {w:>4}: {hist[w]}{tag}")

    print("\n--- classification-prediction convs (cout == num_classes) ---")
    if cls_pred:
        for n, s in sorted(cls_pred):
            print(f"  {list(s)!s:22s}  {_base(n)}")
    else:
        print(f"  NONE found with cout=={num_classes}. Pass the right --num_classes, "
              "or the head fuses cls into a combined output — read the dump below.")

    # Candidate cls-STEM widths: head convs that share a cls-pred's parent scope,
    # minus the prediction conv itself. Scope = base name without its last 2 path
    # components (drops '.../kernel' and the layer name).
    def _scope(name, drop=2):
        return '/'.join(_base(name).split('/')[:-drop])

    stem_widths = None
    if cls_pred:
        widths = set()
        for pn, _ in cls_pred:
            for drop in (2, 3, 1):
                sc = _scope(pn, drop)
                if not sc:
                    continue
                sib = [(n, s) for n, s in head_kernels
                       if _base(n).startswith(sc) and _cout(s) != num_classes]
                if 0 < len(sib) <= 8:      # a plausible per-branch stem group
                    widths.update(_cout(s) for _, s in sib)
                    break
        stem_widths = sorted(widths)
        print("\n--- cls-branch STEM widths (scope-siblings of the cls prediction conv) ---")
        if stem_widths:
            print(f"  candidate stem widths: {set(stem_widths)}")
            if stem_widths == [128]:
                print("  => flat 128 — same as ours (F2 likely DEAD).")
            elif any(w > 128 for w in stem_widths):
                print("  => SCALED (>128 present) — ours (flat 128) is a real capacity cut (F2 REAL).")
            else:
                print("  => verify against the dump below.")
        else:
            print("  could not isolate stem by scope — READ THE DUMP below "
                  "(find the convs feeding the cout==num_classes conv).")

    print("\n--- full head conv dump (shape : name) — read the cls-branch here ---")
    for n, s in sorted(head_kernels, key=lambda x: _base(x[0])):
        print(f"  {list(s)!s:22s}  {_base(n)}")
    print()
    return stem_widths


def _self_test():
    import os
    import tempfile

    def _save(mod):
        pfx = os.path.join(tempfile.mkdtemp(), 'ckpt')
        tf.train.Checkpoint(model=mod).write(pfx)
        return pfx

    # SCALED legacy-style head: generic names (no 'cls'), classifier located by
    # shape (cout==39), stem widths 128/256/512 nested under a per-level scope.
    class _Scaled(tf.Module):
        def __init__(self):
            self.head_class_3_s1_kernel = tf.Variable(tf.zeros([3, 3, 128, 128]))
            self.head_class_3_pred_kernel = tf.Variable(tf.zeros([1, 1, 128, 39]))
            self.head_class_4_s1_kernel = tf.Variable(tf.zeros([3, 3, 256, 256]))
            self.head_class_4_pred_kernel = tf.Variable(tf.zeros([1, 1, 256, 39]))
            self.head_class_5_s1_kernel = tf.Variable(tf.zeros([3, 3, 512, 512]))
            self.head_class_5_pred_kernel = tf.Variable(tf.zeros([1, 1, 512, 39]))
            self.backbone_stage4_conv_kernel = tf.Variable(tf.zeros([3, 3, 256, 512]))  # excluded

    class _Flat(tf.Module):
        def __init__(self):
            self.head_class_3_s1_kernel = tf.Variable(tf.zeros([3, 3, 128, 128]))
            self.head_class_3_pred_kernel = tf.Variable(tf.zeros([1, 1, 128, 39]))
            self.head_class_4_s1_kernel = tf.Variable(tf.zeros([3, 3, 128, 128]))
            self.head_class_4_pred_kernel = tf.Variable(tf.zeros([1, 1, 128, 39]))
            self.head_class_5_s1_kernel = tf.Variable(tf.zeros([3, 3, 128, 128]))
            self.head_class_5_pred_kernel = tf.Variable(tf.zeros([1, 1, 128, 39]))

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    ws = inspect(_save(_Scaled()), num_classes=39)
    check("scaled: stem widths include 256 and 512", ws is not None and 256 in ws and 512 in ws)
    check("scaled: 39 excluded from stem widths", ws is not None and 39 not in ws)
    check("scaled: backbone conv excluded from head dump (512 stem only from head)",
          ws is not None and set(ws) == {128, 256, 512})

    wf = inspect(_save(_Flat()), num_classes=39)
    check("flat: stem widths == {128}", wf == [128])

    # No cls pred of that num_classes -> must NOT crash or falsely conclude.
    empty = inspect(_save(_Flat()), num_classes=7)   # wrong num_classes on purpose
    check("wrong num_classes: returns without a false width claim",
          empty in (None, []) or 39 not in (empty or []))

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Inspect a checkpoint's head/cls-stem conv widths.")
    ap.add_argument('checkpoint', nargs='?', help='Checkpoint path prefix (e.g. /run/ckpt-920304).')
    ap.add_argument('--num_classes', type=int, default=39, help='Classifier output channels (default 39).')
    ap.add_argument('--self_test', action='store_true', help='Run the built-in self-test and exit.')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if _self_test() else 1)
    if not args.checkpoint:
        ap.error("provide a checkpoint path (or --self_test)")
    inspect(args.checkpoint, num_classes=args.num_classes)


if __name__ == '__main__':
    main()
