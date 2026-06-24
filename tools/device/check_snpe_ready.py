"""One-shot pre-conversion check for the SNPE export: un-folded BatchNorm + StridedSlice.

Two faults make `snpe-tensorflow-to-dlc` produce a DLC that quantizes badly or fails:

  1. UN-FOLDED BATCHNORM. If the converted graph still contains FusedBatchNormV3 (instead
     of BN folded into the preceding Conv2D, or expressed as Mul/Sub/Rsqrt/AddV2 constants),
     the converter warns `can only merge 1 encoding for src op: .../FusedBatchNormV3
     .../Conv2D, but found 0` and leaves a STANDALONE BN layer. In float that runs fine; once
     QUANTIZED, that BN (a per-channel scale forced into one per-tensor int8 encoding) craters
     and the error cascades downstream -> the classic "BatchNorm layers are the worst diverged"
     pattern. Fold BN before conversion.

  2. StridedSlice. `snpe-tensorflow-to-dlc` failing in StridedSliceLayerBuilder means the graph
     still contains a StridedSlice op. The fix lives in models/backbone.py (C2f -> tf.split)
     and models/decoder.py (static FPN resize); if it persists, the IMPORTED code is not the
     patched tree (often a pip-installed `models.*` shadowing your checkout).

This script prints, with zero ambiguity:
  1. WHICH models/backbone.py and models/decoder.py are imported, and whether each has the fix.
  2. For an exported saved_model: an op-type histogram, the count of FusedBatchNorm* ops, and
     the count of StridedSlice ops (main graph + every nested function), each with a verdict.

Usage:
    python tools/device/check_snpe_ready.py [path/to/exported/saved_model]
"""

import inspect
import sys
from collections import Counter

# TF op names for an un-folded (inference) batch-norm — none of these should survive into a
# graph that is about to be quantized; they must be folded into Conv2D / expressed as constants.
_BN_OPS = ('FusedBatchNormV3', 'FusedBatchNormV2', 'FusedBatchNorm',
           'BatchNormWithGlobalNormalization', 'BatchNorm')


def _check_sources():
    import models.backbone as b
    import models.decoder as d
    print("=== imported module sources (these are what the export uses) ===")
    print("  models.backbone:", b.__file__)
    print("  models.decoder :", d.__file__)
    bsrc = inspect.getsource(b)
    dsrc = inspect.getsource(d)
    # Fix markers (positive, unique to the patched code):
    #   backbone C2f.call: tf.split for the channel split (old code used tensor indexing).
    #   decoder: the static_resize mechanism — dynamic FPN upsample in the model, made a
    #     compile-time-constant size for the export (old code used a plain
    #     tf.image.resize(..., tf.shape(ref)[1:3]) / the reverted _resize_nn helper).
    b_ok = "tf.split(" in inspect.getsource(b.C2f.call)
    d_ok = "static_resize" in dsrc and "def _upsample" in dsrc
    print("  backbone C2f uses tf.split (no ellipsis slice):", "YES" if b_ok else "NO  <-- OLD CODE")
    print("  decoder has static_resize/_upsample (SNPE-clean):", "YES" if d_ok else "NO  <-- OLD CODE")
    return b_ok and d_ok


def _all_nodes(gd):
    """Every node in the graph: the main graph plus every nested function body."""
    nodes = list(gd.node)
    for f in gd.library.function:
        nodes.extend(f.node_def)
    return nodes


def _scan_saved_model(path):
    from tensorflow.python.saved_model import loader_impl
    gd = loader_impl.parse_saved_model(path).meta_graphs[0].graph_def
    nodes = _all_nodes(gd)
    hist = Counter(n.op for n in nodes)

    print(f"\n=== exported graph: {path} ===")
    print(f"  total ops (main+functions): {len(nodes)}")
    print("  op-type histogram (top 15):")
    for op, c in hist.most_common(15):
        print(f"     {op:28s} {c}")

    # ---- un-folded BatchNorm (the encoding-merge / quantization-cascade fault) ----
    bn = [n for n in nodes if n.op in _BN_OPS]
    print(f"\n  un-folded BatchNorm ops ({'/'.join(_BN_OPS[:3])}...): {len(bn)}")
    if bn:
        for n in bn[:8]:
            print(f"     {n.op}: {n.name}")
        if len(bn) > 8:
            print(f"     ... (+{len(bn) - 8} more)")
    folded = hist.get('Mul', 0) and (hist.get('Rsqrt', 0) or hist.get('Sub', 0))
    print("  BN VERDICT:", "CLEAN — no FusedBatchNorm* (BN folded into Conv / constants)"
          if not bn else
          "*** UN-FOLDED BN PRESENT *** -> the 'merge 1 encoding ... found 0' warning; these "
          "quantize badly. Fold BN into Conv before snpe-tensorflow-to-dlc.")
    if bn and folded:
        print("     (note: Mul/Rsqrt/Sub constants ALSO present — folding is partial; some BN "
              "folded, some not. Re-export so NONE survive.)")

    # ---- StridedSlice (the SNPE builder / stale-code fault) ----
    ss = [n for n in nodes if n.op == "StridedSlice"]
    bad = [n.name for n in ss if n.attr["ellipsis_mask"].i or n.attr["new_axis_mask"].i]
    print(f"\n  StridedSlice ops (main+functions): {len(ss)}"
          + (f"  ({len(bad)} with ellipsis/new_axis mask)" if bad else ""))
    if ss:
        print("  example StridedSlice names:")
        for n in ss[:6]:
            print("     ", n.name,
                  f"(ellipsis={n.attr['ellipsis_mask'].i} new_axis={n.attr['new_axis_mask'].i})")
    print("  SS VERDICT:", "CLEAN — SNPE StridedSliceLayerBuilder cannot fire"
          if not ss else "STALE/OLD — re-export with the patched code (see module paths above)")
    return (len(ss) == 0) and (len(bn) == 0)


if __name__ == "__main__":
    code_ok = _check_sources()
    sm_ok = True
    if len(sys.argv) > 1:
        sm_ok = _scan_saved_model(sys.argv[1])
    else:
        print("\n(no saved_model path given — pass one to scan the exported graph)")
    print("\nSUMMARY:",
          "graph is SNPE-ready (BN folded, no StridedSlice) — good to convert/quantize"
          if (code_ok and sm_ok)
          else "NOT READY — fix the FAIL verdict(s) above (fold BatchNorm into Conv, and/or "
               "re-export the patched StridedSlice-free code) before converting/quantizing")
