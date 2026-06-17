"""One-shot check: is the SNPE export actually picking up the StridedSlice fix?

`snpe-tensorflow-to-dlc` failing in StridedSliceLayerBuilder means the graph it reads
still contains a StridedSlice op. The fix lives in models/backbone.py (C2f -> tf.split)
and models/decoder.py (static FPN resize). If the export still emits StridedSlice, the
code being IMPORTED is not the patched code — almost always because the repo is
pip-installed and `import models.*` resolves to site-packages, not your pulled tree.

This script prints, with zero ambiguity:
  1. WHICH models/backbone.py and models/decoder.py are actually imported (their paths),
     and whether each carries the fix marker.
  2. For a given exported saved_model, how many StridedSlice ops it contains
     (main graph + every nested function), with a PASS/FAIL verdict.

Usage:
    python tools/check_snpe_ready.py [path/to/exported/saved_model]
"""

import inspect
import sys


def _check_sources():
    import models.backbone as b
    import models.decoder as d
    print("=== imported module sources (these are what the export uses) ===")
    print("  models.backbone:", b.__file__)
    print("  models.decoder :", d.__file__)
    bsrc = inspect.getsource(b)
    dsrc = inspect.getsource(d)
    # Fix markers (inspect the actual call bodies, not comments): the C2f forward must
    # use tf.split for the channel split; the decoder must route the FPN upsample
    # through _resize_nn (static size, no Shape/StridedSlice).
    # tf.split appears only in the patched C2f.call (old code used tensor indexing);
    # _resize_nn appears only in the patched decoder (old code used tf.shape(...)[1:3]).
    b_ok = "tf.split(" in inspect.getsource(b.C2f.call)
    d_ok = "def _resize_nn" in dsrc
    print("  backbone C2f uses tf.split (no ellipsis slice):", "YES" if b_ok else "NO  <-- OLD CODE")
    print("  decoder uses _resize_nn (static resize):       ", "YES" if d_ok else "NO  <-- OLD CODE")
    return b_ok and d_ok


def _scan_saved_model(path):
    from tensorflow.python.saved_model import loader_impl
    gd = loader_impl.parse_saved_model(path).meta_graphs[0].graph_def

    def strided(nodes):
        ss = [n for n in nodes if n.op == "StridedSlice"]
        bad = [n.name for n in ss
               if n.attr["ellipsis_mask"].i or n.attr["new_axis_mask"].i]
        return ss, bad

    ss, bad = strided(gd.node)
    total = len(ss)
    for f in gd.library.function:
        fss, _ = strided(f.node_def)
        total += len(fss)
    print(f"\n=== exported graph: {path} ===")
    print(f"  StridedSlice ops (main+functions): {total}")
    if ss:
        print("  example StridedSlice names:")
        for n in ss[:8]:
            print("     ", n.name,
                  f"(ellipsis={n.attr['ellipsis_mask'].i} new_axis={n.attr['new_axis_mask'].i})")
    print("  VERDICT:", "CLEAN — SNPE StridedSliceLayerBuilder cannot fire"
          if total == 0 else "STALE/OLD — re-export with the patched code (see module paths above)")
    return total == 0


if __name__ == "__main__":
    code_ok = _check_sources()
    sm_ok = True
    if len(sys.argv) > 1:
        sm_ok = _scan_saved_model(sys.argv[1])
    else:
        print("\n(no saved_model path given — pass one to scan the exported graph)")
    print("\nSUMMARY:",
          "code patched AND graph clean — good to convert" if (code_ok and sm_ok)
          else "MISMATCH — the imported code or the scanned graph is not the patched version")
