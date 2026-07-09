"""Pinning test: resample_polygons `compact` gates the scattered-sentinel argsort.

resample_polygons can run a stable argsort+gather to compact scattered sentinels to
a prefix. The copy-paste path invalidates out-of-bounds vertices in place and so
hands interleaved -1 holes; it relies on compact=True to fix them. A contiguous-prefix
input needs no sort, so the flag is off by default.

The `compact` flag (default False) gates the sort. This test pins:
  1. On a contiguous-prefix input, compact=False == compact=True (sort is a no-op).
  2. On a SCATTERED-sentinel input, compact=True compacts correctly (no interior
     holes) while compact=False does NOT — so the flag is load-bearing and the
     copy-paste caller MUST pass compact=True.
  3. Wiring: the copy-paste caller passes compact=True.
"""

import inspect

import numpy as np
import tensorflow as tf

from data_pipeline.augmentations import resample_polygons


def _make_prefix(N, P, n_valid, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.full((N, P, 2), -1.0, dtype=np.float32)
    arr[:, :n_valid, :] = rng.uniform(0.0, 1.0, size=(N, n_valid, 2)).astype(np.float32)
    return tf.constant(arr.reshape(N, P * 2), dtype=tf.float32)


def _make_scattered(N, P, n_valid, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.full((N, P, 2), -1.0, dtype=np.float32)
    for i in range(N):
        idxs = rng.choice(P, size=n_valid, replace=False)
        arr[i, idxs, :] = rng.uniform(0.0, 1.0, size=(n_valid, 2)).astype(np.float32)
    return tf.constant(arr.reshape(N, P * 2), dtype=tf.float32)


def test_default_is_compact_false():
    """The default must be compact=False so the decode hot path skips the sort."""
    sig = inspect.signature(resample_polygons)
    assert sig.parameters["compact"].default is False


def test_prefix_input_identical_with_and_without_compact():
    """On a contiguous-prefix input the argsort is a no-op: outputs are identical.

    This is the decode-time invariant — skipping the sort (compact=False) changes
    nothing for prefix-padded inputs (the TFDS contract), it only avoids the cost.
    """
    for N, P, n_valid, K in [(20, 5470, 64, 64), (5, 100, 30, 24), (10, 200, 1, 24)]:
        flat = _make_prefix(N, P, n_valid)
        out_false = resample_polygons(flat, K, compact=False).numpy()
        out_true = resample_polygons(flat, K, compact=True).numpy()
        np.testing.assert_array_equal(
            out_false, out_true,
            err_msg=f"compact changed prefix output (N={N},P={P},n_valid={n_valid})",
        )


def test_scattered_input_needs_compact():
    """On scattered sentinels, compact=True compacts correctly; compact=False does
    not. The flag is load-bearing — the copy-paste caller relies on compact=True.
    """
    N, P, n_valid, K = 20, 5470, 64, 64
    flat = _make_scattered(N, P, n_valid)

    out_true = resample_polygons(flat, K, compact=True).numpy().reshape(N, K, 2)
    out_false = resample_polygons(flat, K, compact=False).numpy().reshape(N, K, 2)

    def interior_holes(out):
        total = 0
        for i in range(N):
            xs = out[i, :, 0]
            v = np.where(xs > -1.0)[0]
            if len(v) == 0:
                continue
            total += int((xs[: v[-1] + 1] <= -1.0).sum())
        return total

    # compact=True: no -1 hole appears before the last valid vertex of any row.
    assert interior_holes(out_true) == 0, (
        "compact=True left interleaved sentinels — compaction failed"
    )
    # compact=False on scattered input is the original corruption (interior holes).
    assert interior_holes(out_false) > 0, (
        "expected compact=False to corrupt scattered input (interior holes); if it "
        "did not, the scattered fixture is not actually scattered"
    )


def test_copy_paste_wires_compact_true():
    """Copy-paste passes compact=True (it can produce scattered sentinels)."""
    import data_pipeline.copy_paste as cp

    cp_src = inspect.getsource(cp)

    # Copy-paste must request compaction (it can produce scattered sentinels).
    assert "compact=True" in cp_src, (
        "copy-paste caller must pass compact=True to handle scattered sentinels"
    )
