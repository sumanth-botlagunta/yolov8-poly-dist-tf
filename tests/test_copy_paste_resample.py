"""Pinning test: copy-paste fits wide object polygons by EVEN RESAMPLE, not truncation.

The copy-paste source decoder does not resample, so a pasted object can carry many
more polygon vertices than the (resampled) background's polygon width. The fit step
used to take the first `n_poly_cols` raw vertices — a contiguous arc that throws
away the far side of the contour and corrupts the PolyYOLO radial target. It now
evenly resamples the valid vertices to the column budget.
"""

import numpy as np
import tensorflow as tf

from data_pipeline.copy_paste import CopyAndPasteModule


def _bg(n_poly_cols):
    return {
        "image": tf.zeros([100, 100, 3], tf.uint8),
        "height": tf.constant(100, tf.int32),
        "width": tf.constant(100, tf.int32),
        "groundtruth_boxes": tf.zeros([0, 4], tf.float32),
        "groundtruth_classes": tf.zeros([0], tf.int64),
        # Background polygons are narrow (e.g. resampled to 64 verts → 128 cols).
        "groundtruth_polygons": tf.fill([0, n_poly_cols], -1.0),
        "groundtruth_is_crowd": tf.zeros([0], tf.bool),
        "groundtruth_area": tf.zeros([0], tf.float32),
        "groundtruth_dontcare": tf.zeros([0], tf.int64),
    }


def _obj_with_wide_polygon(n_pairs):
    # A polygon that traces a full loop: x sweeps 0.1 -> 0.9 -> 0.1, y likewise,
    # so the FAR side of the contour lives in the SECOND half of the vertex list.
    t = np.linspace(0.0, 2 * np.pi, n_pairs, endpoint=False)
    xs = 0.5 + 0.4 * np.cos(t)
    ys = 0.5 + 0.4 * np.sin(t)
    pts = np.stack([xs, ys], axis=-1).reshape(-1).astype(np.float32)
    return {
        "image": tf.fill([40, 40, 4], tf.constant(200, tf.uint8)),
        "orig_bbox": tf.constant([0.1, 0.1, 0.9, 0.9], tf.float32),
        "label": tf.constant(3, tf.int64),
        "points": tf.constant(pts),
    }


def test_wide_object_polygon_is_resampled_not_truncated():
    n_poly_cols = 128                  # background width = 64 verts
    n_pairs = 2000                     # object far wider than the budget
    cnp = CopyAndPasteModule(prob=1.0)
    tf.random.set_seed(0)
    out = cnp._copy_and_paste(_bg(n_poly_cols), _obj_with_wide_polygon(n_pairs))

    polys = out["groundtruth_polygons"].numpy()
    assert polys.shape == (1, n_poly_cols), polys.shape
    pts = polys.reshape(-1, 2)
    valid = pts[pts[:, 0] >= 0.0]
    assert len(valid) > 0

    # The even resample must span the WHOLE contour loop. The first-N-truncation
    # bug keeps only a leading arc (~64/2000 ≈ 3% of the loop): its x- and y-spread
    # would be ~0.002 in bg coords. An even resample spans the full circle, giving
    # a spread an order of magnitude larger on BOTH axes.
    x_spread = float(valid[:, 0].max() - valid[:, 0].min())
    y_spread = float(valid[:, 1].max() - valid[:, 1].min())
    assert x_spread > 0.05 and y_spread > 0.05, (
        f"kept vertices span only x={x_spread:.4f} y={y_spread:.4f} — looks like "
        "a leading-arc truncation, not a full-loop resample"
    )


def test_narrow_object_polygon_is_padded():
    # Object narrower than the bg budget → padded with -1 (unchanged behavior).
    n_poly_cols = 128
    n_pairs = 8
    cnp = CopyAndPasteModule(prob=1.0)
    out = cnp._copy_and_paste(_bg(n_poly_cols), _obj_with_wide_polygon(n_pairs))
    polys = out["groundtruth_polygons"].numpy()
    assert polys.shape == (1, n_poly_cols)
    # Trailing columns must be the -1 pad sentinel.
    assert np.isclose(polys[0, -1], -1.0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
