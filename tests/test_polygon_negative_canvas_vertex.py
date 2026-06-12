"""Pinning test: transform_boxes_polygons treats a legitimately-negative canvas
polygon vertex (mosaic overflow) as a REAL vertex — transform + clip-to-edge — not as
the reserved -1.0 sentinel.

Validity keys off the reserved -1.0 polygon sentinel (`pts[:, :, 0] > -1.0`), NOT
`>= 0.0`. A vertex at a slightly-negative input-normalized coordinate that lands in-view
after clipping is a real vertex and must survive (clipped to the [0,1] edge), keeping the
polygon GT consistent with the box GT (boxes are clipped, not dropped, for the same
overflow case). See docs/design_register.md entry 10.
"""

import numpy as np
import tensorflow as tf

from data_pipeline.augmentations import transform_boxes_polygons


def _identity_M():
    return tf.constant(np.eye(3, dtype=np.float32))  # input px -> output px, same size


def test_negative_canvas_vertex_is_clipped_not_dropped():
    h_in = w_in = 100
    target_h = target_w = 100

    # One object, 3 polygon vertices (6 flat values):
    #   v0: (-0.05, 0.5) -> negative x, in-view after clip-to-edge (overflow case)
    #   v1: (0.5, 0.5)   -> normal interior vertex
    #   v2: (-1.0, -1.0) -> the reserved -1.0 sentinel (true padding)
    polygons = tf.constant([[-0.05, 0.5, 0.5, 0.5, -1.0, -1.0]], dtype=tf.float32)
    boxes = tf.constant([[0.4, 0.4, 0.6, 0.6]], dtype=tf.float32)  # ymin,xmin,ymax,xmax

    _, _, polys_out = transform_boxes_polygons(
        boxes, polygons, _identity_M(), h_in, w_in, target_h, target_w
    )
    out = polys_out.numpy().reshape(-1, 2)

    # v0 survived (not the sentinel) and clipped to the edge x=0.0.
    assert out[0, 0] != -1.0, f"negative-canvas vertex dropped as sentinel: {out[0]}"
    assert out[0, 0] == 0.0, f"negative-canvas x should clip to edge 0.0: {out[0, 0]}"
    assert abs(out[0, 1] - 0.5) < 1e-5, f"y should be preserved at 0.5: {out[0, 1]}"

    # v1 preserved, v2 stays the sentinel.
    assert abs(out[1, 0] - 0.5) < 1e-5 and abs(out[1, 1] - 0.5) < 1e-5
    assert out[2, 0] == -1.0 and out[2, 1] == -1.0, f"true sentinel changed: {out[2]}"
