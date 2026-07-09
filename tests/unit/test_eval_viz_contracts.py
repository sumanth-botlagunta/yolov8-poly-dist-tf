"""Eval/viz behavior contracts.

  - polygon_metrics._radial_to_cartesian infers vertex count from len(radii) so it
    works for any angle_step; PolygonEvaluator accepts num_vertices.
  - viz_utils._draw_polygon defaults its conf_thresh to the same constant
    PolygonEvaluator scores with (shared DEFAULT_POLY_CONF_THRESH).
"""

import inspect

import numpy as np


def test_radial_to_cartesian_infers_vertex_count():
    from eval.polygon_metrics import _radial_to_cartesian

    for n in (12, 24, 36):
        radii = np.ones(n, dtype=np.float32) * 0.5
        out = _radial_to_cartesian(0.5, 0.5, radii, 672, 672)
        assert out.shape == (n, 2), f"expected ({n}, 2), got {out.shape}"
        out_off = _radial_to_cartesian(
            0.5, 0.5, radii, 672, 672, offsets=np.zeros(n, dtype=np.float32)
        )
        assert out_off.shape == (n, 2)


def test_polygon_evaluator_accepts_num_vertices():
    from eval.polygon_metrics import PolygonEvaluator
    ev = PolygonEvaluator(num_vertices=36)
    assert ev._num_vertices == 36


def test_viz_default_matches_evaluator_default():
    from eval.polygon_metrics import DEFAULT_POLY_CONF_THRESH, PolygonEvaluator
    from common.viz_utils import _draw_polygon, render_summary_images

    assert PolygonEvaluator()._conf_thresh == DEFAULT_POLY_CONF_THRESH
    assert (
        inspect.signature(_draw_polygon).parameters["conf_thresh"].default
        == DEFAULT_POLY_CONF_THRESH
    )
    assert (
        inspect.signature(render_summary_images).parameters["conf_thresh"].default
        == DEFAULT_POLY_CONF_THRESH
    )
