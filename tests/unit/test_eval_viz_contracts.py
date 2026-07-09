"""Pinning tests for non-training behavior contracts (eval/viz/docs consistency).

  - polygon_metrics._radial_to_cartesian infers vertex count from len(radii)
    instead of hardcoding 24 (works for configurable angle_step).
  - viz_utils._draw_polygon / render_summary_images take a conf_thresh that
    defaults to the same value PolygonEvaluator scores with (shared constant).
  - README documents checkpoint_interval by formula (train_total_examples //
    global_batch_size), not a baked step count.
  - export_saved_model docstring documents the output schema.
"""

import inspect
import os

import numpy as np
import pytest


REPO = os.path.join(os.path.dirname(__file__), "..", "..")


# ---------------------------------------------------------------------------
# polygon_metrics configurable vertex count
# ---------------------------------------------------------------------------

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


def test_radial_to_cartesian_no_angle_step_constant_import():
    """_ANGLE_STEP module constant should be gone (angle step is now derived)."""
    import eval.polygon_metrics as pm
    assert not hasattr(pm, "_ANGLE_STEP"), "_ANGLE_STEP should be removed"


def test_polygon_evaluator_accepts_num_vertices():
    from eval.polygon_metrics import PolygonEvaluator
    ev = PolygonEvaluator(num_vertices=36)
    assert ev._num_vertices == 36


# ---------------------------------------------------------------------------
# viz_utils conf_thresh threading
# ---------------------------------------------------------------------------

def test_draw_polygon_has_conf_thresh_param():
    from common.viz_utils import _draw_polygon
    sig = inspect.signature(_draw_polygon)
    assert "conf_thresh" in sig.parameters


def test_viz_default_matches_evaluator_default():
    from eval.polygon_metrics import DEFAULT_POLY_CONF_THRESH, PolygonEvaluator
    from common.viz_utils import _draw_polygon, render_summary_images

    assert PolygonEvaluator().__init__.__defaults__  # has defaults
    assert PolygonEvaluator()._conf_thresh == DEFAULT_POLY_CONF_THRESH
    assert (
        inspect.signature(_draw_polygon).parameters["conf_thresh"].default
        == DEFAULT_POLY_CONF_THRESH
    )
    assert (
        inspect.signature(render_summary_images).parameters["conf_thresh"].default
        == DEFAULT_POLY_CONF_THRESH
    )


def test_no_hardcoded_0_4_in_draw_polygon_body():
    from common.viz_utils import _draw_polygon
    src = inspect.getsource(_draw_polygon)
    assert "< 0.4" not in src, "conf gate must use the conf_thresh param, not 0.4"
    assert "conf < conf_thresh" in src


# ---------------------------------------------------------------------------
# README checkpoint interval
# ---------------------------------------------------------------------------

def test_readme_checkpoint_interval_is_formula_not_baked():
    """README describes checkpoint_interval by its formula (one epoch =
    train_total_examples // global_batch_size), not a baked step count."""
    with open(os.path.join(REPO, "README.md")) as f:
        readme = f.read()
    assert "2388" not in readme, "stale checkpoint_interval default 2388 still present"
    assert "2118" not in readme, "checkpoint_interval should be a formula, not the baked 2118"
    assert "train_total_examples // global_batch_size" in readme


# ---------------------------------------------------------------------------
# export schema docs
# ---------------------------------------------------------------------------

def test_export_docstring_documents_output_schema():
    import utils.export.export_saved_model as ex
    doc = ex.__doc__ or ""
    assert "Output Schema" in doc
    for key in ("bbox", "classes", "confidence", "num_detections", "polygons", "distance"):
        assert key in doc, f"output schema missing '{key}'"
