"""Unit tests for utils/export/device_decode.reconstruct_detections.

Pure-numpy (no TensorFlow): builds a tiny synthetic device-contract output dict with
a single hand-placed anchor and asserts the reconstruction yields the geometrically
correct yxyx box, class, and score — plus the legacy_box_order (y-first) reorder.
"""

import math

import numpy as np

from utils.export.device_decode import (
    reconstruct_detections, is_device_contract, is_legacy_contract)

# Tiny model: 32x32 -> levels 4x4 (s8), 2x2 (s16), 1x1 (s32) => N = 21 anchors.
MH = MW = 32
N = sum((MH // s) * (MW // s) for s in (8, 16, 32))   # 21
NUM_CLASSES = 4


def _base_outputs():
    """All anchors suppressed (very negative logits) except none set yet."""
    cls = np.full((N, NUM_CLASSES), -20.0, dtype=np.float32)   # sigmoid ~ 2e-9 < 0.05
    box = np.zeros((N, 4), dtype=np.float32)
    return box, cls


def test_contract_detectors():
    assert is_device_contract(["box", "cls", "poly_conf"]) is True
    assert is_device_contract(["bbox", "num_detections"]) is False
    assert is_legacy_contract(["bbox", "num_detections"]) is True
    assert is_legacy_contract(["box", "cls"]) is False


def test_single_anchor_box_xfirst():
    """One live anchor at grid (0,0) of stride 8 decodes to the exact yxyx box."""
    box, cls = _base_outputs()
    # Anchor 0 (level s8, i=j=0): centre (cx,cy) = (4, 4) pixels.
    # xfirst box = [l, t, r, b] in feature units; *8 -> pixels.
    box[0] = [1.0, 0.5, 1.5, 0.25]     # l=8 t=4 r=12 b=2 px
    cls[0, 2] = 10.0                   # class 2, sigmoid(10) ~ 0.99995

    out = reconstruct_detections({"box": box, "cls": cls}, MH, MW,
                                 legacy_box_order=False)

    assert int(out["num_detections"][0]) == 1
    assert int(out["classes"][0, 0]) == 2
    assert out["confidence"][0, 0] > 0.99
    # xyxy: x1=cx-l=-4->clip0, y1=cy-t=0, x2=cx+r=16, y2=cy+b=6; normalized /32.
    exp = np.array([0.0, 0.0, 6.0 / 32.0, 16.0 / 32.0], dtype=np.float32)  # yxyx
    np.testing.assert_allclose(out["bbox"][0, 0], exp, atol=1e-6)
    # 'polygons'/'distance' absent when those heads are not provided.
    assert "polygons" not in out and "distance" not in out


def test_legacy_box_order_matches_xfirst():
    """A y-first ([t,l,b,r]) export decodes identically once reordered."""
    box_x, cls = _base_outputs()
    box_x[0] = [1.0, 0.5, 1.5, 0.25]   # [l,t,r,b]
    cls[0, 2] = 10.0
    out_x = reconstruct_detections({"box": box_x.copy(), "cls": cls.copy()},
                                   MH, MW, legacy_box_order=False)

    # Same content stored y-first: [t,l,b,r] = permute [1,0,3,2] of [l,t,r,b].
    box_y = box_x.copy()
    box_y[0] = box_x[0][[1, 0, 3, 2]]
    out_y = reconstruct_detections({"box": box_y, "cls": cls.copy()},
                                   MH, MW, legacy_box_order=True)

    np.testing.assert_allclose(out_y["bbox"], out_x["bbox"], atol=1e-6)
    assert int(out_y["num_detections"][0]) == int(out_x["num_detections"][0]) == 1


def test_polygon_and_distance_heads_decode():
    """When poly/dist heads are present they are activated and shaped [1,max,P,3]/[1,max]."""
    box, cls = _base_outputs()
    box[0] = [0.5, 0.5, 0.5, 0.5]
    cls[0, 1] = 8.0
    P = 24
    pa = np.zeros((N, P), np.float32)
    pd = np.zeros((N, P), np.float32)
    pc = np.full((N, P), -20.0, np.float32)   # conf ~ 0
    pc[0, :3] = 10.0                           # first 3 bins valid for the live anchor
    di = np.zeros((N, 1), np.float32)          # log-dist 0 -> exp(0) clipped to [0.5,10]

    out = reconstruct_detections(
        {"box": box, "cls": cls, "poly_angle": pa, "poly_dist": pd,
         "poly_conf": pc, "dist": di}, MH, MW, legacy_box_order=False)

    assert out["polygons"].shape == (1, 300, P, 3)
    assert out["distance"].shape == (1, 300)
    # Channel 0 of polygons is sigmoid(conf): high on the 3 set bins, ~0 elsewhere.
    conf0 = out["polygons"][0, 0, :, 0]
    assert (conf0[:3] > 0.99).all() and (conf0[3:] < 0.01).all()
    # softplus(0) = ln 2 for the raw dist channel, then scaled by the assigned
    # anchor's stride/img_size (anchor 0 is the first stride-8 anchor,
    # model_h=MH=32 -> scale = 8/32 = 0.25) to convert grid units back to a
    # normalized-image radius.
    np.testing.assert_allclose(out["polygons"][0, 0, 0, 1], math.log(2.0) * 8 / MH, atol=1e-5)
    # exp(0)=1 is inside [0.5,10] so distance stays 1.0.
    np.testing.assert_allclose(out["distance"][0, 0], 1.0, atol=1e-5)


def test_no_detections_all_suppressed():
    """All logits below the score threshold -> zero detections, full padding."""
    box, cls = _base_outputs()      # every cls logit -20 -> sigmoid < score_thresh
    out = reconstruct_detections({"box": box, "cls": cls}, MH, MW)
    assert int(out["num_detections"][0]) == 0
    assert out["bbox"].shape == (1, 300, 4)
    assert not out["bbox"].any()
