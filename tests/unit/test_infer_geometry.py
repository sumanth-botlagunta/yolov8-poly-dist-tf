"""Geometry tests for tools/infer.py — the letterbox inverse and polygon decode that
map model-space detections back to original-image pixels."""

import numpy as np

from tools import infer


def test_letterbox_inverse_round_trips_nonsquare():
    H, W, size = 480, 640, 672
    _, r, top, left = infer._letterbox(np.zeros((H, W, 3), np.uint8), size)
    for ox, oy in [(0, 0), (200, 100), (639, 479), (320, 240)]:
        xn = (ox * r + left) / size       # original px -> model normalized
        yn = (oy * r + top) / size
        bx, by = infer._inv_point(xn, yn, size, r, top, left)
        assert abs(bx - ox) < 1e-3 and abs(by - oy) < 1e-3


def test_letterbox_inverse_round_trips_portrait():
    H, W, size = 800, 600, 416
    _, r, top, left = infer._letterbox(np.zeros((H, W, 3), np.uint8), size)
    xn = (250 * r + left) / size
    yn = (700 * r + top) / size
    bx, by = infer._inv_point(xn, yn, size, r, top, left)
    assert abs(bx - 250) < 1e-3 and abs(by - 700) < 1e-3


def test_poly_decode_respects_conf_gate():
    poly = np.zeros((24, 3), np.float32)
    for i in range(0, 24, 2):
        poly[i] = [0.9, 0.1, 0.0]          # occupied bins above the gate
    for i in range(1, 24, 2):
        poly[i] = [0.1, 0.1, 0.0]          # below the gate -> skipped
    verts = infer._poly_vertices_norm(poly, 0.5, 0.5, 0.4)
    assert len(verts) == 12


def test_poly_decode_geometry_at_zero_angle():
    # one vertex at bin 0 (angle 0): x = cx + d, y = cy
    poly = np.zeros((24, 3), np.float32)
    poly[0] = [0.9, 0.2, 0.0]
    verts = infer._poly_vertices_norm(poly, 0.5, 0.5, 0.4)
    assert len(verts) == 1
    vx, vy = verts[0]
    assert abs(vx - 0.7) < 1e-6 and abs(vy - 0.5) < 1e-6
