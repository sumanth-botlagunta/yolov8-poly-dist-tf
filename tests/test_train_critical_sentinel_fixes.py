"""Pinning tests for the TRAINING-affecting sentinel/dead-config fixes.

Polygon validity must key off the reserved -1.0 sentinel (`> -1.0`), NOT `>= 0.0`
(docs/design_register.md entry 10). A legitimately-negative canvas coordinate — an
object near an image edge, or a mosaic-overflow vertex that survived clip-to-edge —
is a REAL vertex and must contribute to the PolyYOLO radial target. The old `>= 0.0`
test silently DROPPED such vertices, zeroing the conf bin they belong to and biasing
the polygon GT for edge objects. These tests fail loudly if the regression returns.

Covered:
  - _preprocess_polygons_v2 (yolo_parser): a due-west vertex at negative x for an
    edge box produces a non-zero conf/dist bin.
  - mosaic._scale_box_poly_to_canvas: a negative-x polygon vertex is carried into the
    canvas (not overwritten with the -1.0 sentinel).
  - aug_rand_angle / aug_rand_perspective removed from ParserConfig (dead config) and
    a stray key in old YAML still loads.
"""

import math
import os

import numpy as np
import tensorflow as tf

from data_pipeline.yolo_parser import V8ParserExtended
from data_pipeline.mosaic import _scale_box_poly_to_canvas
from data_pipeline.augmentations import clip_polygon_coords
from configs.model_config import ParserConfig
from configs.yaml_loader import load_config_from_dict


def _make_parser() -> V8ParserExtended:
    return V8ParserExtended(
        output_size=[64, 64],
        expanded_strides={"3": 8, "4": 16, "5": 32},
        levels=["3", "4", "5"],
        angle_step=15,
    )


def test_preprocess_keeps_negative_x_vertex_for_edge_box():
    """An edge box (center x=0.0) with a due-west vertex at x=-0.02 must occupy the
    180-degree bin. Old `>= 0.0` dropped the vertex -> that bin would be empty."""
    parser = _make_parser()
    # Box yxyx: ymin=0.0, xmin=-0.04, ymax=0.2, xmax=0.04 -> center (cy,cx)=(0.1, 0.0)
    box = tf.constant([[0.0, -0.04, 0.2, 0.04]], tf.float32)
    # Vertices: due-west at (-0.02, 0.1) [bin 12 = 180 deg], plus a -1.0 sentinel pair.
    poly = tf.constant([[-0.02, 0.1, -1.0, -1.0]], tf.float32)

    out = parser._preprocess_polygons_v2(box, poly, angle_step=15).numpy()
    conf = out[0, 2::3]
    dist = out[0, 0::3]

    # atan2(dy=0, dx=-0.02) = pi = 180 deg -> bin 180/15 = 12.
    assert conf[12] == 1.0, f"negative-x west vertex dropped (conf bin 12 = {conf[12]})"
    assert dist[12] > 0.0, f"west vertex radial dist not recorded: {dist[12]}"
    # Exactly one occupied bin (the single valid vertex).
    assert int(conf.sum()) == 1, f"unexpected occupied bins: {np.flatnonzero(conf)}"


def test_preprocess_drops_only_exact_minus_one_sentinel():
    """Control: x exactly == -1.0 is the sentinel and is the ONLY value treated as
    padding. A box at center with one real east vertex + a -1.0 pair => one bin."""
    parser = _make_parser()
    box = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)  # center (0.5, 0.5)
    poly = tf.constant([[1.0, 0.5, -1.0, -1.0]], tf.float32)  # east vertex + sentinel
    out = parser._preprocess_polygons_v2(box, poly, angle_step=15).numpy()
    conf = out[0, 2::3]
    assert conf[0] == 1.0 and int(conf.sum()) == 1, f"sentinel handling wrong: {conf}"


def test_mosaic_canvas_carries_negative_vertex_not_sentinel():
    """_scale_box_poly_to_canvas must carry a negative-x vertex into the canvas (a real
    vertex), not overwrite it with the -1.0 sentinel. Old `>= 0.0` zeroed it to -1.0."""
    H = W = 100
    ex = {
        'groundtruth_boxes': tf.constant([[0.3, 0.3, 0.7, 0.7]], tf.float32),
        # v0: x=-0.02 legit-negative (edge object); v1: 0.5 interior; v2: -1.0 sentinel.
        'groundtruth_polygons': tf.constant([[-0.02, 0.5, 0.5, 0.5, -1.0, -1.0]], tf.float32),
    }
    nh = tf.constant(H); nw = tf.constant(W)
    padh = tf.constant(0); padw = tf.constant(0)
    H2 = tf.constant(2 * H); W2 = tf.constant(2 * W)

    _, polys_c = _scale_box_poly_to_canvas(ex, nh, nw, padh, padw, H2, W2)
    pts = polys_c.numpy().reshape(-1, 2)

    # v0 was carried (scaled), NOT collapsed to the -1.0 sentinel.
    # canvas_x = (x*nw + padw)/W2 = (-0.02*100 + 0)/200 = -0.01
    assert abs(pts[0, 0] - (-0.01)) < 1e-6, f"negative vertex not carried: {pts[0]}"
    assert pts[0, 0] != -1.0, "negative vertex wrongly overwritten with sentinel"
    # v1 interior carried: (0.5*100)/200 = 0.25
    assert abs(pts[1, 0] - 0.25) < 1e-6, f"interior vertex wrong: {pts[1]}"
    # v2 sentinel stays -1.0.
    assert pts[2, 0] == -1.0 and pts[2, 1] == -1.0, f"sentinel changed: {pts[2]}"


def test_clip_polygon_coords_clips_negative_mosaic_overflow_vertex():
    """clip_polygon_coords keys validity off the -1.0 sentinel (`> -1.0`), NOT `>= 0.0`.

    A real mosaic-overflow vertex at a slightly-negative coordinate (e.g. -0.05, which
    is > -1.0) must be clipped into [0, 1], landing on the canvas edge (0.0). The old
    `>= 0.0` check treated it as padding and left it at -0.05, where downstream stages
    then misread it as a -1.0-style sentinel. The exact -1.0 sentinel must be preserved.
    """
    # cols: [x=-0.05 overflow, y=-0.10 overflow, x=0.30 interior, y=1.05 overflow,
    #        x=-1.0 sentinel, y=-1.0 sentinel]
    polygons = tf.constant([[-0.05, -0.10, 0.30, 1.05, -1.0, -1.0]], tf.float32)
    out = clip_polygon_coords(polygons).numpy()[0]

    assert out[0] == 0.0, f"negative-x overflow vertex not clipped to 0.0: {out[0]}"
    assert out[1] == 0.0, f"negative-y overflow vertex not clipped to 0.0: {out[1]}"
    assert abs(out[2] - 0.30) < 1e-6, f"interior vertex altered: {out[2]}"
    assert out[3] == 1.0, f"over-1 overflow vertex not clipped to 1.0: {out[3]}"
    # Padding sentinel must be untouched (-1.0 is NOT > -1.0).
    assert out[4] == -1.0 and out[5] == -1.0, f"sentinel corrupted: {out[4]}, {out[5]}"


def test_dead_aug_fields_removed_from_parser_config():
    pc = ParserConfig()
    assert not hasattr(pc, 'aug_rand_angle'), "dead aug_rand_angle still on ParserConfig"
    assert not hasattr(pc, 'aug_rand_perspective'), "dead aug_rand_perspective still present"


def test_old_yaml_with_dead_aug_fields_still_loads():
    """A legacy config dict carrying aug_rand_angle / aug_rand_perspective must load
    without error and silently ignore them (no leak onto the dataclass)."""
    raw = {
        'task': {
            'train_data': {
                'parser': {
                    'angle_step': 15,
                    'aug_rand_angle': 0.0,        # legacy dead key
                    'aug_rand_perspective': 0.0,  # legacy dead key
                },
            },
        },
    }
    cfg = load_config_from_dict(raw)
    p = cfg.task.train_data.parser
    assert not hasattr(p, 'aug_rand_angle')
    assert not hasattr(p, 'aug_rand_perspective')
