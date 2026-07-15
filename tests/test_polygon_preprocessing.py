"""Tests for polygon format conversion (_preprocess_polygons_v2).

The PolyYOLO target is the interleaved radial format
``[dist, angle, conf] × (360/angle_step)``:

    - PolyYOLO output shape is [N, 360/angle_step * 3].
    - dist is the radial distance from the box center to the bin's vertex.
    - angle is the sub-bin offset (vertex_angle - bin_start)/angle_step in [0, 1)
      on occupied bins, 0.0 on absent bins.
    - conf is 1.0 for bins that received a valid vertex and 0.0 for absent bins;
      absent bins also carry dist == 0.0 (so the dist head learns to collapse them).
"""

import math
import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.yolo_parser import V8ParserExtended


def _make_parser() -> V8ParserExtended:
    return V8ParserExtended(
        output_size=[64, 64],
        expanded_strides={"3": 8, "4": 16, "5": 32},
        levels=["3", "4", "5"],
        angle_step=15,
    )


# One box covering the whole image → center at (0.5, 0.5).
# Radial vector convention is origin − vertex (dx = cx − x, dy = cy − y), so a
# vertex due-east of center (x=1.0, y=0.5) points the radial vector due-WEST
# (180°, bin 12), and a vertex due-south (x=0.5, y=1.0) points the radial
# vector due-NORTH (270°, bin 18) — the vertex's own bin is the OPPOSITE
# compass direction from the vertex itself (bin = old_bin + 12, mod 24).
_BOX = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)        # yxyx normalized
_POLY = tf.constant([[1.0, 0.5, 0.5, 1.0, -1.0, -1.0, -1.0, -1.0]], dtype=tf.float32)


class TestPreprocessPolygonsV2(unittest.TestCase):
    def setUp(self):
        self.parser = _make_parser()
        out = self.parser._preprocess_polygons_v2(_BOX, _POLY, angle_step=15)
        self.out = out.numpy()
        self.dist = self.out[0, 0::3]   # [24]
        self.angle = self.out[0, 1::3]  # [24] sub-bin offset in [0,1)
        self.conf = self.out[0, 2::3]   # [24]

    def test_output_shape(self):
        """24 bins × 3 channels = 72 values per instance."""
        self.assertEqual(self.out.shape, (1, 72))

    def test_conf_binary(self):
        """conf is strictly 0/1; exactly the two occupied bins (12 and 18) are 1."""
        self.assertTrue(set(np.unique(self.conf)).issubset({0.0, 1.0}))
        self.assertEqual(self.conf[12], 1.0)
        self.assertEqual(self.conf[18], 1.0)
        self.assertEqual(int(self.conf.sum()), 2)

    def test_radial_distance_and_absent_bins(self):
        """Occupied bins carry radius 0.5; absent bins carry dist 0 (and conf 0)."""
        self.assertAlmostEqual(self.dist[12], 0.5, places=5)
        self.assertAlmostEqual(self.dist[18], 0.5, places=5)
        absent = np.ones(24, dtype=bool)
        absent[[12, 18]] = False
        np.testing.assert_array_equal(self.dist[absent], 0.0)
        np.testing.assert_array_equal(self.conf[absent], 0.0)
        # The two vertices sit exactly on bin starts (180°, 270° in the
        # flipped origin-minus-vertex convention), so their sub-bin offset is
        # 0; every angle entry is therefore 0 here.
        self.assertAlmostEqual(self.angle[12], 0.0, places=5)
        self.assertAlmostEqual(self.angle[18], 0.0, places=5)
        np.testing.assert_array_equal(self.angle[absent], 0.0)


class TestSubBinAngleOffset(unittest.TestCase):
    """A vertex not on a bin boundary must produce its fractional offset."""

    def test_offset_is_fractional_position_in_bin(self):
        parser = _make_parser()
        box = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)  # centre (0.5,0.5)
        # Vertex placed so that the RADIAL vector (origin − vertex) sits at
        # 7.5° (the middle of bin 0) at radius 0.4: dx = cx - x = r*cos(7.5°),
        # dy = cy - y = r*sin(7.5°) → x = cx - r*cos, y = cy - r*sin.
        r, ang = 0.4, math.radians(7.5)
        x = 0.5 - r * math.cos(ang)
        y = 0.5 - r * math.sin(ang)
        poly = tf.constant([[x, y, -1.0, -1.0]], dtype=tf.float32)
        out = parser._preprocess_polygons_v2(box, poly, angle_step=15).numpy()
        dist, angle, conf = out[0, 0::3], out[0, 1::3], out[0, 2::3]
        self.assertEqual(int(conf.sum()), 1)
        self.assertEqual(conf[0], 1.0)                 # bin 0
        self.assertAlmostEqual(dist[0], r, places=4)   # radial distance preserved
        self.assertAlmostEqual(angle[0], 0.5, places=4)  # 7.5° / 15° = 0.5


def _point_seg_dist(p, a, b):
    """Min distance from point p to segment a→b (all length-2 arrays)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab)) / denom
    t = min(1.0, max(0.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _min_dist_to_contour(p, verts):
    """Min distance from p to the CLOSED contour through `verts` (list of xy)."""
    n = len(verts)
    return min(
        _point_seg_dist(p, verts[i], verts[(i + 1) % n]) for i in range(n)
    )


class TestResamplePolygons(unittest.TestCase):
    """resample_polygons does UNIFORM ARC-LENGTH resampling along the closed contour.

    Uses arc-length resampling, not index-subsampling. An earlier index-subsampling
    code only SELECTED existing vertices, so a sparse polygon (e.g. a 4-corner
    rectangle) kept its few points and the 24-bin radial target collapsed to a
    diamond. Arc-length resampling interpolates points ALONG edges, so the radial
    target now tracks the true boundary. This intentionally CHANGES the radial
    target for sparse-vertex polygons (the fix) while staying within sampling
    tolerance of the old behavior for dense uniform contours.
    """

    def _circle_flat(self, cx, cy, r, n, pad_to):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1).reshape(-1).astype(np.float32)
        return np.concatenate([pts, -np.ones(pad_to - pts.size, np.float32)])

    # (a) Rectangle: arc resampling must POPULATE the bins the long edges cross. ----
    def test_rectangle_arc_resample_fills_bins(self):
        from data_pipeline.augmentations import resample_polygons
        parser = _make_parser()
        # Axis-aligned square, corners only, centered at (0.5, 0.5), half-extent 0.3.
        h = 0.3
        verts = [(0.5 - h, 0.5 - h), (0.5 + h, 0.5 - h),
                 (0.5 + h, 0.5 + h), (0.5 - h, 0.5 + h)]
        polys = tf.constant(_flat_poly(verts, 200))
        boxes = tf.constant([[0.2, 0.2, 0.8, 0.8]], tf.float32)  # yxyx, centre (0.5,0.5)

        # OLD index-subsampling kept only the 4 corners → <= 4 occupied bins.
        raw_out = parser._preprocess_polygons_v2(boxes, polys, 15).numpy()
        self.assertLessEqual(int(raw_out[0, 2::3].sum()), 4)

        red = resample_polygons(polys, 64)
        self.assertEqual(int(red.shape[1]), 128)   # 2 * 64
        out = parser._preprocess_polygons_v2(boxes, red, 15).numpy()
        dist, angle, conf = out[0, 0::3], out[0, 1::3], out[0, 2::3]

        # (a) occupied-bin count must be >= 16 of 24 (was <= 4).
        self.assertGreaterEqual(int(conf.sum()), 16)

        # per-bin distance within 3% of the analytic square-boundary distance.
        for i in range(24):
            if conf[i] > 0.0:
                # exact vertex angle = (bin + sub-bin offset) * angle_step.
                vang = math.radians((i + angle[i]) * 15.0)
                analytic = h / max(abs(math.cos(vang)), abs(math.sin(vang)))
                self.assertLessEqual(
                    abs(dist[i] - analytic) / analytic, 0.03,
                    f"bin {i}: dist {dist[i]:.4f} vs analytic {analytic:.4f}")

    # (b) Dense circle: arc resampling ≈ index subsampling for dense uniform contours.
    def test_dense_circle_radial_target_close(self):
        from data_pipeline.augmentations import resample_polygons
        parser = _make_parser()
        # 200-vertex dense circle. The pre-existing dense expectation: the radial
        # target equals the analytic circle radius (0.3) on every occupied bin, and
        # resampling preserves that to sampling tolerance.
        polys = tf.constant([self._circle_flat(0.5, 0.5, 0.3, 200, 5469 * 2)])
        boxes = tf.constant([[0.2, 0.2, 0.8, 0.8]], tf.float32)
        orig = parser._preprocess_polygons_v2(boxes, polys, 15).numpy()
        out = parser._preprocess_polygons_v2(
            boxes, resample_polygons(polys, 128), 15).numpy()
        # All 24 bins occupied for a dense circle, both before and after.
        self.assertEqual(int(orig[0, 2::3].sum()), 24)
        self.assertEqual(int(out[0, 2::3].sum()), 24)
        # Per-bin distances stay within sampling tolerance of the dense original.
        np.testing.assert_allclose(out[:, 0::3], orig[:, 0::3], atol=1e-3)

    def test_long_polygon_radial_target_close(self):
        from data_pipeline.augmentations import resample_polygons
        parser = _make_parser()
        polys = tf.constant([self._circle_flat(0.5, 0.5, 0.3, 2000, 5469 * 2)])
        boxes = tf.constant([[0.2, 0.2, 0.8, 0.8]], tf.float32)
        orig = parser._preprocess_polygons_v2(boxes, polys, 15).numpy()
        out = parser._preprocess_polygons_v2(boxes, resample_polygons(polys, 128), 15).numpy()
        # downsampled but per-bin distances stay within sampling tolerance.
        np.testing.assert_allclose(out[:, 0::3], orig[:, 0::3], atol=1e-3)

    def test_empty_polygon_is_all_minus_one(self):
        from data_pipeline.augmentations import resample_polygons
        polys = tf.fill([1, 200], -1.0)
        red = resample_polygons(polys, 32).numpy()
        self.assertEqual(red.shape, (1, 64))
        np.testing.assert_array_equal(red, -1.0)

    # (d) Degenerate vertex counts and zero-length segments: no NaN, sentinel safe. --
    def test_zero_one_two_vertex_rows(self):
        from data_pipeline.augmentations import resample_polygons
        K = 8
        # Row 0: c=0 (all sentinel). Row 1: c=1. Row 2: c=2. Row 3: collinear dups.
        row0 = _flat_poly([], 200)[0]
        row1 = _flat_poly([(0.5, 0.5)], 200)[0]
        row2 = _flat_poly([(0.3, 0.3), (0.7, 0.7)], 200)[0]
        row3 = _flat_poly([(0.2, 0.2), (0.2, 0.2), (0.8, 0.2), (0.8, 0.8)], 200)[0]
        polys = tf.constant(np.stack([row0, row1, row2, row3], axis=0))
        out = resample_polygons(polys, K).numpy().reshape(4, K, 2)

        self.assertFalse(np.isnan(out).any())            # no NaN anywhere
        # c=0 → whole row is the -1 sentinel.
        np.testing.assert_array_equal(out[0], -1.0)
        # c=1 → the single point repeated K times; no sentinel introduced.
        self.assertTrue((out[1][:, 0] > -1.0).all())
        np.testing.assert_allclose(out[1], np.tile([0.5, 0.5], (K, 1)), atol=1e-6)
        # c=2 → degenerate two-segment loop; points stay on the line y=x, no NaN.
        self.assertTrue((out[2][:, 0] > -1.0).all())
        np.testing.assert_allclose(out[2][:, 0], out[2][:, 1], atol=1e-5)
        # collinear duplicate vertices (zero-length segment) → no NaN, all valid.
        self.assertTrue((out[3][:, 0] > -1.0).all())

    def test_p_smaller_and_larger_than_k(self):
        """Rows where stored width P < K and P > K must both resample cleanly."""
        from data_pipeline.augmentations import resample_polygons
        # P (pairs) = 4, K = 16 (P < K).
        small = _flat_poly([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], 8)
        out_small = resample_polygons(tf.constant(small), 16).numpy().reshape(16, 2)
        self.assertFalse(np.isnan(out_small).any())
        self.assertTrue((out_small[:, 0] > -1.0).all())   # P<K, all valid
        # P (pairs) = 100, K = 16 (P > K).
        circ = self._circle_flat(0.5, 0.5, 0.3, 100, 200)
        out_big = resample_polygons(tf.constant([circ]), 16).numpy().reshape(16, 2)
        self.assertFalse(np.isnan(out_big).any())
        self.assertTrue((out_big[:, 0] > -1.0).all())

    # (e) Numerical contract: float32, shape, every sample lies ON the input contour.
    def test_dtype_shape_and_points_on_contour(self):
        from data_pipeline.augmentations import resample_polygons
        h = 0.3
        verts = [(0.5 - h, 0.5 - h), (0.5 + h, 0.5 - h),
                 (0.5 + h, 0.5 + h), (0.5 - h, 0.5 + h)]
        polys = tf.constant(_flat_poly(verts, 200))
        red = resample_polygons(polys, 64)
        self.assertEqual(red.dtype, tf.float32)
        self.assertEqual(tuple(red.shape), (1, 128))
        out = red.numpy().reshape(-1, 2)
        valid = out[out[:, 0] > -1.0]
        contour = [np.array(v, np.float32) for v in verts]
        for p in valid:
            self.assertLessEqual(
                _min_dist_to_contour(p, contour), 1e-5,
                f"sampled point {p} is not on the input contour")


class TestRadialRoundTrip(unittest.TestCase):
    """Round trip: encode via _preprocess_polygons_v2, decode via
    eval.polygon_metrics._radial_to_cartesian (conf-gated, sub-bin offsets),
    and check the decoded vertex lands back near the bin's true vertex.

    This pins the encode/decode direction convention end-to-end: both sides
    must agree that the radial vector is origin − vertex (vertex reconstructs
    as center − r·(cos, sin)), otherwise decoded vertices would land ~180°
    away from where they were encoded.
    """

    def test_round_trip_recovers_original_vertices(self):
        from eval.polygon_metrics import DEFAULT_POLY_CONF_THRESH, _radial_to_cartesian

        parser = _make_parser()
        angle_step_deg = 15
        n_bins = 24
        cx, cy = 0.5, 0.5

        # One vertex per bin, each at its own (non-center) sub-bin offset —
        # so this exercises the general sub-bin decode, not just bin centers.
        rng = np.random.default_rng(0)
        offsets = rng.uniform(0.05, 0.95, size=n_bins).astype(np.float32)
        radii = rng.uniform(0.1, 0.4, size=n_bins).astype(np.float32)
        verts = []
        for i in range(n_bins):
            ang = math.radians((i + offsets[i]) * angle_step_deg)
            x = cx - radii[i] * math.cos(ang)
            y = cy - radii[i] * math.sin(ang)
            verts.append((x, y))
        polys = tf.constant(_flat_poly(verts, n_bins * 2))
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)

        out = parser._preprocess_polygons_v2(boxes, polys, angle_step_deg).numpy()
        dist, angle, conf = out[0, 0::3], out[0, 1::3], out[0, 2::3]
        self.assertEqual(int(conf.sum()), n_bins)  # every bin independently occupied

        gate = conf >= DEFAULT_POLY_CONF_THRESH
        self.assertTrue(gate.all())

        # Square image so the isotropic radius decodes identically on both axes.
        w = h = 100
        decoded = _radial_to_cartesian(cx, cy, dist[gate], w, h, offsets=angle[gate])
        expected_px = np.array([[v[0] * w, v[1] * h] for v in np.array(verts)[gate]])

        # Tolerance: half a bin arc at the vertex's own radius, plus float slack.
        half_bin_arc_px = (math.radians(angle_step_deg) / 2.0) * (radii[gate] * w)
        diffs = np.linalg.norm(decoded - expected_px, axis=1)
        self.assertTrue(np.all(diffs < half_bin_arc_px + 1.0),
                         f"max diff {diffs.max():.4f}px exceeds tolerance")


class TestEmptyPolygonAngleTarget(unittest.TestCase):
    """A box with NO valid vertices must produce all-zero polygon targets.

    The sub-bin offset is gated on conf (per-bin validity), so empty polygons get
    angle == 0 everywhere — no spurious offset target drives polygon_angle_loss on
    polygon-less objects (the angle/dist losses are masked by conf anyway).
    """

    def test_no_vertices_targets_all_zero(self):
        parser = _make_parser()
        box = tf.constant([[0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)
        # All vertices invalid (-1 padded).
        empty_poly = tf.constant([[-1.0] * 8], dtype=tf.float32)
        out = parser._preprocess_polygons_v2(box, empty_poly, angle_step=15).numpy()
        angle = out[0, 1::3]
        conf  = out[0, 2::3]
        dist  = out[0, 0::3]
        np.testing.assert_array_equal(angle, 0.0)
        np.testing.assert_array_equal(conf, 0.0)
        np.testing.assert_array_equal(dist, 0.0)

    def test_present_polygon_keeps_validity(self):
        """A present vertex must still set conf=1 even when its sub-bin offset is 0."""
        parser = _make_parser()
        out = parser._preprocess_polygons_v2(_BOX, _POLY, angle_step=15).numpy()
        self.assertEqual(int(out[0, 2::3].sum()), 2)   # two occupied bins (0 and 6)


def _reference_one_hot(boxes, polygons, angle_step):
    """Reference impl: the ORIGINAL one-hot algorithm (pre-segment formulation).

    This is a verbatim copy of the old _preprocess_polygons_v2 body (lines 291-347)
    used to assert exact output equivalence of the new segment formulation. It must
    NOT be "improved" — it encodes the required behavior, including argmax tie-break.
    The dx/dy sign matches the current origin-minus-vertex convention (dx = cx - x,
    dy = cy - y), so this is compared byte-exact against the segment formulation.
    """
    n_angles = 360 // angle_step
    N = tf.shape(boxes)[0]

    cy = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cx = (boxes[:, 1] + boxes[:, 3]) / 2.0

    pts = tf.reshape(polygons, [N, -1, 2])
    valid = pts[:, :, 0] >= 0.0

    dx = cx[:, tf.newaxis] - pts[:, :, 0]
    dy = cy[:, tf.newaxis] - pts[:, :, 1]
    dists = tf.sqrt(dx * dx + dy * dy)

    angles_rad = tf.math.atan2(dy, dx)
    angles_deg = angles_rad * (180.0 / math.pi)
    angles_deg = tf.math.floormod(angles_deg, 360.0)
    bins = tf.cast(tf.math.floor(angles_deg / angle_step), tf.int32)
    bins = tf.clip_by_value(bins, 0, n_angles - 1)

    bins_oh = tf.one_hot(bins, n_angles)
    valid_3d = tf.cast(valid[:, :, tf.newaxis], tf.float32)

    dists_assigned = dists[:, :, tf.newaxis] * bins_oh * valid_3d
    max_dists = tf.reduce_max(dists_assigned, axis=1)
    conf_bins = tf.cast(max_dists > 0.0, tf.float32)

    frac = angles_deg / angle_step - tf.math.floor(angles_deg / angle_step)
    best_pair = tf.argmax(dists_assigned, axis=1, output_type=tf.int32)
    angle_bins = tf.gather(frac, best_pair, batch_dims=1)
    angle_bins = angle_bins * conf_bins

    result = tf.stack([max_dists, angle_bins, conf_bins], axis=-1)
    return tf.reshape(result, [N, n_angles * 3])


def _flat_poly(verts, pad_to):
    """Build a [1, pad_to] flat (x,y) row from a list of (x,y) tuples, -1 padded."""
    flat = np.array(verts, dtype=np.float32).reshape(-1)
    if pad_to < flat.size:
        pad_to = flat.size
    out = np.full((pad_to,), -1.0, dtype=np.float32)
    out[: flat.size] = flat
    return out[np.newaxis, :]


class TestSegmentEquivalence(unittest.TestCase):
    """The segment formulation must be EXACTLY output-equivalent to the one-hot ref."""

    def setUp(self):
        self.parser = _make_parser()

    def _assert_exact(self, boxes, polys, angle_step=15):
        new = self.parser._preprocess_polygons_v2(boxes, polys, angle_step).numpy()
        ref = _reference_one_hot(boxes, polys, angle_step).numpy()
        np.testing.assert_allclose(new, ref, atol=0)
        return new

    # (a) random polygons -------------------------------------------------------
    def test_random_polygons(self):
        rng = np.random.default_rng(0)
        for seed in range(20):
            r = np.random.default_rng(seed)
            for N in (1, 3, 17):
                for P in (4, 50, 500):
                    # Random boxes in [0,1] yxyx (ensure y0<y1, x0<x1).
                    y0 = r.uniform(0.0, 0.5, size=N).astype(np.float32)
                    x0 = r.uniform(0.0, 0.5, size=N).astype(np.float32)
                    y1 = (y0 + r.uniform(0.1, 0.5, size=N)).astype(np.float32)
                    x1 = (x0 + r.uniform(0.1, 0.5, size=N)).astype(np.float32)
                    boxes = tf.constant(np.stack([y0, x0, y1, x1], axis=1))

                    # Random vertices, with a random valid suffix per instance.
                    pts = r.uniform(0.0, 1.0, size=(N, P, 2)).astype(np.float32)
                    n_valid = r.integers(0, P + 1, size=N)
                    for i in range(N):
                        pts[i, n_valid[i]:, :] = -1.0
                    polys = tf.constant(pts.reshape(N, P * 2))
                    self._assert_exact(boxes, polys)

    # (b) adversarial ties ------------------------------------------------------
    def test_adversarial_ties_same_bin_same_radius(self):
        # Center (0.5, 0.5). Two vertices within bin 1 (15°-30°) at IDENTICAL radius.
        cx = cy = 0.5
        r = 0.3
        a1 = math.radians(18.0)
        a2 = math.radians(27.0)  # same bin, same radius, different angle
        verts = [
            (cx + r * math.cos(a1), cy + r * math.sin(a1)),
            (cx + r * math.cos(a2), cy + r * math.sin(a2)),
        ]
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)
        polys = tf.constant(_flat_poly(verts, 8))
        self._assert_exact(boxes, polys)

        # Reversed order — argmax-first must still pick the first listed.
        polys_rev = tf.constant(_flat_poly(verts[::-1], 8))
        self._assert_exact(boxes, polys_rev)

    def test_adversarial_ties_mirrored_in_bin(self):
        # Two vertices mirrored around the bin midline at the same radius.
        cx = cy = 0.5
        r = 0.25
        mid = 7.5
        verts = [
            (cx + r * math.cos(math.radians(mid - 3.0)),
             cy + r * math.sin(math.radians(mid - 3.0))),
            (cx + r * math.cos(math.radians(mid + 3.0)),
             cy + r * math.sin(math.radians(mid + 3.0))),
        ]
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)
        self._assert_exact(boxes, tf.constant(_flat_poly(verts, 8)))

    # (c) boundaries, zero-radius, all-invalid, N=0 -----------------------------
    def test_vertices_on_bin_boundaries(self):
        cx = cy = 0.5
        r = 0.4
        verts = [
            (cx + r * math.cos(math.radians(d)), cy + r * math.sin(math.radians(d)))
            for d in (0.0, 15.0, 30.0, 90.0, 180.0, 345.0)
        ]
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)
        self._assert_exact(boxes, tf.constant(_flat_poly(verts, 24)))

    def test_zero_radius_vertices(self):
        # Vertices exactly at the center → dist 0. Plus one real vertex.
        cx = cy = 0.5
        verts = [(cx, cy), (cx, cy), (0.8, 0.5)]
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)
        self._assert_exact(boxes, tf.constant(_flat_poly(verts, 8)))

    def test_all_zero_radius(self):
        cx = cy = 0.5
        verts = [(cx, cy), (cx, cy)]
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)
        self._assert_exact(boxes, tf.constant(_flat_poly(verts, 8)))

    def test_all_invalid_rows(self):
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.6, 0.6]], tf.float32)
        polys = tf.constant(np.full((2, 16), -1.0, np.float32))
        self._assert_exact(boxes, polys)

    def test_empty_instances(self):
        # N = 0 (no boxes). Segment ops must handle num_segments == 0.
        boxes = tf.zeros([0, 4], tf.float32)
        polys = tf.zeros([0, 16], tf.float32)
        new = self.parser._preprocess_polygons_v2(boxes, polys, 15).numpy()
        ref = _reference_one_hot(boxes, polys, 15).numpy()
        self.assertEqual(new.shape, (0, 72))
        np.testing.assert_allclose(new, ref, atol=0)

    # (d) border-clipped duplicates ---------------------------------------------
    def test_border_clipped_duplicate_corner(self):
        # Several vertices clipped to the SAME corner → identical coords/dist/bin.
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)
        verts = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.2, 0.5)]
        self._assert_exact(boxes, tf.constant(_flat_poly(verts, 12)))

    def test_border_clipped_zero_corner(self):
        boxes = tf.constant([[0.0, 0.0, 1.0, 1.0]], tf.float32)
        verts = [(0.0, 0.0), (0.0, 0.0), (0.5, 0.9)]
        self._assert_exact(boxes, tf.constant(_flat_poly(verts, 8)))


@unittest.skip("micro-benchmark, run via __main__ helper")
class TestMicroBenchmark(unittest.TestCase):
    pass


def _micro_benchmark():
    """Compare old one-hot vs new segment formulation wall time (eager)."""
    import time

    parser = _make_parser()
    N, P, iters = 40, 1986, 100
    rng = np.random.default_rng(123)
    y0 = rng.uniform(0.0, 0.5, size=N).astype(np.float32)
    x0 = rng.uniform(0.0, 0.5, size=N).astype(np.float32)
    y1 = (y0 + rng.uniform(0.1, 0.5, size=N)).astype(np.float32)
    x1 = (x0 + rng.uniform(0.1, 0.5, size=N)).astype(np.float32)
    boxes = tf.constant(np.stack([y0, x0, y1, x1], axis=1))
    pts = rng.uniform(0.0, 1.0, size=(N, P, 2)).astype(np.float32)
    polys = tf.constant(pts.reshape(N, P * 2))

    # Warmup
    _ = _reference_one_hot(boxes, polys, 15)
    _ = parser._preprocess_polygons_v2(boxes, polys, 15)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _reference_one_hot(boxes, polys, 15)
    t_old = (time.perf_counter() - t0) / iters * 1000.0

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = parser._preprocess_polygons_v2(boxes, polys, 15)
    t_new = (time.perf_counter() - t0) / iters * 1000.0

    print(f"\nMicro-benchmark (N={N}, P={P}, {iters} iters, eager):")
    print(f"  OLD one-hot : {t_old:.3f} ms / call")
    print(f"  NEW segment : {t_new:.3f} ms / call")
    print(f"  speedup     : {t_old / t_new:.2f}x")


if __name__ == "__main__":
    _micro_benchmark()
    unittest.main()
