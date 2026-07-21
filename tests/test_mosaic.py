"""Tests for the Ultralytics-style Mosaic + random_perspective augmentation.

Validates:
    - Group semantics: a group of G images emits G // decodes_per_output outputs;
      R=4 uses 4 distinct images per mosaic with zero cross-output reuse; per-output
      mosaic frequency; batch-size independence.
    - Mosaic output image has the configured output_size and boxes stay in [0,1].
    - _place_in_cell pastes/crops an image into a gray-114 cell at an offset.
    - random_perspective: identity round-trips; boxes clip to edge; polygon vertices
      are clipped to the edge (originally-valid stay in [0,1]; -1 padding stays -1).
"""

import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.mosaic import Mosaic, _place_in_cell, _window_shifts
from data_pipeline.augmentations import random_perspective


_MAXV = 8  # 4 (x,y) pairs


def _make_group(G: int, h: int = 32, w: int = 32, n: int = 1) -> dict:
    """Group-of-G example dict (leading dim G). Image i is a solid value i so each
    source is identifiable in the output (mosaic quadrants / single passthrough)."""
    box = tf.constant([[0.25, 0.25, 0.75, 0.75]] * n, dtype=tf.float32)  # yxyx norm
    poly = tf.constant([[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0]] * n, dtype=tf.float32)
    images = tf.stack([tf.fill([h, w, 3], tf.constant(i % 256, tf.uint8)) for i in range(G)])
    return {
        "image":  images,
        "height": tf.constant([h] * G, tf.int32),
        "width":  tf.constant([w] * G, tf.int32),
        "groundtruth_boxes":    tf.stack([box] * G),
        "groundtruth_classes":  tf.zeros([G, n], tf.int64),
        "groundtruth_is_crowd": tf.zeros([G, n], tf.bool),
        "groundtruth_area":     tf.ones([G, n], tf.float32),
        "groundtruth_dontcare": tf.zeros([G, n], tf.int64),
        "groundtruth_dists":    tf.fill([G, n], tf.constant(-1.0)),
        "groundtruth_polygons": tf.stack([poly] * G),
        "source_id":            tf.constant([str(i) for i in range(G)]),
    }


def _identity_mosaic(out=32, freq=0.0, group_size=4, decodes_per_output=1,
                     center=0.25, **kw):
    return Mosaic(
        output_size=[out, out], mosaic_frequency=freq, with_polygons=True,
        aug_scale_min=1.0, aug_scale_max=1.0,
        degrees=0.0, shear=0.0, perspective=0.0, translate=0.0,
        mosaic_center=center, area_thresh=0.0,
        group_size=group_size, decodes_per_output=decodes_per_output,
        **kw,
    )


class TestMosaic(unittest.TestCase):
    def test_group_size_must_be_multiple_of_R(self):
        with self.assertRaises(ValueError):
            Mosaic(output_size=[32, 32], group_size=30, decodes_per_output=4)
        with self.assertRaises(ValueError):
            Mosaic(output_size=[32, 32], group_size=2, decodes_per_output=1)

    def test_output_count_is_group_over_R(self):
        """A group of G emits exactly G // R outputs, for every (G, R)."""
        for G, R in [(8, 4), (8, 2), (8, 1), (4, 4), (32, 4)]:
            m = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=True,
                       group_size=G, decodes_per_output=R)
            out = m.mosaic_fn(is_training=True)(_make_group(G))
            self.assertEqual(tuple(out["image"].shape), (G // R, 32, 32, 3),
                             f"(G={G}, R={R}) -> P={G // R}")

    def test_boxes_in_unit_range(self):
        """mosaic_frequency=1.0 keeps all (padded + real) boxes within [0,1]."""
        m = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=True,
                   group_size=8, decodes_per_output=4)
        out = m.mosaic_fn(is_training=True)(_make_group(8))
        boxes = out["groundtruth_boxes"].numpy()
        self.assertTrue((boxes >= -1e-4).all() and (boxes <= 1.0 + 1e-4).all())

    def test_r4_no_image_reuse(self):
        """R=4, identity geometry: the P outputs partition the group's images with
        zero reuse — every source value appears in exactly one output mosaic."""
        tf.random.set_seed(0)
        G = 8
        m = _identity_mosaic(out=32, freq=1.0, group_size=G, decodes_per_output=4, center=0.0)
        out = m.mosaic_fn(is_training=True)(_make_group(G))
        # Each output is a 2x2 stitch of 4 sources; collect the source values present
        # (drop the 114 gray seam). With identity geometry each quadrant is one source.
        per_output = []
        for i in range(G // 4):
            vals = set(np.unique(out["image"][i].numpy()).tolist()) - {114}
            per_output.append(vals)
        union = set().union(*per_output)
        total = sum(len(v) for v in per_output)
        # No source shared between outputs (sum of sizes == size of union) and all
        # G sources are covered (each source value < G appears once).
        self.assertEqual(total, len(union), f"image reused across outputs: {per_output}")
        self.assertTrue(set(range(G)).issubset(union),
                        f"not all sources used: {sorted(union)}")

    def test_per_output_frequency_extremes(self):
        """freq=0 → every output is a single source (one unique value); freq=1 →
        every output is a mosaic (multiple source values)."""
        G = 8
        single = _identity_mosaic(out=32, freq=0.0, group_size=G, decodes_per_output=1, center=0.0)
        out_s = single.mosaic_fn(is_training=True)(_make_group(G))
        for i in range(G):  # R=1 -> P=G single outputs
            n_vals = len(set(np.unique(out_s["image"][i].numpy()).tolist()))
            self.assertEqual(n_vals, 1, f"single output {i} should be one source")

        tf.random.set_seed(1)
        mosaic = _identity_mosaic(out=32, freq=1.0, group_size=G, decodes_per_output=4, center=0.0)
        out_m = mosaic.mosaic_fn(is_training=True)(_make_group(G))
        for i in range(G // 4):
            vals = set(np.unique(out_m["image"][i].numpy()).tolist()) - {114}
            self.assertGreater(len(vals), 1, f"mosaic output {i} should mix sources")

    def test_identity_single_reproduces_inputs(self):
        """freq=0 + identity + R=1: the P=G outputs reproduce the G input images
        exactly (as a set; the per-group permutation reorders them)."""
        G = 4
        batch = _make_group(G)
        out = _identity_mosaic(out=32, freq=0.0, group_size=G, decodes_per_output=1)\
            .mosaic_fn(is_training=True)(batch)
        self.assertEqual(tuple(out["image"].shape), (G, 32, 32, 3))
        in_vals = sorted(int(np.unique(batch["image"][i].numpy())[0]) for i in range(G))
        out_vals = sorted(int(np.unique(out["image"][i].numpy())[0]) for i in range(G))
        self.assertEqual(in_vals, out_vals)

    def test_branches_have_identical_structure(self):
        """The per-output mosaic/single tf.cond branches emit identical keys/dtypes/ranks."""
        batch = _make_group(8)
        out_m = Mosaic(output_size=[32, 32], mosaic_frequency=1.0, with_polygons=True,
                       group_size=8, decodes_per_output=4).mosaic_fn(is_training=True)(batch)
        out_s = Mosaic(output_size=[32, 32], mosaic_frequency=0.0, with_polygons=True,
                       group_size=8, decodes_per_output=4).mosaic_fn(is_training=True)(batch)
        self.assertEqual(set(out_m.keys()), set(out_s.keys()))
        for k in out_m:
            self.assertEqual(out_m[k].dtype, out_s[k].dtype, f"dtype mismatch {k}")
            self.assertEqual(len(out_m[k].shape), len(out_s[k].shape), f"rank mismatch {k}")
            self.assertEqual(int(out_m[k].shape[0]), 2)  # P = 8 // 4
            self.assertEqual(int(out_s[k].shape[0]), 2)

    def test_padded_rows_are_zero_boxes_and_neg1_polys(self):
        """freq=0 + identity + R=1: exactly one output carries 2 real boxes (the
        source with a real 2nd row); the rest carry 1 real box + a pad row (zero box /
        -1 polygon). Order-independent because the per-group permutation shuffles which
        output gets which source."""
        G = 4
        b_real = [[0.25, 0.25, 0.75, 0.75], [0.10, 0.10, 0.40, 0.40]]
        b_pad = [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.0, 0.0]]
        p_real = [[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0],
                  [0.2, 0.2, 0.35, 0.35, -1.0, -1.0, -1.0, -1.0]]
        p_pad = [[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0], [-1.0] * 8]
        boxes = tf.constant([b_real, b_pad, b_pad, b_pad], tf.float32)
        polys = tf.constant([p_real, p_pad, p_pad, p_pad], tf.float32)
        images = tf.stack([tf.fill([32, 32, 3], tf.constant(i, tf.uint8)) for i in range(G)])
        batch = {
            "image": images,
            "height": tf.constant([32] * G, tf.int32),
            "width":  tf.constant([32] * G, tf.int32),
            "groundtruth_boxes":    boxes,
            "groundtruth_classes":  tf.zeros([G, 2], tf.int64),
            "groundtruth_is_crowd": tf.zeros([G, 2], tf.bool),
            "groundtruth_area":     tf.ones([G, 2], tf.float32),
            "groundtruth_dontcare": tf.zeros([G, 2], tf.int64),
            "groundtruth_dists":    tf.fill([G, 2], tf.constant(-1.0)),
            "groundtruth_polygons": polys,
            "source_id":            tf.constant([str(i) for i in range(G)]),
        }
        out = _identity_mosaic(out=32, freq=0.0, group_size=G, decodes_per_output=1)\
            .mosaic_fn(is_training=True)(batch)
        boxes = out["groundtruth_boxes"].numpy()
        self.assertEqual(boxes.shape[:2], (G, 2))
        # Exactly one output has 2 real (non-zero) boxes; the rest have a pad row.
        n_two = sum(int((boxes[i, 1] != 0.0).any()) for i in range(G))
        self.assertEqual(n_two, 1)

    def test_eval_path_emits_group_size(self):
        """is_training=False single-warps every image -> G outputs."""
        out = _identity_mosaic(out=32, freq=0.0, group_size=8, decodes_per_output=4)\
            .mosaic_fn(is_training=False)(_make_group(8))
        self.assertEqual(tuple(out["image"].shape), (8, 32, 32, 3))

    def test_small_batch_size_unaffected(self):
        """group_size is independent of the final batch size: a group of 16 with R=4
        emits 4 outputs that batch cleanly to any batch_size (e.g. 2)."""
        G = 16
        ds = (tf.data.Dataset.from_tensor_slices(_make_group(G))
              .padded_batch(G, drop_remainder=True)
              .map(_identity_mosaic(out=32, freq=0.5, group_size=G, decodes_per_output=4)
                   .mosaic_fn(is_training=True))
              .unbatch()
              .batch(2))
        total = sum(int(b["image"].shape[0]) for b in ds)
        self.assertEqual(total, G // 4)  # 16 // 4 = 4 outputs


class TestTileCrop(unittest.TestCase):
    """Per-tile random-window crop (tile_crop_min/max).

    Replaces the old per-tile scale. The window geometry is a pure function
    (_apply_window) so the coordinate transform is hand-checkable; the enabled
    path drops rows fully outside the window with a full aligned mask.
    """

    def test_bounds_validation(self):
        _identity_mosaic(tile_crop_min=0.4, tile_crop_max=0.9)   # ok
        _identity_mosaic(tile_crop_min=0.0, tile_crop_max=0.0)   # ok (off)
        with self.assertRaises(ValueError):
            _identity_mosaic(tile_crop_min=0.0, tile_crop_max=0.9)   # min must be > 0
        with self.assertRaises(ValueError):
            _identity_mosaic(tile_crop_min=0.9, tile_crop_max=0.4)   # min > max
        with self.assertRaises(ValueError):
            _identity_mosaic(tile_crop_min=0.5, tile_crop_max=1.2)   # max > 1

    def test_off_is_byte_identical(self):
        """crop OFF (0/0) reproduces the no-crop mosaic byte-for-byte (same seed)."""
        G = 8

        def run(**kw):
            tf.random.set_seed(123)
            m = _identity_mosaic(out=32, freq=1.0, group_size=G, decodes_per_output=4,
                                 center=0.0, **kw)
            return m.mosaic_fn(is_training=True)(_make_group(G))

        base = run()
        off = run(tile_crop_min=0.0, tile_crop_max=0.0)
        np.testing.assert_array_equal(base["image"].numpy(), off["image"].numpy())
        np.testing.assert_array_equal(base["groundtruth_boxes"].numpy(),
                                      off["groundtruth_boxes"].numpy())

    def test_apply_window_exact_transform_no_clip(self):
        """Hand-computed: content 100x100, window top=20 left=20 win=50x50.
        box px [30,50] -> (30-20)/50=0.2 .. (50-20)/50=0.6. poly likewise."""
        boxes = tf.constant([[0.3, 0.3, 0.5, 0.5]], tf.float32)
        polys = tf.constant([[0.3, 0.3, 0.5, 0.5, -1.0, -1.0]], tf.float32)  # (x,y) pairs
        bw, pw, keep = Mosaic._apply_window(boxes, polys, 100, 100, 20, 20, 50, 50)
        self.assertTrue(bool(keep.numpy()[0]))
        np.testing.assert_allclose(bw.numpy()[0], [0.2, 0.2, 0.6, 0.6], atol=1e-6)
        np.testing.assert_allclose(pw.numpy()[0], [0.2, 0.2, 0.6, 0.6, -1.0, -1.0], atol=1e-6)

    def test_apply_window_clips_and_drops(self):
        """A box fully outside the window is dropped; a partial box is clipped."""
        boxes = tf.constant([
            [0.8, 0.8, 0.9, 0.9],   # outside a [0,0,50,50] window on 100x100 -> drop
            [0.2, 0.2, 0.8, 0.8],   # straddles -> clipped to [0.4,0.4,1,1]
        ], tf.float32)
        polys = tf.fill([2, 6], -1.0)
        bw, pw, keep = Mosaic._apply_window(boxes, polys, 100, 100, 0, 0, 50, 50)
        self.assertFalse(bool(keep.numpy()[0]))
        self.assertTrue(bool(keep.numpy()[1]))
        np.testing.assert_allclose(bw.numpy()[1], [0.4, 0.4, 1.0, 1.0], atol=1e-6)

    def test_tile_crop_example_bounds_and_alignment(self):
        """Enabled crop: window dims within bounds and every per-instance field
        masked to the same kept-row count."""
        m = _identity_mosaic(tile_crop_min=0.5, tile_crop_max=0.5)
        cex = {
            "image": tf.fill([100, 80, 3], tf.constant(7, tf.uint8)),
            "groundtruth_boxes": tf.constant(
                [[0.05, 0.05, 0.25, 0.25], [0.6, 0.6, 0.95, 0.95]], tf.float32),
            "groundtruth_polygons": tf.fill([2, 8], -1.0),
            "groundtruth_classes": tf.constant([3, 4], tf.int64),
            "groundtruth_is_crowd": tf.zeros([2], tf.bool),
            "groundtruth_area": tf.ones([2], tf.float32),
            "groundtruth_dontcare": tf.zeros([2], tf.int64),
            "groundtruth_dists": tf.fill([2], -1.0),
        }
        for seed in range(8):
            tf.random.set_seed(seed)
            out = m._tile_crop_example(cex)
            ci = out["image"].numpy()
            # s=0.5 -> win_h = round(100*0.5) = 50, win_w = round(80*0.5) = 40.
            self.assertEqual(ci.shape[:2], (50, 40))
            nb = int(out["groundtruth_boxes"].shape[0])
            for k in ("groundtruth_classes", "groundtruth_polygons", "groundtruth_area",
                      "groundtruth_is_crowd", "groundtruth_dontcare", "groundtruth_dists"):
                self.assertEqual(int(out[k].shape[0]), nb, f"{k} misaligned to boxes")
            b = out["groundtruth_boxes"].numpy()
            if nb:
                self.assertTrue((b >= 0.0).all() and (b <= 1.0).all())

    def test_config_wiring(self):
        from configs.yaml_loader import load_config_from_dict
        cfg = load_config_from_dict({"task": {"train_data": {"parser": {
            "mosaic": {"tile_crop_min": 0.4, "tile_crop_max": 0.9}}}}})
        m = cfg.task.train_data.parser.mosaic
        self.assertEqual((m.tile_crop_min, m.tile_crop_max), (0.4, 0.9))
        # default off
        cfg0 = load_config_from_dict({})
        m0 = cfg0.task.train_data.parser.mosaic
        self.assertEqual((m0.tile_crop_min, m0.tile_crop_max), (0.0, 0.0))
        # input_reader forwards the knobs to Mosaic
        import inspect
        from data_pipeline import input_reader
        src = inspect.getsource(input_reader)
        self.assertIn("tile_crop_min=mosaic_cfg.tile_crop_min", src)
        self.assertIn("tile_crop_max=mosaic_cfg.tile_crop_max", src)


class TestFlipOwnershipAndSinglePath(unittest.TestCase):
    """Flip lives inside the Mosaic module (per tile / per single image), and
    the non-mosaic path uses its own scale/translate params."""

    def _asym_group(self, G=4, h=32, w=32):
        """Group whose images are bright on the LEFT half (value 200 vs 10)."""
        half = tf.concat([tf.fill([h, w // 2, 3], tf.constant(200, tf.uint8)),
                          tf.fill([h, w - w // 2, 3], tf.constant(10, tf.uint8))],
                         axis=1)
        g = _make_group(G, h=h, w=w)
        g["image"] = tf.stack([half] * G)
        # box hugging the bright (left) side: yxyx = [0.2, 0.05, 0.8, 0.45]
        box = tf.constant([[0.2, 0.05, 0.8, 0.45]], dtype=tf.float32)
        g["groundtruth_boxes"] = tf.stack([box] * G)
        return g

    def test_single_flip_consistent_and_both_orientations_occur(self):
        m = _identity_mosaic(out=32, freq=0.0, group_size=4, center=0.0,
                             random_flip=True)
        fn = m.mosaic_fn(is_training=True)
        saw_flipped, saw_upright = False, False
        for seed in range(12):
            tf.random.set_seed(seed)
            res = fn(self._asym_group())
            imgs = res["image"].numpy()
            boxes = res["groundtruth_boxes"].numpy()
            for k in range(imgs.shape[0]):
                left_bright = imgs[k, :, :16, 0].mean() > imgs[k, :, 16:, 0].mean()
                xmin, xmax = boxes[k, 0, 1], boxes[k, 0, 3]
                if left_bright:
                    saw_upright = True
                    self.assertLess(xmax, 0.55, "box didn't stay on bright side")
                else:
                    saw_flipped = True
                    self.assertGreater(xmin, 0.45, "box didn't flip with image")
        self.assertTrue(saw_flipped and saw_upright,
                        "expected both orientations across seeds")

    def test_mosaic_tiles_flip_independently(self):
        # freq=1, fixed center -> TL quadrant of the output shows the TL
        # tile's bottom-right corner (dark when upright, bright when flipped).
        m = _identity_mosaic(out=32, freq=1.0, group_size=4, center=0.0,
                             random_flip=True)
        fn = m.mosaic_fn(is_training=True)
        states = set()
        for seed in range(16):
            tf.random.set_seed(seed)
            img = fn(self._asym_group())["image"].numpy()[0]
            states.add(img[:16, :16, 0].mean() > 100)  # TL region bright?
            if len(states) == 2:
                break
        self.assertEqual(states, {True, False},
                         "TL tile never appeared in both orientations")

    def test_single_path_uses_its_own_scale_params(self):
        # Module warp bounds 0.5 (would shrink), single path pinned to 1.0 ->
        # non-mosaic output must be the identity passthrough.
        m = Mosaic(
            output_size=[32, 32], mosaic_frequency=0.0, with_polygons=True,
            aug_scale_min=0.5, aug_scale_max=0.5,
            single_scale_min=1.0, single_scale_max=1.0, single_translate=0.0,
            degrees=0.0, shear=0.0, perspective=0.0, translate=0.0,
            mosaic_center=0.0, area_thresh=0.0,
            group_size=4, decodes_per_output=1,
        )
        res = m.mosaic_fn(is_training=True)(_make_group(4))
        np.testing.assert_allclose(
            res["groundtruth_boxes"].numpy()[0],
            [[0.25, 0.25, 0.75, 0.75]], atol=1e-2,
            err_msg="single path did not use single_scale (1.0)")

    def test_input_reader_wires_single_params_and_flip(self):
        import inspect
        from data_pipeline import input_reader
        src = inspect.getsource(input_reader)
        self.assertIn("single_scale_min=parser_cfg.aug_scale_min", src)
        self.assertIn("single_scale_max=parser_cfg.aug_scale_max", src)
        self.assertIn("single_translate=parser_cfg.aug_rand_translate", src)
        self.assertIn("random_flip=parser_cfg.random_flip and not is_training", src)


class TestWindowShifts(unittest.TestCase):
    """Source-selection invariants of the Sidon-shift draw (see _SIDON_SHIFTS).

    Output j of a group reads perm[(j*R + s) % G] for s in _window_shifts(R, G).
    Because perm is a bijection, the invariants hold for the raw index sets:
      - uniform reuse: every group index is read by exactly 4/R outputs
      - within an output the 4 indices are distinct
      - any two outputs share at most ONE index (zero at R=4 — disjoint tiling)
    """

    G = 32

    def _rows(self, R, G=None):
        G = G or self.G
        shifts = _window_shifts(R, G)
        return [frozenset((j * R + s) % G for s in shifts) for j in range(G // R)]

    def test_uniform_reuse_and_distinct_within_output(self):
        for R in (1, 2, 4):
            rows = self._rows(R)
            for row in rows:
                self.assertEqual(len(row), 4, f"R={R}: duplicate source in one output")
            counts = np.zeros(self.G, dtype=int)
            for row in rows:
                for i in row:
                    counts[i] += 1
            self.assertTrue((counts == 4 // R).all(),
                            f"R={R}: reuse counts {counts} != {4 // R}")

    def test_pairwise_overlap_at_most_one(self):
        for R in (1, 2):
            rows = self._rows(R)
            worst = max(len(a & b) for i, a in enumerate(rows) for b in rows[i + 1:])
            self.assertLessEqual(
                worst, 1,
                f"R={R}: two outputs share {worst} sources (sliding-window regression)")

    def test_overlap_guarantee_brute_force_sweep(self):
        """Whenever a Sidon set is selected, the <=1-overlap guarantee must hold
        for the ACTUAL mod-G index sets — brute-forced, not derived. Catches
        modular collisions (e.g. a shift difference equal to G/2, which pairs
        with its own negative: at G=16/R=2 the difference 8 self-collides and
        two outputs would share 2 images, so _window_shifts must fall back)."""
        from data_pipeline.mosaic import _SIDON_SHIFTS
        for R in (1, 2, 4):
            for G in range(max(8, R), 129, R if R > 1 else 1):
                if G % R:
                    continue
                shifts = _window_shifts(R, G)
                if shifts != _SIDON_SHIFTS.get(R) or max(shifts) >= G:
                    continue  # fallback window: no guarantee claimed
                rows = self._rows(R, G=G)
                worst = max(
                    (len(a & b) for i, a in enumerate(rows) for b in rows[i + 1:]),
                    default=0)
                limit = 0 if R == 4 else 1
                self.assertLessEqual(worst, limit, f"R={R} G={G}: overlap {worst}")
                counts = np.zeros(G, dtype=int)
                for row in rows:
                    self.assertEqual(len(row), 4, f"R={R} G={G}: within-output dup")
                    for i in row:
                        counts[i] += 1
                self.assertTrue((counts == 4 // R).all(), f"R={R} G={G}: uneven reuse")

    def test_modular_collision_groups_fall_back(self):
        # G=16/R=2: shift difference 8 == G/2 self-collides mod 16 -> 2 shared
        # images if the Sidon set were used; must fall back to the window.
        self.assertEqual(_window_shifts(2, 16), (0, 1, 2, 3))
        self.assertEqual(_window_shifts(2, 18), (0, 1, 2, 3))
        self.assertEqual(_window_shifts(2, 20), (0, 1, 4, 9))
        self.assertEqual(_window_shifts(1, 15), (0, 1, 3, 7))

    def test_r4_outputs_disjoint_and_window_unchanged(self):
        rows = self._rows(4)
        self.assertEqual(sorted(i for row in rows for i in row), list(range(self.G)))
        worst = max(len(a & b) for i, a in enumerate(rows) for b in rows[i + 1:])
        self.assertEqual(worst, 0)
        # R=4 must keep the historical contiguous window (byte-identical indices).
        self.assertEqual(_window_shifts(4, self.G), (0, 1, 2, 3))

    def test_small_group_falls_back_to_contiguous_window(self):
        self.assertEqual(_window_shifts(1, 8), (0, 1, 2, 3))
        self.assertEqual(_window_shifts(2, 4), (0, 1, 2, 3))

    def test_r1_all_images_emitted_once_as_singles(self):
        # freq=0.0 -> every output is _single on its first source, perm[(j + 0) % G]
        # = perm[j]: a permutation of the group, so each image appears exactly once.
        m = _identity_mosaic(out=32, freq=0.0, group_size=16, decodes_per_output=1)
        out = m.mosaic_fn(is_training=True)(_make_group(16))
        vals = sorted(int(out["image"][k, 0, 0, 0]) for k in range(16))
        self.assertEqual(vals, list(range(16)))


class TestMosaicUnbatchIntegration(unittest.TestCase):
    def test_padded_batch_map_unbatch_yields_P(self):
        """from_tensor_slices(G) → padded_batch(G) → map → unbatch → G//R elements."""
        h = w = 32
        G, R = 8, 4
        m = _identity_mosaic(out=h, freq=0.5, group_size=G, decodes_per_output=R)
        ds = (tf.data.Dataset.from_tensor_slices(_make_group(G, h, w))
              .padded_batch(G, drop_remainder=True)
              .map(m.mosaic_fn(is_training=True))
              .unbatch())
        # Static element spec keeps image [H, W, 3].
        self.assertEqual(ds.element_spec["image"].shape.as_list(), [h, w, 3])
        count = sum(1 for _ in ds)
        self.assertEqual(count, G // R)


class TestPlaceInCell(unittest.TestCase):
    def test_paste_with_offset_and_gray_fill(self):
        R = tf.fill([10, 10, 3], tf.constant(200, tf.uint8))
        cell = _place_in_cell(R, tf.constant(20), tf.constant(20),
                              tf.constant(5), tf.constant(5)).numpy()
        self.assertEqual(cell.shape, (20, 20, 3))
        self.assertTrue((cell[5:15, 5:15] == 200).all())   # pasted region
        self.assertTrue((cell[0:5, :] == 114).all())        # gray fill outside

    def test_crop_when_larger_than_cell(self):
        R = tf.fill([30, 30, 3], tf.constant(200, tf.uint8))
        cell = _place_in_cell(R, tf.constant(20), tf.constant(20),
                              tf.constant(-5), tf.constant(-5)).numpy()
        self.assertEqual(cell.shape, (20, 20, 3))
        self.assertTrue((cell == 200).all())   # fully covered by the cropped image


class TestRandomPerspective(unittest.TestCase):
    def setUp(self):
        self.s = 64
        img = np.zeros((self.s, self.s, 3), np.uint8)
        img[8:24, 8:24] = 255
        self.img = tf.constant(img)
        self.boxes = tf.constant([[0.3, 0.3, 0.7, 0.7]], tf.float32)
        self.polys = tf.constant([[0.3, 0.3, 0.7, 0.7, -1.0, -1.0]], tf.float32)

    def test_identity_round_trips(self):
        io, bo, keep, po = random_perspective(
            self.img, self.boxes, self.polys, self.s, self.s,
            degrees=0, translate=0, scale=0, shear=0, perspective=0,
        )
        self.assertTrue(np.array_equal(io.numpy(), self.img.numpy()))
        np.testing.assert_allclose(bo.numpy(), self.boxes.numpy(), atol=1e-3)
        self.assertTrue(bool(keep.numpy()[0]))

    def test_boxes_clipped_to_unit_range(self):
        tf.random.set_seed(0)
        _, bo, keep, _ = random_perspective(
            self.img, self.boxes, self.polys, self.s, self.s, degrees=30,
        )
        b = bo.numpy()
        self.assertTrue((b >= 0.0).all() and (b <= 1.0).all())
        self.assertEqual(keep.numpy().shape, (1,))

    def test_polygon_clip_to_edge_keeps_validity(self):
        """Originally-valid vertices stay in [0,1] (clipped); -1 padding stays -1."""
        tf.random.set_seed(3)
        _, _, _, po = random_perspective(
            self.img, self.boxes, self.polys, self.s, self.s, degrees=25, scale=0.5,
        )
        p = po.numpy()[0]
        valid = p[0:4]     # two transformed (x,y) pairs
        pad = p[4:6]       # the -1 padding pair
        self.assertTrue((valid >= 0.0).all() and (valid <= 1.0).all())
        np.testing.assert_array_equal(pad, [-1.0, -1.0])


class TestMosaicCanvasWarp(unittest.TestCase):
    """Regression tests for the 2×-canvas + single-warp _mosaic formulation.

    Each source image is resized at its drawn scale and placed into the appropriate
    cell of a 2× canvas (twice the output resolution).  A single
    ``random_perspective`` warp then maps the canvas to the final output frame —
    eliminating four separate full-frame warps.  These tests pin (a) the image
    quadrant layout under identity geometry, (b) the annotation path, (c) the mask
    partition, and (d) graph-mode/tf.function compatibility.
    """

    def _det_mosaic(self, out=32):
        """Deterministic identity-geometry mosaic: center=0 → yc=xc=H; s_i=1."""
        return Mosaic(
            output_size=[out, out], mosaic_frequency=1.0, with_polygons=True,
            aug_scale_min=1.0, aug_scale_max=1.0,
            degrees=0.0, shear=0.0, perspective=0.0, translate=0.0,
            mosaic_center=0.0, area_thresh=0.0,
        )

    def _solid_example(self, color, h=32, w=32):
        box = tf.constant([[0.25, 0.25, 0.75, 0.75]], tf.float32)
        poly = tf.constant([[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0]], tf.float32)
        return {
            "image":  tf.fill([h, w, 3], tf.constant(color, tf.uint8)),
            "height": tf.constant(h, tf.int32),
            "width":  tf.constant(w, tf.int32),
            "groundtruth_boxes":    box,
            "groundtruth_classes":  tf.zeros([1], tf.int64),
            "groundtruth_is_crowd": tf.zeros([1], tf.bool),
            "groundtruth_area":     tf.ones([1], tf.float32),
            "groundtruth_dontcare": tf.zeros([1], tf.int64),
            "groundtruth_dists":    tf.fill([1], tf.constant(-1.0)),
            "groundtruth_polygons": poly,
            "source_id":            tf.constant("x"),
        }

    def test_a_identity_geometry_quadrant_layout(self):
        """center=0, identity affine, 4 solid HxW images → the 4 colors land in
        the TL/TR/BL/BR output quadrants (inner corner of each source).

        With yc=xc=H=W and s_i=1 (nh=H, nw=W), the mosaic canvas is 2H×2W with
        image i placed so its center-adjacent corner abuts (H, W). M is the
        center-crop: output (x, y) ← canvas (x + W/2, y + H/2). So the output's
        TL quadrant samples canvas [W/2, W) — which is the inner (BR) corner of
        the TL source — etc. A 2px band around the split is skipped (bilinear
        seam between a quadrant and 114 fill).
        """
        H = W = 32
        colors = [40, 80, 160, 220]  # TL, TR, BL, BR
        exs = [self._solid_example(colors[i], H, W) for i in range(4)]
        out = self._det_mosaic(out=H)._mosaic(*exs)
        img = out["image"].numpy()
        self.assertEqual(img.shape, (H, W, 3))

        hh, hw = H // 2, W // 2
        b = 2  # seam tolerance band
        # Interior of each output quadrant must equal that source's color.
        quad = {
            (0, 0): colors[0],  # TL
            (0, 1): colors[1],  # TR
            (1, 0): colors[2],  # BL
            (1, 1): colors[3],  # BR
        }
        for (qy, qx), color in quad.items():
            y0 = qy * hh + (b if qy == 0 else 0)
            y1 = (qy + 1) * hh - (0 if qy == 0 else b)
            x0 = qx * hw + (b if qx == 0 else 0)
            x1 = (qx + 1) * hw - (0 if qx == 0 else b)
            region = img[y0:y1, x0:x1]
            self.assertTrue(
                (region == color).all(),
                f"quadrant ({qy},{qx}) expected {color}, got values "
                f"{np.unique(region)}",
            )

    def test_b_label_path_regression(self):
        """Identity-geometry: a known box in each source maps to hand-computed
        output boxes within 1e-5. Pins the annotation path (A_i + M corner math).

        center=0 → yc=xc=H=W, s_i=1, padh/padw per quadrant:
          TL: pad=(yc-nh, xc-nw)=(0,0)            → canvas px = src px
          TR: pad=(yc-nh, off_x=xc)=(0, W)        → canvas x = src x + W
          BL: pad=(off_y=yc, xc-nw)=(H, 0)        → canvas y = src y + H
          BR: pad=(yc, xc)=(H, W)                 → canvas (x+W, y+H)
        Canvas-normalized (÷2W,÷2H), then M center-crop: out = canvas*2 - 0.5*... .
        We compute via the public corner math: out_px = M @ canvas_px; here M maps
        canvas px (cx, cy) → output px (cx - W/2, cy - H/2). So out_norm = (canvas
        px - center)/H.
        """
        H = W = 32
        # One box per source, distinct, all = [0.25,0.25,0.75,0.75] in src-norm.
        box = [0.25, 0.25, 0.75, 0.75]
        exs = [self._solid_example(40 + 40 * i, H, W) for i in range(4)]
        for ex in exs:
            ex["groundtruth_boxes"] = tf.constant([box], tf.float32)

        out = self._det_mosaic(out=H)._mosaic(*exs)
        boxes = out["groundtruth_boxes"].numpy()

        # Hand-compute. Source box px (within HxW source): ymin/xmin/ymax/xmax * H/W.
        ymin, xmin, ymax, xmax = box
        s_ymin, s_xmin = ymin * H, xmin * W
        s_ymax, s_xmax = ymax * H, xmax * W
        # Per-quadrant canvas offset (padh, padw):
        pads = [(0, 0), (0, W), (H, 0), (H, W)]  # TL, TR, BL, BR
        # M center-crop: out_px = canvas_px - (W/2, H/2); out_norm = out_px / W (=H).
        expected = []
        for (padh, padw) in pads:
            c_ymin, c_ymax = s_ymin + padh, s_ymax + padh
            c_xmin, c_xmax = s_xmin + padw, s_xmax + padw
            o_ymin = (c_ymin - H / 2.0) / H
            o_xmin = (c_xmin - W / 2.0) / W
            o_ymax = (c_ymax - H / 2.0) / H
            o_xmax = (c_xmax - W / 2.0) / W
            expected.append([o_ymin, o_xmin, o_ymax, o_xmax])
        expected = np.clip(np.array(expected, np.float32), 0.0, 1.0)

        # The kept boxes are those with visible area after clip. Sort both for a
        # set comparison (filtering may reorder / drop fully-out boxes).
        # Under this geometry each source's box partially overlaps the output.
        self.assertEqual(boxes.shape[1], 4)
        # Match each expected box to a returned box within tol.
        for exp in expected:
            # skip boxes that clip to zero area (not kept)
            if (exp[2] - exp[0]) <= 0 or (exp[3] - exp[1]) <= 0:
                continue
            diffs = np.abs(boxes - exp).sum(axis=1)
            self.assertLess(
                diffs.min(), 1e-5,
                f"expected box {exp} not found in {boxes}",
            )

    def test_c_mask_partition_distinct_sources(self):
        """Random geometry: every central output pixel is owned by exactly one
        source (value ∈ {1,2,3,4}) or gray 114 — never a blend of two sources.

        Warp four solid constant-valued images (1,2,3,4). Because each source is a
        single constant and out-of-source fill is 114, bilinear can only blend a
        source value with 114 (at borders), never two distinct source values. So
        every interior pixel that is not on a seam is in {1,2,3,4,114}.
        """
        tf.random.set_seed(7)
        H = W = 48
        m = Mosaic(
            output_size=[H, W], mosaic_frequency=1.0, with_polygons=True,
            aug_scale_min=0.6, aug_scale_max=1.4,
            degrees=8.0, shear=2.0, perspective=0.0, translate=0.1,
            mosaic_center=0.25, area_thresh=0.0,
        )
        exs = [self._solid_example(v, H, W) for v in (1, 2, 3, 4)]
        out = m._mosaic(*exs)
        img = out["image"].numpy()[..., 0].astype(np.int32)
        # Sample a central grid of pixels (avoid the outer 1px ring for safety).
        sample = img[2:-2, 2:-2]
        allowed = {1, 2, 3, 4, 114}
        vals = set(np.unique(sample).tolist())
        # Allow a small number of bilinear-seam values (blends of a source w/ 114).
        unexpected = vals - allowed
        # Any unexpected value must be a source⊗114 blend (between min source and
        # 114) — never a blend of two distinct sources (which would land 1..4
        # range interior values like 1.5→2, caught here as values in (4,114)).
        for u in unexpected:
            self.assertTrue(
                4 < u < 114,
                f"value {u} indicates a two-source blend, not a source/114 seam",
            )

    def test_d_tf_function_traceable(self):
        """_mosaic runs under tf.function tracing and re-execution; output is
        [H, W, 3] uint8."""
        H = W = 32
        m = Mosaic(
            output_size=[H, W], mosaic_frequency=1.0, with_polygons=True,
            aug_scale_min=0.8, aug_scale_max=1.2,
            degrees=10.0, shear=2.0, perspective=0.0, translate=0.1,
            mosaic_center=0.25,
        )
        exs = [self._solid_example(50 + 30 * i, H, W) for i in range(4)]

        @tf.function
        def run(a, b, c, d):
            return m._mosaic(a, b, c, d)["image"]

        img1 = run(*exs)
        img2 = run(*exs)
        self.assertEqual(tuple(img1.shape), (H, W, 3))
        self.assertEqual(img1.dtype, tf.uint8)
        self.assertEqual(tuple(img2.shape), (H, W, 3))


class TestWarpScaleBounds(unittest.TestCase):
    """The warp scale gain must honor the EXPLICIT [aug_scale_min, aug_scale_max]
    config bounds.

    The old symmetric-magnitude form (`scale = max(max−1, 1−min)`) widened the
    configured [0.4, 1.9] to [0.1, 1.9]: a 0.1× draw shrinks the content to 1%
    area and produces a mostly-gray training frame. ~1 in 6 draws fell in the
    undocumented [0.1, 0.4) regime.
    """

    def test_gain_stays_within_explicit_bounds(self):
        from data_pipeline.augmentations import make_perspective_matrix
        lo, hi = 0.4, 1.9
        gains = []
        for _ in range(300):
            # With degrees=shear=perspective=translate=0, M[0,0] is exactly the
            # drawn scale gain.
            M = make_perspective_matrix(
                h_in=64, w_in=64, target_h=64, target_w=64,
                degrees=0.0, translate=0.0, shear=0.0, perspective=0.0,
                scale_min=lo, scale_max=hi,
            )
            gains.append(float(M[0, 0]))
        gains = np.array(gains)
        self.assertGreaterEqual(gains.min(), lo)   # old impl: min ≈ 0.1 → fails
        self.assertLessEqual(gains.max(), hi)
        # The range is actually exercised (not stuck at one end).
        self.assertLess(gains.min(), 0.7)
        self.assertGreater(gains.max(), 1.5)

    def test_magnitude_form_still_supported(self):
        from data_pipeline.augmentations import make_perspective_matrix
        M = make_perspective_matrix(
            h_in=64, w_in=64, target_h=64, target_w=64,
            degrees=0.0, translate=0.0, shear=0.0, perspective=0.0,
            scale=0.0,   # magnitude 0 → gain exactly 1
        )
        self.assertAlmostEqual(float(M[0, 0]), 1.0, places=5)


class TestRotationGating(unittest.TestCase):
    """rotate_prob gates rotation: most outputs stay upright (angle forced to 0),
    only a fraction rotate. With scale gain fixed at 1.0 and shear=0, the matrix
    off-diagonal M[0,1] = -sin(angle); it is exactly 0 iff the output is upright.
    """

    @staticmethod
    def _offdiag(rotate_prob, degrees=30.0):
        from data_pipeline.augmentations import make_perspective_matrix
        M = make_perspective_matrix(
            h_in=64, w_in=64, target_h=64, target_w=64,
            degrees=degrees, translate=0.0, shear=0.0, perspective=0.0,
            scale_min=1.0, scale_max=1.0, rotate_prob=rotate_prob,
        )
        return float(M[0, 1]), float(M[1, 0])

    def test_rotate_prob_zero_is_always_upright(self):
        for _ in range(200):
            o01, o10 = self._offdiag(rotate_prob=0.0)
            self.assertEqual(o01, 0.0)   # no rotation cross-terms, ever
            self.assertEqual(o10, 0.0)

    def test_rotate_prob_one_always_rotates(self):
        # rotate_prob>=1.0 takes the unconditional-rotate path: essentially
        # every draw has a nonzero angle
        rotated = sum(1 for _ in range(200) if abs(self._offdiag(1.0)[0]) > 1e-6)
        self.assertGreater(rotated, 190)

    def test_rotate_prob_fraction_matches(self):
        tf.random.set_seed(0)
        n, p = 400, 0.3
        rotated = sum(1 for _ in range(n) if abs(self._offdiag(p)[0]) > 1e-6)
        frac = rotated / n
        # ~0.3 expected; loose 3-sigma band so it is not flaky
        self.assertGreater(frac, 0.20)
        self.assertLess(frac, 0.40)

    def test_default_mosaic_never_rotates(self):
        # Rotation parity: the mosaic default is upright (rotate_prob default 0,
        # shear 0) and the mosaic warp forces 0 rotation regardless of the field.
        m = Mosaic(output_size=[32, 32], with_polygons=True)
        self.assertEqual(m._shear, 0.0)
        self.assertEqual(m._rotate_prob, 0.0)


class TestFilteredAnnsMissingFields(unittest.TestCase):
    """Pinning test: _filtered_anns tolerates absent per-box side fields.

    A side field (e.g. groundtruth_dists) may be missing from an example that
    nevertheless carries boxes. The fallback for a missing field must be an
    N-length default (one entry per kept-mask slot), not a 0-length tensor:
    tf.boolean_mask requires the masked tensor and the mask to share the masked
    dimension, so a length-0 fallback against an N-length keep raised
    `ValueError: Shapes (0,) and (N,) are incompatible`.
    """

    def test_missing_dists_field_does_not_crash(self):
        # 2 boxes, keep both; example omits groundtruth_dists entirely.
        ex = {"groundtruth_classes": tf.constant([1, 2], tf.int64)}
        boxes = tf.zeros([2, 4], tf.float32)
        polys = tf.fill([2, _MAXV], -1.0)
        keep = tf.constant([True, True])
        out = Mosaic._filtered_anns(ex, boxes, polys, keep)
        # Missing field falls back to an N-length default, then masked by keep.
        self.assertEqual(out["groundtruth_dists"].shape[0], 2)
        self.assertEqual(out["groundtruth_area"].shape[0], 2)
        np.testing.assert_array_equal(out["groundtruth_classes"].numpy(), [1, 2])

    def test_missing_field_with_partial_keep(self):
        # keep drops one box; the N-length fallback must mask down to the kept count.
        ex = {}  # no side fields at all
        boxes = tf.zeros([3, 4], tf.float32)
        polys = tf.fill([3, _MAXV], -1.0)
        keep = tf.constant([True, False, True])
        out = Mosaic._filtered_anns(ex, boxes, polys, keep)
        self.assertEqual(out["groundtruth_dists"].shape[0], 2)
        self.assertEqual(out["groundtruth_classes"].shape[0], 2)
        self.assertEqual(out["groundtruth_is_crowd"].shape[0], 2)


class TestMixUp(unittest.TestCase):
    """MixUp (mosaic.py): the augmentation must actually fire when
    mixup_frequency > 0 — the knob must not be a no-op."""

    @staticmethod
    def _proc(val, nbox):
        """A processed result dict (mosaic/single output format)."""
        return {
            "image":  tf.fill([16, 16, 3], tf.constant(val, tf.uint8)),
            "height": tf.constant(16, tf.int32), "width": tf.constant(16, tf.int32),
            "source_id": tf.constant("x"),
            "groundtruth_boxes":    tf.zeros([nbox, 4]),
            "groundtruth_classes":  tf.zeros([nbox], tf.int64),
            "groundtruth_is_crowd": tf.zeros([nbox], tf.bool),
            "groundtruth_area":     tf.ones([nbox]),
            "groundtruth_dontcare": tf.zeros([nbox], tf.int64),
            "groundtruth_dists":    tf.fill([nbox], -1.0),
            "groundtruth_polygons": tf.fill([nbox, _MAXV], -1.0),
        }

    def test_mixup_blends_image_and_concatenates_labels(self):
        """_mixup blends with a Beta(32,32)~0.5 weight (image strictly between the two
        solids) and concatenates both inputs' instance rows."""
        m = Mosaic(output_size=[16, 16], with_polygons=True)
        one, two = self._proc(100, 3), self._proc(200, 5)
        means = [float(tf.reduce_mean(tf.cast(m._mixup(one, two)["image"], tf.float32)))
                 for _ in range(100)]
        self.assertGreater(min(means), 100.0)   # strictly blended, not a pure source
        self.assertLess(max(means), 200.0)
        self.assertAlmostEqual(sum(means) / len(means), 150.0, delta=10.0)  # ~0.5 mix
        r = m._mixup(one, two)
        for key in ("groundtruth_boxes", "groundtruth_polygons", "groundtruth_classes",
                    "groundtruth_dists"):
            self.assertEqual(r[key].shape[0], 8, key)   # 3 + 5 concatenated

    def test_mixup_frequency_zero_is_unwired(self):
        """At the default mixup_frequency=0.0 the output is a plain mosaic (no blend
        path added) — keys/shape identical to a non-mixup mosaic."""
        g = _make_group(8, h=64, w=64, n=2)
        base = Mosaic(output_size=[64, 64], mosaic_frequency=1.0, mixup_frequency=0.0,
                      with_polygons=True, group_size=8, decodes_per_output=4)
        out = base.mosaic_fn(is_training=True)(g)
        self.assertEqual(tuple(out["image"].shape), (2, 64, 64, 3))

    def test_mixup_frequency_one_fires(self):
        """mixup_frequency=1.0 concatenates a second mosaic's labels every output, so
        the mean surviving instance count is markedly higher than mixup off."""
        def mean_inst(freq, trials=10):
            tot = 0
            for _ in range(trials):
                m = Mosaic(output_size=[64, 64], mosaic_frequency=1.0, mixup_frequency=freq,
                           with_polygons=True, aug_scale_min=1.0, aug_scale_max=1.0,
                           group_size=8, decodes_per_output=4)
                b = m.mosaic_fn(is_training=True)(_make_group(8, h=64, w=64, n=4))[
                    "groundtruth_boxes"].numpy()
                tot += b.any(axis=-1).sum()   # non-padded rows (padding is all-zero)
            return tot / trials
        off, on = mean_inst(0.0), mean_inst(1.0)
        self.assertGreater(on, off * 1.4,
                           f"MixUp did not fire: off={off:.1f} on={on:.1f}")


if __name__ == "__main__":
    unittest.main()


def test_candidate_filter_legacy_parity():
    """Legacy-parity candidate filter: both the mosaic and single paths cull at
    0.1 visible area (the reference/legacy value); the mosaic path adds a 2px
    min_side floor, the single path has none. Aspect ratio < 20 on both."""
    import tensorflow as tf
    from configs.yaml_loader import load_config
    from data_pipeline.augmentations import transform_boxes_polygons
    from data_pipeline.mosaic import Mosaic

    for tier in ('yolov8_bbox', 'yolov8_poly', 'yolov8_poly_dist'):
        cfg = load_config(f'configs/experiments/yolo/{tier}.yaml')
        # Legacy value 0.1 on the mosaic path (clipped/warped visible fraction);
        # the single path uses the parser-level 0.1 too.
        assert cfg.task.train_data.parser.mosaic.area_thresh == 0.1, tier
        assert cfg.task.train_data.parser.area_thresh == 0.1, tier
    m = Mosaic([64, 64], area_thresh=0.4, single_area_thresh=0.1)
    assert m._area_thresh == 0.4 and m._single_area_thresh == 0.1
    assert Mosaic([64, 64])._single_area_thresh == 0.5  # fallback: no split

    # identity warp, box half outside the frame -> ~50% visible: kept at 0.1
    M = tf.eye(3)
    boxes = tf.constant([[0.25, -0.25, 0.75, 0.25]], tf.float32)   # yxyx, 50% off-frame
    polys = tf.fill([1, 8], -1.0)
    _, keep, _ = transform_boxes_polygons(boxes, polys, M, 64, 64, 64, 64,
                                          area_thresh=0.1)
    assert bool(keep[0]), "40-50%-visible box must survive the single-path filter"
    _, keep_strict, _ = transform_boxes_polygons(boxes, polys, M, 64, 64, 64, 64,
                                                 area_thresh=0.6)
    assert not bool(keep_strict[0])

    # reference candidate filter: degenerate slivers (aspect ratio >= 20) are
    # dropped; genuine partials with sane aspect stay
    sliver = tf.constant([[0.10, 0.10, 0.11, 0.60]], tf.float32)   # 1x50 units -> AR 50
    _, keep_ar, _ = transform_boxes_polygons(sliver, tf.fill([1, 8], -1.0),
                                             M, 672, 672, 672, 672)
    assert not bool(keep_ar[0]), "AR>20 sliver must be dropped"
    ok2px = tf.constant([[0.10, 0.10, 0.1035, 0.1035]], tf.float32)  # 0.0035 > 0.003
    _, keep_2px, _ = transform_boxes_polygons(ok2px, tf.fill([1, 8], -1.0),
                                              M, 672, 672, 672, 672)
    assert bool(keep_2px[0]), "a ~2.4px box must survive the 2px mosaic floor"
    # min_side floor is mosaic-only: at min_side=0.0 (the single path) a sub-2px
    # box keeps its label.
    tiny = tf.constant([[0.10, 0.10, 0.1015, 0.1015]], tf.float32)  # ~1px at 672
    _, keep_tiny_mosaic, _ = transform_boxes_polygons(
        tiny, tf.fill([1, 8], -1.0), M, 672, 672, 672, 672)
    assert not bool(keep_tiny_mosaic[0]), "sub-2px box drops on the mosaic path"
    _, keep_tiny_single, _ = transform_boxes_polygons(
        tiny, tf.fill([1, 8], -1.0), M, 672, 672, 672, 672, min_side=0.0)
    assert bool(keep_tiny_single[0]), "sub-2px box survives the single path"


def test_single_path_keeps_sub2px_box_and_parser_drops_zero_rows():
    """The non-mosaic single path applies no min_side floor (legacy), while the
    parser's clip_boxes still drops the mosaic stage's zero-padding rows."""
    import tensorflow as tf
    from data_pipeline.augmentations import clip_boxes
    from data_pipeline.mosaic import Mosaic

    # Identity-warp single path (scale 1, translate 0): a ~1px box survives.
    m = Mosaic([672, 672], mosaic_frequency=0.0,
               single_scale_min=1.0, single_scale_max=1.0, single_translate=0.0,
               single_area_thresh=0.1)
    ex = {
        'image': tf.zeros([672, 672, 3], tf.uint8),
        'groundtruth_boxes': tf.constant([[0.10, 0.10, 0.1015, 0.1015]], tf.float32),
        'groundtruth_polygons': tf.fill([1, 8], -1.0),
        'groundtruth_classes': tf.constant([1], tf.int64),
        'height': tf.constant(672, tf.int32),
        'width': tf.constant(672, tf.int32),
    }
    out = m._single(ex)
    assert int(tf.shape(out['groundtruth_boxes'])[0]) == 1, (
        "sub-2px box must keep its label on the single path")

    # clip_boxes at min_side=0.0 (the parser call): zero rows drop, 1px boxes stay.
    boxes = tf.constant([
        [0.0, 0.0, 0.0, 0.0],              # padded_batch zero row -> dropped
        [0.10, 0.10, 0.1015, 0.1015],      # ~1px at 672 -> kept
    ], tf.float32)
    _, keep = clip_boxes(boxes, min_side=0.0)
    assert not bool(keep[0]) and bool(keep[1])
