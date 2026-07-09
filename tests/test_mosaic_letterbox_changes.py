"""Tests for the legacy-parity augmentation changes:

  1. Letterbox pre-resize (shared augmentations.letterbox_resize) — exact
     coordinate transform, sentinel preservation, gray-114 padding.
  2. Single (non-mosaic) path uses the letterboxed image, so border objects on a
     non-square image survive the pre-resize + translate (were expelled under the
     old squash pre-resize).
  3. Rotation parity — the mosaic canvas never rotates even with degrees set; the
     single path rotates only when enabled (parser rotate / rotate_degrees).
  4. Copy-paste runs INSIDE the mosaic branch (mosaic tiles only): singles never
     receive pastes, and cnp_* fields never leak into the emitted example.
"""

import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.augmentations import letterbox_resize
from data_pipeline.copy_paste import CopyAndPasteModule
from data_pipeline.mosaic import Mosaic


_MAXV = 8


# ---------------------------------------------------------------------------
# 1. Shared letterbox
# ---------------------------------------------------------------------------

class TestSharedLetterbox(unittest.TestCase):
    def test_exact_coords_and_gray_and_sentinel(self):
        """40h x 80w -> 64x64. scale=min(64/40,64/80)=0.8, new_h=32 new_w=64,
        pad_top=(64-32)//2=16 pad_left=0. Content occupies rows [16,48)."""
        img = tf.constant(np.full((40, 80, 3), 200, np.uint8))
        boxes = tf.constant([[0.0, 0.2, 1.0, 0.8]], tf.float32)   # spans full height
        polys = tf.constant([[0.5, 0.0, -1.0, -1.0]], tf.float32)  # vertex (x=0.5,y=0)+pad

        out_img, out_boxes, out_polys = letterbox_resize(img, boxes, polys, 64, 64)
        self.assertEqual(tuple(out_img.shape), (64, 64, 3))

        arr = out_img.numpy()
        # Gray-114 padding on the padded (top/bottom) bands; content elsewhere.
        self.assertTrue((arr[0:16, :, :] == 114).all(), "top pad not gray 114")
        self.assertTrue((arr[48:64, :, :] == 114).all(), "bottom pad not gray 114")
        self.assertTrue((arr[16:48, :, :] == 200).all(), "content band corrupted")

        # box: ymin = 0*32/64 + 16/64 = 0.25 ; ymax = 1*32/64 + 16/64 = 0.75
        #      xmin = 0.2*64/64 + 0 = 0.2 ; xmax = 0.8
        np.testing.assert_allclose(out_boxes.numpy()[0], [0.25, 0.2, 0.75, 0.8], atol=1e-6)
        # vertex: x = 0.5*64/64 = 0.5 ; y = 0.0*32/64 + 16/64 = 0.25 ; sentinel kept
        op = out_polys.numpy()[0]
        np.testing.assert_allclose(op[0:2], [0.5, 0.25], atol=1e-6)
        np.testing.assert_array_equal(op[2:4], [-1.0, -1.0])

    def test_gray_value_matches_eval_parser(self):
        """The shared function and the eval parser must use the same gray fill."""
        from data_pipeline.yolo_parser import V8ParserExtended
        parser = V8ParserExtended(
            output_size=[64, 64], expanded_strides={"3": 8, "4": 16, "5": 32},
            levels=["3", "4", "5"], angle_step=15)
        img = tf.constant(np.full((40, 80, 3), 200, np.uint8))
        boxes = tf.constant([[0.0, 0.2, 1.0, 0.8]], tf.float32)
        polys = tf.constant([[0.5, 0.0, -1.0, -1.0]], tf.float32)
        pim, pb, pp = parser._letterbox_resize(img, boxes, polys)
        sim, sb, sp = letterbox_resize(img, boxes, polys, 64, 64)
        np.testing.assert_array_equal(pim.numpy(), sim.numpy())
        np.testing.assert_allclose(pb.numpy(), sb.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Single path: border object survives letterbox where squash expelled it
# ---------------------------------------------------------------------------

def _single_mosaic(single_translate=0.1, **kw):
    return Mosaic(
        output_size=[64, 64], mosaic_frequency=0.0, with_polygons=True,
        aug_scale_min=1.0, aug_scale_max=1.0,
        single_scale_min=1.0, single_scale_max=1.0, single_translate=single_translate,
        single_area_thresh=0.0, area_thresh=0.0,
        degrees=0.0, shear=0.0, perspective=0.0, translate=0.0,
        mosaic_center=0.0, group_size=4, decodes_per_output=1, **kw)


class TestSinglePathBorderSurvival(unittest.TestCase):
    def _ex(self, image, boxes, h, w):
        return {
            "image": image,
            "height": tf.constant(h, tf.int32),
            "width": tf.constant(w, tf.int32),
            "groundtruth_boxes": boxes,
            "groundtruth_polygons": tf.fill([tf.shape(boxes)[0], _MAXV], -1.0),
            "groundtruth_classes": tf.zeros([tf.shape(boxes)[0]], tf.int64),
            "groundtruth_is_crowd": tf.zeros([tf.shape(boxes)[0]], tf.bool),
            "groundtruth_area": tf.ones([tf.shape(boxes)[0]], tf.float32),
            "groundtruth_dontcare": tf.zeros([tf.shape(boxes)[0]], tf.int64),
            "groundtruth_dists": tf.fill([tf.shape(boxes)[0]], -1.0),
            "source_id": tf.constant("x"),
        }

    def _keep_rate(self, ex):
        m = _single_mosaic()
        fn = lambda: m._single(ex)
        kept = 0
        trials = 24
        for seed in range(trials):
            tf.random.set_seed(seed)
            b = fn()["groundtruth_boxes"].numpy()
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            kept += int((area > 1e-9).any())
        return kept / trials

    def test_letterbox_keeps_top_edge_box_squash_expels(self):
        # Thin box hugging the TOP edge of a non-square (40h x 80w) image.
        raw = tf.constant(np.full((40, 80, 3), 180, np.uint8))
        boxes = tf.constant([[0.0, 0.3, 0.05, 0.7]], tf.float32)

        # Letterbox pre-resize (the new behavior): the top axis is padded, so the
        # box is inset from the output border and survives the +/-0.1 translate.
        lb_img, lb_boxes, lb_polys = letterbox_resize(raw, boxes,
                                                      tf.fill([1, _MAXV], -1.0), 64, 64)
        lb_ex = self._ex(lb_img, lb_boxes, 40, 80)
        lb_ex["groundtruth_polygons"] = lb_polys
        self.assertGreater(self._keep_rate(lb_ex), 0.99,
                           "letterbox border box must survive every translate")

        # Old squash pre-resize: box stays glued to y=0, so an upward translate
        # expels it -> keep rate strictly below 1.
        sq_img = tf.cast(tf.image.resize(tf.cast(raw, tf.float32), [64, 64]), tf.uint8)
        sq_ex = self._ex(sq_img, boxes, 64, 64)
        self.assertLess(self._keep_rate(sq_ex), 1.0,
                        "squash edge box should be expelled by some translates")


# ---------------------------------------------------------------------------
# 3. Rotation parity
# ---------------------------------------------------------------------------

def _solid(color, h=32, w=32, box=None):
    box = box if box is not None else [0.25, 0.25, 0.75, 0.75]
    return {
        "image": tf.fill([h, w, 3], tf.constant(color, tf.uint8)),
        "height": tf.constant(h, tf.int32),
        "width": tf.constant(w, tf.int32),
        "groundtruth_boxes": tf.constant([box], tf.float32),
        "groundtruth_classes": tf.zeros([1], tf.int64),
        "groundtruth_is_crowd": tf.zeros([1], tf.bool),
        "groundtruth_area": tf.ones([1], tf.float32),
        "groundtruth_dontcare": tf.zeros([1], tf.int64),
        "groundtruth_dists": tf.fill([1], -1.0),
        "groundtruth_polygons": tf.constant([[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0]],
                                            tf.float32),
        "source_id": tf.constant("x"),
    }


class TestMosaicRotationParity(unittest.TestCase):
    def test_mosaic_never_rotates_even_with_degrees(self):
        """Even with degrees=90/rotate_prob=1.0 the mosaic canvas warp forces 0
        rotation: identity scale/translate/shear + center=0 gives a pure center-crop,
        so the 4 solid sources land axis-aligned in their quadrants (a real rotation
        would smear colors across the quadrant diagonals)."""
        H = W = 32
        m = Mosaic(
            output_size=[H, W], mosaic_frequency=1.0, with_polygons=True,
            aug_scale_min=1.0, aug_scale_max=1.0,
            degrees=90.0, rotate_prob=1.0, shear=0.0, perspective=0.0, translate=0.0,
            mosaic_center=0.0, area_thresh=0.0, group_size=4, decodes_per_output=1)
        colors = [40, 80, 160, 220]
        exs = [_solid(colors[i], H, W) for i in range(4)]
        for seed in range(8):
            tf.random.set_seed(seed)
            img = m._mosaic(*exs)["image"].numpy()
            hh, hw, b = H // 2, W // 2, 2
            quad = {(0, 0): colors[0], (0, 1): colors[1],
                    (1, 0): colors[2], (1, 1): colors[3]}
            for (qy, qx), color in quad.items():
                y0 = qy * hh + (b if qy == 0 else 0)
                y1 = (qy + 1) * hh - (0 if qy == 0 else b)
                x0 = qx * hw + (b if qx == 0 else 0)
                x1 = (qx + 1) * hw - (0 if qx == 0 else b)
                region = img[y0:y1, x0:x1]
                self.assertTrue((region == color).all(),
                                f"seed {seed} quad ({qy},{qx}) rotated: {np.unique(region)}")


class TestSinglePathRotation(unittest.TestCase):
    def test_disabled_by_default_is_upright(self):
        # No single_rotate -> the single path is the identity passthrough here.
        m = _single_mosaic()
        tf.random.set_seed(0)
        out = m._single(_solid(50, 64, 64, box=[0.4, 0.1, 0.6, 0.9]))
        # translate is 0.1 but with seed fixed we only assert the box stays a wide
        # box (never rotated to tall).
        b = out["groundtruth_boxes"].numpy()[0]
        self.assertGreater(b[3] - b[1], b[2] - b[0], "unexpected rotation with rotate off")

    def test_enabled_rotates_box_and_preserves_center(self):
        # single_rotate with degrees=90: a centered box's AABB stays centered on
        # (0.5,0.5) under any rotation, and a WIDE box becomes TALL for near-90
        # draws (found across seeds). translate off so only rotation moves geometry.
        m = _single_mosaic(single_rotate=True, single_rotate_degrees=90.0,
                           single_translate=0.0)
        saw_tall = False
        for seed in range(30):
            tf.random.set_seed(seed)
            out = m._single(_solid(50, 64, 64, box=[0.45, 0.1, 0.55, 0.9]))
            b = out["groundtruth_boxes"].numpy()[0]
            cy, cx = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            self.assertAlmostEqual(cy, 0.5, delta=0.06, msg=f"seed {seed} y-center moved")
            self.assertAlmostEqual(cx, 0.5, delta=0.06, msg=f"seed {seed} x-center moved")
            if (b[2] - b[0]) > (b[3] - b[1]) + 0.2:
                saw_tall = True
        self.assertTrue(saw_tall, "no near-90 rotation observed across seeds")


# ---------------------------------------------------------------------------
# 4. Copy-paste inside the mosaic (mosaic-only, per tile)
# ---------------------------------------------------------------------------

def _group_with_cnp(G, h=64, w=64, obj=48):
    box = tf.constant([[0.25, 0.25, 0.75, 0.75]], tf.float32)
    poly = tf.constant([[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0]], tf.float32)
    imgs = tf.stack([tf.fill([h, w, 3], tf.constant(i % 256, tf.uint8)) for i in range(G)])
    rgba = np.full((obj, obj, 4), 200, np.uint8)
    rgba[..., 3] = 255
    return {
        "image": imgs,
        "height": tf.constant([h] * G, tf.int32),
        "width": tf.constant([w] * G, tf.int32),
        "groundtruth_boxes": tf.stack([box] * G),
        "groundtruth_classes": tf.zeros([G, 1], tf.int64),
        "groundtruth_is_crowd": tf.zeros([G, 1], tf.bool),
        "groundtruth_area": tf.ones([G, 1], tf.float32),
        "groundtruth_dontcare": tf.zeros([G, 1], tf.int64),
        "groundtruth_dists": tf.fill([G, 1], tf.constant(-1.0)),
        "groundtruth_polygons": tf.stack([poly] * G),
        "source_id": tf.constant([str(i) for i in range(G)]),
        # cnp_* ride-along fields (as input_reader._merge_cnp_fields attaches them).
        "cnp_image": tf.stack([tf.constant(rgba)] * G),
        "cnp_orig_bbox": tf.stack([tf.constant([0.1, 0.1, 0.9, 0.9], tf.float32)] * G),
        "cnp_label": tf.constant([7] * G, tf.int64),
        "cnp_points": tf.stack([tf.constant([0.2, 0.2, 0.8, 0.2, 0.5, 0.8], tf.float32)] * G),
        "cnp_h": tf.constant([obj] * G, tf.int32),
        "cnp_w": tf.constant([obj] * G, tf.int32),
    }


def _n_real(boxes_2d):
    area = (boxes_2d[:, 2] - boxes_2d[:, 0]) * (boxes_2d[:, 3] - boxes_2d[:, 1])
    return int((area > 1e-9).sum())


class TestCopyPasteInsideMosaic(unittest.TestCase):
    def _mod(self, prob):
        return CopyAndPasteModule(prob=prob, min_height=0, min_width=0,
                                  min_resize_ratio=1.0, max_resize_ratio=1.0)

    def _mosaic(self, freq, prob):
        return Mosaic(
            output_size=[64, 64], mosaic_frequency=freq, with_polygons=True,
            aug_scale_min=1.0, aug_scale_max=1.0,
            degrees=0.0, shear=0.0, perspective=0.0, translate=0.0,
            mosaic_center=0.0, area_thresh=0.0, group_size=8, decodes_per_output=4,
            copy_paste_module=self._mod(prob))

    def test_singles_never_get_pastes(self):
        """freq=0 -> every output is a single; the single path ignores cnp, so no
        output ever gains an extra pasted box (real box count stays 1)."""
        m = Mosaic(
            output_size=[64, 64], mosaic_frequency=0.0, with_polygons=True,
            aug_scale_min=1.0, aug_scale_max=1.0,
            single_scale_min=1.0, single_scale_max=1.0, single_translate=0.0,
            single_area_thresh=0.0, area_thresh=0.0,
            degrees=0.0, shear=0.0, perspective=0.0, translate=0.0,
            mosaic_center=0.0, group_size=8, decodes_per_output=1,
            copy_paste_module=self._mod(1.0))  # prob 1.0 — would paste if reached
        out = m.mosaic_fn(is_training=True)(_group_with_cnp(8))
        boxes = out["groundtruth_boxes"].numpy()
        for i in range(boxes.shape[0]):
            self.assertEqual(_n_real(boxes[i]), 1, f"single {i} unexpectedly pasted")

    def test_mosaic_tiles_paste_at_probability(self):
        """prob=1 pastes on every tile -> markedly more surviving boxes than prob=0."""
        def mean_real(prob, trials=8):
            tot = 0
            for t in range(trials):
                tf.random.set_seed(t)
                out = self._mosaic(1.0, prob).mosaic_fn(is_training=True)(_group_with_cnp(8))
                b = out["groundtruth_boxes"].numpy()
                tot += sum(_n_real(b[i]) for i in range(b.shape[0]))
            return tot / trials
        off, on = mean_real(0.0), mean_real(1.0)
        self.assertGreater(on, off + 1.0, f"paste did not fire: off={off} on={on}")

    def test_cnp_keys_absent_from_output(self):
        out = self._mosaic(1.0, 1.0).mosaic_fn(is_training=True)(_group_with_cnp(8))
        for k in out:
            self.assertFalse(k.startswith("cnp_"), f"cnp field {k} leaked to output")

    def test_paste_prob_zero_deterministic_no_paste(self):
        out = self._mosaic(1.0, 0.0).mosaic_fn(is_training=True)(_group_with_cnp(8))
        b = out["groundtruth_boxes"].numpy()
        # No paste at all -> at most the 4 source boxes per mosaic output survive.
        for i in range(b.shape[0]):
            self.assertLessEqual(_n_real(b[i]), 4)


if __name__ == "__main__":
    unittest.main()
