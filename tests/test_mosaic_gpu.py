"""Correctness tests for the GPU-offload mosaic (data_pipeline/mosaic_gpu.py).

The whole point of the offload is that it changes WHERE the warp runs, not WHAT it
produces. These tests pin that the CPU prepare (``_mosaic_prepare`` /
``_single_prepare``) + the deferred GPU warp (``gpu_mosaic_warp``) is BYTE-IDENTICAL
to the stock CPU mosaic (``Mosaic._mosaic`` / ``Mosaic._single``) image and produces
identical labels — for both the mosaic and the single-image branch — plus the flip
split (label flip on the CPU, image flip on the GPU under the same coin) and the
end-to-end ``mosaic_prepare_fn`` payload shape.

Byte-identity holds because ``_mosaic``/``_single`` and their prepare counterparts
both draw the perspective matrix ``M`` through the SAME calls in the SAME order
(``_mosaic_canvas_M`` / ``make_perspective_matrix``), so seeding identically yields
the same ``M``; the GPU warp then feeds the exact inverse-transform vector the CPU
path feeds ``apply_perspective_image``. The op runs on whatever device is present
(CPU here); GPU vs CPU execution of ImageProjectiveTransformV3 is numerically the
same, so these tests are valid without a GPU.
"""

import unittest

import numpy as np
import tensorflow as tf

from data_pipeline.mosaic import Mosaic
from data_pipeline.mosaic_gpu import (
    CANVAS_KEY,
    FLIP_KEY,
    WARP_KEY,
    _flip_labels,
    _mosaic_prepare,
    _single_prepare,
    gpu_mosaic_warp,
    mosaic_prepare_fn,
)


def _solid_example(color, h=64, w=64, n=1):
    """One example: a solid-color HxW image + n boxes/polys (matches test_mosaic)."""
    box = tf.constant([[0.25, 0.25, 0.75, 0.75]] * n, tf.float32)
    poly = tf.constant([[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0]] * n, tf.float32)
    return {
        "image":  tf.fill([h, w, 3], tf.constant(color, tf.uint8)),
        "height": tf.constant(h, tf.int32),
        "width":  tf.constant(w, tf.int32),
        "groundtruth_boxes":    box,
        "groundtruth_classes":  tf.zeros([n], tf.int64),
        "groundtruth_is_crowd": tf.zeros([n], tf.bool),
        "groundtruth_area":     tf.ones([n], tf.float32),
        "groundtruth_dontcare": tf.zeros([n], tf.int64),
        "groundtruth_dists":    tf.fill([n], tf.constant(-1.0)),
        "groundtruth_polygons": poly,
        "source_id":            tf.constant("x"),
    }


def _make_group(G, h=64, w=64, n=1):
    """Group-of-G dict (leading dim G) with identifiable per-image colors."""
    box = tf.constant([[0.25, 0.25, 0.75, 0.75]] * n, tf.float32)
    poly = tf.constant([[0.3, 0.3, 0.6, 0.6, -1.0, -1.0, -1.0, -1.0]] * n, tf.float32)
    images = tf.stack([tf.fill([h, w, 3], tf.constant(20 + 30 * i, tf.uint8)) for i in range(G)])
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


def _mosaic(out=64, seed_scale=(0.6, 1.4), degrees=8.0, center=0.25):
    return Mosaic(
        output_size=[out, out], mosaic_frequency=1.0, with_polygons=True,
        aug_scale_min=seed_scale[0], aug_scale_max=seed_scale[1],
        degrees=degrees, shear=2.0, perspective=0.0, translate=0.1,
        mosaic_center=center, area_thresh=0.0,
        group_size=8, decodes_per_output=4,
    )


def _warp_one(canvas, warp, H, W, flip=False):
    """Run gpu_mosaic_warp on a single (canvas, warp) → [H, W, 3] uint8."""
    out = gpu_mosaic_warp(
        canvas[tf.newaxis], warp[tf.newaxis],
        tf.constant([flip]), H, W,
    )
    return out[0]


class TestMosaicGpuByteIdentity(unittest.TestCase):

    def test_mosaic_image_byte_identical(self):
        """CPU _mosaic image == prepare-canvas + GPU warp image (same M via same seed)."""
        H = W = 64
        m = _mosaic(out=H)
        exs = [_solid_example(20 + 30 * i, H, W) for i in range(4)]

        tf.random.set_seed(123)
        cpu = m._mosaic(*exs)
        img_cpu = cpu["image"].numpy()

        tf.random.set_seed(123)
        prep = _mosaic_prepare(m, exs)
        img_gpu = _warp_one(prep[CANVAS_KEY], prep[WARP_KEY], H, W).numpy()

        self.assertEqual(img_cpu.shape, (H, W, 3))
        self.assertEqual(prep[CANVAS_KEY].shape, (2 * H, 2 * W, 3))
        self.assertEqual(prep[WARP_KEY].shape, (8,))
        np.testing.assert_array_equal(img_cpu, img_gpu)

    def test_mosaic_labels_identical(self):
        """CPU _mosaic and _mosaic_prepare produce identical (pre-flip) labels."""
        H = W = 64
        m = _mosaic(out=H)
        exs = [_solid_example(20 + 30 * i, H, W) for i in range(4)]

        tf.random.set_seed(7)
        cpu = m._mosaic(*exs)
        tf.random.set_seed(7)
        prep = _mosaic_prepare(m, exs)

        np.testing.assert_array_equal(
            cpu["groundtruth_boxes"].numpy(), prep["groundtruth_boxes"].numpy()
        )
        np.testing.assert_array_equal(
            cpu["groundtruth_polygons"].numpy(), prep["groundtruth_polygons"].numpy()
        )
        np.testing.assert_array_equal(
            cpu["groundtruth_classes"].numpy(), prep["groundtruth_classes"].numpy()
        )

    def test_single_image_byte_identical(self):
        """CPU _single image == padded-canvas + GPU warp image (single branch)."""
        H = W = 64
        m = _mosaic(out=H)
        ex = _solid_example(200, H, W)

        tf.random.set_seed(55)
        cpu = m._single(ex)
        img_cpu = cpu["image"].numpy()

        tf.random.set_seed(55)
        prep = _single_prepare(m, ex)
        img_gpu = _warp_one(prep[CANVAS_KEY], prep[WARP_KEY], H, W).numpy()

        self.assertEqual(prep[CANVAS_KEY].shape, (2 * H, 2 * W, 3))
        np.testing.assert_array_equal(img_cpu, img_gpu)

    def test_single_labels_identical(self):
        H = W = 64
        m = _mosaic(out=H)
        ex = _solid_example(200, H, W)

        tf.random.set_seed(9)
        cpu = m._single(ex)
        tf.random.set_seed(9)
        prep = _single_prepare(m, ex)
        np.testing.assert_array_equal(
            cpu["groundtruth_boxes"].numpy(), prep["groundtruth_boxes"].numpy()
        )
        np.testing.assert_array_equal(
            cpu["groundtruth_polygons"].numpy(), prep["groundtruth_polygons"].numpy()
        )


class TestGpuFlip(unittest.TestCase):

    def test_image_flip_matches_flip_left_right(self):
        """gpu_mosaic_warp(flip=True) == flip_left_right of the unflipped warp."""
        H = W = 64
        m = _mosaic(out=H)
        exs = [_solid_example(20 + 30 * i, H, W) for i in range(4)]
        tf.random.set_seed(3)
        prep = _mosaic_prepare(m, exs)

        noflip = _warp_one(prep[CANVAS_KEY], prep[WARP_KEY], H, W, flip=False).numpy()
        yesflip = _warp_one(prep[CANVAS_KEY], prep[WARP_KEY], H, W, flip=True).numpy()
        np.testing.assert_array_equal(yesflip, np.flip(noflip, axis=1))

    def test_label_flip_matches_normalized_rule(self):
        """_flip_labels flips boxes (xmin↔1-xmax) and polygon valid-vertex x↔1-x."""
        boxes = tf.constant([[0.2, 0.1, 0.6, 0.4]], tf.float32)
        polys = tf.constant([[0.1, 0.5, 0.9, 0.5, -1.0, -1.0]], tf.float32)
        bf, pf = _flip_labels(boxes, polys, tf.constant(True))
        np.testing.assert_allclose(bf.numpy(), [[0.2, 0.6, 0.6, 0.9]], atol=1e-6)
        # x flipped (1-x) for valid vertices; y unchanged; -1 sentinel preserved.
        np.testing.assert_allclose(
            pf.numpy(), [[0.9, 0.5, 0.1, 0.5, -1.0, -1.0]], atol=1e-6
        )

    def test_label_flip_noop_when_false(self):
        boxes = tf.constant([[0.2, 0.1, 0.6, 0.4]], tf.float32)
        polys = tf.constant([[0.1, 0.5, -1.0, -1.0]], tf.float32)
        bf, pf = _flip_labels(boxes, polys, tf.constant(False))
        np.testing.assert_array_equal(bf.numpy(), boxes.numpy())
        np.testing.assert_array_equal(pf.numpy(), polys.numpy())


class TestPrepareFnPipeline(unittest.TestCase):

    def test_prepare_fn_payload_shapes(self):
        """padded_batch(G) → mosaic_prepare_fn → emits G//R samples carrying
        canvas[2H,2W,3] + warp[8] + flip[] + final labels."""
        H = W = 64
        G, R = 8, 4
        m = _mosaic(out=H)
        out = mosaic_prepare_fn(m, random_flip=True)(_make_group(G, H, W))

        P = G // R
        self.assertEqual(tuple(out[CANVAS_KEY].shape), (P, 2 * H, 2 * W, 3))
        self.assertEqual(tuple(out[WARP_KEY].shape), (P, 8))
        self.assertEqual(tuple(out[FLIP_KEY].shape), (P,))
        self.assertEqual(out[CANVAS_KEY].dtype, tf.uint8)
        self.assertEqual(out[FLIP_KEY].dtype, tf.bool)
        # Labels present and batched to P.
        self.assertEqual(int(out["groundtruth_boxes"].shape[0]), P)

    def test_prepare_fn_warp_produces_valid_images(self):
        """The emitted canvas/warp/flip warp to a clean [P, H, W, 3] uint8 batch."""
        H = W = 64
        G, R = 8, 4
        m = _mosaic(out=H)
        out = mosaic_prepare_fn(m, random_flip=True)(_make_group(G, H, W))
        imgs = gpu_mosaic_warp(out[CANVAS_KEY], out[WARP_KEY], out[FLIP_KEY], H, W)
        self.assertEqual(tuple(imgs.shape), (G // R, H, W, 3))
        self.assertEqual(imgs.dtype, tf.uint8)

    def test_prepare_fn_traceable_in_tf_data(self):
        """Runs as a real tf.data map (graph mode) → unbatch → P elements."""
        H = W = 64
        G, R = 8, 4
        m = _mosaic(out=H)
        ds = (tf.data.Dataset.from_tensor_slices(_make_group(G, H, W))
              .padded_batch(G, drop_remainder=True)
              .map(mosaic_prepare_fn(m, random_flip=True))
              .unbatch())
        self.assertEqual(ds.element_spec[CANVAS_KEY].shape.as_list(), [2 * H, 2 * W, 3])
        self.assertEqual(sum(1 for _ in ds), G // R)


if __name__ == "__main__":
    unittest.main()
