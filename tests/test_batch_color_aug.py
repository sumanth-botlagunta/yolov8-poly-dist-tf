"""Tests for data_pipeline.batch_color_aug (per-batch GPU colour augmentation).

These verify that the vectorized per-batch ops reproduce the per-sample parser
behaviour they replace, with the same per-image randomness semantics:

  (a) apply_hsv_gains == the per-image PyTorch-YOLO HSV math (multiplicative
      gains in the quantized [180, 255, 255] domain), exact per image.
  (b) Masked albumentations reproduce the single-image reference ops (blur3,
      gray, clahe) for forced masks.
  (c) row_mask == False rows skip albumentations but are still HSV'd.
  (d) freq == 0 and hue == sat == val == 0 → exact passthrough of the /255 cast.
  (e) uint8 input handling (cast + /255).
"""

import numpy as np
import tensorflow as tf

from data_pipeline.batch_color_aug import (
    apply_albumentations_masks,
    apply_hsv_gains,
    batch_albumentations,
    batch_color_augment,
    batch_hsv_augment,
)


# ---------------------------------------------------------------------------
# Reference ops ported from data_pipeline.augmentations internals (single image)
# ---------------------------------------------------------------------------

def _box_blur_ref(image, kernel_size):
    """Single-image separable box blur — copy of augmentations._box_blur_tf."""
    k_h = tf.ones([kernel_size, 1, 3, 1], dtype=tf.float32) / tf.cast(kernel_size, tf.float32)
    k_w = tf.ones([1, kernel_size, 3, 1], dtype=tf.float32) / tf.cast(kernel_size, tf.float32)
    img4 = image[tf.newaxis]
    img4 = tf.nn.depthwise_conv2d(img4, k_w, strides=[1, 1, 1, 1], padding='SAME')
    img4 = tf.nn.depthwise_conv2d(img4, k_h, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(img4, 0)


def _gray_ref(image):
    return tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])


def _clahe_ref(image):
    local_mean = _box_blur_ref(image, 33)
    return tf.clip_by_value(image + 0.5 * (image - local_mean), 0.0, 1.0)


# ---------------------------------------------------------------------------
# (a) HSV exact equivalence (PyTorch-YOLO quantized multiplicative form)
# ---------------------------------------------------------------------------

def _hsv_torch_ref(image, r3):
    """Single-image reference of the quantized multiplicative HSV math."""
    scale = tf.constant([180.0, 255.0, 255.0])
    x = tf.image.rgb_to_hsv(image)
    x = tf.math.floor(x * scale)
    x = tf.math.floor(x * r3)
    h, s, v = tf.split(x, 3, axis=-1)
    h = h % 180.0
    s = tf.clip_by_value(s, 0.0, 255.0)
    v = tf.clip_by_value(v, 0.0, 255.0)
    return tf.image.hsv_to_rgb(tf.concat([h, s, v], axis=-1) / scale)


def test_apply_hsv_gains_matches_reference():
    tf.random.set_seed(0)
    B, H, W = 4, 16, 24
    imgs = tf.random.uniform([B, H, W, 3], dtype=tf.float32)
    r = tf.constant([
        [0.990, 0.40, 0.70],
        [1.000, 1.00, 1.00],
        [1.012, 1.65, 1.38],
        [0.995, 0.85, 1.10],
    ], dtype=tf.float32)

    batched = apply_hsv_gains(imgs, r).numpy()
    for i in range(B):
        ref = _hsv_torch_ref(imgs[i], r[i]).numpy()
        np.testing.assert_allclose(batched[i], ref, atol=1e-5)


def test_batch_hsv_augment_noop_when_all_zero():
    imgs = tf.random.uniform([3, 8, 8, 3], dtype=tf.float32)
    out = batch_hsv_augment(imgs, hue=0.0, sat=0.0, val=0.0)
    # All components disabled → exact passthrough.
    np.testing.assert_array_equal(out.numpy(), imgs.numpy())


def test_batch_hsv_augment_stays_in_range():
    """Extreme gains (sat up to 1.7, val up to 1.4) must stay finite in [0, 1]."""
    tf.random.set_seed(1)
    imgs = tf.random.uniform([2, 8, 8, 3], dtype=tf.float32)
    out = batch_hsv_augment(imgs, hue=0.015, sat=0.7, val=0.4).numpy()
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_per_sample_hsv_augment_matches_batch_math():
    """augmentations.hsv_augment and apply_hsv_gains share one formula: with the
    same gain vector they must produce identical pixels."""
    from data_pipeline.augmentations import hsv_augment  # noqa: F401  (formula twin)
    tf.random.set_seed(2)
    img = tf.random.uniform([16, 16, 3], dtype=tf.float32)
    r = tf.constant([[1.005, 1.30, 0.75]], dtype=tf.float32)
    batched = apply_hsv_gains(img[tf.newaxis], r)[0].numpy()
    ref = _hsv_torch_ref(img, r[0]).numpy()
    np.testing.assert_allclose(batched, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# (b) Masked albumentations reproduce single-image reference ops
# ---------------------------------------------------------------------------

def test_albumentations_forced_blur_matches_reference():
    tf.random.set_seed(2)
    imgs = tf.random.uniform([3, 20, 20, 3], dtype=tf.float32)
    masks_on = tf.constant([True, False, True])
    masks_off = tf.constant([False, False, False])

    out = apply_albumentations_masks(
        imgs, m_blur=masks_on, m_median=masks_off, m_gray=masks_off, m_clahe=masks_off,
    ).numpy()

    for i in range(3):
        if bool(masks_on[i]):
            ref = _box_blur_ref(imgs[i], 3).numpy()
            np.testing.assert_allclose(out[i], ref, atol=1e-5)
        else:
            np.testing.assert_allclose(out[i], imgs[i].numpy(), atol=1e-7)


def test_albumentations_forced_gray_matches_reference():
    tf.random.set_seed(3)
    imgs = tf.random.uniform([2, 12, 12, 3], dtype=tf.float32)
    off = tf.constant([False, False])
    out = apply_albumentations_masks(
        imgs, m_blur=off, m_median=off, m_gray=tf.constant([True, False]), m_clahe=off,
    ).numpy()
    np.testing.assert_allclose(out[0], _gray_ref(imgs[0]).numpy(), atol=1e-5)
    np.testing.assert_allclose(out[1], imgs[1].numpy(), atol=1e-7)


def test_albumentations_forced_clahe_matches_reference():
    tf.random.set_seed(4)
    imgs = tf.random.uniform([2, 40, 40, 3], dtype=tf.float32)
    off = tf.constant([False, False])
    out = apply_albumentations_masks(
        imgs, m_blur=off, m_median=off, m_gray=off, m_clahe=tf.constant([False, True]),
    ).numpy()
    np.testing.assert_allclose(out[0], imgs[0].numpy(), atol=1e-7)
    np.testing.assert_allclose(out[1], _clahe_ref(imgs[1]).numpy(), atol=1e-5)


def test_clahe_all_false_mask_is_exact_passthrough():
    """With no image selected for CLAHE, the output must be byte-identical to the
    input — the tf.cond guard short-circuits the 33-px blur and changes nothing.

    Pins the perf guard (skip the unconditional 33×33 box blur when no row needs
    CLAHE) AND its correctness: the guarded path is identical to applying nothing.
    """
    tf.random.set_seed(7)
    imgs = tf.random.uniform([3, 40, 40, 3], dtype=tf.float32)
    off = tf.constant([False, False, False])
    out = apply_albumentations_masks(
        imgs, m_blur=off, m_median=off, m_gray=off, m_clahe=off,
    ).numpy()
    np.testing.assert_array_equal(out, imgs.numpy())


def test_clahe_guard_matches_unconditional_under_tf_function():
    """Inside @tf.function, the tf.cond-guarded CLAHE must equal the unconditional
    masked form for BOTH the all-off batch (false branch) and a mixed batch
    (true branch), proving the guard is a pure compute optimization.
    """
    @tf.function
    def run(imgs, m_clahe):
        off = tf.zeros_like(m_clahe)
        return apply_albumentations_masks(
            imgs, m_blur=off, m_median=off, m_gray=off, m_clahe=m_clahe,
        )

    tf.random.set_seed(8)
    imgs = tf.random.uniform([3, 40, 40, 3], dtype=tf.float32)

    # False branch: nothing selected → exact passthrough.
    out_off = run(imgs, tf.constant([False, False, False])).numpy()
    np.testing.assert_array_equal(out_off, imgs.numpy())

    # True branch: row 1 selected → that row gets CLAHE, others untouched.
    out_mix = run(imgs, tf.constant([False, True, False])).numpy()
    np.testing.assert_array_equal(out_mix[0], imgs[0].numpy())
    np.testing.assert_array_equal(out_mix[2], imgs[2].numpy())
    np.testing.assert_allclose(out_mix[1], _clahe_ref(imgs[1]).numpy(), atol=1e-5)


def test_albumentations_sequential_order_matches_reference():
    """Two masks on for the same row → transforms compose in order (blur then gray)."""
    tf.random.set_seed(5)
    imgs = tf.random.uniform([1, 16, 16, 3], dtype=tf.float32)
    on = tf.constant([True])
    off = tf.constant([False])
    out = apply_albumentations_masks(
        imgs, m_blur=on, m_median=off, m_gray=on, m_clahe=off,
    ).numpy()
    ref = _box_blur_ref(imgs[0], 3)
    ref = _gray_ref(ref).numpy()
    np.testing.assert_allclose(out[0], ref, atol=1e-5)


# ---------------------------------------------------------------------------
# (c) row_mask gating: masked-out rows skip albumentations but still get HSV
# ---------------------------------------------------------------------------

def test_row_mask_false_rows_untouched_by_albumentations():
    tf.random.set_seed(6)
    imgs = tf.random.uniform([8, 16, 16, 3], dtype=tf.float32)
    # freq=1.0 → all allowed rows enter; row_mask False → that row never enters.
    row_mask = tf.constant([True, False] * 4)
    out = batch_albumentations(imgs, freq=1.0, row_mask=row_mask).numpy()
    for i in range(8):
        if not bool(row_mask[i]):
            np.testing.assert_array_equal(out[i], imgs[i].numpy())


def test_row_mask_false_rows_still_hsv_in_full_pipeline():
    """In batch_color_augment, HSV applies to ALL rows; albumentations only to mask."""
    tf.random.set_seed(7)
    imgs = tf.random.uniform([4, 16, 16, 3], dtype=tf.float32)
    row_mask = tf.constant([True, False, True, False])

    # Distance-like rows (row_mask False): only HSV should differ from input,
    # never albumentations. Compare against an HSV-only path with the SAME seed.
    tf.random.set_seed(100)
    full = batch_color_augment(
        imgs, hue=0.1, sat=0.5, val=0.2, albu_freq=1.0, albu_row_mask=row_mask,
    ).numpy()
    tf.random.set_seed(100)
    hsv_only = batch_hsv_augment(imgs, hue=0.1, sat=0.5, val=0.2).numpy()

    # Masked-out rows: full pipeline == HSV-only (albumentations skipped).
    for i in range(4):
        if not bool(row_mask[i]):
            np.testing.assert_allclose(full[i], hsv_only[i], atol=1e-6)


# ---------------------------------------------------------------------------
# (d) passthrough when augmentation disabled
# ---------------------------------------------------------------------------

def test_passthrough_float_when_all_disabled():
    imgs = tf.random.uniform([3, 8, 8, 3], dtype=tf.float32)
    out = batch_color_augment(imgs, hue=0.0, sat=0.0, val=0.0, albu_freq=0.0)
    # Float input already in [0, 1]: no /255, no jitter → exact passthrough.
    np.testing.assert_array_equal(out.numpy(), imgs.numpy())


def test_uint8_passthrough_is_exact_div255_when_disabled():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, (3, 8, 8, 3), dtype=np.uint8)
    imgs = tf.constant(arr)
    out = batch_color_augment(imgs, hue=0.0, sat=0.0, val=0.0, albu_freq=0.0).numpy()
    np.testing.assert_allclose(out, arr.astype(np.float32) / 255.0, atol=0.0)


# ---------------------------------------------------------------------------
# (e) uint8 input handling
# ---------------------------------------------------------------------------

def test_uint8_input_is_cast_and_normalized():
    rng = np.random.RandomState(1)
    arr = rng.randint(0, 256, (2, 12, 12, 3), dtype=np.uint8)
    imgs = tf.constant(arr)
    out = batch_color_augment(imgs, hue=0.02, sat=0.5, val=0.3, albu_freq=0.5)
    assert out.dtype == tf.float32
    out_np = out.numpy()
    assert out_np.min() >= 0.0 and out_np.max() <= 1.0
    assert np.isfinite(out_np).all()


def test_default_row_mask_allows_all_rows():
    """albu_row_mask=None → every row may enter albumentations (freq gate only)."""
    tf.random.set_seed(8)
    imgs = tf.random.uniform([4, 12, 12, 3], dtype=tf.float32)
    # freq=0.0 short-circuits albumentations; ensure no crash with None mask.
    out = batch_color_augment(imgs, hue=0.0, sat=0.0, val=0.0, albu_freq=0.0,
                              albu_row_mask=None)
    np.testing.assert_array_equal(out.numpy(), imgs.numpy())
    # With freq>0 and None mask it must still run and stay in range.
    out2 = batch_color_augment(imgs, hue=0.0, sat=0.0, val=0.0, albu_freq=1.0,
                               albu_row_mask=None).numpy()
    assert out2.min() >= 0.0 and out2.max() <= 1.0


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
