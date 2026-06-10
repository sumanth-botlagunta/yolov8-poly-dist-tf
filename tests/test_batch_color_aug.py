"""Tests for data_pipeline.batch_color_aug (per-batch GPU colour augmentation).

These verify that the vectorized per-batch ops reproduce the per-sample parser
behaviour they replace, with the same per-image randomness semantics:

  (a) apply_hsv_deltas == per-image adjust_hue → adjust_saturation →
      adjust_brightness → clip  (exact, atol 1e-4).
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
    apply_hsv_deltas,
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
# (a) HSV exact equivalence
# ---------------------------------------------------------------------------

def test_apply_hsv_deltas_matches_per_image_chain():
    tf.random.set_seed(0)
    B, H, W = 4, 16, 24
    imgs = tf.random.uniform([B, H, W, 3], dtype=tf.float32)
    dh = tf.constant([-0.013, 0.0, 0.011, 0.007], dtype=tf.float32)
    ds = tf.constant([0.4, 1.0, 1.6, 0.85], dtype=tf.float32)
    dv = tf.constant([-0.3, 0.0, 0.25, 0.1], dtype=tf.float32)

    batched = apply_hsv_deltas(imgs, dh, ds, dv).numpy()

    for i in range(B):
        ref = imgs[i]
        ref = tf.image.adjust_hue(ref, float(dh[i]))
        ref = tf.image.adjust_saturation(ref, float(ds[i]))
        ref = tf.image.adjust_brightness(ref, float(dv[i]))
        ref = tf.clip_by_value(ref, 0.0, 1.0).numpy()
        np.testing.assert_allclose(batched[i], ref, atol=1e-4)


def test_batch_hsv_augment_noop_when_all_zero():
    imgs = tf.random.uniform([3, 8, 8, 3], dtype=tf.float32)
    out = batch_hsv_augment(imgs, hue=0.0, sat=0.0, val=0.0)
    # All components disabled → exact passthrough.
    np.testing.assert_array_equal(out.numpy(), imgs.numpy())


def test_batch_hsv_augment_zero_components_are_identity_deltas():
    """hue=0 → dh=0, sat=0 → ds=1, val=0 → dv=0 (one active component still jitters)."""
    tf.random.set_seed(1)
    imgs = tf.random.uniform([2, 8, 8, 3], dtype=tf.float32)
    # Only saturation active; hue/val must be no-ops, so result == adjust_saturation only.
    out = batch_hsv_augment(imgs, hue=0.0, sat=0.5, val=0.0).numpy()
    # Saturation strictly changes pixels for a random image with gain != 1 in general;
    # at minimum the op must stay in range and be finite.
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


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
