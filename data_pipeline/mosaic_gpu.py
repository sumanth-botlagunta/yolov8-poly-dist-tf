"""GPU-offload variant of the mosaic stage: build the canvas on the CPU, warp on the GPU.

The per-output ``random_perspective`` warp (``ImageProjectiveTransformV3``) is the
data-pipeline bottleneck on the CPU-throttled training host — a microbenchmark
(``tools/pipeline/bench_mosaic_device.py``) measured it at ~12.7 ms/output on that
CPU versus ~0.18 ms/output on the idle H100 (72× faster). ``tf.data`` ops are
CPU-placed, so the only way to run the warp on the GPU is to move it OUT of the
``tf.data`` map and into the compiled training step.

This module implements that split WITHOUT changing the geometry or labels:

  * **CPU (tf.data), per emitted sample** — ``mosaic_prepare_fn`` builds the same
    2× canvas as the stock mosaic (reusing ``Mosaic._mosaic_canvas_M`` /
    ``Mosaic._single_canvas`` — one source of truth) and computes the FINAL labels
    (warped by the same ``M``, then flipped) on the worker. It emits, per sample:
      - ``mosaic_canvas``  uint8  [2H, 2W, 3]  — the un-warped stitched canvas
      - ``mosaic_warp``    float32 [8]         — the output→canvas transform for
                                                 ImageProjectiveTransformV3 (== the
                                                 inverse of M, the exact vector the
                                                 CPU path feeds ``apply_perspective_image``)
      - ``mosaic_flip``    bool   []           — the horizontal-flip coin
      - the usual label fields (boxes/polygons/classes/… already final)

  * **GPU (train_step), per batch** — ``gpu_mosaic_warp`` runs ONE batched
    ``ImageProjectiveTransformV3`` over the [B, 2H, 2W, 3] canvas with the [B, 8]
    transforms, then applies the per-sample horizontal flip. The result is the
    [B, H, W, 3] uint8 image the model consumes — byte-identical to what the CPU
    mosaic would have produced.

Why the flip is split: in the stock pipeline the parser flips the image AND the
labels AFTER the warp. To keep that semantics exactly, the LABEL flip stays on the
CPU (so the PolyYOLO radial target is built from flipped vertices) and the IMAGE
flip moves to the GPU under the same per-sample coin. ``tf.image.flip_left_right``
(image, ``W-1-x``) and the normalized label flip (``1-x``) are unchanged from the
stock parser — only WHERE they run moves.

Single (non-mosaic) samples pad their [H, W, 3] image into the [2H, 2W, 3] canvas
(``Mosaic._single_canvas``) so singles and mosaics share one batched warp call; the
padded region is never sampled (see that method's docstring).

Used by the GPU-offload benchmark (``tools/pipeline/bench_mosaic_pipeline.py``) and
the ``parser.defer_warp`` mode. The stock CPU ``Mosaic.mosaic_fn`` path is unchanged.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import tensorflow as tf

from data_pipeline.augmentations import (
    matrix_to_inverse_transform,
    transform_boxes_polygons,
)
from data_pipeline.mosaic import _PAD_SPEC, Mosaic


# Keys carried per emitted sample IN ADDITION to the label fields, identifying the
# deferred-warp payload. Consumed by parser.defer_warp and gpu_mosaic_warp.
CANVAS_KEY = "mosaic_canvas"
WARP_KEY = "mosaic_warp"
FLIP_KEY = "mosaic_flip"


# ---------------------------------------------------------------------------
# Label-only horizontal flip (image flip is deferred to the GPU)
# ---------------------------------------------------------------------------

def _flip_labels(
    boxes: tf.Tensor, polygons: tf.Tensor, do_flip: tf.Tensor
) -> tuple:
    """Flip box/polygon x-coords (normalized ``1 - x``) when ``do_flip`` is True.

    Identical label math to ``augmentations.random_horizontal_flip`` (boxes
    ``xmin↔1-xmax``; polygon valid vertices ``x↔1-x``, ``-1`` sentinel kept), but
    driven by an EXTERNAL coin so the matching image flip can run later on the GPU
    under the same decision. ``y`` is unchanged.
    """
    ymin, xmin, ymax, xmax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes_flipped = tf.stack([ymin, 1.0 - xmax, ymax, 1.0 - xmin], axis=1)
    boxes_out = tf.cond(do_flip, lambda: boxes_flipped, lambda: boxes)

    N = tf.shape(polygons)[0]
    max_v = tf.shape(polygons)[1]
    pts = tf.reshape(polygons, [N, -1, 2])
    valid_x = pts[:, :, 0] > -1.0  # sentinel is exactly -1.0 (design_register entry 10)
    x_flipped = tf.where(valid_x, 1.0 - pts[:, :, 0], pts[:, :, 0])
    pts_flipped = tf.stack([x_flipped, pts[:, :, 1]], axis=-1)
    poly_flipped = tf.reshape(pts_flipped, [N, max_v])
    polys_out = tf.cond(do_flip, lambda: poly_flipped, lambda: polygons)

    return boxes_out, polys_out


# ---------------------------------------------------------------------------
# Per-output prepare (canvas + transform + final labels; warp deferred)
# ---------------------------------------------------------------------------

def _mosaic_prepare(m: Mosaic, examples: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
    """Mosaic prepare: build the 2× canvas + final labels; emit canvas/warp (no warp)."""
    H, W = m._H, m._W
    canvas, M, merged_src, boxes_all, polys_all = m._mosaic_canvas_M(
        examples[0], examples[1], examples[2], examples[3]
    )
    # Final labels: warp by the SAME M the GPU will apply to the image (bit-identical
    # to the CPU path's transform_boxes_polygons in _mosaic).
    boxes_out, keep, polys_out = transform_boxes_polygons(
        boxes_all, polys_all, M,
        h_in=2 * H, w_in=2 * W,
        target_h=H, target_w=W,
        area_thresh=m._area_thresh, min_side=0.005,
    )
    anns = m._filtered_anns(merged_src, boxes_out, polys_out, keep)
    anns[CANVAS_KEY] = canvas
    anns[WARP_KEY] = matrix_to_inverse_transform(M)
    return anns


def _single_prepare(m: Mosaic, ex: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Single prepare: pad image into the 2× canvas + final labels; emit canvas/warp."""
    H, W = m._H, m._W
    canvas, M = m._single_canvas(ex)
    boxes = ex.get('groundtruth_boxes', tf.zeros([0, 4]))
    polys = ex.get('groundtruth_polygons', tf.zeros([0, 2]))
    # M is built with h_in=H, w_in=W (the image, not the 2× canvas) — see
    # _single_canvas — so the label transform uses the SAME input dims.
    boxes_out, keep, polys_out = transform_boxes_polygons(
        boxes, polys, M,
        h_in=H, w_in=W,
        target_h=H, target_w=W,
        area_thresh=m._area_thresh, min_side=0.005,
    )
    anns = m._filtered_anns(ex, boxes_out, polys_out, keep)
    anns[CANVAS_KEY] = canvas
    anns[WARP_KEY] = matrix_to_inverse_transform(M)
    return anns


# ---------------------------------------------------------------------------
# Stack the per-output prepare dicts (analog of mosaic._stack_results)
# ---------------------------------------------------------------------------

def _stack_prepare(results: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
    """Stack per-sample prepare dicts to a single group dict (leading dim = len).

    Per-instance label fields are padded to the group-max instance count (per
    ``_PAD_SPEC``) then stacked, exactly like ``mosaic._stack_results``. The
    non-instance fields here are the deferred-warp payload (canvas/warp/flip) +
    ``source_id`` instead of image/height/width.
    """
    n_list = [tf.shape(r['groundtruth_boxes'])[0] for r in results]
    max_n = n_list[0]
    for n_i in n_list[1:]:
        max_n = tf.maximum(max_n, n_i)

    out: Dict[str, tf.Tensor] = {}
    for key, pad_val, dtype in _PAD_SPEC:
        padded = []
        for r, n_i in zip(results, n_list):
            t = tf.cast(r[key], dtype)
            pad_rows = max_n - n_i
            paddings = [[0, pad_rows]] + [[0, 0]] * (len(t.shape) - 1)
            t = tf.pad(t, paddings, constant_values=pad_val)
            padded.append(t)
        out[key] = tf.stack(padded, axis=0)

    out[CANVAS_KEY] = tf.stack([r[CANVAS_KEY] for r in results], axis=0)
    out[WARP_KEY] = tf.stack([r[WARP_KEY] for r in results], axis=0)
    out[FLIP_KEY] = tf.stack([r[FLIP_KEY] for r in results], axis=0)
    out['source_id'] = tf.stack([r['source_id'] for r in results], axis=0)
    return out


# ---------------------------------------------------------------------------
# tf.data map fn (mirrors Mosaic.mosaic_fn windowed selection)
# ---------------------------------------------------------------------------

def mosaic_prepare_fn(m: Mosaic, random_flip: bool = True) -> Callable:
    """Return a ``tf.data`` map fn over a ``padded_batch(group_size)`` group.

    Mirrors ``Mosaic.mosaic_fn`` exactly for IMAGE SELECTION (same per-group
    permutation, same width-4 window stepping by R, same per-output mosaic coin),
    but each output emits the deferred-warp payload (``mosaic_canvas`` /
    ``mosaic_warp`` / ``mosaic_flip``) + the FINAL labels rather than a warped
    image. ``gpu_mosaic_warp`` consumes the payload on the GPU.

    Args:
        m:           a configured ``Mosaic`` (provides geometry + group params).
        random_flip: when True each output draws an independent flip coin and the
                     labels are flipped here (the matching image flip runs on the
                     GPU under the emitted ``mosaic_flip`` bool). When False no flip
                     is applied and ``mosaic_flip`` is all-False.
    """
    G = m._group_size
    P = m._outputs_per_group
    R = m._decodes_per_output

    def _fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def _select(d: Dict, i) -> Dict:
            return {k: tf.gather(v, i) for k, v in d.items()}

        perm = tf.random.shuffle(tf.range(G))
        results = []
        for j in range(P):
            idx = [perm[(R * j + k) % G] for k in range(4)]
            examples = [_select(batch, idx[k]) for k in range(4)]
            do_mosaic = tf.random.uniform([]) < m._mosaic_freq
            out_j = tf.cond(
                do_mosaic,
                lambda ex=examples: _mosaic_prepare(m, ex),
                lambda ex=examples: _single_prepare(m, ex[0]),
            )
            # Label flip under an independent per-output coin; the image flip is
            # deferred to the GPU under the same bool.
            if random_flip:
                do_flip = tf.random.uniform([]) > 0.5
            else:
                do_flip = tf.constant(False)
            boxes_f, polys_f = _flip_labels(
                out_j['groundtruth_boxes'], out_j['groundtruth_polygons'], do_flip
            )
            out_j['groundtruth_boxes'] = boxes_f
            out_j['groundtruth_polygons'] = polys_f
            out_j[FLIP_KEY] = do_flip
            results.append(out_j)
        return _stack_prepare(results)

    return _fn


# ---------------------------------------------------------------------------
# GPU batched warp (runs inside the compiled train_step)
# ---------------------------------------------------------------------------

def gpu_mosaic_warp(
    canvas: tf.Tensor,
    transforms: tf.Tensor,
    flip: tf.Tensor,
    output_h: int,
    output_w: int,
) -> tf.Tensor:
    """Warp a batch of 2× canvases to output-size images on the current device.

    Runs ONE batched ``ImageProjectiveTransformV3`` (gray-114 CONSTANT fill,
    BILINEAR) — the op the CPU pipeline ran per-output, now amortized over the
    batch on the GPU — then applies the per-sample horizontal flip
    (``tf.image.flip_left_right``, the same ``W-1-x`` the stock parser used).

    Args:
        canvas:     uint8/float [B, 2H, 2W, 3] stitched canvases from the prepare fn.
        transforms: float32 [B, 8] output→canvas vectors (== mosaic_warp).
        flip:       bool [B] per-sample horizontal-flip coin (== mosaic_flip).
        output_h/w: model input size (H, W).

    Returns:
        uint8 [B, H, W, 3] — the warped (and flipped) images the model consumes.
    """
    images_f = tf.cast(canvas, tf.float32)
    warped = tf.raw_ops.ImageProjectiveTransformV3(
        images=images_f,
        transforms=transforms,
        output_shape=tf.stack([tf.cast(output_h, tf.int32), tf.cast(output_w, tf.int32)]),
        fill_value=114.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )  # [B, H, W, 3] float32
    flipped = tf.image.flip_left_right(warped)
    out = tf.where(flip[:, tf.newaxis, tf.newaxis, tf.newaxis], flipped, warped)
    # Truncating cast (NOT round) to match apply_perspective_image's
    # tf.cast(..., uint8) exactly — the GPU warp is then byte-identical to the
    # stock CPU mosaic image (verified in tests/test_mosaic_gpu.py).
    out = tf.cast(out, tf.uint8)
    out.set_shape([None, output_h, output_w, 3])
    return out
