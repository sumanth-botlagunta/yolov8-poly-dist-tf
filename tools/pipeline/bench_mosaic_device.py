"""Microbenchmark: mosaic geometry on CPU vs GPU (random input, no datasets).

Isolates the two heavy ops the mosaic stage runs per emitted sample —
  • 4 × tf.image.resize  (each source image -> its drawn scale)
  • 1 × ImageProjectiveTransform  (the random_perspective warp, 2x canvas -> output)
— and times them **batched** on each available device, plus the host->device transfer
cost the GPU path would add. Purely synthetic (random tensors), so it runs anywhere with
just TensorFlow and needs none of the repo / datasets.

Why: the mosaic stage is the pipeline bottleneck on the CPU-throttled training host, and the
question is whether moving its geometry to the (idle) GPU is worth a refactor. This prints the
device speedup for those ops and the transfer overhead, so the decision is data-driven.

Run on the GPU training host for the meaningful comparison:
    python -m tools.pipeline.bench_mosaic_device --batch 128 --iters 30
    python -m tools.pipeline.bench_mosaic_device --batch 128 --iters 30 --size 672

What the numbers mean (printed at the end):
    per-batch geometry ms, outputs/sec, and per-output ms on CPU vs GPU. GPU-offload replaces
    the CPU per-output geometry cost with (GPU geometry + host->device transfer of 4 images).
"""

import argparse
import time

import numpy as np
import tensorflow as tf


def _build_mosaic_geom_fn(device: str, H: int, W: int, scale: float):
    """tf.function doing the mosaic geometry for a batch of B outputs on *device*.

    src:        [B, 4, H, W, 3] float32   (4 source images per output)
    transforms: [B, 8]          float32   (per-output projective warp, output<-canvas)
    returns:    [B, H, W, 3]    float32   (the warped mosaics)
    """
    Hs = Ws = int(round(scale * H))          # per-image drawn scale (representative)

    @tf.function(reduce_retracing=True)
    def mosaic_geom(src, transforms):
        with tf.device(device):
            B = tf.shape(src)[0]
            flat = tf.reshape(src, [B * 4, H, W, 3])
            # (1) the 4 per-image resizes, batched as B*4 resizes
            resized = tf.image.resize(flat, [Hs, Ws], method="bilinear")
            # place each resized image into an H×W cell (top-left); 4 cells -> 2H×2W canvas
            padded = tf.image.pad_to_bounding_box(resized, 0, 0, H, W)   # [B*4, H, W, 3]
            q = tf.reshape(padded, [B, 4, H, W, 3])
            top = tf.concat([q[:, 0], q[:, 1]], axis=2)                  # [B, H, 2W, 3]
            bot = tf.concat([q[:, 2], q[:, 3]], axis=2)
            canvas = tf.concat([top, bot], axis=1)                       # [B, 2H, 2W, 3]
            # (2) the single perspective warp, 2H×2W canvas -> H×W output, batched as B warps
            out = tf.raw_ops.ImageProjectiveTransformV3(
                images=canvas, transforms=transforms,
                output_shape=[H, W], fill_value=114.0,
                interpolation="BILINEAR", fill_mode="CONSTANT",
            )
        return out

    return mosaic_geom


def _time(fn, *args, iters: int, warmup: int = 3) -> float:
    """Return mean seconds/iter, forcing completion (sync) once at the end."""
    for _ in range(warmup):
        out = fn(*args)
    _ = out.numpy()                       # ensure warmup/trace finished
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    _ = out.numpy()                       # GPU executes in order -> this waits for all `iters`
    return (time.perf_counter() - t0) / iters


def _time_transfer(host_uint8, device: str, iters: int, warmup: int = 3) -> float:
    """Mean seconds to copy a host uint8 array to *device* (the GPU path's extra cost)."""
    def copy():
        with tf.device(device):
            d = tf.constant(host_uint8)
            return tf.cast(d, tf.float32)
    for _ in range(warmup):
        _ = copy().numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = copy()
    _ = out.numpy()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser(description="Mosaic-geometry CPU vs GPU microbenchmark")
    ap.add_argument("--batch", type=int, default=128, help="mosaic outputs per batch (B)")
    ap.add_argument("--size", type=int, default=672, help="output H=W")
    ap.add_argument("--scale", type=float, default=0.7, help="per-image resize scale (drawn)")
    ap.add_argument("--iters", type=int, default=30, help="timed iterations")
    a = ap.parse_args()
    B, H, W = a.batch, a.size, a.size

    gpus = tf.config.list_physical_devices("GPU")
    devices = ["/CPU:0"] + (["/GPU:0"] if gpus else [])
    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs visible: {[g.name for g in gpus] or 'none'}")
    print(f"Batch B={B}  output={H}x{W}  per-image scale={a.scale}  iters={a.iters}\n")

    # Synthetic inputs (host side).
    rng = np.random.default_rng(0)
    src_host = rng.integers(0, 255, size=[B, 4, H, W, 3], dtype=np.uint8)
    # center-crop-ish projective transform (output<-canvas); value is irrelevant for timing.
    transforms = np.tile(
        np.array([[1, 0, W / 2.0, 0, 1, H / 2.0, 0, 0]], np.float32), [B, 1])
    transforms_t = tf.constant(transforms)

    src_mb = src_host.nbytes / 1e6
    print(f"Source tensor per batch: {src_mb:.0f} MB  "
          f"(4 imgs/output × {B} = {B*4} images) — this is what the GPU path must transfer.\n")

    results = {}
    for dev in devices:
        fn = _build_mosaic_geom_fn(dev, H, W, a.scale)
        src_t = tf.constant(src_host.astype(np.float32))   # pre-placed (geometry-only timing)
        sec = _time(fn, src_t, transforms_t, iters=a.iters)
        results[dev] = sec
        print(f"[{dev}] mosaic geometry: {sec*1e3:8.1f} ms/batch   "
              f"{B/sec:8.1f} outputs/sec   {sec/B*1e3:6.2f} ms/output")

    # Host->device transfer cost (only the GPU path pays this).
    if "/GPU:0" in devices:
        tsec = _time_transfer(src_host, "/GPU:0", iters=a.iters)
        print(f"\n[host->GPU] transfer 4×{B} images: {tsec*1e3:7.1f} ms/batch "
              f"({src_mb/tsec/1e3:.1f} GB/s)")

    # ---- interpretation ----
    print("\n" + "=" * 70)
    cpu = results["/CPU:0"]
    print(f"CPU mosaic geometry:        {cpu/B*1e3:6.2f} ms/output  ({B/cpu:.0f} out/sec)")
    if "/GPU:0" in results:
        gpu = results["/GPU:0"]
        tsec = _time_transfer(src_host, "/GPU:0", iters=a.iters)
        gpu_eff = gpu + tsec                       # GPU path also pays the transfer
        print(f"GPU mosaic geometry:        {gpu/B*1e3:6.2f} ms/output  ({B/gpu:.0f} out/sec)")
        print(f"GPU geometry + transfer:    {gpu_eff/B*1e3:6.2f} ms/output  ({B/gpu_eff:.0f} out/sec)")
        print(f"Speedup (geometry only):    {cpu/gpu:5.1f}x")
        print(f"Speedup (incl. transfer):   {cpu/gpu_eff:5.1f}x")
        print("\nNote: GPU-offload removes the CPU per-output geometry cost above, but the")
        print("decode + pre-resize (R× per output) stays on the CPU, and the GPU now also")
        print("competes with the model step. Combine this with your diagnose_pipeline stage")
        print("numbers (decode ms/image) to estimate the real epoch time.")
    else:
        print("No GPU visible here — run this on the training host to get the GPU column.")
    print("=" * 70)


if __name__ == "__main__":
    main()
