"""End-to-end pipeline benchmark: CPU mosaic warp vs GPU-offload warp.

Where ``tools/pipeline/bench_mosaic_device.py`` measured the mosaic geometry in
ISOLATION on random tensors (CPU ~12.7 ms/output vs GPU ~0.18 ms), this runs the
REAL training pipeline both ways and reports the resulting end-to-end throughput
and projected epoch time — the number that actually decides whether the offload is
worth it.

Two modes (``--mode``):

  * ``baseline`` — the stock pipeline: the per-output ``random_perspective`` warp
    runs inside ``tf.data`` on the CPU workers (``Mosaic.mosaic_fn``), and the
    parser emits the finished [H, W, 3] image.
  * ``offload`` — the mosaic stage emits the un-warped 2× canvas + warp transform +
    flip coin (``mosaic_gpu.mosaic_prepare_fn``), the parser runs in ``defer_warp``
    mode, and the WARP runs on the GPU inside the training step
    (``mosaic_gpu.gpu_mosaic_warp``), fed via ``prefetch_to_device`` when a GPU is
    present.

Both modes run the SAME model forward/backward + per-batch colour augmentation, so
the only difference is where the warp executes. For a fair comparison the distance
stream merge is disabled in both (mosaic is a detection-stream property; the
distance stream does not go through mosaic).

Reports per mode: data-only throughput (iterate the dataset, no model), end-to-end
throughput (dataset + warp + colour + model step), ms/step, and the projected
epoch time from the config's ``steps_per_loop``.

Run on the GPU training host for the meaningful comparison:
    python -m tools.pipeline.bench_mosaic_pipeline --mode both --steps 60 --warmup 15
    python -m tools.pipeline.bench_mosaic_pipeline --mode offload --batch 64 --steps 40

NOTE: the GPU column is only meaningful on a machine with a GPU. With no GPU the
offload warp falls back to the CPU (so it will NOT be faster locally) — locally use
``--steps 5 --batch 4`` just to smoke-test that the offload pipeline builds and the
step runs end-to-end.
"""

import argparse
import os
import time

import tensorflow as tf

from configs.yaml_loader import load_config
from data_pipeline.input_reader import build_input_reader_from_config
from data_pipeline.mosaic_gpu import gpu_mosaic_warp
from data_pipeline.batch_color_aug import batch_color_augment

_DEFAULT_CONFIG = "configs/experiments/yolo/yolov8_poly_dist.yaml"


def _has_gpu() -> bool:
    return len(tf.config.list_physical_devices("GPU")) > 0


def _build_dataset(config, gpu_offload: bool, batch_override: int | None):
    """Build the detection-only training dataset for one mode.

    The distance stream is disabled (``_distance_reader = None``) so both modes
    measure the same detection/mosaic path; the offload mode also can't merge the
    distance stream (its 672² image is not a canvas to warp).
    """
    train_data = config.task.train_data
    if batch_override:
        train_data.global_batch_size = batch_override

    reader = build_input_reader_from_config(
        data_cfg=train_data,
        task_cfg=config.task,
        is_training=True,
        gpu_offload=gpu_offload,
    )
    reader._distance_reader = None  # detection-only (fair + offload-compatible)
    ds = reader(None)

    if gpu_offload and _has_gpu():
        # Pin the canvas batch + labels to the GPU so the host→device transfer
        # overlaps the previous step (the transfer is the offload's main cost).
        ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    return ds


def _build_step(config, mode: str, with_model: bool):
    """Return (step_fn, model, optimizer). step_fn(images, labels) runs one train step.

    In offload mode the step first warps the canvas → image on the current device
    (the GPU under prefetch_to_device), then runs the identical colour-aug + model
    + loss + optimizer path as the baseline.
    """
    from train.task import YoloV8Task

    H, W = config.task.model.input_size[:2]
    p = config.task.train_data.parser
    clip_norm = config.task.gradient_clip_norm
    is_offload = mode == "offload"

    if not with_model:
        @tf.function
        def warp_only(images, labels):
            if is_offload:
                images = gpu_mosaic_warp(
                    images, labels["mosaic_warp"], labels["mosaic_flip"], H, W
                )
            # Touch the tensor so the pipeline + warp actually execute.
            return tf.reduce_mean(tf.cast(images[:, 0, 0, 0], tf.float32))
        return warp_only, None, None

    task = YoloV8Task(config)
    model = task.build_model()
    optimizer = task.build_optimizer()
    loss_fn = task.build_losses()

    @tf.function
    def step(images, labels):
        if is_offload:
            images = gpu_mosaic_warp(
                images, labels["mosaic_warp"], labels["mosaic_flip"], H, W
            )
        images = batch_color_augment(
            images,
            hue=p.aug_rand_hue,
            sat=p.aug_rand_saturation,
            val=p.aug_rand_brightness,
            albu_freq=p.albumentations_frequency,
            albu_row_mask=tf.equal(labels["ignore_bg"], 0),
        )
        with tf.GradientTape() as tape:
            feats = model(images, training=True)
            total = loss_fn(feats, labels)[0]
        grads = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables), clip_norm=clip_norm
        )
        return total

    return step, model, optimizer


def _time_loop(ds, step_fn, n_steps: int, warmup: int):
    """Run warmup + timed steps; return (steps_per_sec, ms_per_step)."""
    it = iter(ds)
    for _ in range(warmup):
        images, labels = next(it)
        out = step_fn(images, labels)
    _ = out.numpy()  # sync the warmup tail (finish tracing + queued compute)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        images, labels = next(it)
        out = step_fn(images, labels)
    _ = out.numpy()  # single sync: pipeline-overlapped steady-state throughput
    dt = time.perf_counter() - t0
    return n_steps / dt, dt / n_steps * 1e3


def _time_data_only(ds, n_steps: int, warmup: int):
    """Iterate the dataset only (no step); return data-production steps_per_sec."""
    it = iter(ds)
    for _ in range(warmup):
        _ = next(it)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        images, _labels = next(it)
        # Eager next() already materializes the batch; touch the shape to be sure.
        _ = images.shape
    dt = time.perf_counter() - t0
    return n_steps / dt


def _run_mode(config, mode: str, args, batch_size: int):
    print(f"\n{'='*70}\n[{mode.upper()}]")
    ds = _build_dataset(config, gpu_offload=(mode == "offload"), batch_override=args.batch)
    step_fn, _model, _opt = _build_step(config, mode, with_model=not args.no_model)

    data_sps = _time_data_only(ds, args.steps, args.warmup)
    print(f"  data-only:   {data_sps:7.2f} steps/s   {data_sps*batch_size:8.1f} img/s")

    if args.no_model:
        return {"mode": mode, "data_sps": data_sps, "e2e_sps": None}

    e2e_sps, ms_step = _time_loop(ds, step_fn, args.steps, args.warmup)
    spl = config.trainer.steps_per_loop or 1
    epoch_min = spl / e2e_sps / 60.0
    print(f"  end-to-end:  {e2e_sps:7.2f} steps/s   {e2e_sps*batch_size:8.1f} img/s   "
          f"{ms_step:7.1f} ms/step")
    print(f"  projected epoch: {epoch_min:6.1f} min   "
          f"(steps_per_loop={spl}, batch={batch_size})")
    return {"mode": mode, "data_sps": data_sps, "e2e_sps": e2e_sps,
            "ms_step": ms_step, "epoch_min": epoch_min}


def main():
    ap = argparse.ArgumentParser(description="CPU-mosaic vs GPU-offload pipeline benchmark")
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--mode", choices=["baseline", "offload", "both"], default="both")
    ap.add_argument("--steps", type=int, default=60, help="timed steps per mode")
    ap.add_argument("--warmup", type=int, default=15, help="warmup steps per mode")
    ap.add_argument("--batch", type=int, default=None, help="override global_batch_size")
    ap.add_argument("--no-model", action="store_true",
                    help="data + warp only (skip model fwd/bwd) — isolates the data side")
    args = ap.parse_args()

    config = load_config(args.config)

    # Apply the same runtime settings the trainer uses (thread caps + precision),
    # so the model step's cost matches production.
    from scripts.run_train import _apply_runtime_config
    _apply_runtime_config(config.runtime, debug=False)

    batch_size = args.batch or config.task.train_data.global_batch_size
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}   GPUs: {[g.name for g in gpus] or 'none'}")
    print(f"config={args.config}")
    print(f"batch={batch_size}  steps={args.steps}  warmup={args.warmup}  "
          f"model={'off' if args.no_model else 'on'}")
    if not gpus:
        print("WARNING: no GPU — the offload warp runs on the CPU, so offload will "
              "NOT be faster here. This run only validates that the pipeline builds "
              "and the step executes. Run on the GPU host for the real comparison.")

    modes = ["baseline", "offload"] if args.mode == "both" else [args.mode]
    results = [_run_mode(config, m, args, batch_size) for m in modes]

    if len(results) == 2 and all(r.get("e2e_sps") for r in results):
        base, off = results[0], results[1]
        speedup = off["e2e_sps"] / base["e2e_sps"]
        print(f"\n{'='*70}\nSUMMARY")
        print(f"  baseline epoch: {base['epoch_min']:6.1f} min   "
              f"({base['e2e_sps']:.2f} steps/s)")
        print(f"  offload  epoch: {off['epoch_min']:6.1f} min   "
              f"({off['e2e_sps']:.2f} steps/s)")
        print(f"  end-to-end speedup: {speedup:.2f}x")
        print("=" * 70)


if __name__ == "__main__":
    main()
