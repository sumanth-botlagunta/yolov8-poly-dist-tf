"""Stage-by-stage data-pipeline throughput attribution (CPU only, no model).

Builds the training pipeline cumulatively (read, +decode, +copy-paste, +mosaic,
+parser, +batch, full merged stream) and reports each stage's samples/sec so the
bottleneck is identified directly. Also prints TFDS dataset sizes and stored image
encodings. Rates only decrease down the table; the first big drop is the
bottleneck stage.

Usage:
    python utils/pipeline/diagnose_pipeline.py \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        [--samples 768] [--batches 10] [--threadpool-sweep 0,13,26]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _options(threadpool: int):
    import tensorflow as tf
    opts = tf.data.Options()
    opts.deterministic = False
    if threadpool > 0:
        opts.threading.private_threadpool_size = threadpool
    return opts


def _measure(ds, n_elems: int, samples_per_elem, warmup: int = 16) -> float:
    """Iterate `warmup` then `n_elems` elements; return samples/sec.

    `samples_per_elem` is 1 for unbatched stages, or a callable(elem)->int
    for batched stages. Eager iteration fully materializes every element.
    """
    it = iter(ds)
    for _ in range(warmup):
        next(it)
    t0 = time.perf_counter()
    count = 0
    for _ in range(n_elems):
        elem = next(it)
        count += samples_per_elem(elem) if callable(samples_per_elem) else samples_per_elem
    dt = time.perf_counter() - t0
    return count / dt


def _batch_count(elem) -> int:
    img = elem[0] if isinstance(elem, (tuple, list)) else elem['image']
    return int(img.shape[0])


def _print_dataset_info(cfg) -> None:
    import tensorflow_datasets as tfds
    td = cfg.task.train_data
    names = [n.strip() for n in td.tfds_name.split(',')]
    splits = [s.strip() for s in td.tfds_split.split(',')]
    extra = []
    if td.tfds_for_cnp:
        extra.append((td.tfds_for_cnp, td.tfds_for_cnp_split))
    if getattr(td, 'distance_data', None) is not None:
        extra.append((td.distance_data.tfds_name, td.distance_data.tfds_split))

    print("\n--- TFDS datasets (size + stored image encoding) ---")
    total_det = 0
    for i, (name, split) in enumerate(list(zip(names, splits)) + extra):
        try:
            b = tfds.builder(name, data_dir=td.tfds_data_dir)
            n = b.info.splits[split].num_examples
            feat = b.info.features['image']
            enc = getattr(feat, '_encoding_format', None) or getattr(feat, 'encoding_format', '?')
            shape = getattr(feat, 'shape', '?')
            print(f"  {name:<35s} {split:<8s} {n:>9,d} examples   image: {enc} {shape}")
            if i < len(names):
                total_det += n
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  {name:<35s} {split:<8s} ERROR: {e}")
    bs = td.global_batch_size
    print(f"  detection total: {total_det:,d}  → steps/epoch at batch {bs}: {total_det // bs}")
    spl = cfg.trainer.steps_per_loop
    if total_det // bs != spl:
        print(f"  *** MISMATCH: config train_total_examples gives steps_per_loop={spl} "
              f"but actual data gives {total_det // bs} — fix train_total_examples in the YAML ***")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--samples', type=int, default=768,
                    help='elements to time per UNBATCHED stage (default 768)')
    ap.add_argument('--batches', type=int, default=10,
                    help='batches to time per BATCHED stage (default 10)')
    ap.add_argument('--threadpool-sweep', default='',
                    help='comma-separated private_threadpool_size values to A/B '
                         'on the full merged stream, e.g. "0,13,26"')
    args = ap.parse_args()

    import tensorflow as tf  # noqa: F401 - after sys.path setup
    from configs.yaml_loader import load_config
    from data_pipeline.input_reader import build_input_reader_from_config

    cfg = load_config(args.config)
    td = cfg.task.train_data
    tp = getattr(td, 'private_threadpool_size', 0)

    _print_dataset_info(cfg)

    reader = build_input_reader_from_config(
        data_cfg=td, task_cfg=cfg.task, is_training=True,
    )
    opts = _options(tp)

    # ---- Cumulative stage datasets (mirrors InputReader._build_detection_dataset) ----
    raw = [d.repeat() for d in reader._load_tfds_datasets()]
    if len(raw) == 1:
        base = raw[0]
    else:
        weights = reader._normalize_weights(reader._sampling_weights, len(raw))
        base = tf.data.Dataset.sample_from_datasets(raw, weights=weights, seed=reader._seed)
    base = base.shuffle(reader._shuffle_buffer_size, seed=reader._seed,
                        reshuffle_each_iteration=True)

    AUTOTUNE = tf.data.AUTOTUNE
    stages = []  # (label, dataset, samples_per_elem, n_elems)

    s1 = base.prefetch(AUTOTUNE).with_options(opts)
    stages.append(("1. read + sample + shuffle (encoded records)", s1, 1, args.samples))

    # NOTE: stage order MUST mirror InputReader._build_detection_dataset,
    # currently: decode -> pre-resize -> copy-paste -> padded_batch(group_size) ->
    # mosaic -> unbatch -> shuffle(128) -> parser -> batch. Keep in sync when the
    # pipeline changes, or the attribution lies.
    s2 = base.map(reader._decoder.decode, num_parallel_calls=AUTOTUNE)
    stages.append(("2. + decode (image bytes → pixels)",
                   s2.prefetch(AUTOTUNE).with_options(opts), 1, args.samples))

    s3 = s2
    if reader._mosaic_module is not None:
        H, W = reader._mosaic_module._H, reader._mosaic_module._W

        def _pre_resize(ex, H=H, W=W):
            img_in = ex['image']
            shp = tf.shape(img_in)

            def _resize():
                return tf.cast(
                    tf.image.resize(tf.cast(img_in, tf.float32), [H, W], method='bilinear'),
                    tf.uint8,
                )

            img = tf.cond(
                tf.logical_and(tf.equal(shp[0], H), tf.equal(shp[1], W)),
                lambda: img_in, _resize,
            )
            img.set_shape([H, W, 3])
            return {**ex, 'image': img}

        s3 = s2.map(_pre_resize, num_parallel_calls=AUTOTUNE)
        stages.append(("3. + pre-resize → %d²" % H,
                       s3.prefetch(AUTOTUNE).with_options(opts), 1, args.samples))

    s4 = s3
    if reader._copy_paste_module is not None and reader._cnp_tfds_name:
        cnp = reader._load_cnp_dataset()
        s4 = tf.data.Dataset.zip((s3, cnp)).map(
            reader._copy_paste_module.process_fn(is_training=True),
            num_parallel_calls=AUTOTUNE,
        )
        stages.append(("4. + copy-paste (zip cnp + composite)",
                       s4.prefetch(AUTOTUNE).with_options(opts), 1, args.samples))

    s5 = s4
    if reader._mosaic_module is not None:
        # Mirror the real pipeline: padded_batch(group_size) -> mosaic (G in -> G//R out).
        g = reader._mosaic_module._group_size
        r = reader._mosaic_module._decodes_per_output
        s5 = (
            s4
            .padded_batch(g, drop_remainder=True)
            .map(reader._mosaic_module.mosaic_fn(is_training=True),
                 num_parallel_calls=AUTOTUNE)
            .unbatch()
            .shuffle(max(256, 4 * g), seed=reader._seed, reshuffle_each_iteration=True)
        )
        stages.append((f"5. + mosaic({g}->{g // r}) + unbatch + shuffle",
                       s5.prefetch(AUTOTUNE).with_options(opts), 1, args.samples))

    s6 = s5.map(reader._parser.parse_fn(is_training=True), num_parallel_calls=AUTOTUNE)
    stages.append(("6. + parser (flip/polygon targets; color aug is on-GPU)",
                   s6.prefetch(AUTOTUNE).with_options(opts), 1, args.samples))

    s7 = s6.batch(td.global_batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    stages.append(("7. + batch(%d)" % td.global_batch_size,
                   s7.with_options(opts), _batch_count, args.batches))

    s8 = reader()  # full merged stream incl. distance zip + options
    stages.append(("8. FULL merged stream (detection + distance)",
                   s8, _batch_count, args.batches))

    print(f"\n--- Stage attribution (threadpool={tp}, deterministic=False) ---")
    print("    NOTE: cumulative — the first big rate drop is the bottleneck stage.\n")
    results = []
    for label, ds, per_elem, n in stages:
        try:
            rate = _measure(ds, n, per_elem)
            results.append((label, rate))
            print(f"  {label:<55s} {rate:9.1f} imgs/sec")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:<55s} ERROR: {e}")

    if results:
        print("\n--- Drop analysis ---")
        for (la, ra), (lb, rb) in zip(results, results[1:]):
            drop = (1 - rb / ra) * 100 if ra > 0 else 0
            marker = "  <<< BOTTLENECK CANDIDATE" if drop > 30 else ""
            print(f"  {la.split('.')[0]} → {lb.split('.')[0]}: "
                  f"{ra:8.1f} → {rb:8.1f} imgs/sec  ({drop:+.0f}% drop){marker}")
        full = results[-1][1]
        # Merged per-step image count = detection batch + the (optional) distance batch.
        dd = getattr(cfg.task.train_data, 'distance_data', None)
        merged_batch = cfg.task.train_data.global_batch_size + (dd.global_batch_size if dd else 0)
        steps = cfg.trainer.steps_per_loop
        epoch_min = steps * (merged_batch / max(full, 1e-9)) / 60.0
        print(f"\n  Full-stream rate: {full:.0f} imgs/sec → pipeline-bound epoch "
              f"≈ {epoch_min:.1f} min for {steps} steps × {merged_batch} imgs "
              f"(target ≤20 min needs ≥{steps * merged_batch / 1200:.0f} imgs/sec)")

    if args.threadpool_sweep:
        print("\n--- private_threadpool_size sweep (full merged stream) ---")
        for v in [int(x) for x in args.threadpool_sweep.split(',')]:
            td.private_threadpool_size = v
            r = build_input_reader_from_config(
                data_cfg=td, task_cfg=cfg.task, is_training=True,
            )
            try:
                rate = _measure(r(), args.batches, _batch_count)
                print(f"  threadpool={v:<4d} {rate:9.1f} imgs/sec")
            except Exception as e:  # noqa: BLE001
                print(f"  threadpool={v:<4d} ERROR: {e}")


if __name__ == '__main__':
    main()
