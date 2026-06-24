#!/usr/bin/env python3
"""One-time offline re-encoder: store detection TFDS images pre-resized to 672x672.

Why
---
The training input pipeline has two fixed, unavoidable CPU costs per detection
example that survive every other optimization:

  1. full-resolution JPEG/PNG **decode** (TFDS `Image` feature), and
  2. the full-resolution -> 672x672 **bilinear resize**
     (`data_pipeline/input_reader._pre_resize_for_mosaic`).

On a 13-core-capped training host these two ops dominate the per-example decode
budget. This tool eliminates both *offline, once* by writing a new TFDS dataset
variant whose images are already stored as 672x672 JPEG. After the swap:

  * decode is of a tiny 672x672 JPEG instead of the full capture, and
  * `_pre_resize_for_mosaic` short-circuits via its `tf.cond` (image is already
    exactly HxW), so the resize op never runs.

The re-encoded pixels are produced with EXACTLY the same op sequence the live
pipeline uses (`cast float32 -> tf.image.resize(..., 'bilinear') -> cast uint8`),
so training pixels are identical **up to one extra JPEG round-trip** at the
`Image` feature's quality (95 by default in this tfds). At q95 that round-trip is
visually lossless. (TFDS's `Image(encoding_format='jpeg')` re-encodes uint8
arrays at quality 95; there is no public q knob on the feature, so we document
the q95 round-trip rather than parameterize it.)

The original capture dimensions are preserved as new int64 scalar features
`orig_height` / `orig_width`. The copy-paste resolution correction in the parser
needs the *original* capture size, which `tf.shape()` can no longer recover once
the image is stored small; `data_pipeline.tfds_decoders.PolygonDecoder.decode`
already prefers `orig_height`/`orig_width` when present.

Output dataset name: ``<source_name>_672`` (same version), written into the same
``--data_dir`` via ``tfds.dataset_builders.TfDataBuilder`` (the
"store as tfds dataset" ad-hoc builder API).

Usage
-----
    python tools/pipeline/reencode_tfds_672.py \\
        --data_dir /path/tensorflow_datasets \\
        --datasets cleaner_polygon2026:2.0.0,field_misrecog2026:1.0.0,station_misrecog:1.1.0 \\
        --size 672 \\
        [--splits all]

This is a single-host streaming job: no ``.cache()``, the TFDS writer handles
sharding. Memory stays flat regardless of dataset size.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _parse_name_version(spec: str) -> Tuple[str, Optional[str]]:
    """Split a ``name:version`` (or bare ``name``) spec."""
    if ":" in spec:
        name, version = spec.split(":", 1)
        return name, version
    return spec, None


def _image_num_channels(image_feature) -> Optional[int]:
    """Return the channel count of a tfds Image feature, or None if unknown."""
    shape = getattr(image_feature, "shape", None)
    if shape is None or len(shape) < 3:
        return None
    return shape[-1]


def _build_target_features(
    src_features: tfds.features.FeaturesDict,
    size: int,
) -> tfds.features.FeaturesDict:
    """Clone the source features, replacing image and adding orig dims.

    Every non-image top-level feature connector (including the nested ``objects``
    Sequence) is passed through unchanged. ``image`` becomes a fixed-size
    672x672 JPEG; ``orig_height`` / ``orig_width`` are added as int64 scalars.
    """
    new_features: Dict[str, object] = dict(src_features)  # top-level connectors
    new_features["image"] = tfds.features.Image(
        shape=(size, size, 3), encoding_format="jpeg"
    )
    new_features["orig_height"] = tfds.features.Tensor(shape=(), dtype=np.int64)
    new_features["orig_width"] = tfds.features.Tensor(shape=(), dtype=np.int64)
    return tfds.features.FeaturesDict(new_features)


def _make_reencode_map_fn(size: int):
    """Build the per-example tf.data map fn (resize + record orig dims)."""

    def _map(ex):
        img_in = ex["image"]
        shp = tf.shape(img_in)
        orig_h = tf.cast(shp[0], tf.int64)
        orig_w = tf.cast(shp[1], tf.int64)

        # EXACTLY mirror input_reader._pre_resize_for_mosaic so the stored pixels
        # match the on-the-fly path (modulo the JPEG round-trip on write):
        #   cast float32 -> bilinear resize -> cast uint8.
        img = tf.cast(
            tf.image.resize(tf.cast(img_in, tf.float32), [size, size], method="bilinear"),
            tf.uint8,
        )
        img.set_shape([size, size, 3])

        out = dict(ex)
        out["image"] = img
        out["orig_height"] = orig_h
        out["orig_width"] = orig_w
        return out

    return _map


def _resolve_splits(builder, requested: str) -> List[str]:
    """Resolve the requested splits arg to a concrete list of split names."""
    available = list(builder.info.splits.keys())
    if requested == "all":
        return available
    wanted = [s.strip() for s in requested.split(",") if s.strip()]
    missing = [s for s in wanted if s not in available]
    if missing:
        raise ValueError(
            f"Requested splits {missing} not in dataset (have {available})."
        )
    return wanted


def reencode_one(
    name: str,
    data_dir: str,
    size: int,
    splits: str = "all",
    version: Optional[str] = None,
) -> str:
    """Re-encode one source dataset to ``<name>_672`` with 672x672 JPEG images.

    Args:
        name: source dataset name, e.g. ``cleaner_polygon2026`` (a ``name:version``
            spec is also accepted and overrides ``version``).
        data_dir: TFDS data dir holding the source and receiving the output.
        size: target square edge (e.g. 672).
        splits: ``"all"`` or a comma-separated subset of split names.
        version: source version string (e.g. ``2.0.0``); if None, TFDS default.

    Returns:
        The output dataset name (``<name>_672``).

    Raises:
        ValueError: on per-split count mismatch or a 4-channel (RGBA) source.
    """
    parsed_name, parsed_version = _parse_name_version(name)
    name = parsed_name
    if parsed_version is not None:
        version = parsed_version

    spec = f"{name}:{version}" if version else name
    src_builder = tfds.builder(spec, data_dir=data_dir)

    src_features = src_builder.info.features
    image_feature = src_features["image"]
    n_channels = _image_num_channels(image_feature)
    if n_channels == 4:
        raise ValueError(
            f"Refusing to re-encode '{spec}': image has 4 channels (RGBA). "
            "RGBA copy-paste datasets (e.g. cleaner_copy_paste) are not supported "
            "by this 3-channel 672 re-encoder; skip it."
        )

    out_name = f"{name}_672"
    out_version = version or str(src_builder.info.version)

    requested_splits = _resolve_splits(src_builder, splits)

    target_features = _build_target_features(src_features, size)
    map_fn = _make_reencode_map_fn(size)

    src_counts: Dict[str, int] = {}
    split_datasets: Dict[str, tf.data.Dataset] = {}
    for split in requested_splits:
        src_counts[split] = int(src_builder.info.splits[split].num_examples)

        # Stream: decode -> resize+record-orig -> hand to writer. No .cache().
        ds = src_builder.as_dataset(split=split, shuffle_files=False)
        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Per-split progress logging without materializing anything extra: count
        # via a side-effecting scan-free enumerate during write is not possible
        # through TfDataBuilder, so we log by re-iterating cheaply below instead.
        split_datasets[split] = ds

    print(f"[{out_name}] re-encoding splits {requested_splits} "
          f"(source counts: {src_counts}) ...")

    builder = tfds.dataset_builders.TfDataBuilder(
        name=out_name,
        version=out_version,
        features=target_features,
        split_datasets=split_datasets,
        data_dir=data_dir,
        description=(
            f"{name} with images pre-resized to {size}x{size} JPEG (q95) and "
            f"original capture dims stored as orig_height/orig_width. "
            f"Generated by tools/pipeline/reencode_tfds_672.py."
        ),
    )
    builder.download_and_prepare()

    # Verify counts per split against the source.
    out_builder = tfds.builder(f"{out_name}:{out_version}", data_dir=data_dir)
    print(f"[{out_name}] verifying example counts ...")
    from tools.shared.progress import Progress
    for split in requested_splits:
        out_n = int(out_builder.info.splits[split].num_examples)
        src_n = src_counts[split]
        # Progress re-iteration so long writes show life on big splits.
        seen = 0
        vbar = Progress(total=out_n, desc=f"verify {out_name}/{split}", unit='ex')
        for _ in out_builder.as_dataset(split=split, shuffle_files=False):
            seen += 1
            vbar.update(1)
        vbar.close()
        print(f"[{out_name}/{split}] examples: source={src_n} "
              f"output_info={out_n} scanned={seen}")
        assert out_n == src_n == seen, (
            f"Count mismatch on split '{split}': source={src_n}, "
            f"output_info={out_n}, scanned={seen}"
        )

    print(f"[{out_name}] DONE. All split counts match the source.")
    print(f"[{out_name}] YAML change required -- swap tfds_name to the _672 variant:")
    print(f"    tfds_name: {name}:{out_version}   ->   tfds_name: {out_name}:{out_version}")
    return out_name


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-encode detection TFDS datasets with 672x672 JPEG images.",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="TFDS data dir holding sources and receiving the _672 outputs.",
    )
    parser.add_argument(
        "--datasets", required=True,
        help="Comma-separated name:version specs, e.g. "
             "cleaner_polygon2026:2.0.0,field_misrecog2026:1.0.0",
    )
    parser.add_argument(
        "--size", type=int, default=672,
        help="Target square edge in pixels (default 672).",
    )
    parser.add_argument(
        "--splits", default="all",
        help="'all' (default) or a comma-separated subset of split names.",
    )
    args = parser.parse_args(argv)

    specs = [s.strip() for s in args.datasets.split(",") if s.strip()]
    done: List[str] = []
    skipped: List[str] = []
    for spec in specs:
        try:
            out_name = reencode_one(
                name=spec, data_dir=args.data_dir, size=args.size, splits=args.splits,
            )
            done.append(out_name)
        except ValueError as e:
            print(f"SKIP {spec}: {e}")
            skipped.append(spec)

    print("\n==== SUMMARY ====")
    print(f"Re-encoded: {done}")
    if skipped:
        print(f"Skipped:    {skipped}")
    print("Swap the tfds_name entries in your data YAML to the _672 names above, "
          "then retrain. The pipeline auto-detects orig_height/orig_width and "
          "skips the redundant on-the-fly resize.")


if __name__ == "__main__":
    main()
