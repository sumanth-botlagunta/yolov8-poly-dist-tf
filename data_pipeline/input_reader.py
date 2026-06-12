"""Main data reader with multi-TFDS weighted sampling support.

Handles two parallel data streams:
  1. Detection stream: multiple TFDS sources merged via sample_from_datasets.
  2. Distance stream (optional): servingbot_polygon TFDS, batched separately
     and concatenated onto the detection batch (ignore_bg=1 on those rows).

Pipeline order for training:
    tfds.load(names, SkipDecoding) → repeat each source → sample_from_datasets(weights)
    → shuffle (encoded records) → decode
    → zip(cnp_dataset) → copy_paste(prob)
    → batch(4) → mosaic (4 in → 4 out) → unbatch → shuffle(small)
    → parser.parse_fn(is_training=True)
    → batch(global_batch_size)
    → prefetch(AUTOTUNE)

Training-stream invariants:
  - Each SOURCE dataset is repeated before sample_from_datasets so the [95,2,3]
    sampling weights stay stationary forever (repeating the merged stream would
    replay the tail-skew that appears as small sources exhaust). The training
    stream is therefore infinite; the trainer runs a fixed steps_per_loop steps
    per epoch (one nominal pass = train_total_examples / batch).
  - Images stay ENCODED (SkipDecoding) through shuffle, so the shuffle buffer
    holds KBs of JPEG bytes per element instead of MBs of decoded pixels; the
    decoders' tf.string branch decodes inside the parallel decode map.
  - Mosaic emits 4 samples per 4-group (no decoded image is discarded); a small
    post-unbatch shuffle breaks up the 4-sample correlation clusters.

Distance stream (when distance_reader is provided):
    servingbot_polygon → dist_parser → batch(16) → prefetch
    Merged at the task level via zip + concat on batch dim.

Classes:
    InputReader: Builds a merged tf.data pipeline from one or more TFDS datasets.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

log = logging.getLogger(__name__)

_AUTOTUNE = tf.data.AUTOTUNE


def _concat_batch_dicts(
    det: tuple,
    dist: tuple,
) -> tuple:
    """Concatenate detection and distance batches along the batch dimension.

    Both are (image, labels) tuples.  Distance labels already have ignore_bg=1
    set by V8DistanceParser.

    Returns:
        (images [det+dist, H, W, 3], merged_labels_dict)
    """
    det_img, det_labels = det
    dist_img, dist_labels = dist
    # Guard against schema drift: iterating only det_labels would silently drop any
    # key the distance parser adds but the detection parser doesn't (or crash on a
    # missing key). Require identical label schemas.
    if set(det_labels) != set(dist_labels):
        raise ValueError(
            "Detection/distance label schema mismatch — keys must match exactly. "
            f"detection={sorted(det_labels)} distance={sorted(dist_labels)}"
        )
    merged_img = tf.concat([det_img, dist_img], axis=0)
    merged_labels = {
        k: tf.concat([det_labels[k], dist_labels[k]], axis=0)
        for k in det_labels
    }
    return merged_img, merged_labels


class InputReader:
    """Build a merged tf.data pipeline from one or more TFDS datasets.

    For training the pipeline order is:
        decode → copy-paste (prob=0.2) → mosaic (freq=0.5) → parser

    For evaluation:
        decode → parser (no augmentation)

    The distance stream is decoded and parsed independently then zipped and
    concatenated with the detection batch before being returned.
    """

    def __init__(
        self,
        tfds_names: List[str],
        tfds_split: List[str],
        tfds_data_dir: str,
        tfds_sampling_weights: Optional[List[float]] = None,
        global_batch_size: int = 128,
        is_training: bool = True,
        decoder=None,
        parser=None,
        copy_paste_module=None,
        mosaic_module=None,
        distance_reader: Optional["InputReader"] = None,
        cnp_tfds_name: Optional[str] = None,
        cnp_tfds_split: Optional[str] = None,
        cnp_decoder=None,
        seed: Optional[int] = None,
        shuffle_buffer_size: int = 1500,
        drop_remainder: bool = True,
        tfds_download: bool = True,
        private_threadpool_size: int = 0,
    ):
        self._tfds_names = tfds_names
        self._tfds_split = tfds_split
        self._tfds_data_dir = tfds_data_dir
        self._sampling_weights = tfds_sampling_weights
        self._global_batch_size = global_batch_size
        self._is_training = is_training
        self._decoder = decoder
        self._parser = parser
        self._copy_paste_module = copy_paste_module
        self._mosaic_module = mosaic_module
        self._distance_reader = distance_reader
        self._cnp_tfds_name = cnp_tfds_name
        self._cnp_tfds_split = cnp_tfds_split
        self._cnp_decoder = cnp_decoder
        self._seed = seed
        self._shuffle_buffer_size = shuffle_buffer_size
        self._drop_remainder = drop_remainder
        self._tfds_download = tfds_download
        self._private_threadpool_size = private_threadpool_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, ctx: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
        """Return the fully constructed tf.data.Dataset.

        Args:
            ctx: Optional distribution input context from MirroredStrategy.
                 When provided, the global batch size is split across replicas.
        """
        batch_size = self._per_replica_batch_size(ctx)

        if self._is_training:
            det_ds = self._build_detection_dataset(batch_size)
        else:
            det_ds = self._build_eval_dataset(batch_size)

        ds = det_ds
        if self._is_training and self._distance_reader is not None:
            dist_batch_size = self._distance_reader._per_replica_batch_size(ctx)
            dist_ds = self._distance_reader._build_distance_dataset(dist_batch_size)
            ds = self._merge_streams(det_ds, dist_ds)

        if self._is_training:
            # Performance options for the whole input graph (options set on the
            # terminal dataset propagate upstream at finalization, including the
            # zipped distance stream). deterministic=False removes head-of-line
            # blocking in the parallel maps (forfeits seeded sample order — the
            # per-op augmentation randomness is unaffected). The private
            # threadpool caps tf.data's worker count on machines whose visible
            # core count exceeds the actual CPU quota (cgroup caps).
            options = tf.data.Options()
            options.deterministic = False
            if self._private_threadpool_size > 0:
                options.threading.private_threadpool_size = self._private_threadpool_size
            ds = ds.with_options(options)

        return ds

    # ------------------------------------------------------------------
    # Stream builders
    # ------------------------------------------------------------------

    def _build_detection_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Build the weighted-sampled detection stream (infinite via repeat)."""
        raw_datasets = self._load_tfds_datasets()

        # Repeat each SOURCE before sampling so the sampling weights stay
        # stationary: sample_from_datasets keeps drawing from the remaining
        # sources as smaller ones exhaust, so repeating the merged stream would
        # replay a tail skewed toward the largest source every cycle. The
        # resulting stream is infinite; epoch length is enforced by the trainer
        # (steps_per_loop), not by data exhaustion.
        raw_datasets = [d.repeat() for d in raw_datasets]

        if len(raw_datasets) == 1:
            ds = raw_datasets[0]
        else:
            weights = self._normalize_weights(self._sampling_weights, len(raw_datasets))
            ds = tf.data.Dataset.sample_from_datasets(
                raw_datasets, weights=weights, seed=self._seed
            )

        # Detection source shuffle: seed=self._seed (the base seed). The cnp source
        # shuffle uses self._seed+1 and the post-unbatch shuffle uses self._seed+2 so
        # the three shuffle stages draw from DISTINCT RNG streams — sharing one seed
        # makes the permutations correlated, which can partially undo each stage's
        # decorrelation of the previous one.
        ds = ds.shuffle(self._shuffle_buffer_size, seed=self._seed, reshuffle_each_iteration=True)

        if self._decoder is not None:
            ds = ds.map(self._decoder.decode, num_parallel_calls=_AUTOTUNE)

        # Pre-resize to the fixed output shape BEFORE copy-paste so the composite
        # runs on a 672² image instead of the full-resolution original (the
        # composite cost scales with background pixels — measured ~18 ms·core per
        # image at full res). CopyAndPasteModule reads the original dims from the
        # 'height'/'width' fields (preserved by this map) and scales the pasted
        # object by (new/orig) per axis, so the object's RELATIVE size and
        # placement distribution are exactly what compositing at full resolution
        # then resizing would have produced. The resize is also required before
        # padded_batch(4) anyway (variable spatial dims cannot be stacked).
        if self._mosaic_module is not None:
            _H, _W = self._mosaic_module._H, self._mosaic_module._W

            def _pre_resize_for_mosaic(ex, H=_H, W=_W):
                # Skip the (expensive, full-image float32) resize when the image
                # is already exactly the target size — true for every record of
                # a pre-resized dataset variant (tools/reencode_tfds_672.py) and
                # the occasional natively-sized capture. Runtime tf.cond because
                # decoded shapes are dynamic; the identity branch is ~free.
                img_in = ex['image']
                shp = tf.shape(img_in)

                def _resize():
                    return tf.cast(
                        tf.image.resize(tf.cast(img_in, tf.float32), [H, W], method='bilinear'),
                        tf.uint8,
                    )

                img = tf.cond(
                    tf.logical_and(tf.equal(shp[0], H), tf.equal(shp[1], W)),
                    lambda: img_in,
                    _resize,
                )
                img.set_shape([H, W, 3])
                return {**ex, 'image': img}

            ds = ds.map(_pre_resize_for_mosaic, num_parallel_calls=_AUTOTUNE)

        # Copy-Paste: zip with CNP dataset BEFORE mosaic (on pre-resized images).
        if self._copy_paste_module is not None and self._cnp_tfds_name:
            cnp_ds = self._load_cnp_dataset()
            ds = tf.data.Dataset.zip((ds, cnp_ds))
            copy_paste_fn = self._copy_paste_module.process_fn(is_training=True)
            ds = ds.map(copy_paste_fn, num_parallel_calls=_AUTOTUNE)

        # Mosaic: batch(4) → combine (4 in → 4 out) → unbatch.
        if self._mosaic_module is not None:
            mosaic_fn = self._mosaic_module.mosaic_fn(is_training=True)
            # Explicit padding_values for EVERY key in the decoder element spec
            # (PolygonDecoder/ServingBot output, preserved by copy-paste). Without
            # this, padded_batch pads every numeric field with 0, which is WRONG
            # for groundtruth_polygons: 0.0 is a valid (top-left) vertex coordinate,
            # so 0-padded rows would read as real vertices instead of the reserved
            # -1.0 sentinel (see docs/design_register entry 10) and corrupt the
            # PolyYOLO radial target. We pin -1.0 for polygons and the natural empty
            # value for every other field. Keyed by name so it survives spec
            # reordering; dtypes match the decoder exactly.
            _padding_values = {
                'image': tf.constant(0, tf.uint8),
                'source_id': tf.constant('', tf.string),
                'height': tf.constant(0, tf.int32),
                'width': tf.constant(0, tf.int32),
                'groundtruth_boxes': tf.constant(0.0, tf.float32),
                'groundtruth_classes': tf.constant(0, tf.int64),
                'groundtruth_polygons': tf.constant(-1.0, tf.float32),  # sentinel
                'groundtruth_is_crowd': tf.constant(False, tf.bool),
                'groundtruth_area': tf.constant(0.0, tf.float32),
                'groundtruth_dontcare': tf.constant(0, tf.int64),
                'groundtruth_dists': tf.constant(0.0, tf.float32),
            }
            ds = (
                ds
                .padded_batch(4, drop_remainder=True, padding_values=_padding_values)
                .map(mosaic_fn, num_parallel_calls=_AUTOTUNE)
                .unbatch()
                # mosaic_fn emits 4 samples per 4-group (the 4 mosaics share
                # source images, and the group coin flip makes mosaic/non-mosaic
                # arrive in runs of 4) — a small shuffle breaks those clusters up
                # before batching. ~128 × 1.4 MB decoded samples ≈ 180 MB buffer.
                # seed=self._seed+2: a DISTINCT seed from the pre-decode source
                # shuffle (seed) and the cnp source shuffle (seed+1) so the three
                # shuffle stages do not share an RNG stream (correlated permutations
                # across stages would partially undo each other's decorrelation).
                .shuffle(
                    128,
                    seed=None if self._seed is None else self._seed + 2,
                    reshuffle_each_iteration=True,
                )
            )

        if self._parser is not None:
            ds = ds.map(
                self._parser.parse_fn(is_training=True),
                num_parallel_calls=_AUTOTUNE,
            )

        ds = (
            ds
            .batch(batch_size, drop_remainder=self._drop_remainder)
            .prefetch(_AUTOTUNE)
        )
        return ds

    def _build_eval_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Build the evaluation dataset (no augmentation, no mosaic)."""
        raw_datasets = self._load_tfds_datasets()
        if len(raw_datasets) == 1:
            ds = raw_datasets[0]
        else:
            # Concatenate all eval datasets for full coverage.
            ds = raw_datasets[0]
            for extra in raw_datasets[1:]:
                ds = ds.concatenate(extra)

        if self._decoder is not None:
            ds = ds.map(self._decoder.decode, num_parallel_calls=_AUTOTUNE)

        if self._parser is not None:
            ds = ds.map(
                self._parser.parse_fn(is_training=False),
                num_parallel_calls=_AUTOTUNE,
            )

        ds = (
            ds
            .batch(batch_size, drop_remainder=self._drop_remainder)
            .prefetch(_AUTOTUNE)
        )
        return ds

    def _build_distance_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Build the distance-only stream (servingbot_polygon), batched to 16."""
        raw_datasets = self._load_tfds_datasets()
        ds = raw_datasets[0]  # distance stream is always a single TFDS

        ds = ds.shuffle(self._shuffle_buffer_size, seed=self._seed, reshuffle_each_iteration=True)
        ds = ds.repeat()  # repeat so the zip never exhausts

        if self._decoder is not None:
            ds = ds.map(self._decoder.decode, num_parallel_calls=_AUTOTUNE)

        if self._parser is not None:
            ds = ds.map(
                self._parser.parse_fn(is_training=True),
                num_parallel_calls=_AUTOTUNE,
            )

        ds = (
            ds
            .batch(batch_size, drop_remainder=self._drop_remainder)
            .prefetch(_AUTOTUNE)
        )
        return ds

    def _merge_streams(
        self,
        det_dataset: tf.data.Dataset,
        dist_dataset: tf.data.Dataset,
    ) -> tf.data.Dataset:
        """Zip detection + distance batches and concatenate on the batch dim."""
        return (
            tf.data.Dataset
            .zip((det_dataset, dist_dataset))
            .map(_concat_batch_dicts, num_parallel_calls=_AUTOTUNE)
            # Final terminal prefetch: overlap the batch-concat with the training
            # step. The sub-streams prefetch internally, but the merged stream is
            # what the training loop actually consumes.
            .prefetch(_AUTOTUNE)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tfds_datasets(self) -> List[tf.data.Dataset]:
        datasets = []
        for name, split in zip(self._tfds_names, self._tfds_split):
            try:
                ds = tfds.load(
                    name=name,
                    split=split,
                    data_dir=self._tfds_data_dir,
                    as_supervised=False,
                    download=self._tfds_download,
                    # Keep images as encoded bytes through shuffle; the decoders'
                    # tf.string branch decodes inside the parallel decode map.
                    # (A shuffle buffer of decoded images costs MBs per element.)
                    decoders={'image': tfds.decode.SkipDecoding()},
                )
                datasets.append(ds)
                log.info("Loaded TFDS: %s [%s]", name, split)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load TFDS dataset '{name}' split='{split}' "
                    f"from data_dir='{self._tfds_data_dir}'. "
                    f"Run /check-env to verify dataset availability. Error: {e}"
                ) from e
        return datasets

    def _load_cnp_dataset(self) -> tf.data.Dataset:
        """Load the copy-paste source dataset (infinite, shuffled)."""
        ds = tfds.load(
            name=self._cnp_tfds_name,
            split=self._cnp_tfds_split,
            data_dir=self._tfds_data_dir,
            as_supervised=False,
            download=self._tfds_download,
            # Encoded bytes through shuffle (RGBA PNG crops); CopyPasteDecoder's
            # tf.string branch decodes with channels=4 in the parallel map.
            decoders={'image': tfds.decode.SkipDecoding()},
        )
        # cnp source shuffle: seed=self._seed+1 — a DISTINCT seed from the detection
        # source shuffle (self._seed) and the post-unbatch shuffle (self._seed+2).
        # The cnp stream is zipped with the detection stream for copy-paste; sharing a
        # seed would lock the cnp permutation in lockstep with the detection one,
        # pairing the same background/paste-object indices every epoch.
        ds = ds.shuffle(
            500,
            seed=None if self._seed is None else self._seed + 1,
            reshuffle_each_iteration=True,
        ).repeat()
        if self._cnp_decoder is not None:
            ds = ds.map(self._cnp_decoder.decode, num_parallel_calls=_AUTOTUNE)
        return ds

    def _per_replica_batch_size(self, ctx: Optional[tf.distribute.InputContext]) -> int:
        if ctx is None:
            return self._global_batch_size
        return ctx.get_per_replica_batch_size(self._global_batch_size)

    @staticmethod
    def _normalize_weights(
        weights: Optional[List[float]], n: int
    ) -> List[float]:
        if weights is None:
            return [1.0 / n] * n
        total = sum(weights)
        return [w / total for w in weights]


# ---------------------------------------------------------------------------
# Factory helper used by YoloV8Task.build_inputs()
# ---------------------------------------------------------------------------

def build_input_reader_from_config(
    data_cfg,
    task_cfg,
    is_training: bool,
    decoder=None,
    parser=None,
    copy_paste_module=None,
    mosaic_module=None,
    distance_reader: Optional[InputReader] = None,
    cnp_decoder=None,
) -> InputReader:
    """Construct an InputReader from DataConfig + TaskConfig dataclasses.

    All pipeline components (decoder, parser, mosaic, copy-paste, distance
    reader) are built from config when not explicitly provided.  task.py calls
    this function without passing any components, so this is where they must be
    instantiated — failing to do so leaves parser=None and raw variable-size
    images reach batch() directly, causing a shape-mismatch crash.
    """
    names = [n.strip() for n in data_cfg.tfds_name.split(',')]
    splits = [s.strip() for s in data_cfg.tfds_split.split(',')]

    output_size = task_cfg.model.input_size[:2]   # [H, W]
    num_classes = task_cfg.num_classes
    parser_cfg  = data_cfg.parser
    min_level   = task_cfg.model.backbone.min_level  # 3
    max_level   = task_cfg.model.backbone.max_level  # 5

    # Decoder: normalise raw TFDS feature dicts into our standard schema.
    if decoder is None:
        from data_pipeline.tfds_decoders import PolygonDecoder
        decoder = PolygonDecoder(
            max_vertices=parser_cfg.max_vertices,
            num_classes=num_classes,
            resample_points=parser_cfg.resample_points,
        )

    # Parser: augment + resize images and build fixed-shape label tensors.
    # Without this every image stays at its native resolution and batch() fails.
    if parser is None:
        from data_pipeline.yolo_parser import V8ParserExtended
        levels = [str(l) for l in range(min_level, max_level + 1)]
        expanded_strides = {
            str(l): 8 * (2 ** (l - min_level))
            for l in range(min_level, max_level + 1)
        }
        parser = V8ParserExtended(
            output_size=output_size,
            expanded_strides=expanded_strides,
            levels=levels,
            max_vertices=parser_cfg.max_vertices,
            angle_step=parser_cfg.angle_step,
            with_polygons=parser_cfg.with_polygons,
            dummy_distance=parser_cfg.dummy_distance,
            skip_crowd_during_training=parser_cfg.skip_crowd_during_training,
            max_num_instances=parser_cfg.max_num_instances,
            aug_rand_hue=parser_cfg.aug_rand_hue,
            aug_rand_saturation=parser_cfg.aug_rand_saturation,
            aug_rand_brightness=parser_cfg.aug_rand_brightness,
            aug_rand_translate=parser_cfg.aug_rand_translate,
            aug_scale_min=parser_cfg.aug_scale_min,
            aug_scale_max=parser_cfg.aug_scale_max,
            random_flip=parser_cfg.random_flip,
            letter_box=parser_cfg.letter_box,
            resize_with_random_method=parser_cfg.resize_with_random_method,
            albumentations_frequency=parser_cfg.albumentations_frequency,
            area_thresh=parser_cfg.area_thresh,
            eval_gray_border=parser_cfg.eval_gray_border,
        )

    # Mosaic (training only).
    if mosaic_module is None and is_training:
        from data_pipeline.mosaic import Mosaic
        mosaic_cfg = parser_cfg.mosaic
        mosaic_module = Mosaic(
            output_size=output_size,
            mosaic_frequency=mosaic_cfg.mosaic_frequency,
            mixup_frequency=mosaic_cfg.mixup_frequency,
            mosaic_center=mosaic_cfg.mosaic_center,
            aug_scale_min=mosaic_cfg.aug_scale_min,
            aug_scale_max=mosaic_cfg.aug_scale_max,
            area_thresh=mosaic_cfg.area_thresh,
            mosaic_crop_mode=mosaic_cfg.mosaic_crop_mode,
            with_polygons=parser_cfg.with_polygons,
            degrees=mosaic_cfg.degrees,
            shear=mosaic_cfg.shear,
            perspective=mosaic_cfg.perspective,
            translate=mosaic_cfg.translate,
        )

    # Copy-paste (training only, when a source dataset is configured).
    if copy_paste_module is None and is_training and data_cfg.tfds_for_cnp:
        from data_pipeline.copy_paste import CopyAndPasteModule
        from data_pipeline.tfds_decoders import CopyPasteDecoder
        copy_paste_module = CopyAndPasteModule(prob=data_cfg.prob_copy_n_paste)
        if cnp_decoder is None:
            cnp_decoder = CopyPasteDecoder(num_classes=num_classes)

    # Distance reader (training only, when distance_data is configured).
    if distance_reader is None and is_training and getattr(data_cfg, 'distance_data', None) is not None:
        from data_pipeline.distance_parser import V8DistanceParser
        from data_pipeline.tfds_decoders import ServingBotDetDecoder
        dist_cfg = data_cfg.distance_data
        dist_decoder = ServingBotDetDecoder(
            num_classes=num_classes,
            resample_points=dist_cfg.parser.resample_points,
        )
        dist_parser = V8DistanceParser(
            output_size=output_size,
            max_num_instances=dist_cfg.parser.max_num_instances,
            angle_step=dist_cfg.parser.angle_step,
            with_polygons=dist_cfg.with_polygons,
            min_meter=task_cfg.min_distance,
            max_meter=task_cfg.max_distance,
            aug_rand_hue=dist_cfg.parser.aug_rand_hue,
            aug_rand_saturation=dist_cfg.parser.aug_rand_saturation,
            aug_rand_brightness=dist_cfg.parser.aug_rand_brightness,
            random_flip=dist_cfg.parser.random_flip,
            skip_crowd_during_training=dist_cfg.parser.skip_crowd_during_training,
        )
        distance_reader = InputReader(
            tfds_names=[dist_cfg.tfds_name],
            tfds_split=[dist_cfg.tfds_split],
            tfds_data_dir=dist_cfg.tfds_data_dir,
            global_batch_size=dist_cfg.global_batch_size,
            is_training=True,
            decoder=dist_decoder,
            parser=dist_parser,
            seed=data_cfg.seed,
            shuffle_buffer_size=dist_cfg.shuffle_buffer_size,
            drop_remainder=dist_cfg.drop_remainder,
        )

    return InputReader(
        tfds_names=names,
        tfds_split=splits,
        tfds_data_dir=data_cfg.tfds_data_dir,
        tfds_sampling_weights=data_cfg.tfds_sampling_weights,
        global_batch_size=data_cfg.global_batch_size,
        is_training=is_training,
        decoder=decoder,
        parser=parser,
        copy_paste_module=copy_paste_module,
        mosaic_module=mosaic_module,
        distance_reader=distance_reader,
        cnp_tfds_name=data_cfg.tfds_for_cnp,
        cnp_tfds_split=data_cfg.tfds_for_cnp_split,
        cnp_decoder=cnp_decoder,
        seed=data_cfg.seed,
        shuffle_buffer_size=data_cfg.shuffle_buffer_size,
        drop_remainder=data_cfg.drop_remainder,
        tfds_download=True,
        private_threadpool_size=getattr(data_cfg, 'private_threadpool_size', 0),
    )
