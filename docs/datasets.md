# Datasets

Training reads several [TensorFlow Datasets](https://www.tensorflow.org/datasets) (TFDS)
builders. They are project-specific (built and distributed by the team, not public TFDS).
This page documents what each one is, the schema each decoder expects, and where they must
live. Training requires them on disk.

## Where TFDS looks

All datasets are loaded from `tfds_data_dir` (set per stream in the experiment YAML, default
`/home/user/tensorflow_datasets/`). TFDS resolves `name:version` to
`<tfds_data_dir>/<name>/<version>/`. Set the same path via the `TFDS_DATA_DIR` env var for tools
that read it. Each dataset must already be built (the `dataset_info.json` + `.tfrecord` shards
present) under that directory.

## The datasets used

| Role | TFDS (in `yolov8_poly_dist.yaml`) | Decoder | Stream |
|------|-----------------------------------|---------|--------|
| Detection (train) | `cleaner_polygon2026:2.0.0`, `field_misrecog2026:1.0.0`, `station_misrecog:1.1.0` | `PolygonDecoder` | multi-source weighted sampling (`tfds_sampling_weights`) |
| Detection (val) | `cleaner_polygon2026:2.0.0` (split `test`) | `PolygonDecoder` | validation |
| Copy-paste source | `cleaner_copy_paste:1.0.0` (split `train_f`) | `CopyPasteDecoder` | merged into detection before mosaic (`prob_copy_n_paste`) |
| Distance | `servingbot_polygon:1.0.1` (split `train`) | `ServingBotDetDecoder` | separate stream, training only, merged on the batch dim |

The detection train stream samples from multiple TFDS builders with stationary weights: set
`train_data.tfds_sampling_weights` to a list aligned with the comma-separated `tfds_name`.

## Expected schemas

The decoders access fields directly (TF Model Garden `MSCOCODecoder` style:
`data['image']`, `data['image/id']`, `data['objects'][field]`). A dataset must expose exactly
these features. See `data_pipeline/tfds_decoders.py` (authoritative).

**Detection - `cleaner_polygon2026` / `field_misrecog2026` / `station_misrecog`** (identical):
```
image:               uint8   [H, W, 3]
image/filename:      string
image/id:            int64
objects/bbox:        float32 [N, 4]      ymin/xmin/ymax/xmax, normalized [0,1]
objects/label:       int64   [N]         class id (0-based; see configs/class_map.py)
objects/area:        int64   [N]
objects/id:          int64   [N]
objects/is_crowd:    bool    [N]
objects/is_dontcare: bool    [N]
objects/points:      float32 [N, 3972]   polygon xy interleaved, -1 padded
```

**Distance - `servingbot_polygon`** - same as detection except:
```
objects/points:    float32 [N, 10940]
objects/distance:  float32 [N]          per-object distance in meters (valid range [0.5, 10.0])
(no objects/is_dontcare)
```

**Copy-paste - `cleaner_copy_paste`** - flat (no nested `objects`):
```
image:          uint8   [H, W, 4]   RGBA (the alpha channel is the object mask)
image/filename: string
image/id:       int64
label:          int64   scalar       single object per image
obj_id:         int64   scalar
orig_bbox:      float32 [4]
points:         float32 [3972]       polygon xy interleaved, -1 padded
```

## Classes

39 detection classes, defined in `configs/class_map.py` (`DETECTION_CLASSES`, index = the
`objects/label` id). `num_classes` in the YAML must match. The distance dataset uses a class
remap (`SERVINGBOT_CLASS_REMAP`) applied at decode.

## Pre-resized 672x672 variants (optional, faster)

Decode + resize of full-resolution images can bottleneck the pipeline. Pre-resized `<name>_672`
dataset copies store 672x672 JPEG plus `orig_height`/`orig_width` (which `PolygonDecoder`
prefers for copy-paste scaling). If such variants exist under `tfds_data_dir`, point the YAML
at the `_672` names (the commented switch-over lines are already in the tier YAMLs).
Detection sets only: the distance parser letterboxes (aspect-preserving), so
`servingbot_polygon` must stay full-resolution.

## Init checkpoint

The shipped configs warm-start backbone + decoder from
`task.init_checkpoint: initial_checkpoint_folder/ckpt-920304`. `init_checkpoint` must be a
checkpoint produced by this codebase; the selected modules (`init_checkpoint_modules`, default
backbone + decoder) are restored via the EMA-aware full-model loader while the rest keep their
fresh init (see [configuration.md](configuration.md)). Provide that checkpoint, or set
`init_checkpoint: null` to train from scratch.
