# Documentation

Developer documentation for the **YOLOv8 Polygon + Distance (TensorFlow)** codebase.
For a quick start (setup, training/eval/export commands), see the top-level
[README.md](../README.md).

## Guides (task-oriented — start here)

Step-by-step procedures for the core flows. The reference docs below are the deep-dives.

| Guide | Walks you through |
|-------|-------------------|
| [guides/training.md](guides/training.md) | Launching, monitoring, resuming, and stopping a training run end to end. |
| [guides/validation.md](guides/validation.md) | Evaluating a checkpoint, reading the metrics, and picking the best one. |
| [guides/finetuning.md](guides/finetuning.md) | Warm-starting a new run from an existing checkpoint and tuning it. |
| [guides/deployment.md](guides/deployment.md) | SavedModel → SNPE/DLC export, quantization, and verifying device == host. |
| [guides/inference.md](guides/inference.md) | Running a checkpoint/SavedModel on a folder of images (predictions JSON + visuals). |

## Reference

| Doc | Covers |
|-----|--------|
| [architecture.md](architecture.md) | Model structure: backbone, FPN-PAN decoder, the 6 heads, anchors/strides, detection generator, polygon formats. |
| [datasets.md](datasets.md) | The required TFDS datasets, their schemas, where they live, the 672² variants, and the init checkpoint. |
| [data_pipeline.md](data_pipeline.md) | The tf.data pipeline end to end: TFDS decoding, copy-paste, mosaic, augmentations, parsers, the distance-stream merge, and the three polygon formats. |
| [losses.md](losses.md) | TAL assignment, CIoU/DFL/cls, polygon and distance losses, the gains, and the normalization conventions (incl. the documented deviations). |
| [training.md](training.md) | Configs, the optimizer/EMA, the training loop, checkpoints, mixed precision/XLA, and distributed training. |
| [configuration.md](configuration.md) | Every config section/field explained: how YAMLs load, the dataclass layout, defaults, and validated invariants. |
| [scripts.md](scripts.md) | Every runnable script and analysis notebook: purpose, inputs explained, and a copy-paste command. |
| [metrics.md](metrics.md) | Glossary of every metric `eval` prints (mAP/F1, polygon, distance). |
| [device_export.md](device_export.md) | On-device (Qualcomm SNPE/DLC) export workflow and the box channel-order contract. |
| [troubleshooting.md](troubleshooting.md) | Common training/eval/export failures and what to check. |
| [testing.md](testing.md) | Test layout, how to run subsets, and what needs TFDS. |

## Conventions used in these docs
- Tensor coordinate order is called out explicitly (`yxyx` vs `xyxy`, normalized vs pixels) because mismatches are the most common source of bugs.
- File references use `path:symbol` so they are easy to grep.
