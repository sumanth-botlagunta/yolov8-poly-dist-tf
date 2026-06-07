# Documentation

Developer documentation for the **YOLOv8 Polygon + Distance (TensorFlow)** codebase.
For a quick start (setup, training/eval/export commands), see the top-level
[README.md](../README.md). For the Claude-Code-oriented project summary, see
[CLAUDE.md](../CLAUDE.md).

## Contents

| Doc | Covers |
|-----|--------|
| [architecture.md](architecture.md) | Model structure: backbone, FPN-PAN decoder, the 6 heads, anchors/strides, detection generator. |
| [data_pipeline.md](data_pipeline.md) | The tf.data pipeline end to end: TFDS decoding, copy-paste, mosaic, augmentations, parsers, the distance-stream merge, and the three polygon formats. |
| [losses.md](losses.md) | TAL assignment, CIoU/DFL/cls, polygon and distance losses, the gains, and the normalization conventions (incl. the documented deviations). |
| [training.md](training.md) | Configs, the optimizer/EMA, the training loop, checkpoints, mixed precision/XLA, and distributed training. |
| [testing.md](testing.md) | Test layout, how to run subsets, what needs TFDS, and CI. |

## Conventions used in these docs
- Tensor coordinate order is called out explicitly (`yxyx` vs `xyxy`, normalized vs pixels) because mismatches are the most common source of bugs.
- File references use `path:symbol` so they are easy to grep.
