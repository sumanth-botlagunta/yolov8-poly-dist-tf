# /visualize-aug — Inspect augmented training images

During training, augmented images are logged automatically to TensorBoard every epoch
under `train/augmentations`. This is the primary way to inspect augmentation quality.
For deeper per-stage debugging, use the pipeline-debugger agent.

## TensorBoard (primary — no extra script needed)

After starting a training run, open TensorBoard and navigate to the **Images** tab:

```bash
tensorboard --logdir runs/{output_dir}/tb_events/
```

Look for: `train/augmentations` (raw augmented batch) and `val/predictions`
(model predictions overlaid on validation images).

## What to verify

- Bounding boxes are tight around objects (not clipped or shifted)
- Polygon vertices align with object boundaries
- After mosaic: 4 images stitched, all labels present
- After flip: polygon x-coordinates correctly mirrored
- After letterbox: gray padding at edges, correct aspect ratio
- No black or zero-filled patches (corrupted augmentation)

## Debugging a specific stage

If you suspect a bug in a specific augmentation stage (mosaic, copy-paste, affine, HSV),
use the pipeline-debugger agent. It will reproduce the issue with a synthetic tensor
in eager mode and trace coordinate transforms step by step.

If any stage shows misaligned boxes or polygons, check the coordinate transform code
before proceeding — these bugs compound across stages.
