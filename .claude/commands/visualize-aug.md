# /visualize-aug — Save augmentation stage visualizations to disk

Runs `tools/visualize_augmentations.py` to save images at each augmentation
stage (raw decode → copy-paste → mosaic → affine → HSV → albumentations).
Critical for debugging polygon/box alignment bugs.

## Usage

```
/visualize-aug                 # save 20 samples to /tmp/aug_debug/
/visualize-aug --n 50          # save 50 samples
/visualize-aug --stage mosaic  # save only the mosaic stage output
```

## What to run

```bash
python tools/visualize_augmentations.py \
  --config configs/experiments/yolo/yolov8_poly_dist.yaml \
  --output /tmp/aug_debug/ \
  --n 20 \
  [--stage all|mosaic|copy_paste|affine|hsv|albumentations]
```

## What to verify visually

- Bounding boxes are tight around objects (not clipped or shifted)
- Polygon vertices align with object boundaries
- After mosaic: 4 images stitched, all labels present
- After flip: polygon x-coordinates correctly mirrored
- After letterbox: gray padding, aspect ratio preserved
- No black or zero-filled patches (corrupted augmentation)

If any stage shows misaligned boxes or polygons, that stage has a bug.
Flag it and check the coordinate transform code before proceeding.
