# /train — Launch a training run

Starts training using `scripts/train.py`. Always reads the config from a YAML file.

## Usage

```
/train bbox                    # train yolov8_bbox tier
/train poly                    # train yolov8_poly tier
/train poly_dist               # train yolov8_poly_dist (full)
/train bbox --debug            # debug mode: 10 steps, visualize augmentations
/train poly --resume           # auto-resume from latest checkpoint in output_dir
```

## What to run

Map the shorthand to the config path:
- `bbox`      → `configs/experiments/yolo/yolov8_bbox.yaml`
- `poly`      → `configs/experiments/yolo/yolov8_poly.yaml`
- `poly_dist` → `configs/experiments/yolo/yolov8_poly_dist.yaml`

Default output dir: `runs/{experiment_name}_{timestamp}/`

```bash
python scripts/train.py \
  --config configs/experiments/yolo/yolov8_{tier}.yaml \
  --output_dir runs/{tier}_{timestamp} \
  [--debug] [--resume]
```

Before launching, verify:
1. TFDS datasets are accessible (`tfds.builder(name, data_dir=...)` should not raise)
2. GPU is visible (`tf.config.list_physical_devices('GPU')` should return ≥1 device)
3. Disk space ≥ 20 GB free (for 300 checkpoints)

Report: config used, output dir, initial loss values after step 1.
