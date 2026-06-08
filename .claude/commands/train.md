# /train — Launch a training run

Starts training using `scripts/run_train.py`. Always reads the config from a YAML file.

## Usage

```
/train bbox                    # train yolov8_bbox tier
/train poly                    # train yolov8_poly tier
/train poly_dist               # train yolov8_poly_dist (full)
/train poly_dist --resume      # auto-resume from latest checkpoint in output_dir
/train poly_dist --resume_from runs/run1/ckpt-2388   # resume from specific checkpoint
```

## What to run

Map the shorthand to the config path:
- `bbox`      → `configs/experiments/yolo/yolov8_bbox.yaml`
- `poly`      → `configs/experiments/yolo/yolov8_poly.yaml`
- `poly_dist` → `configs/experiments/yolo/yolov8_poly_dist.yaml`

Default output dir: `runs/{experiment_name}_{timestamp}/`

```bash
python scripts/run_train.py \
  --config configs/experiments/yolo/yolov8_{tier}.yaml \
  --output_dir runs/{tier}_{timestamp} \
  [--resume] \
  [--resume_from /path/to/specific/ckpt]
```

Logs are written to both the console and `{output_dir}/train.log` automatically.
TensorBoard events land in `{output_dir}/tb_events/`. Augmented training images
are logged to TensorBoard under `train/augmentations` each epoch.

Before launching, verify:
1. TFDS datasets are accessible (run `/check-env` first)
2. GPU is visible (`tf.config.list_physical_devices('GPU')` should return ≥1 device)
3. Disk space ≥ 20 GB free (300 checkpoints × ~500 MB each)

Report: config used, output dir, initial loss values after step 1.
