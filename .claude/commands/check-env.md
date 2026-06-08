# /check-env — Verify the training environment is correctly set up

Runs a series of environment checks before starting a training run. Use this
at the start of each cloud session to catch problems early.

## What to run

Run each of these checks in sequence:

```bash
# 1. Python and TF version
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# 2. CUDA / cuDNN
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# 3. Package imports
python -c "import dacite, yaml, albumentations, cv2, pycocotools, sklearn; print('all imports OK')"

# 4. TFDS dataset availability
python -c "
import tensorflow_datasets as tfds
for name in ['cleaner_polygon2026:2.0.0', 'field_misrecog2026:1.0.0', 'station_misrecog:1.1.0', 'servingbot_polygon:1.0.1', 'cleaner_copy_paste:1.0.0']:
    try:
        b = tfds.builder(name, data_dir='/home/user/tensorflow_datasets/')
        print(f'OK: {name}')
    except Exception as e:
        print(f'MISSING: {name} — {e}')
"

# 5. Config loading
python -c "from configs.yaml_loader import load_config; cfg = load_config('configs/experiments/yolo/yolov8_poly_dist.yaml'); print('Config OK — classes:', cfg.task.model.num_classes)"

# 6. Disk space
df -h .

# 7. GPU memory
python -c "import tensorflow as tf; [print(tf.config.experimental.get_memory_info(f'GPU:{i}')) for i in range(len(tf.config.list_physical_devices('GPU')))]"
```

## What to report

- TF version (expected: 2.16.1)
- GPU count (expected: ≥ 2)
- All 5 TFDS datasets reachable (MISSING = blocker before training)
- Disk free ≥ 20 GB
- Any import errors

Stop and fix any MISSING or ERROR before starting training.
