"""Training entry point for YOLOv8 polygon + distance model.

Usage:
    python scripts/run_train.py \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --output_dir /tmp/yolo_run

Flags:
    --config      Path to experiment YAML (required).
    --output_dir  Directory for checkpoints and TensorBoard events (required).
    --debug       Run eagerly and enable verbose logging (overrides runtime config).
"""

import logging as stdlib_logging
import os

from absl import app, flags, logging
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',     None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('output_dir', None, 'Output directory for checkpoints and logs.', required=True)
    flags.DEFINE_bool  ('debug',      False, 'Enable eager execution and verbose logging.')
    flags.DEFINE_string('resume_from', None, 'Resume from a specific checkpoint (overrides auto-latest).')
except flags.DuplicateFlagError:
    pass


def _setup_file_logging(log_path: str) -> None:
    """Write all Python logging (including absl) to a persistent log file."""
    handler = stdlib_logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setFormatter(stdlib_logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    stdlib_logging.root.addHandler(handler)
    stdlib_logging.root.setLevel(stdlib_logging.INFO)
    try:
        from absl import logging as absl_logging
        absl_logging.use_python_logging()
    except Exception:
        pass


def _build_strategy(runtime_cfg) -> tf.distribute.Strategy:
    """Build distribution strategy from RuntimeConfig."""
    strategy_name = runtime_cfg.distribution_strategy.lower()
    num_gpus      = runtime_cfg.num_gpus

    if strategy_name == "one_device":
        return tf.distribute.OneDeviceStrategy("/gpu:0")

    # MirroredStrategy: use num_gpus GPUs if specified, else all available.
    if num_gpus > 0:
        gpus = tf.config.list_logical_devices('GPU')
        devices = [g.name for g in gpus[:num_gpus]]
        return tf.distribute.MirroredStrategy(devices=devices)

    return tf.distribute.MirroredStrategy()


def _validate_config(config, output_dir: str) -> None:
    """Validate mandatory config fields before any GPU or model work begins.

    Raises ValueError with a clear message on the first problem found so the
    user sees exactly what is wrong before waiting for model build or TFDS load.
    """
    import os
    errors = []

    task    = config.task
    trainer = config.trainer

    # --- core training fields ---
    if task.num_classes <= 0:
        errors.append(f"task.num_classes must be > 0, got {task.num_classes}")
    if trainer.train_total_examples <= 0:
        errors.append(
            f"trainer.train_total_examples must be > 0, got {trainer.train_total_examples}. "
            "Set it to the total number of training images."
        )
    if trainer.train_epochs <= 0:
        errors.append(f"trainer.train_epochs must be > 0, got {trainer.train_epochs}")

    # --- dataset directories ---
    for label, data_cfg in [
        ("train_data",      task.train_data),
        ("validation_data", task.validation_data),
    ]:
        d = data_cfg.tfds_data_dir
        if not os.path.isdir(d):
            errors.append(
                f"task.{label}.tfds_data_dir does not exist: '{d}'. "
                "Update the YAML or create/symlink the directory."
            )

    dist_data = getattr(task.train_data, 'distance_data', None)
    if dist_data is not None:
        d = dist_data.tfds_data_dir
        if not os.path.isdir(d):
            errors.append(
                f"task.train_data.distance_data.tfds_data_dir does not exist: '{d}'"
            )

    # --- init checkpoint ---
    ckpt = task.init_checkpoint
    if ckpt:
        index_file = ckpt + ".index"
        if not os.path.exists(index_file):
            errors.append(
                f"task.init_checkpoint not found: '{ckpt}' "
                f"(looked for '{index_file}'). "
                "Update the path or remove the field to train from scratch."
            )

    # --- model consistency ---
    if task.model.output_poly_size != (360 // task.model.angle_step):
        errors.append(
            f"task.model.output_poly_size ({task.model.output_poly_size}) "
            f"must equal 360 // angle_step ({360 // task.model.angle_step})"
        )

    # --- output directory writable ---
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        errors.append(f"Cannot create output_dir '{output_dir}': {e}")

    if errors:
        msg = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"Config validation failed ({len(errors)} error(s)):\n{msg}")

    logging.info("Config validation passed.")


def _apply_runtime_config(runtime_cfg, debug: bool) -> None:
    """Apply framework-level settings from RuntimeConfig before building the model."""
    run_eagerly = debug or runtime_cfg.run_eagerly
    if run_eagerly:
        tf.config.run_functions_eagerly(True)
        logging.set_verbosity(logging.DEBUG)

    if runtime_cfg.enable_xla:
        tf.config.optimizer.set_jit(True)
        logging.info("XLA JIT compilation enabled.")

    # Normalize so trailing whitespace / case / common aliases can't silently
    # bypass the dtype handling and fall through to float32 unannounced.
    precision = (runtime_cfg.mixed_precision_dtype or "float32").strip().lower()
    if precision in ("float16", "fp16", "half", "mixed_float16"):
        # float16 needs dynamic loss scaling to avoid gradient underflow, but the
        # custom SGDTorch optimizer + bare GradientTape training step do not apply
        # any loss scaling. Enabling mixed_float16 here would train poorly / stall
        # silently. Reject it until loss scaling is wired up; bfloat16 is supported
        # (no loss scaling required) and is the recommended Tensor-Core path.
        raise NotImplementedError(
            "mixed_precision_dtype='float16' is not supported: the training step "
            "has no loss scaling, so float16 gradients would underflow. Use "
            "'bfloat16' (no loss scaling needed) or 'float32'."
        )
    elif precision in ("bfloat16", "bf16", "mixed_bfloat16"):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        logging.info("Mixed precision: bfloat16 policy active.")
    elif precision in ("float32", "fp32", ""):
        pass  # default policy; nothing to do
    else:
        raise ValueError(
            f"Unknown mixed_precision_dtype={runtime_cfg.mixed_precision_dtype!r}. "
            "Expected one of: 'float32', 'bfloat16', 'float16'."
        )


def main(_):
    from configs.yaml_loader import load_config
    from train.task import YoloV8Task
    from train.trainer import YoloV8Trainer

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    _setup_file_logging(os.path.join(FLAGS.output_dir, 'train.log'))

    config = load_config(FLAGS.config)
    _validate_config(config, FLAGS.output_dir)

    _apply_runtime_config(config.runtime, FLAGS.debug)
    strategy = _build_strategy(config.runtime)
    logging.info("Distribution strategy: %s  (%d replica(s))",
                 config.runtime.distribution_strategy,
                 strategy.num_replicas_in_sync)

    task    = YoloV8Task(config)
    trainer = YoloV8Trainer(
        task=task,
        config=config,
        output_dir=FLAGS.output_dir,
        strategy=strategy,
        debug=FLAGS.debug,
        resume_from=FLAGS.resume_from,
    )

    trainer.train(total_epochs=config.trainer.train_epochs)


if __name__ == '__main__':
    app.run(main)
