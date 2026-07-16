"""Training entry point for YOLOv8 polygon + distance model.

Usage:
    python -m train.run_train \
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
    flags.DEFINE_string('finetune_from', None,
                        'Fine-tune: seed a fresh run from a trained checkpoint (full model, '
                        'EMA/deployed weights; fresh optimizer/EMA/LR). Overrides the config '
                        'task.finetune_from. Distinct from --resume_from (same run, continues).')
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
    """Validate config invariants before any GPU or model work begins.

    Collects every violation and raises ValueError listing them, so problems
    surface before the model build or TFDS load.
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
    if getattr(trainer, 'grad_accum_steps', 1) < 1:
        errors.append(f"trainer.grad_accum_steps must be >= 1, got {trainer.grad_accum_steps}")
    if trainer.train_epochs <= 0:
        errors.append(f"trainer.train_epochs must be > 0, got {trainer.train_epochs}")
    if trainer.steps_per_loop <= 0:
        errors.append(
            f"trainer.steps_per_loop must be > 0 (derived from train_total_examples "
            f"// global_batch_size), got {trainer.steps_per_loop}. The training loop "
            "runs exactly steps_per_loop steps per epoch — without it epochs have "
            "no defined length."
        )
    # decay_steps is an explicit YAML value (not derived); if it drifts from the
    # optimizer-update count the cosine schedule ends early or never anneals. The
    # schedule advances once per optimizer update, which under gradient
    # accumulation (grad_accum_steps=N) is one update per N micro-steps, so it
    # should span train_steps // N updates. At N=1 that equals train_steps.
    n_accum = max(1, getattr(trainer, 'grad_accum_steps', 1))
    decay = trainer.optimizer_config.learning_rate.decay_steps
    expected_decay = trainer.train_steps // n_accum
    if trainer.train_steps > 0 and decay != expected_decay:
        accum_note = "" if n_accum == 1 else f" // grad_accum_steps {n_accum}"
        logging.warning(
            "LR decay_steps (%d) != optimizer updates (%d = train_steps %d%s, "
            "steps_per_loop %d × epochs %d). The cosine schedule will %s. Set "
            "decay_steps to %d in the YAML unless this is intentional.",
            decay, expected_decay, trainer.train_steps, accum_note,
            trainer.steps_per_loop, trainer.train_epochs,
            "reach its floor before training ends" if decay < expected_decay
            else "never reach its floor",
            expected_decay,
        )

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
    finetune = getattr(task, 'finetune_from', None)
    if finetune:
        if task.init_checkpoint:
            errors.append(
                "task.finetune_from and task.init_checkpoint are mutually exclusive "
                "(fine-tune loads the FULL model; init_checkpoint is for transfer-init). "
                "Set only one."
            )
        if not os.path.exists(finetune + ".index"):
            errors.append(
                f"task.finetune_from not found: '{finetune}' (looked for '{finetune}.index')."
            )

    freeze = getattr(task, 'freeze_modules', None) or []
    _valid_modules = {'backbone', 'decoder', 'head'}
    bad = [m for m in freeze if m not in _valid_modules]
    if bad:
        errors.append(f"task.freeze_modules has unknown module(s) {bad}; "
                      f"valid: {sorted(_valid_modules)}.")
    if set(freeze) >= _valid_modules:
        errors.append("task.freeze_modules freezes every module — nothing to train. "
                      "Leave at least one (e.g. the head) unfrozen.")
    if getattr(task, 'freeze_backbone_layers', 0) < 0:
        errors.append("task.freeze_backbone_layers must be >= 0, got "
                      f"{task.freeze_backbone_layers}")

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
    input_size = task.model.input_size
    if len(input_size) >= 2 and input_size[0] != input_size[1]:
        # The polygon evaluator assumes a square input (radial vertex decode uses
        # one scale for both axes); a non-square config trains fine then crashes at
        # the first validation.
        errors.append(
            f"task.model.input_size must be square (H == W), got "
            f"{input_size[0]}x{input_size[1]}; the polygon evaluator requires it"
        )
    if task.model.output_poly_size != (360 // task.model.angle_step):
        errors.append(
            f"task.model.output_poly_size ({task.model.output_poly_size}) "
            f"must equal 360 // angle_step ({360 // task.model.angle_step})"
        )

    # --- mosaic group / diversity ---
    mosaic_cfg = getattr(getattr(task.train_data, "parser", None), "mosaic", None)
    if mosaic_cfg is not None:
        g, r = mosaic_cfg.group_size, mosaic_cfg.decodes_per_output
        if g < 4:
            errors.append(f"mosaic.group_size ({g}) must be >= 4")
        if r < 1:
            errors.append(f"mosaic.decodes_per_output ({r}) must be >= 1")
        elif g % r != 0:
            errors.append(
                f"mosaic.group_size ({g}) must be a multiple of "
                f"mosaic.decodes_per_output ({r})"
            )
        tc_min = getattr(mosaic_cfg, "tile_crop_min", 0.0)
        tc_max = getattr(mosaic_cfg, "tile_crop_max", 0.0)
        if tc_max > 0.0 and not 0.0 < tc_min <= tc_max <= 1.0:
            # A crop window is a fraction of the tile content, so max > 1 is
            # meaningless and max > 0 requires min > 0 (a well-formed [min, max]).
            errors.append(
                f"mosaic.tile_crop bounds invalid: need 0 < min <= max <= 1 "
                f"(or 0/0 to disable), got [{tc_min}, {tc_max}]"
            )
        if 1 <= r < 4:
            # R<4 is supported (not an error). Sidon-shift source selection caps
            # any two outputs of a group at one shared image, so R<4 avoids
            # near-duplicate outputs; each image just recurs in 4/R outputs, i.e.
            # an epoch consumes R/4 as many distinct images as R=4.
            logging.info(
                "mosaic.decodes_per_output=%d (< 4): each decoded image is "
                "reused in %d outputs per group (Sidon selection, <=1 shared "
                "source between any two outputs). R=4 maximizes distinct images "
                "per epoch at ~%dx the decode cost.", r, 4 // r, 4 // r)

    # --- single-image (non-mosaic) pre-warp rotation ---
    parser_cfg = getattr(task.train_data, "parser", None)
    if parser_cfg is not None:
        rotate = getattr(parser_cfg, "rotate", False)
        rotate_degrees = getattr(parser_cfg, "rotate_degrees", None)
        deg_ok = (
            isinstance(rotate_degrees, (int, float))
            and not isinstance(rotate_degrees, bool)
            and rotate_degrees > 0
        )
        if rotate and not deg_ok:
            errors.append(
                f"parser.rotate is true but parser.rotate_degrees "
                f"({rotate_degrees!r}) must be a number > 0"
            )
        elif not rotate and rotate_degrees is not None and not deg_ok:
            errors.append(
                f"parser.rotate_degrees ({rotate_degrees!r}) must be a number "
                f"> 0 when set"
            )

    # --- validation stream sanity ---
    vd = getattr(task, "validation_data", None)
    if vd is not None and getattr(vd, "is_training", False):
        # is_training=True on the val stream builds the infinite training
        # pipeline, so `for inputs in val_ds` never terminates and validation
        # hangs. DataConfig defaults is_training to True, so a YAML that omits
        # the key hits exactly this.
        errors.append(
            "validation_data.is_training must be false (an infinite training "
            "stream as the val set hangs validation forever)"
        )
    if vd is not None and getattr(vd, "drop_remainder", False):
        # A dropped final partial batch silently omits images from the metrics;
        # validation must score every image.
        errors.append(
            "validation_data.drop_remainder must be false (validation must score "
            "every image; a dropped final partial batch omits images from the metrics)"
        )
    # --- multi-TFDS sampling weights ---
    td = task.train_data
    weights = getattr(td, "tfds_sampling_weights", None)
    names = [n for n in str(getattr(td, "tfds_name", "")).split(",") if n.strip()]
    if weights and len(names) > 1 and len(weights) != len(names):
        errors.append(
            f"len(tfds_sampling_weights) ({len(weights)}) != number of datasets "
            f"in tfds_name ({len(names)}) — weights are index-aligned, a "
            "mismatch crashes deep inside sample_from_datasets"
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

    # Thread-pool caps must run before the TF context initializes (before any op
    # executes). On cgroup-capped machines TF sizes its pools to the visible core
    # count while the process can use only a fraction, so its threads thrash.
    inter = getattr(runtime_cfg, 'inter_op_threads', 0)
    intra = getattr(runtime_cfg, 'intra_op_threads', 0)
    if inter > 0:
        tf.config.threading.set_inter_op_parallelism_threads(inter)
        logging.info("inter_op_parallelism_threads = %d", inter)
    if intra > 0:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        logging.info("intra_op_parallelism_threads = %d", intra)

    if runtime_cfg.enable_xla:
        tf.config.optimizer.set_jit(True)
        logging.info("XLA JIT compilation enabled.")

    # Normalize so trailing whitespace / case / common aliases can't silently
    # bypass the dtype handling and fall through to float32 unannounced.
    precision = (runtime_cfg.mixed_precision_dtype or "float32").strip().lower()
    if precision in ("float16", "fp16", "half", "mixed_float16"):
        # float16 needs dynamic loss scaling to avoid gradient underflow, which
        # the SGDTorch optimizer + bare GradientTape step do not apply. Reject it
        # until loss scaling is wired up; bfloat16 needs none and is the
        # recommended Tensor-Core path.
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
    if FLAGS.finetune_from:                      # CLI overrides the config field
        config.task.finetune_from = FLAGS.finetune_from
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
