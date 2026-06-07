"""Training entry point for YOLOv8 polygon + distance model.

Usage:
    python scripts/train.py \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --output_dir /tmp/yolo_run

Flags:
    --config      Path to experiment YAML (required).
    --output_dir  Directory for checkpoints and TensorBoard events (required).
    --debug       Run eagerly and enable verbose logging.
"""

from absl import app, flags, logging
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',     None, 'Path to experiment YAML config.', required=True)
    flags.DEFINE_string('output_dir', None, 'Output directory for checkpoints and logs.', required=True)
    flags.DEFINE_bool  ('debug',      False, 'Enable eager execution and verbose logging.')
except flags.DuplicateFlagError:
    pass


def main(_):
    if FLAGS.debug:
        tf.config.run_functions_eagerly(True)
        logging.set_verbosity(logging.DEBUG)

    from configs.yaml_loader import load_config
    from train.task import YoloV8Task
    from train.trainer import YoloV8Trainer

    config = load_config(FLAGS.config)

    strategy = tf.distribute.MirroredStrategy()
    logging.info("Running with %d replica(s).", strategy.num_replicas_in_sync)

    task    = YoloV8Task(config)
    trainer = YoloV8Trainer(
        task=task,
        config=config,
        output_dir=FLAGS.output_dir,
        strategy=strategy,
        debug=FLAGS.debug,
    )

    trainer.train(total_epochs=config.trainer.train_epochs)


if __name__ == '__main__':
    app.run(main)
