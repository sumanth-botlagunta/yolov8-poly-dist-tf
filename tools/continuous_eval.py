"""Continuous evaluation: watches an output directory for new checkpoints and evaluates them.

Usage:
    python tools/continuous_eval.py \
        --config  configs/experiments/yolo/yolov8_poly_dist.yaml \
        --watch_dir /output/run1/ \
        --interval 300
"""

import json
import logging
import os
import time

from absl import app, flags, logging as absl_logging
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('config',    None, 'Path to experiment YAML config.', required=True)
flags.DEFINE_string('watch_dir', None, 'Directory to watch for checkpoints.', required=True)
flags.DEFINE_integer('interval', 300, 'Seconds between polls.')
flags.DEFINE_integer('max_evals', 0, 'Max number of evaluations (0 = unlimited).')

log = logging.getLogger(__name__)


def _eval_checkpoint(config, ckpt_path: str) -> dict:
    from models.yolo_v8 import build_yolov8
    from train.task import YoloV8Task
    from eval.coco_metrics import COCOEvaluator

    task = YoloV8Task(config)
    model = build_yolov8(config.task.model)
    model.deploy = True
    model.build_and_init(config.task.model.input_size)

    # Prefer EMA weights (what the trainer validates with and tools/eval.py uses)
    # so the continuous-eval curve matches the official eval rather than the
    # noisier raw-weight curve.
    from tools.ckpt_loading import restore_eval_weights
    kind = restore_eval_weights(model, ckpt_path)
    log.info("Loaded checkpoint (%s weights): %s", kind, ckpt_path)

    val_ds   = task.build_inputs(config.task.validation_data)
    img_size = tuple(config.task.model.input_size[:2])
    coco_ev  = COCOEvaluator(num_classes=config.task.num_classes, image_size=img_size)

    from train.task import normalize_images

    for step, (images, labels) in enumerate(val_ds):
        # Eval parser emits uint8; the model needs float32 [0, 1].
        preds = model(normalize_images(images), training=False)
        coco_ev.update(preds, labels)
        if step % 50 == 0:
            log.info("  eval step %d", step)

    metrics = coco_ev.evaluate()
    metrics['checkpoint'] = ckpt_path
    metrics['timestamp']  = time.strftime('%Y-%m-%dT%H:%M:%S')
    return metrics


def main(_):
    from configs.yaml_loader import load_config
    config = load_config(FLAGS.config)

    log_path    = os.path.join(FLAGS.watch_dir, 'eval_log.jsonl')
    seen        = set()
    eval_count  = 0

    log.info("Watching %s every %ds. Results → %s", FLAGS.watch_dir, FLAGS.interval, log_path)

    while True:
        latest = tf.train.latest_checkpoint(FLAGS.watch_dir)
        if latest and latest not in seen:
            seen.add(latest)
            log.info("New checkpoint: %s — evaluating...", latest)
            try:
                metrics = _eval_checkpoint(config, latest)
                with open(log_path, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                log.info("mAP=%.4f  F1@50=%.4f", metrics.get('mAP', 0), metrics.get('F1score50', 0))
                eval_count += 1
            except Exception as e:
                log.error("Eval failed for %s: %s", latest, e)

        if FLAGS.max_evals > 0 and eval_count >= FLAGS.max_evals:
            break
        time.sleep(FLAGS.interval)


if __name__ == '__main__':
    app.run(main)
