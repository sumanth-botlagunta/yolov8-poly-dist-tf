"""Localize WHERE the device-DLC export diverges from the in-repo model.

`tools/export_device_dlc.py --verify` compares the final SavedModel against the
in-memory model. When they disagree it cannot say *which* stage introduced the
difference. This tool runs the SAME image through every stage and prints a compact,
NON-CONFIDENTIAL report (policy names, dtypes, mismatch %, max abs diff — no image
content, no weights), so the failing stage is obvious.

Stages compared on one deterministic synthetic image:
    A  model(img/255)                 eager reference (what training/eval uses)
    B  serving_fn(img255)             the tf.function, BEFORE freezing (graph trace)
    C  reloaded SavedModel(img255)    after freeze + re-import + simple_save

Interpretation:
    A == B == C            -> export is faithful; the --verify failure is elsewhere.
    A == B,  B != C        -> the FREEZE / re-import step drops or mis-binds weights.
    A != B                 -> tracing the model under tf.function changes numerics
                             (per-layer dtype policy / graph-mode precision).
It also dumps the set of distinct layer dtype policies and the model-output dtypes,
which reveals a per-layer bfloat16 policy that a global-policy check cannot see.

Usage:
    python tools/diagnose_device_export.py \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /path/to/ckpt-N --input_size 672,416
"""

import logging

from absl import app, flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config',     None, 'Experiment YAML.', required=True)
    flags.DEFINE_string('checkpoint', None, 'Checkpoint prefix.', required=True)
    flags.DEFINE_string('input_size', '672,416', 'Device H,W.')
    flags.DEFINE_bool('normalize', True, 'Bake /255 (match the export default).')
except flags.DuplicateFlagError:
    pass

log = logging.getLogger(__name__)


def _stats(name, a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return f"{name:18s} SHAPE DIFF {a.shape} vs {b.shape}"
    mism = float(np.mean(~np.isclose(a, b, rtol=1e-5, atol=1e-4)) * 100.0)
    return (f"{name:18s} mism={mism:6.2f}%  max|d|={np.abs(a - b).max():.3e}  "
            f"max|val|={np.abs(a).max():.3e}")


def main(_):
    import tempfile
    from configs.yaml_loader import load_config
    from models.yolo_v8 import build_yolov8
    from tools.ckpt_loading import restore_eval_weights
    from tools import export_device_dlc as ed

    H, W = (int(x) for x in FLAGS.input_size.split(','))
    tf.keras.mixed_precision.set_global_policy('float32')

    cfg = load_config(FLAGS.config)
    mc = cfg.task.model
    mc.input_size = [H, W, 3]
    model = build_yolov8(mc)
    model.deploy = False
    model.build_and_init([H, W, 3])
    kind = restore_eval_weights(model, FLAGS.checkpoint)

    n_classes = mc.num_classes
    poly = mc.output_poly_size
    head_chan = [('box', 64), ('cls', n_classes)]
    if mc.with_polygons:
        head_chan += [('poly_angle', poly), ('poly_dist', poly), ('poly_conf', poly)]
    if mc.with_distance:
        head_chan += [('dist', 1)]
    do_norm = FLAGS.normalize

    print("\n================ device-export divergence report ================")
    print(f"checkpoint kind          : {kind}")
    print(f"global policy            : {tf.keras.mixed_precision.global_policy().name} "
          f"(compute={tf.keras.mixed_precision.global_policy().compute_dtype})")

    # Distinct per-layer dtype policies (reveals a per-layer bf16 a global check misses).
    def _all_layers(m):
        seen = []
        stack = list(getattr(m, 'layers', []) or
                     [m.backbone, m.decoder, m.head])
        while stack:
            lyr = stack.pop()
            seen.append(lyr)
            stack.extend(getattr(lyr, 'layers', []) or [])
        return seen

    pols = {}
    try:
        for lyr in _all_layers(model):
            p = getattr(getattr(lyr, 'dtype_policy', None), 'name', None)
            if p:
                pols[p] = pols.get(p, 0) + 1
    except Exception as e:  # pragma: no cover - introspection best-effort
        pols = {f"<introspection failed: {e}>": 0}
    print(f"layer dtype policies     : {pols}")

    rng = np.random.RandomState(0)
    img255 = rng.uniform(0, 255, size=[1, H, W, 3]).astype(np.float32)
    img_in = tf.constant(img255) / 255.0 if do_norm else tf.constant(img255)

    # A: eager reference. Each head value is a per-level dict {'3','4','5'}.
    A = model(img_in, training=False)
    dt = ', '.join('{}:{}'.format(k, A[k]['3'].dtype.name) for k, _ in head_chan)
    print("model output dtypes      : {" + dt + "}")

    # A again: determinism of the eager model itself.
    A2 = model(img_in, training=False)
    print("\n--- A vs A (eager determinism) ---")
    for n, c in head_chan:
        print("  " + _stats(n, ed._concat_levels(A[n], c).numpy(),
                            ed._concat_levels(A2[n], c).numpy()))

    # B: the tf.function (graph trace) BEFORE freezing.
    serving_fn = ed.build_serving_fn(model, H, W, head_chan, do_norm)
    B = serving_fn(tf.constant(img255))
    print("\n--- A vs B (eager vs tf.function trace, both pre-freeze) ---")
    for n, c in head_chan:
        man = ed._concat_levels(A[n], c).numpy()
        print("  " + _stats(n, B[n].numpy(), man))

    # C: full freeze + re-import + simple_save, then reload.
    out_dir = tempfile.mkdtemp(prefix='diag_dev_')
    ed._save_named_savedmodel(serving_fn, [n for n, _ in head_chan], out_dir)
    fn = tf.saved_model.load(out_dir).signatures['serving_default']
    C = fn(input_image=tf.constant(img255))
    print("\n--- B vs C (pre-freeze tf.function vs reloaded SavedModel) ---")
    for n, c in head_chan:
        print("  " + _stats(n, C[n].numpy(), B[n].numpy()))
    print("\n--- A vs C (eager reference vs reloaded SavedModel = what --verify checks) ---")
    for n, c in head_chan:
        man = ed._concat_levels(A[n], c).numpy()
        print("  " + _stats(n, C[n].numpy(), man))
    print("=================================================================\n")
    print("Read-off: if 'A vs B' is clean but 'B vs C' is large -> the freeze/re-import\n"
          "drops weights. If 'A vs B' is large -> graph-mode/per-layer precision. If all\n"
          "clean but --verify failed -> mismatch is image/normalize specific.\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(main)
