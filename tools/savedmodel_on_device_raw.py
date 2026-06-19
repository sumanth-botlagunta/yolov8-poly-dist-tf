"""Run the KNOWN-GOOD device SavedModel on the ACTUAL device .raw input files and draw
the detections — to test the input-format hypothesis WITHOUT the DLC or SNPE.

Why this exists
---------------
host eval (repo pipeline) = F1 0.68, but the DLC on CPU = 0.18. CPU rules out
quantization; the padding fix moved it ~0, ruling out a conversion spatial-shift. A gap
that is invisible to host eval AND to --verify AND independent of padding most often
means the bytes fed to the DLC on-device are NOT what the model trained on — a swapped
R/B channel order or a different resize/letterbox geometry. host eval never sees that
because it builds its own (correct) input; the DLC eats whatever the on-device raw-image
generator produced.

This script feeds the SavedModel the SAME .raw file the device feeds the DLC and shows
the detections. The SavedModel is the known-good model (host 0.68). So:

  detections land on the objects        -> the input bytes are fine; the bug is the
                                           SavedModel->DLC conversion or the on-device
                                           decode harness (not the input).
  detections are garbage as-is, but good with --swap_rb
                                        -> the device .raw is BGR; fix the raw generator
                                           (or add a channel swap to the export).
  garbage both ways                     -> resize / letterbox / scale geometry mismatch
                                           between the device raw and training.

No DLC, no SNPE, no device — just TensorFlow + the .raw files you already feed net-run.

Usage:
    python tools/savedmodel_on_device_raw.py \
        --saved_model /path/to/device/saved_model \
        --raw         /path/to/raw_dir_or_one.raw \
        --out_dir     /tmp/sm_on_raw \
        --num_classes 39 \
        --dtype float32          # the .raw dtype the device writes ([0,255] floats)
        # add --swap_rb to test the BGR hypothesis; --normalize_baked False if /255 NOT baked
"""

import argparse
import glob
import os

import numpy as np
import tensorflow as tf

# Reuse the exact reconstruction the device decoder must do (LTRB pre-stride -> stride +
# anchor -> yxyx; sigmoid; top-1; per-class NMS) so the boxes here match what the harness
# should produce from these same head tensors.
from tools.validate_device_export import _reconstruct

_CLASS_NAMES = None  # filled from configs.class_map if importable


def _load_class_names():
    global _CLASS_NAMES
    try:
        from configs.class_map import DETECTION_CLASSES
        # DETECTION_CLASSES is a {index: name} dict in this repo; list() of a dict
        # yields its KEYS (ints), so map index->name explicitly. Also tolerate a
        # plain list for forward-compat.
        if isinstance(DETECTION_CLASSES, dict):
            _CLASS_NAMES = [str(DETECTION_CLASSES[i]) for i in sorted(DETECTION_CLASSES)]
        else:
            _CLASS_NAMES = [str(x) for x in DETECTION_CLASSES]
    except Exception:
        _CLASS_NAMES = None


def _draw(img_hw3_uint8, boxes_yxyx_norm, classes, scores, out_path, score_thr):
    """Draw the kept detections on the image and save a PNG. Uses cv2 (headless, the
    repo dep); falls back to PIL; if neither is present, drawing is skipped (the printed
    detections are the decisive part anyway)."""
    H, W = img_hw3_uint8.shape[:2]
    n = 0
    try:
        import cv2
        # cv2 expects BGR for writing; the .raw is in the model's channel order (RGB),
        # so flip just for the saved file so colors look natural.
        canvas = img_hw3_uint8[..., ::-1].copy()
        for (y1, x1, y2, x2), c, s in zip(boxes_yxyx_norm, classes, scores):
            if s < score_thr:
                continue
            n += 1
            p1 = (int(x1 * W), int(y1 * H)); p2 = (int(x2 * W), int(y2 * H))
            cv2.rectangle(canvas, p1, p2, (0, 0, 255), 2)
            name = _CLASS_NAMES[int(c)] if (_CLASS_NAMES and int(c) < len(_CLASS_NAMES)) else str(int(c))
            cv2.putText(canvas, f"{name} {s:.2f}", (p1[0] + 2, max(10, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(out_path, canvas)
        return n
    except Exception as e:
        print(f"  (drawing skipped: {e}; printed detections below are the decisive part)")
        return sum(1 for s in scores if s >= score_thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_model', required=True)
    ap.add_argument('--raw', required=True, help='one .raw file or a directory of them')
    ap.add_argument('--out_dir', default='/tmp/sm_on_raw')
    ap.add_argument('--num_classes', type=int, default=39)
    ap.add_argument('--dtype', default='float32', choices=['float32', 'uint8'],
                    help='dtype of the .raw image file the device writes')
    ap.add_argument('--normalize_baked', default='true', choices=['true', 'false'],
                    help='the SavedModel bakes /255 (feed raw [0,255]). If false, this '
                         'script divides by 255 before feeding.')
    ap.add_argument('--swap_rb', action='store_true',
                    help='swap channels 0<->2 before inference (test the BGR hypothesis)')
    ap.add_argument('--score_thr', type=float, default=0.25, help='draw/print threshold')
    ap.add_argument('--topk', type=int, default=10, help='print this many top detections')
    a = ap.parse_args()
    _load_class_names()

    loaded = tf.saved_model.load(a.saved_model)
    fn = loaded.signatures['serving_default']
    H, W = int(fn.inputs[0].shape[1]), int(fn.inputs[0].shape[2])
    print(f"SavedModel native input: {H}x{W}   swap_rb={a.swap_rb}   "
          f"normalize_baked={a.normalize_baked}")

    files = ([a.raw] if os.path.isfile(a.raw)
             else sorted(glob.glob(os.path.join(a.raw, '*.raw'))))
    if not files:
        raise SystemExit(f"no .raw files found at {a.raw}")
    os.makedirs(a.out_dir, exist_ok=True)
    np_dtype = np.uint8 if a.dtype == 'uint8' else np.float32
    expect = H * W * 3

    for f in files:
        flat = np.fromfile(f, np_dtype).astype(np.float32)
        if flat.size != expect:
            print(f"  SKIP {os.path.basename(f)}: {flat.size} values, expected {expect} "
                  f"({H}x{W}x3). Wrong --dtype or wrong size?")
            continue
        img = flat.reshape(1, H, W, 3)
        if a.swap_rb:
            img = img[..., ::-1].copy()
        feed = img if a.normalize_baked == 'true' else img / 255.0
        out = fn(input_image=tf.constant(feed.astype(np.float32)))

        # Per-node stats table (so the head outputs are readable / reportable).
        order_nodes = ['box', 'cls', 'poly_angle', 'poly_dist', 'poly_conf', 'dist']
        keys = [k for k in order_nodes if k in out] + [k for k in out if k not in order_nodes]
        print("\n" + "=" * 88)
        print(f" {os.path.basename(f)}   node outputs   (swap_rb={a.swap_rb})")
        print("=" * 88)
        print(f"{'node':12s}{'shape':>16s}{'min':>11s}{'max':>11s}{'mean':>11s}{'std':>11s}")
        print("-" * 88)
        for k in keys:
            v = out[k].numpy().astype(np.float32)
            fl = v.reshape(-1)
            print(f"{k:12s}{str(list(v.shape)):>16s}{fl.min():>11.4f}{fl.max():>11.4f}"
                  f"{fl.mean():>11.4f}{fl.std():>11.4f}")
        print("-" * 88)

        pred = _reconstruct(out, H, W, a.num_classes, score_thresh=0.01)
        nd = int(pred['num_detections'][0])
        boxes = pred['bbox'].numpy()[0, :nd]
        cls = pred['classes'].numpy()[0, :nd]
        scr = pred['confidence'].numpy()[0, :nd]

        # Build a uint8 image for drawing ([0,255] floats -> uint8; clip for safety).
        vis = np.clip(img[0], 0, 255).astype(np.uint8)
        png = os.path.join(a.out_dir, os.path.splitext(os.path.basename(f))[0] +
                           ('_swaprb' if a.swap_rb else '') + '.png')
        n_drawn = _draw(vis, boxes, cls, scr, png, a.score_thr)

        n_above = int((scr >= a.score_thr).sum())
        print(f"\n{os.path.basename(f)}: {nd} dets total, {n_above} >= {a.score_thr} "
              f"(drew {n_drawn}) -> {png}")
        order = np.argsort(-scr)[:a.topk]
        for r in order:
            name = (_CLASS_NAMES[int(cls[r])] if (_CLASS_NAMES and int(cls[r]) < len(_CLASS_NAMES))
                    else str(int(cls[r])))
            print(f"    {str(name):18s} score={float(scr[r]):.3f}  yxyx={np.round(boxes[r], 4)}")

    print(f"\nOpen the PNGs in {a.out_dir}. If boxes land on objects, the device input "
          f"bytes are fine and the bug is conversion/decode. If they're garbage but good "
          f"with --swap_rb, the device .raw is BGR.")


if __name__ == '__main__':
    main()
