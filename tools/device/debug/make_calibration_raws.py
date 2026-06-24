"""Build a calibration raw-image set + input_list for snpe-dlc-quantize.

TEMPORARY HELPER for re-quantizing the device DLC with a proper calibration set (the
default 26-image set gives noisy/narrow int8 ranges -> broad per-layer divergence). Sample
N representative images, letterbox them to the device input format, write float32 [0,255]
.raw files, and emit the input_list txt — in one command.

IMPORTANT — calibration distribution
  PTQ calibration estimates each tensor's int8 min/max range, so the images must MATCH the
  deployment distribution. Best: a held-out set of YOUR-domain images (e.g. training images,
  which are never in your eval split) — same domain, no eval leakage. A generic set (COCO)
  is a domain mismatch and usually quantizes WORSE on your scenes; use it only for a quick
  A/B. Either way the preprocessing here MUST equal your device raw generator.

IMPORTANT — format must match inference (else ranges are miscalibrated)
  --channel_order  rgb|bgr   the order the device feeds the model (RGB unless your generator
                             writes BGR — savedmodel_on_device_raw --swap_rb tells you which)
  --pad            114|0     letterbox pad value your generator uses
  --dtype          float32   device raw dtype ([0,255], NOT /255). float32 -> 3,354,624 bytes
                             per 672x416x3 file; uint8 -> 838,656.
  When unsure, run your EXISTING raw generator on the sampled images instead of this script —
  that guarantees byte-identical format.

Usage
-----
  # (best) your held-out/training images:
  python -m tools.device.make_calibration_raws \
      --src /path/to/train_images --out_dir /calib_raw --list calib_300.txt --n 300 \
      --input_size 672,416 --pad 114 --channel_order rgb --dtype float32

  # (quick A/B) COCO val2017 — download first, then point --src at it:
  wget -c http://images.cocodataset.org/zips/val2017.zip -O /tmp/val2017.zip
  unzip -q /tmp/val2017.zip -d /tmp/coco            # -> /tmp/coco/val2017/*.jpg
  python -m tools.device.make_calibration_raws --src /tmp/coco/val2017 \
      --out_dir /calib_raw --list calib_300.txt --n 300

  # then re-quantize (int8, memory-safe levers — no int16):
  snpe-dlc-quantize --input_list calib_300.txt --input_dlc model_pre.dlc \
      --output_dlc model_quant.dlc --use_enhanced_quantizer \
      --use_per_channel_quantization --algorithms cle bc
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np

_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')


def _letterbox(im, H, W, pad):
    """Aspect-preserving resize into HxW with constant `pad` (gray-114 by convention)."""
    h0, w0 = im.shape[:2]
    r = min(H / h0, W / w0)
    rh, rw = max(1, round(h0 * r)), max(1, round(w0 * r))
    canvas = np.full((H, W, 3), pad, np.float32)
    top, left = (H - rh) // 2, (W - rw) // 2
    canvas[top:top + rh, left:left + rw] = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='directory of source images (jpg/png)')
    ap.add_argument('--out_dir', required=True, help='where the .raw files are written')
    ap.add_argument('--list', required=True, help='output input_list txt (one raw path per line)')
    ap.add_argument('--n', type=int, default=300, help='how many images to sample')
    ap.add_argument('--input_size', default='672,416', help='H,W (device input size)')
    ap.add_argument('--pad', type=float, default=114.0, help='letterbox pad value (match device)')
    ap.add_argument('--channel_order', default='rgb', choices=['rgb', 'bgr'],
                    help='channel order the device feeds the model (cv2 reads BGR)')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'uint8'],
                    help='device raw dtype; [0,255], NOT /255')
    ap.add_argument('--abs_paths', default='true', choices=['true', 'false'],
                    help='write absolute paths into the list (SNPE usually wants absolute)')
    ap.add_argument('--seed', type=int, default=0, help='sampling seed (reproducible)')
    a = ap.parse_args()
    H, W = (int(x) for x in a.input_size.split(','))

    files = []
    for e in _EXTS:
        files.extend(glob.glob(os.path.join(a.src, '**', e), recursive=True))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no images found under {a.src}")
    if len(files) < a.n:
        print(f"WARNING: only {len(files)} images available (< {a.n}); using all of them.")
    random.Random(a.seed).shuffle(files)
    pick = files[:a.n]

    os.makedirs(a.out_dir, exist_ok=True)
    np_dtype = np.float32 if a.dtype == 'float32' else np.uint8
    expect = H * W * 3 * (4 if a.dtype == 'float32' else 1)
    written = 0
    with open(a.list, 'w') as lst:
        for p in pick:
            im = cv2.imread(p, cv2.IMREAD_COLOR)         # BGR
            if im is None:
                continue
            if a.channel_order == 'rgb':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            canvas = _letterbox(im, H, W, a.pad)         # float32 [0,255]
            arr = np.ascontiguousarray(canvas, np_dtype)
            out = os.path.join(a.out_dir, os.path.splitext(os.path.basename(p))[0] + '.raw')
            arr.tofile(out)
            lst.write((os.path.abspath(out) if a.abs_paths == 'true' else out) + '\n')
            written += 1

    print(f"wrote {written} raw images -> {a.out_dir}")
    print(f"input_list ({written} entries) -> {a.list}")
    print(f"each raw must be {expect} bytes ({H}x{W}x3 {a.dtype}); "
          f"channel_order={a.channel_order} pad={a.pad}")
    print("VERIFY the format matches your device raws (channel order / pad / dtype) before "
          "quantizing — a mismatch silently miscalibrates the ranges.")


if __name__ == '__main__':
    main()
