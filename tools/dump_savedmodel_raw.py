"""Run the device SavedModel on ONE raw input image and dump <node>:0.raw files, in the
same layout SNPE net-run produces — so you can diff the SavedModel against the DLC with
tools/compare_dlc_raw.py WITHOUT needing the legacy DLC.

The SavedModel is proven faithful to the trained model (export --verify, box geometry to
2e-7). So:
  SavedModel == DLC  -> the conversion is faithful; the 0.17 is in the eval/extraction
                        harness or the input, not the DLC.
  SavedModel != DLC  -> snpe-tensorflow-to-dlc changed the values -> conversion bug; the
                        per-node diff shows which head.

Run this in the environment where TensorFlow works (the one that gives host F1 0.71).

Usage:
    python tools/dump_savedmodel_raw.py \
        --saved_model /path/to/device/saved_model \
        --raw_image   /path/to/one_672x416_image.raw   # the SAME raw fed to the DLC
        --out_dir     /tmp/expected
    # then:
    python tools/compare_dlc_raw.py --legacy /tmp/expected/Result_0 \
        --new /path/to/dlc_dsp_result/Result_0 --input_size 672,416
"""

import argparse
import os

import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_model', required=True, help='the exported device SavedModel')
    ap.add_argument('--raw_image', required=True,
                    help='one input image .raw (the SAME file fed to the DLC)')
    ap.add_argument('--out_dir', required=True, help='writes <out_dir>/Result_0/<node>:0.raw')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'uint8'],
                    help="dtype of the raw image file (DLC raw images are usually float32 [0,255])")
    a = ap.parse_args()

    loaded = tf.saved_model.load(a.saved_model)
    fn = loaded.signatures['serving_default']
    in_shape = fn.inputs[0].shape                      # [1, H, W, 3]
    H, W = int(in_shape[1]), int(in_shape[2])
    print(f"SavedModel native input: {H}x{W}")

    img = np.fromfile(a.raw_image, np.uint8 if a.dtype == 'uint8' else np.float32).astype(np.float32)
    expect = H * W * 3
    if img.size != expect:
        raise SystemExit(f"raw image has {img.size} floats, expected {expect} ({H}x{W}x3). "
                         f"Check --dtype and that this is the 672x416 raw fed to the DLC.")
    img = img.reshape(1, H, W, 3)

    out = fn(input_image=tf.constant(img))
    rd = os.path.join(a.out_dir, 'Result_0')
    os.makedirs(rd, exist_ok=True)
    for k in out:
        arr = np.ascontiguousarray(out[k].numpy().astype(np.float32))
        arr.tofile(os.path.join(rd, f'{k}:0.raw'))
        print(f"  wrote {k}:0.raw  shape {arr.shape}  ({arr.size} floats)")
    print(f"\nDone -> {rd}")
    print(f"Now diff against the DLC:\n"
          f"  python tools/compare_dlc_raw.py --legacy {rd} "
          f"--new <dlc .../Result_N> --input_size {H},{W}")


if __name__ == '__main__':
    main()
