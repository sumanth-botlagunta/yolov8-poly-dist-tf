"""Build a prediction JSON from the device SavedModel over a TFDS — the HOST-side twin of
``tools/device/gen_pred_json_from_dlc.py``.

Purpose (the diagnostic)
------------------------
Checkpoint/host eval is good but the DLC on CPU does not match it (no relation even
un-quantized). To localize the gap we need the SAME prediction JSON produced two ways from
the SAME decode/transform, differing ONLY in the raw source:

    OLD (device):  image .raw -> SNPE net-run -> DLC raw heads -> gen_pred_json_from_dlc.py
    NEW (host):    TFDS image -> letterbox  -> SavedModel heads -> THIS script

Both call the IDENTICAL numpy decode (``_decode``) and un-letterbox (``_to_original``)
imported straight from ``gen_pred_json_from_dlc`` — so the JSON schema and the geometry are
byte-for-byte the same recipe. Diff the two JSONs against the same GT:

    NEW good, OLD bad   -> the SavedModel is fine; the gap is the SavedModel->DLC conversion
                           / quantization, OR the on-device input bytes (resize/letterbox/
                           channel order of the device raw generator).
    NEW == OLD (both)   -> the decode or the un-letterbox transform is the problem (shared by
                           both), not the DLC.
    NEW bad alone       -> the host SavedModel itself (or this script's letterbox) is wrong.

Input geometry
--------------
The SavedModel takes ``input_image`` float32 ``[1, H, W, 3]`` in ``[0, 255]`` (``/255`` is
baked in; see docs/design_register.md entry 12). Each TFDS image is letterboxed to ``HxW``
using the SAME per-image ``info_ratio`` / ``info_tblr`` recorded in the transform pkl that the
device raw generator wrote, so the host input reproduces the device input geometry. The
``--pad_value`` MUST match what the device raw generator pads with (unknown to this repo —
set it to match; YOLO convention is 114, many raw pipelines use 0). Detections are clipped to
the original image region by ``_to_original``, so pad-area boxes drop out regardless.

Matching: TFDS examples carry ``image/filename``; pkl entries carry ``file_name``. They are
matched by BASENAME. The pkl is keyed by the zero-padded frame index ('%06d'); that key is
used for the ``fname_idx`` image-id option.

Usage
-----
    # 1) export the device SavedModel (the exact artifact that becomes the DLC)
    python -m tools.device.export_device_dlc --config <yaml> --checkpoint <ckpt> \
        --output_dir /export/sm --input_size 672,416 --verify

    # 2) host predictions over the eval TFDS, SAME decode as the DLC path
    python -m tools.device.gen_pred_json_from_savedmodel \
        --saved_model   /export/sm \
        --tfds_name     cleaner_polygon2026:2.0.0 --tfds_split test \
        --tfds_data_dir ~/tensorflow_datasets \
        --transform_pkl /path/cleaner_eval_672x416_transform_info.pkl \
        --output_json   /tmp/pred_from_savedmodel.json \
        --input_size 672,416 --num_classes 39 \
        --conf_threshold 0.001 --nms_iou 0.65 --pad_value 114

Then score /tmp/pred_from_savedmodel.json and the DLC JSON against the same GT and compare.
"""

import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Reuse the EXACT decode + un-letterbox from the DLC reference so the only difference between
# the two JSONs is the raw source (and the input letterbox), never the decode recipe.
from tools.device.gen_pred_json_from_dlc import _STRIDES, _decode, _to_original


def _letterbox(img_hwc, H, W, ratio, tblr, pad_value, interpolation):
    """Resize+pad an original-resolution uint8 image to HxW using the pkl's ratio/tblr.

    Reproduces the device raw generator's geometry EXACTLY: the resized size is implied by
    the recorded pads (resized_h = H - top - bottom, resized_w = W - left - right), so the
    same ratio is honored without recomputing/rounding it here. Returns float32 [H, W, 3] in
    [0, 255].
    """
    top, bottom, left, right = (int(round(float(v))) for v in tblr)
    rh = H - top - bottom
    rw = W - left - right
    if rh <= 0 or rw <= 0:
        raise ValueError(f"bad pads {tblr} for size {H}x{W} (resized {rh}x{rw})")
    resized = tf.image.resize(tf.cast(img_hwc, tf.float32), [rh, rw],
                              method=interpolation, antialias=False)
    padded = tf.pad(resized, [[top, bottom], [left, right], [0, 0]],
                    constant_values=float(pad_value))
    padded = tf.ensure_shape(padded, [H, W, 3])
    return padded


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saved_model', required=True, help='the exported device SavedModel')
    ap.add_argument('--tfds_name', required=True, help="e.g. 'cleaner_polygon2026:2.0.0'")
    ap.add_argument('--tfds_split', default='test')
    ap.add_argument('--tfds_data_dir', default=None, help='TFDS root (default: TFDS_DATA_DIR)')
    ap.add_argument('--transform_pkl', required=True)
    ap.add_argument('--output_json', required=True)
    ap.add_argument('--input_size', default='672,416', help='H,W (device size)')
    ap.add_argument('--num_classes', type=int, default=39)
    ap.add_argument('--conf_threshold', type=float, default=0.001)
    ap.add_argument('--nms_iou', type=float, default=0.65)
    ap.add_argument('--category_offset', type=int, default=0)
    ap.add_argument('--image_id_field', default='file_name',
                    choices=['file_name', 'file_stem', 'fname_idx'])
    ap.add_argument('--pad_value', type=float, default=114.0,
                    help='letterbox pad value — MUST match the device raw generator (114 or 0)')
    ap.add_argument('--interpolation', default='bilinear',
                    choices=['bilinear', 'nearest', 'bicubic', 'area'])
    ap.add_argument('--normalize_baked', default='true', choices=['true', 'false'],
                    help='Feed the SavedModel raw [0,255] (default true) — correct for the '
                         '[0,255]-trained model + not-baked export. false -> divide by 255 '
                         '(legacy [0,1] model).')
    ap.add_argument('--box_order', default='yfirst', choices=['yfirst', 'xfirst'],
                    help="channel order of the SavedModel's box head. 'yfirst' = the "
                         "legacy/DLC order ([t,l,b,r], export_device_dlc --legacy_box_order "
                         "default), reordered to x-first before decode (shared with "
                         "gen_pred_json_from_dlc). Use 'xfirst' ONLY if you exported with "
                         "--legacy_box_order=False. MISMATCH HERE TRANSPOSES EVERY BOX.")
    ap.add_argument('--swap_rb', action='store_true',
                    help='swap channels 0<->2 before inference (test the BGR hypothesis)')
    ap.add_argument('--limit', type=int, default=0, help='process at most N images (0 = all)')
    a = ap.parse_args()
    H, W = (int(x) for x in a.input_size.split(','))
    N = sum((H // s) * (W // s) for s in _STRIDES)

    # ---- transform pkl: key '%06d' -> {file_name, info_ratio, info_tblr, ...} ----
    with open(a.transform_pkl, 'rb') as f:
        tinfo = pickle.load(f)
    print(f"transform entries: {len(tinfo)}   example key: {sorted(tinfo)[0]!r}")
    ex0 = tinfo[sorted(tinfo)[0]]
    print(f"example entry keys: {list(ex0.keys())}  "
          f"info_ratio={ex0.get('info_ratio')} info_tblr={ex0.get('info_tblr')}")
    # Reverse map: image basename -> (frame_key, entry). Matches TFDS image/filename basename.
    by_name = {}
    for k, e in tinfo.items():
        by_name[os.path.basename(e['file_name'])] = (k, e)
    print(f"indexed {len(by_name)} transform entries by basename")

    # ---- SavedModel ----
    loaded = tf.saved_model.load(a.saved_model)
    fn = loaded.signatures['serving_default']
    sm_h, sm_w = int(fn.inputs[0].shape[1]), int(fn.inputs[0].shape[2])
    if (sm_h, sm_w) != (H, W):
        print(f"WARNING: SavedModel native input {sm_h}x{sm_w} != --input_size {H}x{W}; "
              f"using the SavedModel's size.")
        H, W = sm_h, sm_w
        N = sum((H // s) * (W // s) for s in _STRIDES)
    print(f"SavedModel input: {H}x{W}   normalize_baked={a.normalize_baked}  "
          f"swap_rb={a.swap_rb}  pad_value={a.pad_value}  interp={a.interpolation}")

    # ---- TFDS (decoded images at original resolution) ----
    ds = tfds.load(a.tfds_name, split=a.tfds_split, data_dir=a.tfds_data_dir,
                   shuffle_files=False)

    preds = []
    seen = matched = done = no_tf = 0
    for ex in tfds.as_numpy(ds):
        seen += 1
        if a.limit and done >= a.limit:
            break
        fname = ex.get('image/filename')
        if fname is None:
            continue
        fname = fname.decode() if isinstance(fname, (bytes, bytearray)) else str(fname)
        base = os.path.basename(fname)
        hit = by_name.get(base)
        if hit is None:
            no_tf += 1
            continue
        key, entry = hit
        matched += 1

        img = ex['image']                                   # uint8 [h0, w0, 3] original
        feed = _letterbox(img, H, W, entry['info_ratio'], entry['info_tblr'],
                          a.pad_value, a.interpolation)      # float32 [H, W, 3] in [0,255]
        if a.swap_rb:
            feed = feed[..., ::-1]
        if a.normalize_baked == 'false':
            feed = feed / 255.0
        out = fn(input_image=tf.constant(feed[None], tf.float32))

        box = out['box'].numpy().reshape(N, 4)
        cls = out['cls'].numpy().reshape(N, a.num_classes)
        # box reorder is handled inside _decode (shared with gen_pred_json_from_dlc).
        xyxy, score, klass = _decode(box, cls, H, W, a.conf_threshold,
                                     a.nms_iou, a.num_classes, box_order=a.box_order)
        done += 1
        if len(xyxy) == 0:
            continue
        xywh = _to_original(xyxy, entry, H, W)              # xywh in ORIGINAL pixels
        stem = os.path.splitext(entry['file_name'])[0]
        image_id = {'file_name': entry['file_name'], 'file_stem': stem,
                    'fname_idx': key}[a.image_id_field]
        for j in range(len(xywh)):
            preds.append({
                'image_id': image_id,
                'bbox': [round(float(v), 3) for v in xywh[j]],
                'category_id': int(klass[j]) + a.category_offset,
                'score': round(float(score[j]), 5),
            })

    with open(a.output_json, 'w') as f:
        json.dump(preds, f)
    print(f"\nscanned {seen} TFDS images; matched {matched} to the transform pkl; "
          f"ran {done}; wrote {len(preds)} detections -> {a.output_json}")
    if no_tf:
        print(f"NOTE: {no_tf} TFDS image(s) had no matching transform entry (basename mismatch).")
    print("Entry schema matches gen_pred_json_from_dlc.py exactly "
          "{image_id, bbox:[x,y,w,h] orig px, category_id (0-based+offset), score}.")


if __name__ == '__main__':
    main()
