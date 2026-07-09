"""Per-term census of the post-warp candidate filter on REAL training data.

Zero arguments — paths baked in (override with EP_CONFIG / EP_GROUPS).

Runs N mosaic groups through the real Mosaic module eagerly, and for every
box fed to the warp calls the production ``transform_boxes_polygons`` with
the SAME matrix under isolated filter settings, attributing each drop to the
term responsible:

  visible          survived every term (what training actually keeps now)
  area<0.1         lost >90% of its area (dropped by legacy AND both configs)
  area 0.1-0.5     kept by the reference/legacy filter, DELETED by the old 0.5
                   rule — the contradictory-supervision population
  side<2px         degenerate after clip (sub-2px side)
  ar>=20           degenerate sliver (dropped by legacy + the new filter,
                   kept by the old config, trained as a positive)

Run from the repo root:  python count_filter_drops.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG = os.environ.get('EP_CONFIG', 'configs/experiments/yolo/yolov8_nov2_model.yaml')
N_GROUPS = int(os.environ.get('EP_GROUPS', '50'))

import numpy as np
import tensorflow as tf

from configs.yaml_loader import load_config
from data_pipeline import augmentations as A
from data_pipeline import tfds_decoders

cfg = load_config(CONFIG)
td = cfg.task.train_data
pc = td.parser
mc = pc.mosaic
H, W = cfg.task.model.input_size[:2]

_orig = A.transform_boxes_polygons
counts = Counter()
sizes = Counter()
cls_in = Counter()
cls_kept = Counter()     # size bucket of boxes deleted ONLY by the old 0.5 rule


def _bucket(side_px):
    for b in (8, 16, 32, 64, 128, 256, 10_000):
        if side_px <= b:
            return f'<={b}px' if b < 10_000 else '>256px'


def _instrumented(boxes, polygons, M, *args, **kw):
    # Production result first (whatever params the pipeline passed).
    counts['_wrapper_calls'] += 1
    out = _orig(boxes, polygons, M, *args, **kw)

    # Same matrix, isolated terms. area/side/ar "off" = permissive extremes.
    base_kw = {k: v for k, v in kw.items()
               if k not in ('area_thresh', 'min_side', 'max_aspect_ratio')}

    def keep_with(area, side, ar):
        _, k, _ = _orig(boxes, polygons, M, *args, area_thresh=area,
                        min_side=side, max_aspect_ratio=ar, **base_kw)
        return k.numpy()

    base = keep_with(1e-9, 1e-9, 1e9)          # only degenerate/empty dropped
    k_area01 = keep_with(0.1, 1e-9, 1e9)
    k_area05 = keep_with(0.5, 1e-9, 1e9)
    k_side = keep_with(1e-9, 0.003, 1e9)
    k_ar = keep_with(1e-9, 1e-9, 20.0)

    b = boxes.numpy()
    for i in range(len(b)):
        if (b[i] <= 0).all():                   # padding row
            continue
        counts['boxes_total'] += 1
        if not base[i]:
            counts['left_frame_entirely'] += 1
            continue
        if not k_area01[i]:
            counts['area<0.1 (all recipes drop)'] += 1
        elif not k_area05[i]:
            counts['area 0.1-0.5 (OLD 0.5 deleted; legacy/new keep)'] += 1
            side = min(b[i][2] - b[i][0], b[i][3] - b[i][1]) * H
            sizes[_bucket(side)] += 1
        if not k_side[i]:
            counts['side<2px'] += 1
        if not k_ar[i]:
            counts['ar>=20 sliver'] += 1
        if k_area01[i] and k_side[i] and k_ar[i]:
            counts['visible (new filter keeps)'] += 1
    return out


A.transform_boxes_polygons = _instrumented
# Import mosaic AFTER patching: it binds the augmentation functions at
# import time (from-import), so the patch must exist first.
from data_pipeline import mosaic as mosaic_mod
mosaic_mod.transform_boxes_polygons = _instrumented

def run_pass(freq, label, per_class=False):
    counts.clear(); sizes.clear(); cls_in.clear(); cls_kept.clear()
    m = mosaic_mod.Mosaic(
    output_size=[H, W], mosaic_frequency=freq,
    mosaic_center=mc.mosaic_center,
    aug_scale_min=mc.aug_scale_min, aug_scale_max=mc.aug_scale_max,
    area_thresh=mc.area_thresh, with_polygons=pc.with_polygons,
    degrees=mc.degrees, translate=mc.translate, rotate_prob=mc.rotate_prob,
    tile_scale_min=mc.tile_scale_min, tile_scale_max=mc.tile_scale_max,
    group_size=mc.group_size, decodes_per_output=mc.decodes_per_output,
    single_scale_min=pc.aug_scale_min, single_scale_max=pc.aug_scale_max,
    single_translate=pc.aug_rand_translate,
    single_area_thresh=pc.area_thresh, random_flip=True)
    fn = m.mosaic_fn(True)

    ds = _build_ds()
    print(f'\n===== {label} (mosaic_frequency={freq}) =====')
    for gi, batch in enumerate(ds):
        if gi >= N_GROUPS:
            break
        bx = batch['groundtruth_boxes'].numpy()
        cl = batch['groundtruth_classes'].numpy()
        valid = (bx[..., 2] - bx[..., 0]) * (bx[..., 3] - bx[..., 1]) > 0
        for c in cl[valid]:
            cls_in[int(c)] += 1
        out = fn(batch)
        oc = out['groundtruth_classes'].numpy()
        ob = out['groundtruth_boxes'].numpy()
        ov = (ob[..., 2] - ob[..., 0]) * (ob[..., 3] - ob[..., 1]) > 0
        for c in oc[ov]:
            cls_kept[int(c)] += 1
        if (gi + 1) % 10 == 0:
            print(f'  ... {gi + 1}/{N_GROUPS} groups')
    _report(per_class)


import tensorflow_datasets as tfds
name = os.environ.get('EP_TFDS', td.tfds_name.split(',')[0].strip())
split = os.environ.get('EP_SPLIT', td.tfds_split.split(',')[0].strip())
decoder = tfds_decoders.PolygonDecoder(
    num_classes=cfg.task.num_classes,
    class_remap_json_path=td.class_remap_json_path,
    resample_points=pc.resample_points)
def _build_ds():
    ds = tfds.load(name, split=split,
                   data_dir=os.environ.get('EP_DATA_DIR', td.tfds_data_dir),
                   decoders={'image': tfds.decode.SkipDecoding()})
    ds = ds.map(decoder.decode)
    ds = ds.map(_pre_resize)
    if int(os.environ.get('EP_REPEAT', '0')):
        ds = ds.repeat(int(os.environ['EP_REPEAT']))
    return ds.padded_batch(mc.group_size, drop_remainder=True)


def _pre_resize(ex):
    ex = dict(ex)
    ex['image'] = tf.cast(tf.image.resize(ex['image'], [H, W]), tf.uint8)
    return ex


def _report(per_class=False):
    wcalls = counts.pop('_wrapper_calls', 0)
    total = counts.pop('boxes_total', 0)
    print(f'warp calls instrumented: {wcalls}')
    print(f'boxes fed to the warp: {total}')
    for k, v in counts.most_common():
        print(f'  {k:48s} {v:8d}  ({100 * v / max(total, 1):5.1f}%)')
    print('size of boxes the OLD 0.5 rule deleted (min side, px at 672):')
    for k in ('<=8px', '<=16px', '<=32px', '<=64px', '<=128px', '<=256px', '>256px'):
        if sizes.get(k):
            print(f'  {k:>8s}: {sizes[k]}')
    if not per_class:
        return
    try:
        from configs.class_map import DETECTION_CLASSES
        names = {i: DETECTION_CLASSES[i] for i in sorted(DETECTION_CLASSES)} \
            if isinstance(DETECTION_CLASSES, dict) else dict(enumerate(DETECTION_CLASSES))
    except Exception:
        names = {}
    rows = []
    for c, n_in in cls_in.items():
        kept = cls_kept.get(c, 0)
        rows.append((kept / max(n_in, 1), c, n_in, kept))
    if rows:
        print('per-class retention into training targets (lowest first):')
        for r, c, n_in, kept in sorted(rows)[:15]:
            print(f'  {c:3d} {names.get(c, "?"):22s} in={n_in:7d} kept={kept:7d} '
                  f'retention={100*r:5.1f}%')


print(f'config={CONFIG}\nsource={name}[{split}]  groups={N_GROUPS} '
      f'(x{mc.group_size} images)')
run_pass(1.0, 'MOSAIC path')
run_pass(0.0, 'SINGLE-image path (the one the old shared 0.5 hit vs legacy)', per_class=True)
print('\nReading: "area 0.1-0.5" is the population the old config trained AS '
      'BACKGROUND while visible; "ar>=20" is what the old config trained as '
      'positives but legacy/new drop. Judge each path against its own table; '
      'the SINGLE path is where the old config diverged from legacy.')
