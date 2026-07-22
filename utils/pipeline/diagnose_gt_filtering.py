"""Measure how many ground-truth objects the training augmentation drops.

Runs the REAL Mosaic module's single-image path and mosaic path over the
detection sources and counts, per class, how many GT objects are filtered out
(candidate filter: visible-area fraction, ~2px min side on the mosaic path,
aspect ratio < 20, plus objects cut off the canvas). Copy-paste is disabled so
only original GTs are measured.

Two phases:
  * counting -- runs GRAPH-COMPILED and parallel (fast) over the whole dataset
    (or --limit N), differencing input vs kept GT class counts per path.
  * samples  -- a small EAGER pass (--sample_scan images) captures the filter
    keep mask so a handful of annotated images can be saved (kept green,
    dropped red). Eager is needed only here (to capture per-box drops).

    python -m utils.pipeline.diagnose_gt_filtering \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --output_dir /tmp/gt_filter [--limit 0] [--num_samples 50] [--sample_scan 4000]
"""

import os
from collections import Counter

from absl import app, flags, logging as absl_logging
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string('config', None, 'Experiment YAML.', required=True)
    flags.DEFINE_string('split', 'train', "Split (train/test).")
    flags.DEFINE_string('output_dir', '/tmp/gt_filter', 'Where sample images go.')
    flags.DEFINE_integer('limit', 0, 'Count only N images per source (0 = whole dataset).')
    flags.DEFINE_integer('num_samples', 50, 'Annotated sample images to save.')
    flags.DEFINE_integer('sample_scan', 4000, 'Images scanned (eager) to draw the samples from.')
    flags.DEFINE_integer('seed', 0, 'RNG seed for the reservoir sampler.')
except flags.DuplicateFlagError:
    pass

_AUTOTUNE = tf.data.AUTOTUNE
_CAP = {}   # filled by the capture wrapper during the eager sample pass


def _install_capture():
    """Wrap transform_boxes_polygons (both namespaces) to stash (boxes, keep)."""
    import data_pipeline.augmentations as au
    import data_pipeline.mosaic as mo
    orig = au.transform_boxes_polygons

    def wrapped(boxes, polys, M, *a, **kw):
        boxes_clip, keep, polys_out = orig(boxes, polys, M, *a, **kw)
        _CAP['boxes'] = boxes_clip
        _CAP['keep'] = keep
        return boxes_clip, keep, polys_out

    au.transform_boxes_polygons = wrapped   # single path (via random_perspective)
    mo.transform_boxes_polygons = wrapped   # mosaic path (direct call)


def _cls_counter(classes) -> Counter:
    a = np.asarray(classes).reshape(-1)
    return Counter(int(c) for c in a)


def _safe_iter(ds, tag):
    """Iterate a dataset, stopping cleanly (partial results, a warning) if a
    corrupt element raises a TF OpError that ignore_errors did not absorb --
    never crash the whole run."""
    it = iter(ds)
    n = 0
    while True:
        try:
            item = next(it)
        except StopIteration:
            return
        except tf.errors.OpError as e:
            absl_logging.warning("%s: stopped after %d items (%s) -- partial", tag, n, type(e).__name__)
            return
        n += 1
        yield item


class Reservoir:
    """Uniform reservoir sampler."""

    def __init__(self, k, rng):
        self.k, self.rng, self.seen, self.items = k, rng, 0, []

    def offer(self, item):
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randint(0, self.seen - 1)
            if j < self.k:
                self.items[j] = item


def _draw_and_save(items, names, out_dir):
    try:
        import cv2
    except ImportError:
        absl_logging.warning("cv2 unavailable; skipping sample-image save.")
        return 0
    os.makedirs(out_dir, exist_ok=True)
    for idx, (img, boxes, keep, path, drops) in enumerate(items):
        im = np.asarray(img)
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        H, W = im.shape[:2]
        b = np.asarray(boxes)
        k = np.asarray(keep).astype(bool)
        for i in range(b.shape[0]):
            y1, x1, y2, x2 = b[i]
            color = (0, 200, 0) if k[i] else (0, 0, 255)   # green kept, red dropped
            cv2.rectangle(im, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), color, 2)
        label = f"{path}  dropped: {', '.join(names.get(c, str(c)) for c in drops) or 'none'}"
        cv2.putText(im, label[:120], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"{idx:03d}_{path}.png"), im)
    return len(items)


def _print_table(title, in_c, kept_c, names):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  {'id':>3} {'class':<20}{'GTs in':>10}{'kept':>10}{'dropped':>10}{'drop%':>8}")
    print("  " + "-" * 62)
    rows = []
    for c in in_c:
        gi, gk = in_c[c], kept_c.get(c, 0)
        gd = gi - gk
        rows.append((100.0 * gd / gi if gi else 0.0, c, gi, gk, gd))
    for pct, c, gi, gk, gd in sorted(rows, reverse=True):   # worst drop first
        print(f"  {c:>3} {names.get(c, str(c)):<20}{gi:>10}{gk:>10}{gd:>10}{pct:>7.1f}%")
    ti, tk = sum(in_c.values()), sum(kept_c.values())
    td = ti - tk
    print("  " + "-" * 62)
    print(f"  {'':>3} {'ALL':<20}{ti:>10}{tk:>10}{td:>10}{(100.0*td/ti if ti else 0):>7.1f}%")


def main(_):
    from configs.yaml_loader import load_config
    from data_pipeline.input_reader import build_input_reader_from_config
    from data_pipeline.augmentations import letterbox_resize
    from common.progress import Progress
    import tensorflow_datasets as tfds
    try:
        from configs.class_map import DETECTION_CLASSES
        names = {int(i): str(DETECTION_CLASSES[i]) for i in range(len(DETECTION_CLASSES))}
    except Exception:
        names = {}

    cfg = load_config(FLAGS.config)
    data_cfg = cfg.task.train_data if FLAGS.split == 'train' else cfg.task.validation_data

    # Real components; copy-paste OFF so only original GTs are measured.
    reader = build_input_reader_from_config(data_cfg, cfg.task, is_training=True)
    mosaic = reader._mosaic_module
    decoder = reader._decoder
    mosaic._copy_paste_module = None
    H, W = mosaic._H, mosaic._W

    def _preresize(ex):
        img = ex['image']
        boxes = ex.get('groundtruth_boxes', tf.zeros([0, 4], tf.float32))
        polys = ex.get('groundtruth_polygons', tf.zeros([0, 2], tf.float32))
        shp = tf.shape(img)
        im, bo, po = tf.cond(
            tf.logical_and(tf.equal(shp[0], H), tf.equal(shp[1], W)),
            lambda: (tf.ensure_shape(img, [H, W, 3]), boxes, polys),
            lambda: letterbox_resize(img, boxes, polys, H, W))
        im.set_shape([H, W, 3])
        return {**ex, 'image': im, 'groundtruth_boxes': bo, 'groundtruth_polygons': po}

    def _source_ds(src, sp, shuffle=False):
        # shuffle=True (the sample pass) reshuffles shards + a record buffer so the
        # saved sample images are a RANDOM subset of the dataset, not the first N.
        # Counting (shuffle=False) is order-independent, so it skips the shuffle.
        ds = tfds.load(src, split=sp, data_dir=data_cfg.tfds_data_dir, shuffle_files=shuffle,
                       decoders={'image': tfds.decode.SkipDecoding()},
                       read_config=tfds.ReadConfig(assert_cardinality=False))
        # Datasets are known partially corrupted. assert_cardinality=False tolerates
        # clean truncation; ignore_errors after EACH stage skips a record that raises
        # at read (source), decode, or preprocess and continues.
        ds = ds.ignore_errors(log_warning=True)            # source / TFRecord read
        if shuffle:
            ds = ds.shuffle(8000, seed=FLAGS.seed)          # encoded records: ~cheap
        ds = ds.map(decoder.decode, num_parallel_calls=_AUTOTUNE)
        ds = ds.map(_preresize, num_parallel_calls=_AUTOTUNE)
        return ds.ignore_errors(log_warning=True)          # decode / preprocess

    def _total(src, sp):
        try:
            return tfds.builder(src, data_dir=data_cfg.tfds_data_dir).info.splits[sp].num_examples
        except Exception:
            return None

    sources = [s.strip() for s in data_cfg.tfds_name.split(',')]
    splits = [s.strip() for s in data_cfg.tfds_split.split(',')]
    grand = {'si': Counter(), 'sk': Counter(), 'mi': Counter(), 'mk': Counter()}

    # ================= PHASE 1: counting (graph-compiled, fast) =================
    for src, sp in zip(sources, splits):
        ds = _source_ds(src, sp)
        tot = _total(src, sp)
        if FLAGS.limit:
            tot = FLAGS.limit if tot is None else min(FLAGS.limit, tot)
        s_in, s_kept, m_in, m_kept = Counter(), Counter(), Counter(), Counter()

        # SINGLE path: one graph map per image.
        sds = ds.map(
            lambda ex: (ex['groundtruth_classes'], mosaic._single(ex)['groundtruth_classes']),
            num_parallel_calls=_AUTOTUNE).prefetch(_AUTOTUNE)
        pb = Progress(total=tot, desc=f'{src} single', unit='img')
        n = 0
        for in_c, out_c in _safe_iter(sds, f'{src} single'):
            s_in += _cls_counter(in_c); s_kept += _cls_counter(out_c)
            n += 1; pb.update(1)
            if FLAGS.limit and n >= FLAGS.limit:
                break
        pb.close()

        # MOSAIC path: padded_batch 4 (per-image object counts vary, so batch() can't
        # stack them -- pad to the group max, like the real pipeline), one graph map
        # per group. The mosaic input objects are the SAME images the single pass
        # already counted, so reuse s_in as the mosaic input and count only the kept
        # output here. Padding rows are 0-area boxes -> dropped by the mosaic filter,
        # so they never enter the kept count.
        def _mfn(b):
            e = [{k: v[i] for k, v in b.items()} for i in range(4)]
            return mosaic._mosaic(e[0], e[1], e[2], e[3])['groundtruth_classes']
        mds = ds.padded_batch(4, drop_remainder=True).map(_mfn, num_parallel_calls=_AUTOTUNE).prefetch(_AUTOTUNE)
        pb2 = Progress(total=(tot // 4 if tot else None), desc=f'{src} mosaic', unit='grp')
        n = 0
        for out_c in _safe_iter(mds, f'{src} mosaic'):
            m_kept += _cls_counter(out_c)
            n += 1; pb2.update(1)
            if FLAGS.limit and n >= FLAGS.limit // 4:
                break
        pb2.close()
        m_in = s_in   # mosaic sees the same images as the single pass

        _print_table(f"SINGLE path  |  source: {src}", s_in, s_kept, names)
        _print_table(f"MOSAIC path  |  source: {src}", m_in, m_kept, names)
        grand['si'] += s_in; grand['sk'] += s_kept
        grand['mi'] += m_in; grand['mk'] += m_kept

    _print_table("SINGLE path  |  ALL SOURCES", grand['si'], grand['sk'], names)
    _print_table("MOSAIC path  |  ALL SOURCES", grand['mi'], grand['mk'], names)

    # ================= PHASE 2: sample images (small eager pass) =================
    if FLAGS.num_samples > 0 and FLAGS.sample_scan > 0:
        tf.config.run_functions_eagerly(True)   # so the capture wrapper fires per image
        _install_capture()
        rng = np.random.RandomState(FLAGS.seed)
        res = Reservoir(FLAGS.num_samples, rng)
        pb3 = Progress(total=FLAGS.sample_scan, desc='scanning for samples', unit='img')
        scanned = 0
        for src, sp in zip(sources, splits):
            if scanned >= FLAGS.sample_scan:
                break
            buf = []
            for ex in _safe_iter(_source_ds(src, sp, shuffle=True), f'{src} samples'):
                _CAP.clear()
                r = mosaic._single(ex)
                if 'keep' in _CAP:
                    k = np.asarray(_CAP['keep']).astype(bool)
                    if (~k).any():
                        dr = _cls_counter(np.asarray(ex['groundtruth_classes']).reshape(-1)[~k])
                        res.offer((r['image'].numpy(), _CAP['boxes'].numpy(), k, 'single', list(dr)))
                buf.append(ex)
                if len(buf) == 4:
                    _CAP.clear()
                    mr = mosaic._mosaic(buf[0], buf[1], buf[2], buf[3])
                    if 'keep' in _CAP:
                        k = np.asarray(_CAP['keep']).astype(bool)
                        merged = np.concatenate(
                            [np.asarray(e['groundtruth_classes']).reshape(-1) for e in buf])
                        if (~k).any() and merged.shape[0] == k.shape[0]:
                            dr = _cls_counter(merged[~k])
                            res.offer((mr['image'].numpy(), _CAP['boxes'].numpy(), k, 'mosaic', list(dr)))
                    buf = []
                scanned += 1; pb3.update(1)
                if scanned >= FLAGS.sample_scan:
                    break
        pb3.close()
        saved = _draw_and_save(res.items, names, os.path.join(FLAGS.output_dir, 'samples'))
        print(f"\nSaved {saved} annotated sample images (green=kept, red=dropped) to "
              f"{os.path.join(FLAGS.output_dir, 'samples')}")


if __name__ == '__main__':
    app.run(main)
