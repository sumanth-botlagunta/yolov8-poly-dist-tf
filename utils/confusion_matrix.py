"""Per-class detection confusion matrix for a checkpoint or a SavedModel.

Runs the validation (or train) split through a model and accumulates a
`(num_classes + 1) x (num_classes + 1)` confusion matrix, where the extra
row/column is the `background` class. The model comes from either a trained
checkpoint (with its config + EMA weights) or an exported SavedModel; the
validation dataset is always built from the config, so `--config` is required
in both modes.

The exported SavedModel is the device-contract artifact (utils/export/
export_saved_model.py): its signature exposes flat per-anchor heads (box/cls/...)
and takes `input_image` pixels in [0, 255]. This tool reconstructs the boxes /
classes / scores it needs from those heads (utils/export/device_decode.py). An
older post-processed SavedModel (with `num_detections` in its signature) is still
consumed directly.

Matrix orientation is `matrix[predicted, ground_truth]`:
  * a matched detection increments `matrix[pred_class, gt_class]` (diagonal =
    correct class, off-diagonal = misclassification),
  * an unmatched ground truth (false negative) increments the background row:
    `matrix[background, gt_class]`,
  * an unmatched prediction (false positive) increments the background column:
    `matrix[pred_class, background]`.

Matching is greedy, highest-score-first, class-agnostic on IoU (so cross-class
confusion is visible), at IoU >= `--iou`; predictions below `--conf` are
dropped before matching. Crowd / don't-care ground truths are handled with the
same policy as `eval/coco_metrics.py` (see `ConfusionMatrix.update`).

Usage:
    # from a checkpoint (EMA weights preferred)
    python -m utils.confusion_matrix \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --checkpoint /run/ckpt-100000 \
        --split val --conf 0.25 --iou 0.5 \
        --output_csv /tmp/cm.csv --output_png /tmp/cm.png

    # from an exported SavedModel (config still supplies the val data)
    python -m utils.confusion_matrix \
        --config configs/experiments/yolo/yolov8_poly_dist.yaml \
        --saved_model /export/saved_model \
        --split val
"""

import logging
import os

import numpy as np

log = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of `path` if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

# Mirror the project's crowd-policy category ids (eval/coco_metrics.py). Imported
# so the two stay in lock-step; falls back to a literal if the import path moves.
try:
    from eval.coco_metrics import DEFAULT_ISCROWD_LABELS
except Exception:  # pragma: no cover - trivial fallback
    DEFAULT_ISCROWD_LABELS = [6, 13, 24, 36, 37]


# ----------------------------------------------------------------------------
# Pure numpy matching + matrix accumulation (no TensorFlow, importable for tests)
# ----------------------------------------------------------------------------

def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """Pairwise IoU between two sets of yxyx boxes.

    Args:
        boxes_a: [Na, 4] array of (y1, x1, y2, x2).
        boxes_b: [Nb, 4] array of (y1, x1, y2, x2).

    Returns:
        [Na, Nb] IoU matrix (0 where either box is degenerate or they miss).
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    ay1, ax1, ay2, ax2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    by1, bx1, by2, bx2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area_a = np.clip(ay2 - ay1, 0, None) * np.clip(ax2 - ax1, 0, None)
    area_b = np.clip(by2 - by1, 0, None) * np.clip(bx2 - bx1, 0, None)

    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    inter = np.clip(iy2 - iy1, 0, None) * np.clip(ix2 - ix1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections(pred_boxes, pred_scores, gt_boxes, iou_thresh: float = 0.5) -> np.ndarray:
    """Greedy highest-score-first, class-agnostic IoU matching.

    Predictions are visited in descending score order; each claims the
    highest-IoU still-unclaimed ground truth whose IoU >= `iou_thresh`. Class
    labels are ignored during matching so cross-class confusion is preserved.

    Args:
        pred_boxes: [Np, 4] yxyx predicted boxes.
        pred_scores: [Np] confidences (used only for match order).
        gt_boxes: [Ng, 4] yxyx ground-truth boxes.
        iou_thresh: minimum IoU for a positive match.

    Returns:
        [Np] int array; entry p is the matched GT index or -1 if unmatched.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    n_pred, n_gt = pred_boxes.shape[0], gt_boxes.shape[0]
    matches = np.full(n_pred, -1, dtype=np.int64)
    if n_pred == 0 or n_gt == 0:
        return matches

    iou = iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float64).reshape(-1), kind='stable')
    taken = np.zeros(n_gt, dtype=bool)
    for p in order:
        row = iou[p].copy()
        row[taken] = -1.0
        g = int(np.argmax(row))
        if row[g] >= iou_thresh:
            taken[g] = True
            matches[p] = g
    return matches


class ConfusionMatrix:
    """Accumulates a per-class detection confusion matrix.

    The stored matrix is `[num_classes + 1, num_classes + 1]` indexed
    `[predicted, ground_truth]`; index `num_classes` is the background class.

    Crowd / don't-care policy mirrors `eval/coco_metrics.py`:
      * a crowd GT whose class is in `iscrowds_labels` is dropped entirely when
        `ignore_iscrowds` is set (it is neither a match target nor a false
        negative);
      * a don't-care GT (when `ignore_dontcare` is set) is not scored as a
        target, but it absorbs any leftover prediction that overlaps it at
        IoU >= `iou_thresh` so the prediction is not counted as a false
        positive. When `ignore_dontcare` is off, don't-care GTs are scored as
        ordinary GTs.
    """

    def __init__(self, num_classes: int, class_names=None, iou_thresh: float = 0.5,
                 conf_thresh: float = 0.25, ignore_dontcare: bool = True,
                 ignore_iscrowds: bool = False, iscrowds_labels=None):
        self.num_classes = int(num_classes)
        self.bg = int(num_classes)
        self.iou_thresh = float(iou_thresh)
        self.conf_thresh = float(conf_thresh)
        self.ignore_dontcare = bool(ignore_dontcare)
        self.ignore_iscrowds = bool(ignore_iscrowds)
        self.iscrowds_labels = set(iscrowds_labels) if iscrowds_labels is not None \
            else set(DEFAULT_ISCROWD_LABELS)
        self.class_names = list(class_names) if class_names is not None else None
        self.matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int64)

    # ------------------------------------------------------------------
    def update(self, pred_boxes, pred_classes, pred_scores,
               gt_boxes, gt_classes, gt_is_crowd=None, gt_is_dontcare=None) -> None:
        """Accumulate one image's predictions and ground truths.

        All boxes are yxyx (any consistent units; IoU is scale-free). Predictions
        below `conf_thresh` are dropped before matching.
        """
        pb = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        pc = np.asarray(pred_classes).reshape(-1).astype(np.int64)
        ps = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
        gb = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        gc = np.asarray(gt_classes).reshape(-1).astype(np.int64)
        n_gt = gb.shape[0]

        crowd = (np.asarray(gt_is_crowd).reshape(-1).astype(bool)
                 if gt_is_crowd is not None else np.zeros(n_gt, dtype=bool))
        dontcare = (np.asarray(gt_is_dontcare).reshape(-1).astype(bool)
                    if gt_is_dontcare is not None else np.zeros(n_gt, dtype=bool))

        # Confidence gate on predictions (before matching).
        keep = ps >= self.conf_thresh
        pb, pc, ps = pb[keep], pc[keep], ps[keep]

        # Partition GT into skipped-crowd / absorbing-dontcare / scored-normal.
        if n_gt:
            in_crowd_labels = np.array([int(c) in self.iscrowds_labels for c in gc], dtype=bool)
        else:
            in_crowd_labels = np.zeros(0, dtype=bool)
        skip_mask = (crowd & in_crowd_labels) if self.ignore_iscrowds else np.zeros(n_gt, dtype=bool)
        dc_mask = (dontcare & ~skip_mask) if self.ignore_dontcare else np.zeros(n_gt, dtype=bool)
        normal_mask = ~skip_mask & ~dc_mask

        ng_boxes, ng_classes = gb[normal_mask], gc[normal_mask]
        dc_boxes = gb[dc_mask]

        # Match kept predictions against normal GT.
        matches = match_detections(pb, ps, ng_boxes, self.iou_thresh)
        matched_pred = matches >= 0
        matched_gt = set(int(g) for g in matches[matched_pred])

        for p in np.where(matched_pred)[0]:
            self._inc(int(pc[p]), int(ng_classes[matches[p]]))

        # Unmatched normal GT -> false negative (background row).
        for g in range(ng_classes.shape[0]):
            if g not in matched_gt:
                self._inc(self.bg, int(ng_classes[g]))

        # Unmatched predictions -> absorbed by a dontcare GT, else false positive.
        unmatched = np.where(~matched_pred)[0]
        if unmatched.size:
            absorbed = np.zeros(unmatched.size, dtype=bool)
            if dc_boxes.shape[0]:
                iou_dc = iou_matrix(pb[unmatched], dc_boxes)
                absorbed = iou_dc.max(axis=1) >= self.iou_thresh
            for local, p in enumerate(unmatched):
                if not absorbed[local]:
                    self._inc(int(pc[p]), self.bg)

    def _inc(self, pred_idx: int, gt_idx: int) -> None:
        """Increment a cell, clamping out-of-range class ids into background."""
        if not (0 <= pred_idx <= self.num_classes):
            pred_idx = self.bg
        if not (0 <= gt_idx <= self.num_classes):
            gt_idx = self.bg
        self.matrix[pred_idx, gt_idx] += 1

    # ------------------------------------------------------------------
    def as_array(self) -> np.ndarray:
        """Return a copy of the raw integer matrix `[predicted, ground_truth]`."""
        return self.matrix.copy()

    def _label(self, idx: int) -> str:
        if idx == self.bg:
            return 'background'
        if self.class_names and idx < len(self.class_names):
            return str(self.class_names[idx])
        return str(idx)

    def top_confusions(self, k: int = 20):
        """Top off-diagonal cells as `(pred_label, gt_label, count)`, count-desc."""
        out = []
        n = self.num_classes + 1
        for i in range(n):
            for j in range(n):
                if i == j or self.matrix[i, j] == 0:
                    continue
                out.append((self._label(i), self._label(j), int(self.matrix[i, j])))
        out.sort(key=lambda t: t[2], reverse=True)
        return out[:k]

    def per_class_report(self):
        """Per-GT-class stats, worst-recall first.

        For each ground-truth class j (column j), reads straight off the matrix:
          * n_gt   = sum over the column (every GT of class j lands in some row:
                     a predicted class on a match, or the background row on a miss),
          * recall = matrix[j, j] / n_gt   (matched AND correctly classified),
          * miss   = matrix[background, j] / n_gt   (detected as nothing),
          * top-confusion = the largest off-diagonal, non-background row i for
                     this column (the class the model most often calls a j), and
          * fp     = matrix[j, background]   (class-j predictions matching no GT).
        Classes with no GT in the split are omitted.
        """
        bg = self.bg
        rows = []
        for j in range(self.num_classes):
            n_gt = int(self.matrix[:, j].sum())
            if n_gt == 0:
                continue
            best_i, best_c = -1, 0
            for i in range(self.num_classes):
                if i != j and self.matrix[i, j] > best_c:
                    best_c, best_i = int(self.matrix[i, j]), i
            rows.append({
                'idx': j, 'name': self._label(j), 'n_gt': n_gt,
                'recall': int(self.matrix[j, j]) / n_gt,
                'miss_frac': int(self.matrix[bg, j]) / n_gt,
                'conf_idx': best_i, 'conf_frac': (best_c / n_gt) if best_i >= 0 else 0.0,
                'fp': int(self.matrix[j, bg]),
            })
        rows.sort(key=lambda r: r['recall'])
        return rows

    def format_per_class(self, max_name: int = 16) -> str:
        """Readable per-class table (the terminal headline; the full grid is in the CSV/PNG)."""
        rows = self.per_class_report()
        lines = [
            f'Per-class detection report   conf>={self.conf_thresh:.2f}  iou>={self.iou_thresh:.2f}'
            '   (sorted: worst recall first)',
            f'  {"class":<{max_name}} {"n_gt":>6} {"recall":>7} {"miss→bg":>8}   '
            f'{"most-confused-with":<{max_name}} {"":>5} {"FP":>6}',
        ]
        for r in rows:
            if r['conf_idx'] >= 0:
                conf = f'{self._label(r["conf_idx"])[:max_name]:<{max_name}} {r["conf_frac"]*100:>4.0f}%'
            else:
                conf = f'{"-":<{max_name}} {"":>5}'
            lines.append(
                f'  {r["name"][:max_name]:<{max_name}} {r["n_gt"]:>6} '
                f'{r["recall"]*100:>6.1f}% {r["miss_frac"]*100:>7.1f}%   {conf} {r["fp"]:>6}'
            )
        return '\n'.join(lines)

    def format_table(self, max_name: int = 14, top: int = 20) -> str:
        """Render a row-normalized (%) table plus a top-confusions summary."""
        n = self.num_classes + 1
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        norm = np.where(row_sums > 0, self.matrix / np.maximum(row_sums, 1) * 100.0, 0.0)

        def short(idx):
            return self._label(idx)[:max_name]

        col_w = 6
        head = ' ' * (max_name + 1) + ''.join(f'{j:>{col_w}d}' for j in range(n))
        lines = [
            'Confusion matrix  rows = predicted, cols = ground truth '
            '(last index = background)',
            f'iou_thresh={self.iou_thresh:.2f}  conf_thresh={self.conf_thresh:.2f}  '
            f'values = row-normalized %',
            head,
        ]
        for i in range(n):
            cells = ''.join(f'{norm[i, j]:>{col_w}.0f}' for j in range(n))
            lines.append(f'{short(i):<{max_name}} {cells}   {short(i)}')

        lines.append('')
        lines.append('Legend (col/row index -> class):')
        for j in range(n):
            lines.append(f'  {j:>3d}  {self._label(j)}')

        confs = self.top_confusions(top)
        lines.append('')
        lines.append(f'Top {len(confs)} confusions (raw counts):')
        lines.append(f'  {"predicted":<{max_name}} {"ground_truth":<{max_name}} count')
        for pred_l, gt_l, cnt in confs:
            lines.append(f'  {pred_l[:max_name]:<{max_name}} {gt_l[:max_name]:<{max_name}} {cnt}')
        return '\n'.join(lines)

    def save_csv(self, path: str) -> None:
        """Write the raw integer matrix with a labeled header row/column."""
        _ensure_parent_dir(path)
        n = self.num_classes + 1
        header = ['pred\\gt'] + [self._label(j) for j in range(n)]
        rows = [header]
        for i in range(n):
            rows.append([self._label(i)] + [str(int(self.matrix[i, j])) for j in range(n)])
        with open(path, 'w') as f:
            f.write('\n'.join(','.join(r) for r in rows) + '\n')

    def save_png(self, path: str) -> bool:
        """Write a row-normalized heat map. Returns False if matplotlib is absent."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover - environment dependent
            log.warning('matplotlib unavailable (%s); skipping --output_png', e)
            return False

        _ensure_parent_dir(path)
        n = self.num_classes + 1
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        norm = np.where(row_sums > 0, self.matrix / np.maximum(row_sums, 1), 0.0)
        labels = [self._label(j) for j in range(n)]

        # Scale the figure and annotation font with class count so cells stay legible.
        cell = max(0.38, min(0.6, 22.0 / n))
        fig, ax = plt.subplots(figsize=(max(9, n * cell + 3), max(8, n * cell + 2)))
        im = ax.imshow(norm, cmap='Blues', vmin=0.0, vmax=1.0)
        ax.set_xlabel('ground truth')
        ax.set_ylabel('predicted')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)

        # Annotate each non-trivial cell with the row-normalized percentage so the
        # heat map carries numbers, not just colour. Zeros and <1% are left blank to
        # cut clutter; text flips to white on dark (high) cells for contrast.
        ann_fs = max(3.0, min(7.0, 26.0 / n))
        for i in range(n):
            for j in range(n):
                v = norm[i, j]
                if v < 0.01:
                    continue
                ax.text(j, i, f'{v*100:.0f}', ha='center', va='center',
                        fontsize=ann_fs, color=('white' if v > 0.5 else 'black'))

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True


# ----------------------------------------------------------------------------
# CLI (TensorFlow-heavy; imported lazily so --help and the pure helpers stay fast)
# ----------------------------------------------------------------------------

def _load_checkpoint_model(config, ckpt_path: str):
    from models.yolo_v8 import build_yolov8
    from common.ckpt_loading import restore_eval_weights

    model = build_yolov8(config.task.model)
    model.deploy = True
    model.build_and_init(config.task.model.input_size)
    kind = restore_eval_weights(model, ckpt_path)
    log.info('Checkpoint restored (%s weights): %s', kind, ckpt_path)
    return model


def _run(FLAGS) -> None:
    import dataclasses

    import tensorflow as tf

    from configs.yaml_loader import load_config
    from common.progress import Progress
    from common.runtime_setup import apply_eval_precision_policy
    from train.task import YoloV8Task, normalize_images

    if bool(FLAGS.checkpoint) == bool(FLAGS.saved_model):
        raise SystemExit('Provide exactly one of --checkpoint or --saved_model.')

    tf.config.run_functions_eagerly(False)
    config = load_config(FLAGS.config)
    apply_eval_precision_policy(config)
    task = YoloV8Task(config)
    task_cfg = config.task

    # Model source: checkpoint (build + restore EMA) or exported SavedModel.
    if FLAGS.saved_model:
        from utils.export.device_decode import (
            reconstruct_detections, is_device_contract, is_legacy_contract)
        loaded = tf.saved_model.load(FLAGS.saved_model)
        infer = loaded.signatures['serving_default']
        out_keys = list(infer.structured_outputs.keys())
        sig_in = infer.inputs[0].shape

        if is_device_contract(out_keys):
            mh, mw = int(sig_in[1]), int(sig_in[2])
            legacy_order = (FLAGS.device_box_order == 'yfirst')
            log.info('Device-contract SavedModel (box_order=%s) — reconstructing '
                     'detections from flat heads.', FLAGS.device_box_order)

            def predict(images):
                # The device SavedModel takes one image at a time ([1, H, W, 3]) with
                # pixels in [0, 255]; reconstruct each and stack back to a batch.
                imgs = tf.cast(images, tf.float32)
                per = [reconstruct_detections(dict(infer(input_image=imgs[i:i + 1])),
                                              mh, mw, legacy_box_order=legacy_order)
                       for i in range(int(imgs.shape[0]))]
                return {k: tf.constant(np.concatenate([p[k] for p in per], axis=0))
                        for k in ('bbox', 'classes', 'confidence', 'num_detections')}
        elif is_legacy_contract(out_keys):
            def predict(images):
                return infer(normalize_images(images))
        else:
            raise SystemExit(
                f'SavedModel signature outputs {sorted(out_keys)} match neither the device '
                "contract (flat 'box'+'cls' heads) nor a post-processed deploy dict "
                "(with 'num_detections'). Re-export with utils/export/export_saved_model.py.")
    else:
        model = _load_checkpoint_model(config, FLAGS.checkpoint)

        def predict(images):
            return model(normalize_images(images), training=False)

    # Validation dataset (eval mode: no training augmentation), from the config.
    data_cfg = task_cfg.train_data if FLAGS.split == 'train' else task_cfg.validation_data
    data_cfg = dataclasses.replace(data_cfg, is_training=False)
    val_ds = task.build_inputs(data_cfg)

    try:
        from configs.class_map import DETECTION_CLASSES
        names = [str(DETECTION_CLASSES[i]) for i in sorted(DETECTION_CLASSES)]
    except Exception:
        names = None

    cm = ConfusionMatrix(
        num_classes=task_cfg.num_classes,
        class_names=names,
        iou_thresh=FLAGS.iou,
        conf_thresh=FLAGS.conf,
        ignore_dontcare=task_cfg.ignore_dontcare,
        ignore_iscrowds=task_cfg.ignore_iscrowds,
        iscrowds_labels=task_cfg.iscrowds_labels,
    )

    # Derive the batch count so the progress bar shows a real bar + %/ETA (the
    # val split is finite; drop_remainder is off, so round up).
    import math
    n_examples = (task_cfg.train_total_examples if FLAGS.split == 'train'
                  else task_cfg.validation_total_examples)
    bs = int(getattr(data_cfg, 'global_batch_size', 0) or 0)
    total_batches = int(math.ceil(n_examples / bs)) if (n_examples and bs) else None

    pbar = Progress(total=total_batches, desc='Confusion', unit='batch')
    for images, labels in val_ds:
        preds = predict(images)
        B = int(preds['num_detections'].shape[0])
        ic = labels.get('is_crowd')
        idc = labels.get('is_dontcare')
        for i in range(B):
            nd = int(preds['num_detections'][i])
            ng = int(labels['n_gt'][i])
            cm.update(
                pred_boxes=preds['bbox'][i].numpy()[:nd],
                pred_classes=preds['classes'][i].numpy()[:nd],
                pred_scores=preds['confidence'][i].numpy()[:nd],
                gt_boxes=labels['bbox'][i].numpy()[:ng],
                gt_classes=labels['classes'][i].numpy()[:ng],
                gt_is_crowd=(ic[i].numpy()[:ng] if ic is not None else None),
                gt_is_dontcare=(idc[i].numpy()[:ng] if idc is not None else None),
            )
        pbar.update(1)
    pbar.close()

    # Headline: the readable per-class report + top confusions. The full 41x41
    # grid is unreadable in a terminal, so it goes to the CSV/PNG instead (a
    # numbered heat map), not stdout.
    print()
    print(cm.format_per_class())
    print()
    confs = cm.top_confusions(FLAGS.top)
    print(f'Top {len(confs)} raw confusions  (predicted <- ground_truth : count):')
    for pred_l, gt_l, cnt in confs:
        print(f'  {pred_l:<18} <- {gt_l:<18} {cnt}')

    if FLAGS.output_csv:
        cm.save_csv(FLAGS.output_csv)
        log.info('Raw matrix written to %s', FLAGS.output_csv)
    if FLAGS.output_png:
        if cm.save_png(FLAGS.output_png):
            log.info('Heat map written to %s', FLAGS.output_png)


def main(_):
    from absl import flags
    _run(flags.FLAGS)


def _define_flags():
    from absl import flags
    try:
        flags.DEFINE_string('config', None, 'Path to experiment YAML config.', required=True)
        flags.DEFINE_string('checkpoint', None, 'Checkpoint path prefix.')
        flags.DEFINE_string('saved_model', None, 'Exported SavedModel dir.')
        flags.DEFINE_string('split', 'val', "Eval split: 'val', 'test', or 'train'.")
        flags.DEFINE_enum('device_box_order', 'yfirst', ['yfirst', 'xfirst'],
                          "Box-channel order of a device-contract SavedModel's box head: "
                          "'yfirst' ([t,l,b,r], the export's legacy_box_order=True default) "
                          "or 'xfirst' ([l,t,r,b], a --legacy_box_order=False export).")
        flags.DEFINE_float('conf', 0.25, 'Min detection confidence, applied before matching.')
        flags.DEFINE_float('iou', 0.5, 'IoU threshold for a positive match.')
        flags.DEFINE_string('output_csv', None, 'Write the raw integer matrix (CSV) here.')
        flags.DEFINE_string('output_png', None, 'Write a row-normalized heat map (PNG) here.')
        flags.DEFINE_integer('top', 20, 'Top confusions to print in the summary.')
    except flags.DuplicateFlagError:
        pass


if __name__ == '__main__':
    from absl import app
    _define_flags()
    app.run(main)
