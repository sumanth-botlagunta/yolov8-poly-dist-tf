"""Failure-case mining for evaluation — surface *why* a class is weak, not just the number.

During eval, ``FailureCollector`` greedily matches detections to GT per class (COCO-style)
and records three failure kinds, keeping the worst-K per class:

  * **FP**  — a confident detection with no GT match (sorted by score; the worst are the
              high-confidence false positives).
  * **FN**  — a GT object no detection matched (a miss; sorted by GT area).
  * **low-IoU** — a correct-class match that is poorly localized (sorted by lowest IoU).

``write()`` renders each kept case as an annotated image under
``<out_dir>/<NN_name>/<kind>_<rank>_*.png`` (GT green, the failing box red/orange, with the
class/score/IoU labelled), so you can open a weak class and look at its actual mistakes.

Memory: only the worst-K records per (class, kind) are retained; an evicted record's image
is dropped. Pairs with the ``per_class/`` TensorBoard metrics (which weak class → look here).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np


def _iou_yxyx(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two yxyx boxes (any consistent units)."""
    iy1, ix1 = max(a[0], b[0]), max(a[1], b[1])
    iy2, ix2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Record:
    __slots__ = ('badness', 'image', 'pred_box', 'gt_box', 'score', 'iou', 'img_id')

    def __init__(self, badness, image, pred_box, gt_box, score, iou, img_id):
        self.badness = badness        # higher = worse (sort key, kept top-K)
        self.image = image            # uint8 HxWx3 (RGB)
        self.pred_box = pred_box      # yxyx norm or None
        self.gt_box = gt_box          # yxyx norm or None
        self.score = score
        self.iou = iou
        self.img_id = img_id


class FailureCollector:
    def __init__(self, class_names: Optional[List[str]] = None, per_class: int = 8,
                 match_iou: float = 0.5, lowiou_below: float = 0.7,
                 score_thresh: float = 0.25):
        self.class_names = class_names
        self.per_class = per_class
        self.match_iou = match_iou
        self.lowiou_below = lowiou_below
        self.score_thresh = score_thresh
        # {(class, kind): [ _Record ... ]} kept sorted desc by badness, capped at per_class
        self._kept: Dict = {}
        self._img_id = 0

    def _add(self, cls: int, kind: str, rec: _Record) -> None:
        key = (cls, kind)
        bucket = self._kept.setdefault(key, [])
        bucket.append(rec)
        bucket.sort(key=lambda r: r.badness, reverse=True)
        if len(bucket) > self.per_class:
            bucket.pop()              # drop the least-bad (its image is released)

    def update(self, image_uint8: np.ndarray, pred: dict, gt: dict) -> None:
        """Match one image's detections vs GT and record failures.

        pred: {'bbox' [N,4] yxyx-norm, 'classes' [N], 'confidence' [N], 'num_detections'}.
        gt:   {'bbox' [M,4] yxyx-norm, 'classes' [M], 'n_gt'}.
        """
        img_id = self._img_id
        self._img_id += 1

        nd = int(pred['num_detections'])
        pb = np.asarray(pred['bbox'])[:nd]
        pc = np.asarray(pred['classes'])[:nd].astype(int)
        ps = np.asarray(pred['confidence'])[:nd]
        ng = int(gt['n_gt'])
        gb = np.asarray(gt['bbox'])[:ng]
        gc = np.asarray(gt['classes'])[:ng].astype(int)

        classes = set(pc.tolist()) | set(gc.tolist())
        for c in classes:
            det_idx = [i for i in range(nd) if pc[i] == c and ps[i] >= self.score_thresh]
            det_idx.sort(key=lambda i: -ps[i])           # high score first (COCO order)
            gt_idx = [j for j in range(ng) if gc[j] == c]
            matched = set()
            for i in det_idx:
                best_iou, best_j = 0.0, -1
                for j in gt_idx:
                    if j in matched:
                        continue
                    iou = _iou_yxyx(pb[i], gb[j])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= self.match_iou:
                    matched.add(best_j)
                    if best_iou < self.lowiou_below:      # matched but poorly localized
                        self._add(c, 'lowiou', _Record(
                            1.0 - best_iou, image_uint8, pb[i], gb[best_j],
                            float(ps[i]), best_iou, img_id))
                else:                                     # confident false positive
                    self._add(c, 'fp', _Record(
                        float(ps[i]), image_uint8, pb[i], None, float(ps[i]), 0.0, img_id))
            for j in gt_idx:                              # missed GT
                if j not in matched:
                    area = float((gb[j][2] - gb[j][0]) * (gb[j][3] - gb[j][1]))
                    self._add(c, 'fn', _Record(
                        area, image_uint8, None, gb[j], 0.0, 0.0, img_id))

    def _name(self, c: int) -> str:
        if self.class_names and 0 <= c < len(self.class_names):
            return f"{c:02d}_{self.class_names[c]}"
        return f"{c:02d}"

    def write(self, out_dir: str) -> int:
        """Render all kept failures to annotated PNGs. Returns the count written."""
        import cv2
        os.makedirs(out_dir, exist_ok=True)
        _COLORS = {'fp': (0, 0, 255), 'fn': (0, 220, 255), 'lowiou': (0, 140, 255)}  # BGR
        written = 0
        for (cls, kind), bucket in sorted(self._kept.items()):
            cdir = os.path.join(out_dir, self._name(cls))
            os.makedirs(cdir, exist_ok=True)
            for rank, r in enumerate(bucket):
                img = cv2.cvtColor(np.ascontiguousarray(r.image), cv2.COLOR_RGB2BGR)
                H, W = img.shape[:2]
                if r.gt_box is not None:                  # GT in green
                    y1, x1, y2, x2 = r.gt_box
                    cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)),
                                  (0, 200, 0), 2)
                if r.pred_box is not None:                # failing prediction
                    y1, x1, y2, x2 = r.pred_box
                    cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)),
                                  _COLORS[kind], 2)
                tag = {'fp': f"FP score={r.score:.2f}", 'fn': "MISSED GT",
                       'lowiou': f"low IoU={r.iou:.2f}"}[kind]
                cv2.putText(img, f"{self._name(cls)} | {tag}", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLORS[kind], 1, cv2.LINE_AA)
                path = os.path.join(cdir, f"{kind}_{rank:02d}_img{r.img_id}.png")
                cv2.imwrite(path, img)
                written += 1
        return written

    def summary(self) -> Dict[str, int]:
        out = {'fp': 0, 'fn': 0, 'lowiou': 0}
        for (_, kind), bucket in self._kept.items():
            out[kind] += len(bucket)
        return out
