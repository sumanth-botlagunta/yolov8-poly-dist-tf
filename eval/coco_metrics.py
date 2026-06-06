"""COCO-style detection evaluator.

Wraps pycocotools to compute mAP, mAP50, AR, and F1@50 over accumulated
prediction/GT batches.  All bounding boxes are expected in yxyx-normalized
format from the model (matching the parser convention).

Classes:
    COCOEvaluator: Accumulates predictions + GT, computes mAP on evaluate().
"""

import io
import json
import logging
from contextlib import redirect_stdout
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class COCOEvaluator:
    """Wraps pycocotools for per-epoch mAP computation.

    Usage::

        evaluator = COCOEvaluator(num_classes=39, image_size=(672, 672))
        for batch in val_ds:
            evaluator.update(predictions, groundtruths)
        metrics = evaluator.evaluate()  # {'mAP', 'mAP50', 'AR100', 'F1score50'}
        evaluator.reset()

    Args:
        num_classes: Number of detection categories (0-indexed).
        image_size:  (H, W) of the input image in pixels — used to convert
                     normalized boxes to absolute pixel coordinates.
    """

    def __init__(self, num_classes: int, image_size=(672, 672)):
        self._num_classes = num_classes
        self._H, self._W = image_size[0], image_size[1]
        self._dt_anns: List[dict] = []
        self._gt_anns: List[dict] = []
        self._gt_imgs: List[dict] = []
        # pycocotools stores matched GT ID in dtMatches; 0 is falsy so any
        # annotation with id=0 would be treated as unmatched.  Start at 1.
        self._img_id  = 1
        self._ann_id  = 1

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(self, predictions: dict, groundtruths: dict) -> None:
        """Accumulate one batch of predictions and GT.

        Args:
            predictions:  Dict with keys 'bbox' [B,N,4], 'classes' [B,N],
                          'confidence' [B,N], 'num_detections' [B].
                          All boxes are yxyx-normalized.
            groundtruths: Dict with keys 'bbox' [B,M,4] yxyx-normalized,
                          'classes' [B,M] int64, 'n_gt' [B].
        """
        import tensorflow as tf

        batch_size = int(predictions['num_detections'].shape[0])
        H, W = self._H, self._W

        for i in range(batch_size):
            img_id  = self._img_id
            n_det   = int(predictions['num_detections'][i])
            n_gt    = int(groundtruths['n_gt'][i])

            self._gt_imgs.append({'id': img_id, 'height': H, 'width': W})

            # ---- GT annotations ----
            for j in range(n_gt):
                y1, x1, y2, x2 = [float(v) for v in groundtruths['bbox'][i, j]]
                cat = int(groundtruths['classes'][i, j])
                xywh = [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H]
                self._gt_anns.append({
                    'id':          self._ann_id,
                    'image_id':    img_id,
                    'category_id': cat,
                    'bbox':        xywh,
                    'area':        xywh[2] * xywh[3],
                    'iscrowd':     0,
                })
                self._ann_id += 1

            # ---- Detection results ----
            for j in range(n_det):
                y1, x1, y2, x2 = [float(v) for v in predictions['bbox'][i, j]]
                cat   = int(predictions['classes'][i, j])
                score = float(predictions['confidence'][i, j])
                xywh  = [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H]
                self._dt_anns.append({
                    'image_id':    img_id,
                    'category_id': cat,
                    'bbox':        xywh,
                    'score':       score,
                })

            self._img_id += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Compute mAP and related metrics from accumulated data.

        Returns:
            Dict with keys: mAP, mAP50, AR100, F1score50, and optionally
            per_class_AP50 (dict str→float).
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not self._gt_anns:
            log.warning("COCOEvaluator.evaluate() called with no GT annotations.")
            return {'mAP': 0.0, 'mAP50': 0.0, 'AR100': 0.0, 'F1score50': 0.0}

        cats = [{'id': c, 'name': str(c)} for c in range(self._num_classes)]
        gt_dict = {
            'images':      self._gt_imgs,
            'annotations': self._gt_anns,
            'categories':  cats,
        }

        coco_gt = COCO()
        with redirect_stdout(io.StringIO()):
            coco_gt.dataset = gt_dict
            coco_gt.createIndex()

        if self._dt_anns:
            with redirect_stdout(io.StringIO()):
                coco_dt = coco_gt.loadRes(self._dt_anns)
        else:
            return {'mAP': 0.0, 'mAP50': 0.0, 'AR100': 0.0, 'F1score50': 0.0}

        # ---- Standard eval (IoU 0.50:0.95) ----
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        with redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
        ev.summarize()

        map_val   = float(ev.stats[0])
        map50_val = float(ev.stats[1])
        ar100_val = float(ev.stats[8])

        # ---- F1@50: separate eval at IoU=0.5 only ----
        ev50 = COCOeval(coco_gt, coco_dt, 'bbox')
        ev50.params.iouThrs = np.array([0.5])
        with redirect_stdout(io.StringIO()):
            ev50.evaluate()
            ev50.accumulate()

        f1_50 = self._peak_f1(ev50)

        return {
            'mAP':        map_val,
            'mAP50':      map50_val,
            'AR100':      ar100_val,
            'F1score50':  f1_50,
        }

    def reset(self) -> None:
        """Clear all accumulated predictions and GT."""
        self._dt_anns.clear()
        self._gt_anns.clear()
        self._gt_imgs.clear()
        self._img_id = 0
        self._ann_id = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _peak_f1(coco_eval) -> float:
        """Mean peak-F1 over all classes from a COCOeval run at single IoU."""
        # precision shape: [T, R, K, A, M] — T=1 (single IoU), R=101 recall pts
        precision = coco_eval.eval.get('precision')
        if precision is None or precision.size == 0:
            return 0.0

        # precision[0, :, :, 0, 2] → [R, K] at IoU=0.5, all areas, maxDets=100
        prec = precision[0, :, :, 0, 2]   # [101, num_classes]
        recall_thrs = coco_eval.params.recThrs  # [101]

        class_f1 = []
        for k in range(prec.shape[1]):
            p = prec[:, k]
            r = recall_thrs
            valid = p >= 0
            if not valid.any():
                continue
            p, r = p[valid], r[valid]
            denom = p + r
            # Avoid division by zero
            f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
            class_f1.append(float(f1.max()))

        return float(np.mean(class_f1)) if class_f1 else 0.0
