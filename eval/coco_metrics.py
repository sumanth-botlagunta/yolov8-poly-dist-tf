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
        self._ev50    = None

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

        f1_50, best_thresh = self._peak_f1(ev50)
        self._ev50 = ev50  # keep for per_category_ap50()

        return {
            'mAP':              map_val,
            'mAP50':            map50_val,
            'AR100':            ar100_val,
            'F1score50':        f1_50,
            'best_conf_thresh': best_thresh,
        }

    def per_category_ap50(self) -> Dict[int, float]:
        """Per-category AP@50.  Call after evaluate()."""
        ev50 = getattr(self, '_ev50', None)
        if ev50 is None or ev50.eval is None:
            return {}
        prec = ev50.eval['precision']  # [T=1, R=101, K, A=1, M=1]
        result: Dict[int, float] = {}
        for k, cat_id in enumerate(ev50.params.catIds):
            p = prec[0, :, k, 0, 2]
            valid = p[p >= 0]
            result[int(cat_id)] = float(valid.mean()) if valid.size > 0 else 0.0
        return result

    def reset(self) -> None:
        """Clear all accumulated predictions and GT."""
        self._dt_anns.clear()
        self._gt_anns.clear()
        self._gt_imgs.clear()
        self._img_id = 1
        self._ann_id = 1
        self._ev50   = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _peak_f1(coco_eval):
        """Mean peak-F1 and best confidence threshold over all classes.

        Returns:
            (mean_peak_f1: float, best_conf_thresh: float)
        """
        precision = coco_eval.eval.get('precision')
        if precision is None or precision.size == 0:
            return 0.0, 0.0

        prec = precision[0, :, :, 0, 2]   # [101, num_classes]
        recall_thrs = coco_eval.params.recThrs  # [101]

        class_f1        = []
        best_recall_idx = []
        for k in range(prec.shape[1]):
            p = prec[:, k]
            r = recall_thrs
            valid = p >= 0
            if not valid.any():
                continue
            p, r = p[valid], r[valid]
            denom = p + r
            with np.errstate(divide='ignore', invalid='ignore'):
                f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
            peak_idx = int(f1.argmax())
            class_f1.append(float(f1[peak_idx]))
            best_recall_idx.append(float(r[peak_idx]))

        mean_f1 = float(np.mean(class_f1)) if class_f1 else 0.0
        best_thresh = float(np.mean(best_recall_idx)) if best_recall_idx else 0.0
        return mean_f1, best_thresh
