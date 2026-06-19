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

    def __init__(
        self,
        num_classes: int,
        image_size=(672, 672),
        ignore_dontcare: bool = True,
        ignore_iscrowds: bool = False,
        iscrowds_labels: Optional[List[int]] = None,
    ):
        self._num_classes    = num_classes
        self._H, self._W     = image_size[0], image_size[1]
        self._ignore_dontcare = ignore_dontcare
        self._ignore_iscrowds = ignore_iscrowds
        self._iscrowds_labels = set(iscrowds_labels) if iscrowds_labels else set()
        self._dt_anns: List[dict] = []
        self._gt_anns: List[dict] = []
        self._gt_imgs: List[dict] = []
        # pycocotools stores matched GT ID in dtMatches; 0 is falsy so any
        # annotation with id=0 would be treated as unmatched.  Start at 1.
        self._img_id  = 1
        self._ann_id  = 1
        self._ev50    = None
        self._ev      = None   # full eval (all IoU thresholds)

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
                          'classes' [B,M] int64, 'n_gt' [B],
                          optionally 'is_crowd' [B,M] bool,
                          optionally 'is_dontcare' [B,M] bool.

        GT handling:
            is_crowd + class in iscrowds_labels → skip GT entirely (not a missed detection).
            is_dontcare → iscrowd=1 in COCO: absorbs overlapping detections (IoU>0.5)
                          without counting them as FP, but is itself not a TP.
        """
        import tensorflow as tf

        batch_size   = int(predictions['num_detections'].shape[0])
        H, W         = self._H, self._W
        is_crowd_arr    = groundtruths.get('is_crowd')
        is_dontcare_arr = groundtruths.get('is_dontcare')

        for i in range(batch_size):
            img_id  = self._img_id
            n_det   = int(predictions['num_detections'][i])
            n_gt    = int(groundtruths['n_gt'][i])

            self._gt_imgs.append({'id': img_id, 'height': H, 'width': W})

            # ---- GT annotations ----
            for j in range(n_gt):
                cat       = int(groundtruths['classes'][i, j])
                is_crowd  = bool(is_crowd_arr[i, j])    if is_crowd_arr    is not None else False
                is_dc     = bool(is_dontcare_arr[i, j]) if is_dontcare_arr is not None else False

                # iscrowd objects whose class is in the crowd-class list → skip entirely
                if self._ignore_iscrowds and is_crowd and cat in self._iscrowds_labels:
                    continue

                y1, x1, y2, x2 = [float(v) for v in groundtruths['bbox'][i, j]]
                xywh = [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H]

                # dontcare → iscrowd=1: pycocotools absorbs overlapping detections
                # without counting them as FP or TP
                iscrowd_val = 1 if (self._ignore_dontcare and is_dc) else 0

                self._gt_anns.append({
                    'id':          self._ann_id,
                    'image_id':    img_id,
                    'category_id': cat,
                    'bbox':        xywh,
                    'area':        xywh[2] * xywh[3],
                    'iscrowd':     iscrowd_val,
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

        # Clear any cached eval objects up front so an empty-detection early
        # return below can't leave stale per-category metrics from a prior call.
        self._ev = None
        self._ev50 = None

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
            ev.summarize()  # keep inside redirect: still populates ev.stats
        self._ev = ev   # keep for per_category_full_metrics()

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
        self._ev50 = ev50  # keep for per_category_ap50() / per_category_full_metrics()

        # Mean precision/recall at each class's peak-F1 operating point (same point
        # F1score50 is read from), so the logged scalars are mutually consistent.
        best = self.per_category_best_f1()
        prec_50 = float(np.mean([b['precision'] for b in best])) if best else 0.0
        rec_50  = float(np.mean([b['recall']    for b in best])) if best else 0.0

        return {
            'mAP':              map_val,
            'mAP50':            map50_val,
            'AR100':            ar100_val,
            'F1score50':        f1_50,
            'precision50':      prec_50,
            'recall50':         rec_50,
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

    def per_category_full_metrics(self) -> Dict[int, Dict[str, float]]:
        """Per-category full COCO metrics (12 per category) after evaluate().

        Mirrors the old-codebase ``_retrieve_per_category_metrics`` output:
            ap, ap50, ap75, ap_s, ap_m, ap_l  (precision-based)
            ar1, ar10, ar100, ar_s, ar_m, ar_l (recall-based)

        Returns:
            Dict mapping category_id → dict of metric_name → float.
            Empty if evaluate() has not been called.
        """
        ev = getattr(self, '_ev', None)
        if ev is None or ev.eval is None:
            return {}

        prec = ev.eval['precision']   # [T=10, R=101, K, A=4, M=3]
        rec  = ev.eval['recall']      # [T=10, K, A=4, M=3]

        def _mean(arr):
            v = arr[arr >= 0]
            return float(v.mean()) if v.size > 0 else 0.0

        result: Dict[int, Dict[str, float]] = {}
        for k, cat_id in enumerate(ev.params.catIds):
            result[int(cat_id)] = {
                'ap':    _mean(prec[:, :, k, 0, 2]),   # AP@50:95
                'ap50':  _mean(prec[0,  :, k, 0, 2]),  # AP@50
                'ap75':  _mean(prec[5,  :, k, 0, 2]),  # AP@75
                'ap_s':  _mean(prec[:, :, k, 1, 2]),   # AP small
                'ap_m':  _mean(prec[:, :, k, 2, 2]),   # AP medium
                'ap_l':  _mean(prec[:, :, k, 3, 2]),   # AP large
                'ar1':   _mean(rec[:,  k, 0, 0]),       # AR@1
                'ar10':  _mean(rec[:,  k, 0, 1]),       # AR@10
                'ar100': _mean(rec[:,  k, 0, 2]),       # AR@100
                'ar_s':  _mean(rec[:,  k, 1, 2]),       # AR small
                'ar_m':  _mean(rec[:,  k, 2, 2]),       # AR medium
                'ar_l':  _mean(rec[:,  k, 3, 2]),       # AR large
            }
        return result

    # ------------------------------------------------------------------
    # Per-category F1 / precision / recall tables (for the saved report)
    # ------------------------------------------------------------------

    def _ev50_pr_arrays(self):
        """(precision, scores, recThrs) at IoU=0.5, area='all', maxDets=100.

        precision/scores are [R=101, K] (recall-point × class); recThrs is [101].
        Same slice F1score50 is read from, so the report's mean F1 equals the
        logged F1score50. Returns (None, None, None) if evaluate() hasn't run.
        """
        ev50 = getattr(self, '_ev50', None)
        if ev50 is None or ev50.eval is None:
            return None, None, None
        prec = ev50.eval['precision'][0, :, :, 0, 2]            # [101, K]
        sc   = ev50.eval.get('scores')
        scores = sc[0, :, :, 0, 2] if sc is not None else None   # [101, K]
        return prec, scores, ev50.params.recThrs

    def _gt_counts(self) -> Dict[int, Dict[str, int]]:
        """Per-category {num_gt, dontcare} from the accumulated GT annotations.
        (iscrowd==1 marks dontcare here; raw crowd is filtered before accumulation.)"""
        counts: Dict[int, Dict[str, int]] = {}
        for a in self._gt_anns:
            c = counts.setdefault(int(a['category_id']), {'num_gt': 0, 'dontcare': 0})
            c['num_gt'] += 1
            if a.get('iscrowd'):
                c['dontcare'] += 1
        return counts

    def per_category_best_f1(self) -> List[Dict[str, float]]:
        """Per-category peak F1 with its precision / recall / conf threshold. After evaluate()."""
        prec, scores, rec = self._ev50_pr_arrays()
        if prec is None:
            return []
        out = []
        for k, cat in enumerate(self._ev50.params.catIds):
            p = prec[:, k]
            valid = p >= 0
            if not valid.any():
                out.append({'category': int(cat), 'f1': 0.0, 'precision': 0.0,
                            'recall': 0.0, 'conf_threshold': 0.0})
                continue
            pv, rv = p[valid], rec[valid]
            denom = pv + rv
            with np.errstate(divide='ignore', invalid='ignore'):
                f1 = np.where(denom > 0, 2 * pv * rv / denom, 0.0)
            i = int(f1.argmax())
            thr = 0.0
            if scores is not None:
                sv = scores[:, k][valid]
                thr = float(sv[i]) if sv[i] >= 0 else 0.0
            out.append({'category': int(cat), 'f1': float(f1[i]),
                        'precision': float(pv[i]), 'recall': float(rv[i]),
                        'conf_threshold': thr})
        return out

    def per_category_conf_sweep(self, conf_grid) -> List[Dict[str, float]]:
        """Per-category F1/precision/recall at each confidence threshold. After evaluate().

        For threshold t, includes all detections with score >= t: that is the
        highest-recall PR point whose score is still >= t (scores decrease as recall
        rises). Reads precision/recall there. Empty when t exceeds all scores.
        """
        prec, scores, rec = self._ev50_pr_arrays()
        if prec is None or scores is None:
            return []
        out = []
        for k, cat in enumerate(self._ev50.params.catIds):
            p = prec[:, k]
            s = scores[:, k]
            valid = p >= 0
            for t in conf_grid:
                sel = valid & (s >= t)
                if sel.any():
                    idx = int(np.where(sel)[0].max())     # max-recall point with score >= t
                    pp, rr = float(p[idx]), float(rec[idx])
                else:
                    pp, rr = 0.0, 0.0
                f1 = (2 * pp * rr / (pp + rr)) if (pp + rr) > 0 else 0.0
                out.append({'category': int(cat), 'thresh': round(float(t), 4),
                            'f1': f1, 'precision': pp, 'recall': rr})
        return out

    def metrics_tables(self, conf_grid=None) -> Dict:
        """Full per-category F1 report: averaged means + best-conf table + all-conf
        sweep + per-category AP / GT counts. Numbers are full-precision floats; a
        machine-readable structure callers serialize to JSON / csv / xlsx / txt.
        """
        if conf_grid is None:
            conf_grid = [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05)]
        best  = self.per_category_best_f1()
        sweep = self.per_category_conf_sweep(conf_grid)
        ap    = self.per_category_full_metrics()
        gtc   = self._gt_counts()
        per_cat_ap = [{
            'category': k,
            'ap':       ap.get(k, {}).get('ap', 0.0),
            'ap50':     ap.get(k, {}).get('ap50', 0.0),
            'num_gt':   gtc.get(k, {}).get('num_gt', 0),
            'dontcare': gtc.get(k, {}).get('dontcare', 0),
        } for k in sorted(ap)] if ap else []

        def _mean(key):
            vals = [b[key] for b in best]
            return float(np.mean(vals)) if vals else 0.0

        return {
            'iou_thresh': 0.5,
            'area':       'all',
            'max_dets':   100,
            'conf_grid':  list(conf_grid),
            'mean':       {'f1': _mean('f1'), 'precision': _mean('precision'),
                           'recall': _mean('recall')},
            'best_conf':  best,
            'all_conf':   sweep,
            'per_category_ap': per_cat_ap,
        }

    def reset(self) -> None:
        """Clear all accumulated predictions and GT."""
        self._dt_anns.clear()
        self._gt_anns.clear()
        self._gt_imgs.clear()
        self._img_id = 1
        self._ann_id = 1
        self._ev50   = None
        self._ev     = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _peak_f1(coco_eval):
        """Mean peak-F1 and mean confidence threshold at peak-F1 over all classes.

        Uses the ``scores`` tensor from COCOeval (shape [T, R, K, A, M]) which
        stores the detection confidence at each precision-recall curve point.
        This gives the actual NMS score threshold that achieves peak F1 —
        matching the old-codebase behavior of reporting a usable confidence value.

        Returns:
            (mean_peak_f1: float, mean_conf_thresh: float)
        """
        precision = coco_eval.eval.get('precision')
        if precision is None or precision.size == 0:
            return 0.0, 0.0

        prec   = precision[0, :, :, 0, 2]              # [101, num_classes]
        scores = coco_eval.eval.get('scores')           # [T, R, K, A, M] or None
        scores_arr = scores[0, :, :, 0, 2] if scores is not None else None  # [101, K]

        class_f1     = []
        class_thresh = []
        for k in range(prec.shape[1]):
            p     = prec[:, k]
            r     = coco_eval.params.recThrs   # [101]
            valid = p >= 0
            if not valid.any():
                continue
            p_v, r_v = p[valid], r[valid]
            denom = p_v + r_v
            with np.errstate(divide='ignore', invalid='ignore'):
                f1 = np.where(denom > 0, 2 * p_v * r_v / denom, 0.0)
            peak_idx = int(f1.argmax())
            class_f1.append(float(f1[peak_idx]))

            # Confidence threshold at peak F1: look up from scores tensor.
            # The scores array maps each PR curve point to the detection score
            # that achieved that recall — this is the usable NMS threshold.
            if scores_arr is not None:
                s = scores_arr[:, k][valid]
                thresh = float(s[peak_idx]) if s[peak_idx] >= 0 else 0.0
            else:
                thresh = 0.0
            class_thresh.append(thresh)

        mean_f1     = float(np.mean(class_f1))     if class_f1     else 0.0
        mean_thresh = float(np.mean(class_thresh)) if class_thresh else 0.0
        return mean_f1, mean_thresh
