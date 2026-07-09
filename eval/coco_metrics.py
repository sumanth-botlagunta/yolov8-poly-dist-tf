"""COCO-style detection evaluator.

Computes mAP, mAP50, AR, and F1@50 over accumulated prediction/GT batches using a
single :class:`~eval.coco_eval_custom.COCOevalCustom` instance, so every detection
metric shares one match table, crowd policy, and don't-care absorption pass. All
bounding boxes are expected in yxyx-normalized format from the model (matching the
parser convention).

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

# Category ids treated as crowd regions by the project's crowd policy. Used as the
# default ``iscrowds_labels`` when a caller does not supply its own list.
DEFAULT_ISCROWD_LABELS = [6, 13, 24, 36, 37]


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
        find_best_score_thresh: bool = True,
        score_thresh_step: float = 0.05,
        f1_max_dets: int = 10,
    ):
        # Every metric comes from one COCOevalCustom: mAP / mAP50 / AR100 from its
        # precision/recall summary, and F1score50 from a confidence-threshold sweep
        # (step 0.05) on the cumulative precision/recall with a hallucination-GT recall
        # correction, at IoU=0.5 / area='all' / maxDets=10, macro-averaged over
        # categories with a valid (>= 0) bestF1. The GT-build policy is exposed as
        # flags: the defaults absorb dontcare regions and keep crowd GT; setting
        # ``ignore_iscrowds`` instead drops crowd GT. ``iscrowds_labels`` defaults to
        # the project's crowd-policy category ids.
        self._num_classes    = num_classes
        self._H, self._W     = image_size[0], image_size[1]
        self._ignore_dontcare = ignore_dontcare
        self._ignore_iscrowds = ignore_iscrowds
        labels = iscrowds_labels if iscrowds_labels is not None else DEFAULT_ISCROWD_LABELS
        self._iscrowds_labels = set(labels)
        self._find_best_score_thresh = find_best_score_thresh
        self._score_thresh_step = score_thresh_step
        self._f1_max_dets       = f1_max_dets
        self._dt_anns: List[dict] = []
        self._gt_anns: List[dict] = []
        self._gt_imgs: List[dict] = []
        # pycocotools stores the matched GT id in dtMatches; 0 is falsy, so an
        # annotation with id=0 would read as unmatched. Start ids at 1.
        self._img_id  = 1
        self._ann_id  = 1
        self._ev      = None   # single COCOevalCustom (all IoU thresholds + F1 sweep)

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
            is_dontcare → carried as a separate 'dontcare' field: absorbs overlapping
                          detections (IoU>=0.5) without counting them as FP, but is
                          itself not a TP, and is removed from the recall denominator.
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

            # GT annotations
            for j in range(n_gt):
                cat       = int(groundtruths['classes'][i, j])
                is_crowd  = bool(is_crowd_arr[i, j])    if is_crowd_arr    is not None else False
                is_dc     = bool(is_dontcare_arr[i, j]) if is_dontcare_arr is not None else False

                # iscrowd objects whose class is in the crowd-class list are skipped
                if self._ignore_iscrowds and is_crowd and cat in self._iscrowds_labels:
                    continue

                y1, x1, y2, x2 = [float(v) for v in groundtruths['bbox'][i, j]]
                xywh = [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H]

                # Carry dontcare as a separate field, not collapsed into iscrowd.
                # COCOevalCustom keys dontcare absorption off ann['dontcare'] (IoU
                # >= 0.5 fixed), while iscrowd stays = raw is_crowd for the crowd
                # policy (COCOevalCustom._prepare additionally sets iscrowd=1 for
                # categories in iscrowds_labels).
                dontcare_val = 1 if is_dc else 0
                iscrowd_val  = 1 if is_crowd else 0

                self._gt_anns.append({
                    'id':          self._ann_id,
                    'image_id':    img_id,
                    'category_id': cat,
                    'bbox':        xywh,
                    'area':        xywh[2] * xywh[3],
                    'iscrowd':     iscrowd_val,
                    'dontcare':    dontcare_val,
                })
                self._ann_id += 1

            # Detection results
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

    def evaluate(self) -> Dict[str, float]:
        """Compute mAP and related metrics from accumulated data.

        All metrics come from one :class:`COCOevalCustom`: mAP / mAP50 / AR100 from
        its summary slots, F1score50 from the confidence-threshold sweep.

        Returns:
            Dict with keys: mAP, mAP50, AR100, F1score50, precision50, recall50,
            best_conf_thresh.
        """
        from pycocotools.coco import COCO

        if not self._gt_anns:
            log.warning("COCOEvaluator.evaluate() called with no GT annotations.")
            return {'mAP': 0.0, 'mAP50': 0.0, 'AR100': 0.0, 'F1score50': 0.0,
                    'precision50': 0.0, 'recall50': 0.0, 'best_conf_thresh': 0.0}

        from .coco_eval_custom import COCOevalCustom

        cats = [{'id': c, 'name': str(c)} for c in range(self._num_classes)]

        # GT carries raw iscrowd (= is_crowd) plus a separate 'dontcare' field;
        # COCOevalCustom absorbs dontcare at IoU>=0.5 via dtMatchesDc and sets
        # iscrowd=1 for iscrowds_labels categories in _prepare.
        gt_dict = {
            'images':      self._gt_imgs,
            'annotations': self._gt_anns,
            'categories':  cats,
        }

        # Clear any cached eval object up front so an empty-detection early
        # return below can't leave stale per-category metrics from a prior call.
        self._ev = None

        coco_gt = COCO()
        with redirect_stdout(io.StringIO()):
            coco_gt.dataset = gt_dict
            coco_gt.createIndex()

        if self._dt_anns:
            with redirect_stdout(io.StringIO()):
                coco_dt = coco_gt.loadRes(self._dt_anns)
        else:
            return {'mAP': 0.0, 'mAP50': 0.0, 'AR100': 0.0, 'F1score50': 0.0,
                    'precision50': 0.0, 'recall50': 0.0, 'best_conf_thresh': 0.0}

        image_ids = [img['id'] for img in self._gt_imgs]

        # Single evaluator: full PR summary (IoU 0.50:0.95) + F1 sweep.
        ev = COCOevalCustom(
            coco_gt, coco_dt, iouType='bbox',
            find_best_score_thresh=self._find_best_score_thresh,
            ignore_dontcare=self._ignore_dontcare,
            ignore_iscrowds=self._ignore_iscrowds,
            iscrowds_labels=sorted(self._iscrowds_labels) if self._iscrowds_labels else None,
            iou_thresh_dontcare=0.5,
            score_thresh_step=self._score_thresh_step,
        )
        ev.params.imgIds = image_ids
        # Ensure the F1 maxDets slot exists in params.maxDets (sorted), as the F1
        # readout selects the slot whose value == f1_max_dets.
        ev.params.maxDets = sorted(set(list(ev.params.maxDets) + [self._f1_max_dets]))
        with redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate(find_best_score_thresh=self._find_best_score_thresh)
            ev.summarize()  # fills ev.stats (12 COCO slots + F1@.5 in slot 12)

        self._ev = ev   # keep for per_category_* / metrics_tables()

        map_val   = float(ev.stats[0])
        map50_val = float(ev.stats[1])
        ar100_val = float(ev.stats[8])
        f1_50     = float(ev.stats[12])

        # Mean precision/recall at each class's best-F1 operating point, averaged over
        # the SAME categories F1score50 uses (valid bestF1 >= 0), so the three scalars
        # share a denominator (mean of per-category bestF1 == F1score50).
        best = self.per_category_best_f1()
        valid_best = [b for b in best if b.get('valid', True)]
        prec_50 = float(np.mean([b['precision'] for b in valid_best])) if valid_best else 0.0
        rec_50  = float(np.mean([b['recall']    for b in valid_best])) if valid_best else 0.0
        best_thresh = (
            float(np.mean([b['conf_threshold'] for b in valid_best])) if valid_best else 0.0)

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
        ev = getattr(self, '_ev', None)
        if ev is None or ev.eval is None:
            return {}
        prec = ev.eval['precision']  # [T, R, K, A, M]
        t = np.where(0.5 == ev.params.iouThrs)[0][0]
        m100 = [i for i, m in enumerate(ev.params.maxDets) if m == 100][0]
        result: Dict[int, float] = {}
        for k, cat_id in enumerate(ev.params.catIds):
            p = prec[t, :, k, 0, m100]
            valid = p[p >= 0]
            result[int(cat_id)] = float(valid.mean()) if valid.size > 0 else 0.0
        return result

    def per_category_full_metrics(self) -> Dict[int, Dict[str, float]]:
        """Per-category full COCO metrics (12 per category) after evaluate().

        Reports the standard 12 COCO metrics per category:
            ap, ap50, ap75, ap_s, ap_m, ap_l  (precision-based)
            ar1, ar10, ar100, ar_s, ar_m, ar_l (recall-based)

        Returns:
            Dict mapping category_id → dict of metric_name → float.
            Empty if evaluate() has not been called.
        """
        ev = getattr(self, '_ev', None)
        if ev is None or ev.eval is None:
            return {}

        prec = ev.eval['precision']   # [T, R, K, A, M]
        rec  = ev.eval['recall']      # [T, K, A, M]

        # Resolve maxDets / IoU slots by value (params.maxDets may be augmented).
        def _mIdx(v):
            idx = [i for i, m in enumerate(ev.params.maxDets) if m == v]
            return idx[0] if idx else len(ev.params.maxDets) - 1
        m1, m10, m100 = _mIdx(1), _mIdx(10), _mIdx(100)
        t50 = int(np.where(0.5 == ev.params.iouThrs)[0][0])
        t75 = int(np.where(0.75 == ev.params.iouThrs)[0][0])

        def _mean(arr):
            v = arr[arr >= 0]
            return float(v.mean()) if v.size > 0 else 0.0

        result: Dict[int, Dict[str, float]] = {}
        for k, cat_id in enumerate(ev.params.catIds):
            result[int(cat_id)] = {
                'ap':    _mean(prec[:,   :, k, 0, m100]),  # AP@50:95
                'ap50':  _mean(prec[t50, :, k, 0, m100]),  # AP@50
                'ap75':  _mean(prec[t75, :, k, 0, m100]),  # AP@75
                'ap_s':  _mean(prec[:,   :, k, 1, m100]),  # AP small
                'ap_m':  _mean(prec[:,   :, k, 2, m100]),  # AP medium
                'ap_l':  _mean(prec[:,   :, k, 3, m100]),  # AP large
                'ar1':   _mean(rec[:,  k, 0, m1]),          # AR@1
                'ar10':  _mean(rec[:,  k, 0, m10]),         # AR@10
                'ar100': _mean(rec[:,  k, 0, m100]),        # AR@100
                'ar_s':  _mean(rec[:,  k, 1, m100]),        # AR small
                'ar_m':  _mean(rec[:,  k, 2, m100]),        # AR medium
                'ar_l':  _mean(rec[:,  k, 3, m100]),        # AR large
            }
        return result

    def _pr_arrays_at_50(self):
        """(precision, scores, recThrs) at IoU=0.5, area='all', maxDets=f1_max_dets.

        Uses the same detection budget (``f1_max_dets``, default 10) as the headline
        F1score50 and the per-category best-F1 table, so the all-conf sweep table stays
        consistent with them. precision/scores are [R=101, K] (recall-point × class);
        recThrs is [101].
        Returns (None, None, None) if evaluate() hasn't run.
        """
        ev = getattr(self, '_ev', None)
        if ev is None or ev.eval is None:
            return None, None, None
        t = int(np.where(0.5 == ev.params.iouThrs)[0][0])
        md = [i for i, m in enumerate(ev.params.maxDets) if m == self._f1_max_dets][0]
        prec = ev.eval['precision'][t, :, :, 0, md]          # [101, K]
        sc   = ev.eval.get('scores')
        scores = sc[t, :, :, 0, md] if sc is not None else None   # [101, K]
        return prec, scores, ev.params.recThrs

    def _gt_counts(self) -> Dict[int, Dict[str, int]]:
        """Per-category {num_gt, dontcare} from the accumulated GT annotations.
        Dontcare is now a separate field (not collapsed into iscrowd)."""
        counts: Dict[int, Dict[str, int]] = {}
        for a in self._gt_anns:
            c = counts.setdefault(int(a['category_id']), {'num_gt': 0, 'dontcare': 0})
            c['num_gt'] += 1
            if a.get('dontcare'):
                c['dontcare'] += 1
        return counts

    def per_category_best_f1(self) -> List[Dict[str, float]]:
        """Per-category best F1 (confidence-sweep) with its precision / recall /
        conf threshold, at (IoU=0.5, area='all', maxDets=f1_max_dets). Categories
        with no valid bestF1 are flagged ``valid=False`` so the macro means exclude
        them — matching F1score50. After evaluate()."""
        ev = getattr(self, '_ev', None)
        if ev is None or not getattr(ev, 'eval', None):
            return []
        return ev.per_category_best_f1(
            iouThr=0.5, areaRng='all', maxDets=self._f1_max_dets)

    def per_category_conf_sweep(self, conf_grid, envelope=False) -> List[Dict[str, float]]:
        """Per-category F1/precision/recall at each confidence threshold. After evaluate().

        Default (``envelope=False``): reads the raw confidence sweep grid stored by
        ``COCOevalCustom.accumulate`` (sweep_f1 / sweep_precision / sweep_recall at
        IoU=0.5, area='all', maxDets=``f1_max_dets``) — the exact operating-point
        counts the headline F1score50 / per-category best-F1 are selected from, so the
        all-conf table agrees with the best-conf table for the same (class, threshold).
        Each requested ``conf_grid`` value maps to the stored threshold grid (identical
        by default); a threshold not in the grid snaps to the nearest stored one and the
        reported ``thresh`` is that stored value. Cells with no detection above the
        threshold (stored -1) report f1/precision/recall = 0.0.

        ``envelope=True`` keeps the legacy behavior: reads COCO's interpolated
        (monotone-envelope) precision array with a ``score >= t`` selection. Kept for
        back-compat / diagnostics; it can disagree with the best-conf table.
        """
        if envelope:
            return self._per_category_conf_sweep_envelope(conf_grid)

        ev = getattr(self, '_ev', None)
        if ev is None or not getattr(ev, 'eval', None) or 'sweep_f1' not in ev.eval:
            return []
        t = int(np.where(0.5 == ev.params.iouThrs)[0][0])
        aind = [i for i, a in enumerate(ev.params.areaRngLbl) if a == 'all'][0]
        md = [i for i, m in enumerate(ev.params.maxDets) if m == self._f1_max_dets][0]
        grid = np.asarray(ev.eval['sweep_thresholds'], dtype=float)
        sf1 = ev.eval['sweep_f1'][t, :, aind, md, :]          # [K, S]
        spr = ev.eval['sweep_precision'][t, :, aind, md, :]   # [K, S]
        srr = ev.eval['sweep_recall'][t, :, aind, md, :]      # [K, S]

        out = []
        for k, cat in enumerate(ev.params.catIds):
            for t_req in conf_grid:
                # Same grid by default; otherwise snap to the nearest stored threshold
                # and report that stored value.
                si = int(np.argmin(np.abs(grid - float(t_req))))
                thr = float(grid[si])
                f1, pp, rr = float(sf1[k, si]), float(spr[k, si]), float(srr[k, si])
                if f1 < 0 or pp < 0 or rr < 0:   # no detection above threshold
                    f1, pp, rr = 0.0, 0.0, 0.0
                out.append({'category': int(cat), 'thresh': round(thr, 4),
                            'f1': f1, 'precision': pp, 'recall': rr})
        return out

    def _per_category_conf_sweep_envelope(self, conf_grid) -> List[Dict[str, float]]:
        """Legacy all-conf sweep from COCO's interpolated envelope precision.

        For threshold t, includes all detections with score >= t: that is the
        highest-recall PR point whose score is still >= t (scores decrease as recall
        rises). Reads precision/recall there. Empty when t exceeds all scores.
        """
        prec, scores, rec = self._pr_arrays_at_50()
        if prec is None or scores is None:
            return []
        out = []
        for k, cat in enumerate(self._ev.params.catIds):
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

    def metrics_tables(self, conf_grid=None, envelope_sweep=False) -> Dict:
        """Full per-category F1 report: averaged means + best-conf table + all-conf
        sweep + per-category AP / GT counts. Numbers are full-precision floats; a
        machine-readable structure callers serialize to JSON / csv / xlsx / txt.

        ``envelope_sweep=False`` (default) builds the all-conf table from the raw
        operating-point sweep so it agrees with the headline F1score50 / best-conf
        table; ``envelope_sweep=True`` uses COCO's interpolated envelope precision. The
        chosen source is reported as ``sweep_source`` = 'raw' or 'coco_envelope'.
        """
        if conf_grid is None:
            # Floor 0.10 to match the best-F1 sweep grid (COCOevalCustom._scoreTreshCand);
            # below that is too low-confidence to be a useful operating point.
            conf_grid = [round(float(x), 2) for x in np.arange(0.1, 1.0, 0.05)]
        best  = self.per_category_best_f1()
        sweep = self.per_category_conf_sweep(conf_grid, envelope=envelope_sweep)
        ap    = self.per_category_full_metrics()
        gtc   = self._gt_counts()
        per_cat_ap = [{
            'category': k,
            'ap':       ap.get(k, {}).get('ap', 0.0),
            'ap50':     ap.get(k, {}).get('ap50', 0.0),
            'num_gt':   gtc.get(k, {}).get('num_gt', 0),
            'dontcare': gtc.get(k, {}).get('dontcare', 0),
        } for k in sorted(ap)] if ap else []

        # Average over classes with a valid PR point only, so the report's mean F1
        # equals the logged F1score50 (empty classes are still listed in `best_conf`).
        valid_best = [b for b in best if b.get('valid', True)]

        def _mean(key):
            vals = [b[key] for b in valid_best]
            return float(np.mean(vals)) if vals else 0.0

        return {
            'iou_thresh': 0.5,
            'area':       'all',
            'max_dets':   self._f1_max_dets,
            'conf_grid':  list(conf_grid),
            'mean':       {'f1': _mean('f1'), 'precision': _mean('precision'),
                           'recall': _mean('recall')},
            'sweep_source': 'coco_envelope' if envelope_sweep else 'raw',
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
        self._ev     = None
