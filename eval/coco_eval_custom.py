"""Custom COCO detection evaluator with a confidence-sweep best-F1 metric.

``COCOevalCustom`` subclasses :class:`pycocotools.cocoeval.COCOeval` and adds two
structural extensions:

1. **Don't-care absorption** (`evaluateImgDontcare`): don't-care GTs are flagged by a
   separate ``ann['dontcare']==1`` field (NOT ``iscrowd``). A detection unmatched to a
   real GT but overlapping a don't-care GT at **IoU >= ``iou_thresh_dontcare`` (default
   0.5, FIXED at all IoU thresholds)** is recorded in ``dtMatchesDc`` and thereby
   excluded from false positives, and don't-care GTs are removed from ``npig``. This
   keeps ``iscrowd`` free for the separate crowd policy.

2. **Best-F1 via confidence-threshold sweep** (in ``accumulate``): for each threshold
   ``s`` in ``np.arange(0.1, 1.0, score_thresh_step)``, take the last detection with
   ``score > s`` (strict), read the raw cumulative precision/recall there, and compute
   ``2*p*r/(p+r+eps)``; keep the max over the sweep. The recall denominator carries a
   **hallucination-GT** correction: a TP whose matched GT id was already matched by a
   higher-scored detection inflates the denominator (``npig + cumsum(isHalluGt)``).

The macro-average best-F1 at (IoU=0.5, area='all', maxDets=10) over categories with a
valid (>= 0) best-F1 is the project's ``F1score50`` scalar used for best-checkpoint
selection. Formula constants: ``eps = np.spacing(1)``, strict ``>`` threshold,
last-index-above-threshold, ``mergesort`` by ``-score``.
"""

import copy
import datetime
import time
from collections import defaultdict

import numpy as np
from pycocotools.cocoeval import COCOeval


class COCOevalCustom(COCOeval):
    """COCOeval with custom dontcare handling + confidence-sweep best-F1.

    Args:
        cocoGt / cocoDt:        pycocotools COCO objects (GT / detections).
        iouType:                'bbox' / 'segm' / 'keypoints'.
        find_best_score_thresh: enable the F1 confidence sweep in accumulate().
        ignore_dontcare:        enable the custom evaluateImgDontcare + dontcare
                                FP-absorption / npig-subtraction path.
        ignore_iscrowds:        iscrowd GTs get ``ignore=1`` (dropped, not FN).
        iscrowds_labels:        GT categories in this set are treated as iscrowd.
        iou_thresh_dontcare:    fixed IoU threshold for dontcare absorption (0.5).
        score_thresh_step:      step of the confidence sweep grid (arange(0.1,1.0,step)).
    """

    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox',
                 verbose_eval_results_per_category=False,
                 find_best_score_thresh=True,
                 ignore_dontcare=False,
                 ignore_iscrowds=True,
                 iscrowds_labels=None,
                 iou_thresh_dontcare=0.5,
                 score_thresh_step=0.05):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        self._verbose_eval_results_per_category = verbose_eval_results_per_category
        self._find_best_score_thresh = find_best_score_thresh
        self._ignore_dontcare = ignore_dontcare
        self._iou_thresh_dontcare = iou_thresh_dontcare
        # arange(0.1, 1.0, step) -> [0.10, 0.15, ..., 0.95] for step=0.05. Floor is
        # 0.10: anything below is too low-confidence to be a useful operating point.
        # The report's all-conf grid (coco_metrics.metrics_tables) uses the SAME 0.10
        # floor so the best-conf table and the all-conf sweep agree.
        self._scoreTreshCand = np.arange(0.1, 1.0, score_thresh_step)
        self._ignore_iscrowds = ignore_iscrowds
        self._iscrowds_labels = (
            set(int(x) for x in iscrowds_labels)
            if iscrowds_labels is not None else None)

    # ------------------------------------------------------------------
    # _prepare — set ignore/iscrowd/dontcare flags
    # ------------------------------------------------------------------
    def _prepare(self, ignore_iscrowds=True, iscrowds_labels=None):
        """Prepare ._gts and ._dts for evaluation based on params."""
        def _toMask(anns, coco):
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['iscrowd'] = gt['iscrowd'] if 'iscrowd' in gt else 0
            # ensure a dontcare flag exists for evaluateImgDontcare
            gt['dontcare'] = gt['dontcare'] if 'dontcare' in gt else 0

            if iscrowds_labels is not None and len(iscrowds_labels) > 0:
                if int(gt['category_id']) in iscrowds_labels:
                    gt['iscrowd'] = 1
            if ignore_iscrowds:
                gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)
        self.eval = {}

    # ------------------------------------------------------------------
    # evaluateImgDontcare — custom per-image match w/ dontcare absorption
    # ------------------------------------------------------------------
    def evaluateImgDontcare(self, imgId, catId, aRng, maxDet):
        """Perform evaluation for a single category and image (dontcare-aware)."""
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious (reindexed by the gt sort above)
        ious = (self.ious[imgId, catId][:, gtind]
                if len(self.ious[imgId, catId]) > 0
                else self.ious[imgId, catId])

        # After the gt[gtind] reorder, gt is sorted ignore-LAST. The dontcare/nondc
        # split indexes into that *reordered* list, so it must iterate ascending
        # positions `range(len(gt))` — NOT `for i in gtind` (the permutation's order),
        # which would scramble the ignore-last ordering the matching loop's early-out
        # at `gtIg[m]==0 and gtIg[gind]==1` relies on (an ignore GT visited before a
        # real GT could absorb a detection that should match the real GT). Matches
        # stock pycocotools' `for gind, g in enumerate(gt)`.
        gtind_nondc = []
        gtind_dc = []
        for i in range(len(gt)):
            if int(gt[i]['dontcare']) == 1:
                gtind_dc.append(i)
            else:
                gtind_nondc.append(i)

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        dtmdc = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        gtIgDc = np.array([g['dontcare'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind in gtind_nondc:
                        g = gt[gind]
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if no real-GT match: try to absorb on a dontcare GT
                    if m == -1:
                        for gind in gtind_dc:
                            if gtIg[gind] == 1:
                                break
                            # dontcare gts do not care whether matched; no max-iou search.
                            if ious[dind, gind] >= self._iou_thresh_dontcare:
                                dtmdc[tind, dind] = gt[gind]['id']
                                break
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array(
            [d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]
        ).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        return {
            'image_id':    imgId,
            'category_id': catId,
            'aRng':        aRng,
            'maxDet':      maxDet,
            'dtIds':       [d['id'] for d in dt],
            'gtIds':       [g['id'] for g in gt],
            'dtMatches':   dtm,
            'dtMatchesDc': dtmdc,
            'gtMatches':   gtm,
            'dtScores':    [d['score'] for d in dt],
            'gtIgnore':    gtIg,
            'gtIgnoreDc':  gtIgDc,
            'dtIgnore':    dtIg,
        }

    # ------------------------------------------------------------------
    # evaluate — same orchestration, but dispatches to evaluateImgDontcare
    # ------------------------------------------------------------------
    def evaluate(self):
        """Run per-image evaluation; store results in self.evalImgs."""
        p = self.params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare(ignore_iscrowds=self._ignore_iscrowds,
                      iscrowds_labels=self._iscrowds_labels)
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds
                     for catId in catIds}

        if self._ignore_dontcare:
            evaluateImg = self.evaluateImgDontcare
        else:
            evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds]
        self._paramsEval = copy.deepcopy(self.params)

    # ------------------------------------------------------------------
    # accumulate — standard PR plus the custom F1 sweep + hgt + dontcare
    # ------------------------------------------------------------------
    def accumulate(self, p=None, find_best_score_thresh=True):
        """Accumulate per-image evaluation results into self.eval.

        Alongside the precision/recall/scores arrays this also fills:
          best_fiscore / best_fiscoreTresh / best_fiscorePrecision /
          best_fiscoreRecall  — each [T, K, A, M] — from the confidence sweep, plus
          the FULL sweep grid it was selected from: sweep_f1 / sweep_precision /
          sweep_recall — each [T, K, A, M, S] (S = len(_scoreTreshCand)) — and
          sweep_thresholds [S]. -1 marks a (cell, threshold) with no detection above
          that threshold. The report's all-conf table reads this raw grid so it agrees
          byte-for-byte with the best-F1 operating point.
        """
        if not self.evalImgs:
            raise Exception('Please run evaluate() first')
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        if find_best_score_thresh:
            bestF1Score = -np.ones((T, K, A, M))
            bestF1ScoreTresh = -np.ones((T, K, A, M))
            bestPrecision = -np.ones((T, K, A, M))
            bestRecall = -np.ones((T, K, A, M))
            # Full confidence sweep grid stored alongside the best-F1 readout so the
            # report's all-conf table reads the SAME raw operating-point p/r/F1 the
            # best-F1 selection sees (not COCO's interpolated envelope precision).
            # Shape [T, K, A, M, S]; -1 marks a (cell, threshold) with no detection
            # above that threshold. Filled inside the sweep loop below — no extra pass.
            S = len(self._scoreTreshCand)
            sweepF1 = -np.ones((T, K, A, M, S))
            sweepPrecision = -np.ones((T, K, A, M, S))
            sweepRecall = -np.ones((T, K, A, M, S))

        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # mergesort to be consistent with the Matlab implementation
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    if self._ignore_dontcare:
                        dtmDc = np.concatenate(
                            [e['dtMatchesDc'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    if self._ignore_dontcare:
                        # npig = tp + fn; dontcare GTs subtracted from npig
                        gtIgDc = np.concatenate([e['gtIgnoreDc'] for e in E])
                        npig = np.count_nonzero(
                            np.logical_and(gtIg == 0, gtIgDc == 0))
                    else:
                        npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    if self._ignore_dontcare:
                        fps = np.logical_and(fps, np.logical_not(dtmDc))

                    # hallucination-GT correction: a TP whose matched GT id was
                    # already matched by a higher-scored det inflates recall denom.
                    isHalluGtAll = []
                    for tps_, dtm_ in zip(tps, dtm):
                        matchedGtIds = set()
                        isHalluGtSub = []
                        for tp, dtGtId in zip(tps_, dtm_):
                            if tp:
                                if dtGtId in matchedGtIds:
                                    isHalluGtSub.append(1)
                                else:
                                    isHalluGtSub.append(0)
                                    matchedGtIds.add(dtGtId)
                            else:
                                isHalluGtSub.append(0)
                        isHalluGtAll.append(isHalluGtSub)
                    isHalluGtAll = np.array(isHalluGtAll)

                    hgt_sum = np.cumsum(isHalluGtAll, axis=1).astype(dtype=float)
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp, hgt) in enumerate(zip(tp_sum, fp_sum, hgt_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / (npig + hgt)
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if find_best_score_thresh:
                            maxF1Score = -1.0
                            maxF1ScoreTresh = -1.0
                            bestPr = -1.0
                            bestRc = -1.0

                            if nd:
                                for si, scoreTresh in enumerate(self._scoreTreshCand):
                                    mask = dtScoresSorted > scoreTresh
                                    if np.count_nonzero(mask) > 0:
                                        lastInd = np.where(mask)[0][-1]
                                        pr_val = pr[lastInd]
                                        rc_val = rc[lastInd]
                                        f1Score = (
                                            2 * (pr_val * rc_val)
                                            / (pr_val + rc_val + np.spacing(1)))
                                        # Store this operating point in the full grid
                                        # (cells with no det above thresh stay -1).
                                        sweepF1[t, k, a, m, si] = f1Score
                                        sweepPrecision[t, k, a, m, si] = pr_val
                                        sweepRecall[t, k, a, m, si] = rc_val
                                        if f1Score > maxF1Score:
                                            maxF1Score = f1Score
                                            maxF1ScoreTresh = scoreTresh
                                            bestPr = pr_val
                                            bestRc = rc_val

                                if maxF1Score < np.spacing(1):
                                    maxF1Score = -1.0
                                    maxF1ScoreTresh = -1.0
                                    bestPr = -1.0
                                    bestRc = -1.0

                                bestF1Score[t, k, a, m] = maxF1Score
                                bestF1ScoreTresh[t, k, a, m] = maxF1ScoreTresh
                                bestPrecision[t, k, a, m] = bestPr
                                bestRecall[t, k, a, m] = bestRc

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        pr = pr.tolist()
                        q = q.tolist()
                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds_r = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds_r):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except Exception:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)

        self.eval = {
            'params':    p,
            'counts':    [T, R, K, A, M],
            'date':      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':    recall,
            'scores':    scores,
        }
        if find_best_score_thresh:
            self.eval['best_fiscore'] = bestF1Score
            self.eval['best_fiscoreTresh'] = bestF1ScoreTresh
            self.eval['best_fiscorePrecision'] = bestPrecision
            self.eval['best_fiscoreRecall'] = bestRecall
            self.eval['sweep_f1'] = sweepF1
            self.eval['sweep_precision'] = sweepPrecision
            self.eval['sweep_recall'] = sweepRecall
            self.eval['sweep_thresholds'] = np.asarray(self._scoreTreshCand)

    # ------------------------------------------------------------------
    # bestF1 readout — macro-average over valid categories
    # ------------------------------------------------------------------
    def summarize_best_f1(self, iouThr=0.5, areaRng='all', maxDets=10):
        """Macro-average bestF1 over valid (>= 0) categories.

        Computes the ``F1score50`` scalar at (iouThr=0.5, areaRng='all', maxDets=10).
        Returns -1.0 if no category has a valid bestF1 (categories with no valid F1
        are excluded from the macro-average).
        """
        p = self.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        bestF1Score = self.eval['best_fiscore']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            bestF1Score = bestF1Score[t]
        bestF1Score = bestF1Score[:, :, aind, mind]   # [iou, cat, 1]

        catIds = p.catIds if p.useCats else [-1]
        iouThrs = [iouThr] if iouThr is not None else p.iouThrs

        avg_mF1Score = 0.0
        validScoreNum = 0
        for ioui, _iouthr in enumerate(iouThrs):
            mF1Score = 0.0
            validCatNum = 0
            for cati, catId in enumerate(catIds):
                if catId == -1:
                    break
                if bestF1Score[ioui, cati, 0] >= 0:
                    validCatNum += 1
                    mF1Score += bestF1Score[ioui, cati, 0]
            mF1Score = mF1Score / validCatNum if validCatNum > 0 else -1.0
            if mF1Score > -1.0:
                avg_mF1Score += mF1Score
                validScoreNum += 1

        if validScoreNum > 0:
            avg_mF1Score /= validScoreNum
        else:
            avg_mF1Score = -1.0
        return float(avg_mF1Score)

    # ------------------------------------------------------------------
    # summarize — fill self.stats with the standard COCO detection summary
    # ------------------------------------------------------------------
    def summarize(self):
        """Fill ``self.stats`` from the accumulated precision/recall arrays.

        Populates the 12 standard COCO detection summary slots plus a 13th slot
        carrying the best-F1 scalar:

            stats[0]  AP @[.50:.95] | area=all   | maxDets=100
            stats[1]  AP @.50       | area=all   | maxDets=100
            stats[2]  AP @.75       | area=all   | maxDets=100
            stats[3]  AP @[.50:.95] | area=small | maxDets=100
            stats[4]  AP @[.50:.95] | area=medium| maxDets=100
            stats[5]  AP @[.50:.95] | area=large | maxDets=100
            stats[6]  AR @[.50:.95] | area=all   | maxDets=1
            stats[7]  AR @[.50:.95] | area=all   | maxDets=10
            stats[8]  AR @[.50:.95] | area=all   | maxDets=100
            stats[9]  AR @[.50:.95] | area=small | maxDets=100
            stats[10] AR @[.50:.95] | area=medium| maxDets=100
            stats[11] AR @[.50:.95] | area=large | maxDets=100
            stats[12] F1 @.50       | area=all   | maxDets=10  (best-F1 sweep)

        Means are taken over entries ``>= 0`` (the COCO convention that drops
        recall points with no GT / detections). ``maxDets`` is selected by value
        against ``params.maxDets`` so the slot indices are robust to the array
        being augmented with extra thresholds.
        """
        p = self.params

        def _mDetIdx(value):
            idx = [i for i, m in enumerate(p.maxDets) if m == value]
            return idx[0] if idx else None

        def _ap(iouThr=None, areaRng='all', maxDets=100):
            precision = self.eval['precision']        # [T, R, K, A, M]
            aind = [i for i, a in enumerate(p.areaRngLbl) if a == areaRng]
            mind = _mDetIdx(maxDets)
            if mind is None or not aind:
                return -1.0
            s = precision
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
            valid = s[s > -1]
            return float(valid.mean()) if valid.size else -1.0

        def _ar(iouThr=None, areaRng='all', maxDets=100):
            recall = self.eval['recall']              # [T, K, A, M]
            aind = [i for i, a in enumerate(p.areaRngLbl) if a == areaRng]
            mind = _mDetIdx(maxDets)
            if mind is None or not aind:
                return -1.0
            s = recall
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
            valid = s[s > -1]
            return float(valid.mean()) if valid.size else -1.0

        stats = np.zeros((13,))
        stats[0] = _ap(maxDets=100)
        stats[1] = _ap(iouThr=0.5, maxDets=100)
        stats[2] = _ap(iouThr=0.75, maxDets=100)
        stats[3] = _ap(areaRng='small', maxDets=100)
        stats[4] = _ap(areaRng='medium', maxDets=100)
        stats[5] = _ap(areaRng='large', maxDets=100)
        stats[6] = _ar(maxDets=1)
        stats[7] = _ar(maxDets=10)
        stats[8] = _ar(maxDets=100)
        stats[9] = _ar(areaRng='small', maxDets=100)
        stats[10] = _ar(areaRng='medium', maxDets=100)
        stats[11] = _ar(areaRng='large', maxDets=100)
        if 'best_fiscore' in self.eval:
            f1 = self.summarize_best_f1(iouThr=0.5, areaRng='all', maxDets=10)
            stats[12] = f1 if f1 >= 0 else 0.0
        else:
            stats[12] = 0.0
        self.stats = stats
        return stats

    def per_category_best_f1(self, iouThr=0.5, areaRng='all', maxDets=10):
        """Per-category bestF1 / precision / recall / conf-threshold.

        Returns a list of dicts keyed by category id with the same (IoU, area,
        maxDets) slice that ``summarize_best_f1`` averages, flagging categories with
        no valid bestF1 (value < 0) as ``valid=False``.
        """
        p = self.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        def _slice(name):
            arr = self.eval[name]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                arr = arr[t]
            return arr[:, :, aind, mind]   # [iou, cat, 1]

        bF1 = _slice('best_fiscore')
        bTr = _slice('best_fiscoreTresh')
        bPr = _slice('best_fiscorePrecision')
        bRc = _slice('best_fiscoreRecall')

        catIds = p.catIds if p.useCats else [-1]
        out = []
        for cati, catId in enumerate(catIds):
            if catId == -1:
                break
            f1 = float(bF1[0, cati, 0])
            valid = f1 >= 0
            out.append({
                'category':       int(catId),
                'f1':             f1 if valid else 0.0,
                'precision':      float(bPr[0, cati, 0]) if valid else 0.0,
                'recall':         float(bRc[0, cati, 0]) if valid else 0.0,
                'conf_threshold': float(bTr[0, cati, 0]) if valid else 0.0,
                'valid':          bool(valid),
            })
        return out
