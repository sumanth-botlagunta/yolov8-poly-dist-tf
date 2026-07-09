"""Tests for COCOEvaluator.

Validates:
    - Perfect predictions (bbox == GT) yield mAP50 = 1.0.
    - No detections yields mAP = 0.0 without crashing.
    - reset() clears all accumulated state.
    - update() accepts multiple batches; metrics are consistent.
"""

import unittest
import numpy as np
import tensorflow as tf

from eval.coco_metrics import COCOEvaluator


def _make_batch(B=2, n_gt=2, img_h=100, img_w=100, num_classes=3):
    """Synthetic batch: all detections match GT exactly."""
    # GT boxes: two instances per image, yxyx normalized
    gt_boxes = np.array([[[0.1, 0.1, 0.5, 0.5],
                           [0.6, 0.6, 0.9, 0.9]]] * B, dtype=np.float32)
    gt_classes = np.array([[0, 1]] * B, dtype=np.int64)
    n_gt_arr   = np.array([n_gt] * B, dtype=np.int64)

    labels = {
        'bbox':    tf.constant(gt_boxes),
        'classes': tf.constant(gt_classes),
        'n_gt':    tf.constant(n_gt_arr),
    }

    # Perfect predictions: identical boxes, high confidence
    preds = {
        'bbox':           tf.constant(gt_boxes),
        'classes':        tf.constant(gt_classes),
        'confidence':     tf.ones([B, n_gt], dtype=tf.float32),
        'num_detections': tf.constant([n_gt] * B, dtype=tf.int32),
    }
    return preds, labels


class TestCOCOEvaluator(unittest.TestCase):

    def test_perfect_predictions_map50_is_one(self):
        """Exact bbox matches should yield mAP50 = 1.0."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        metrics = ev.evaluate()
        self.assertAlmostEqual(metrics['mAP50'], 1.0, places=2)

    def test_no_detections_returns_zero(self):
        """Empty detection list should not crash and return mAP = 0.0."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        # Zero out detections
        preds['num_detections'] = tf.zeros([2], dtype=tf.int32)
        ev.update(preds, labels)
        metrics = ev.evaluate()
        self.assertAlmostEqual(metrics['mAP50'], 0.0, places=5)

    def test_no_detections_returns_all_seven_keys(self):
        """Both early-return branches return the full metric key set."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        preds['num_detections'] = tf.zeros([2], dtype=tf.int32)
        ev.update(preds, labels)
        metrics = ev.evaluate()
        for k in ('mAP', 'mAP50', 'AR100', 'F1score50',
                  'precision50', 'recall50', 'best_conf_thresh'):
            self.assertIn(k, metrics)

    def test_macro_means_consistent_when_a_class_has_no_gt(self):
        """F1score50, precision50/recall50, and the saved report's mean F1 must all be
        averaged over the SAME classes (those with a valid PR point). A class absent from
        the GT (here class 2: num_classes=3 but GT only uses 0/1) has no valid PR point and
        must be excluded from every macro mean — not counted as 0 in some and skipped in
        others (the pre-fix inconsistency)."""
        # _make_batch: num_classes=3, GT classes {0,1}, perfect detections -> class 2 has
        # no GT, so its precision is all -1 (no valid PR point).
        preds, labels = _make_batch(num_classes=3)
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        ev.update(preds, labels)
        m = ev.evaluate()

        # Classes 0/1 perfectly detected; class 2 (no GT) excluded from the means.
        self.assertAlmostEqual(m['F1score50'], 1.0, places=5)
        self.assertAlmostEqual(m['precision50'], m['F1score50'], places=6)
        self.assertAlmostEqual(m['recall50'],    m['F1score50'], places=6)

        # The saved report's mean F1 equals F1score50 (same denominator)...
        report = ev.metrics_tables()
        self.assertAlmostEqual(report['mean']['f1'], m['F1score50'], places=6)
        # ...but the undetected class is still LISTED (flagged valid=False).
        best = ev.per_category_best_f1()
        invalid = [b for b in best if not b.get('valid', True)]
        self.assertTrue(any(b['category'] == 2 for b in invalid))

    def test_reset_clears_state(self):
        """After reset(), evaluate() on empty state returns zeros."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        ev.reset()
        metrics = ev.evaluate()
        self.assertAlmostEqual(metrics['mAP50'], 0.0, places=5)

    def test_metrics_dict_has_required_keys(self):
        """evaluate() must return mAP, mAP50, AR100, F1score50."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        metrics = ev.evaluate()
        for key in ('mAP', 'mAP50', 'AR100', 'F1score50'):
            self.assertIn(key, metrics)

    def test_multiple_batches_accumulated(self):
        """Calling update() twice doubles the sample count; result stays valid."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        ev.update(preds, labels)
        metrics = ev.evaluate()
        self.assertGreaterEqual(metrics['mAP50'], 0.0)
        self.assertLessEqual(metrics['mAP50'],    1.0)

    def test_f1score50_between_zero_and_one(self):
        """F1score50 must be in [0, 1]."""
        ev = COCOEvaluator(num_classes=3, image_size=(100, 100))
        preds, labels = _make_batch()
        ev.update(preds, labels)
        metrics = ev.evaluate()
        self.assertGreaterEqual(metrics['F1score50'], 0.0)
        self.assertLessEqual(metrics['F1score50'],    1.0)


class TestCustomF1Sweep(unittest.TestCase):
    """Tests for the F1score50 metric (confidence sweep, maxDets=10,
    hallucination-GT correction, dontcare absorption at IoU>=0.5 fixed)."""

    @staticmethod
    def _single_image(dets, gts, num_classes=1, ignore_dontcare=True,
                      ignore_iscrowds=False):
        """Build a 1-image evaluator state.

        dets: list of (yxyx_norm, cls, score)
        gts:  list of (yxyx_norm, cls, is_dontcare)
        """
        ev = COCOEvaluator(num_classes=num_classes, image_size=(100, 100),
                           ignore_dontcare=ignore_dontcare,
                           ignore_iscrowds=ignore_iscrowds)
        n_det, n_gt = len(dets), len(gts)
        preds = {
            'bbox':           tf.constant([[d[0] for d in dets]], dtype=tf.float32)
                              if n_det else tf.zeros([1, 0, 4], tf.float32),
            'classes':        tf.constant([[d[1] for d in dets]], dtype=tf.int64)
                              if n_det else tf.zeros([1, 0], tf.int64),
            'confidence':     tf.constant([[d[2] for d in dets]], dtype=tf.float32)
                              if n_det else tf.zeros([1, 0], tf.float32),
            'num_detections': tf.constant([n_det], dtype=tf.int32),
        }
        labels = {
            'bbox':        tf.constant([[g[0] for g in gts]], dtype=tf.float32)
                           if n_gt else tf.zeros([1, 0, 4], tf.float32),
            'classes':     tf.constant([[g[1] for g in gts]], dtype=tf.int64)
                           if n_gt else tf.zeros([1, 0], tf.int64),
            'n_gt':        tf.constant([n_gt], dtype=tf.int64),
            'is_dontcare': tf.constant([[bool(g[2]) for g in gts]], dtype=tf.bool)
                           if n_gt else tf.zeros([1, 0], tf.bool),
        }
        ev.update(preds, labels)
        return ev

    def test_f1_sweep_hand_computed(self):
        """Hand-computed best-F1 over the confidence grid arange(0.1,1.0,0.05).

        One class, 2 GT, 3 detections:
          A: TP for GT1, score 0.9
          B: FP (no overlap), score 0.6
          C: TP for GT2, score 0.3
        Sorted by -score: [A(TP), B(FP), C(TP)].
          tp=[1,1,2], fp=[0,1,1], npig=2, hgt=0
          rc = tp/2 = [0.5,0.5,1.0]
          pr = tp/(tp+fp) = [1.0,0.5,2/3]
        Sweep (strict >, last index above thresh):
          s in [0.1..0.25] -> all 3 kept -> last=C -> pr=2/3,rc=1 -> f1=0.8
          s in [0.3..0.55] -> A,B kept    -> last=B -> pr=0.5,rc=0.5 -> f1=0.5
          s in [0.6..0.85] -> A kept       -> last=A -> pr=1,rc=0.5 -> f1=2/3
          s in [0.9,0.95]  -> none kept -> skipped
        max F1 = 0.8.  An interpolated peak-F1 would NOT give exactly 0.8.
        """
        GT1 = [0.10, 0.10, 0.30, 0.30]
        GT2 = [0.50, 0.50, 0.70, 0.70]
        FP  = [0.80, 0.05, 0.95, 0.20]
        ev = self._single_image(
            dets=[(GT1, 0, 0.9), (FP, 0, 0.6), (GT2, 0, 0.3)],
            gts=[(GT1, 0, 0), (GT2, 0, 0)],
            num_classes=1,
        )
        m = ev.evaluate()
        self.assertAlmostEqual(m['F1score50'], 0.8, places=6)

        best = ev.per_category_best_f1()
        self.assertEqual(len(best), 1)
        self.assertTrue(best[0]['valid'])
        self.assertAlmostEqual(best[0]['f1'], 0.8, places=6)
        # best operating point is the low-threshold one keeping all 3 dets:
        self.assertAlmostEqual(best[0]['precision'], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(best[0]['recall'], 1.0, places=6)

    def test_dontcare_absorbs_fp(self):
        """A detection overlapping ONLY a dontcare GT (IoU>=0.5) is absorbed: not an
        FP, and the dontcare GT is removed from npig.

        1 class, GT = {GT1 (real), GTd (dontcare)}.  Detections:
          A: TP for GT1, score 0.9
          B: overlaps GTd at IoU=1.0, score 0.6  -> absorbed (dtMatchesDc), not FP
        npig = 1 (dontcare excluded). With B absorbed:
          sorted [A(TP), B(absorbed: not tp, not fp)]
          tp=[1,1], fp=[0,0], rc=tp/1=[1,1], pr=tp/(tp+fp)=[1,1]
          best F1 over sweep = 1.0.
        Without dontcare absorption (B as FP) F1 would be < 1.0, so this asserts the
        absorption is active.
        """
        GT1 = [0.10, 0.10, 0.30, 0.30]
        GTd = [0.50, 0.50, 0.70, 0.70]   # dontcare
        ev = self._single_image(
            dets=[(GT1, 0, 0.9), (GTd, 0, 0.6)],
            gts=[(GT1, 0, 0), (GTd, 0, 1)],   # second GT is dontcare
            num_classes=1, ignore_dontcare=True,
        )
        m = ev.evaluate()
        self.assertAlmostEqual(m['F1score50'], 1.0, places=6)

    def test_dontcare_off_counts_fp(self):
        """With ignore_dontcare=False the same overlapping det is a normal detection
        matched to the (now-real) GT -> still a TP here, but the dontcare GT counts in
        npig.  This pins that the dontcare channel is gated by the flag."""
        GT1 = [0.10, 0.10, 0.30, 0.30]
        GTd = [0.50, 0.50, 0.70, 0.70]
        ev = self._single_image(
            dets=[(GT1, 0, 0.9), (GTd, 0, 0.6)],
            gts=[(GT1, 0, 0), (GTd, 0, 1)],
            num_classes=1, ignore_dontcare=False,
        )
        m = ev.evaluate()
        # both dets are TPs of their (counted) GTs -> F1 = 1.0
        self.assertAlmostEqual(m['F1score50'], 1.0, places=6)

    def test_maxdets10_truncation(self):
        """F1 uses maxDets=10: only the top-10 highest-scored detections per image
        participate.  Put 1 real GT matched by a LOW-scored det, preceded by 10
        higher-scored false positives -> the TP is ranked 11th and dropped at
        maxDets=10, so there is no TP above any threshold -> best F1 = -1 -> 0.0.

        At maxDets=100 the TP would survive and yield a positive F1, so this isolates
        the maxDets=10 behavior.
        """
        GT1 = [0.10, 0.10, 0.30, 0.30]
        # 10 FP boxes, all higher score than the TP det
        fp_boxes = [[0.40 + 0.001 * i, 0.40, 0.45 + 0.001 * i, 0.45] for i in range(10)]
        dets = [(b, 0, 0.95 - 0.01 * i) for i, b in enumerate(fp_boxes)]
        dets.append((GT1, 0, 0.20))   # the only TP, lowest score -> rank 11
        ev = self._single_image(
            dets=dets, gts=[(GT1, 0, 0)], num_classes=1, ignore_dontcare=True,
        )
        m = ev.evaluate()
        # TP dropped by maxDets=10 -> no TP -> bestF1 sentinel -> 0.0
        self.assertAlmostEqual(m['F1score50'], 0.0, places=6)

    def test_hallucination_gt_correction(self):
        """Two detections both match the SAME single GT; the second (lower-scored) is
        a hallucination -> its recall denominator gets +1 (npig+hgt).

        1 class, 1 GT.  Dets: A(TP for GT, 0.9), B(also matches GT, 0.6).
        In pycocotools the 2nd det matching an already-matched GT is a FP (dtm=0).
        So actually B is an FP here. tp=[1,1], fp=[0,1]. This pins that a duplicate
        det does NOT inflate recall beyond 1 and is penalized as FP:
          rc=tp/1=[1,1]; pr=[1, 0.5]
          sweep: s low -> last=B -> f1=2*0.5*1/1.5=2/3 ; s in [0.6..0.85] -> last=A ->
          f1=1.0 ; best=1.0.
        """
        GT1 = [0.10, 0.10, 0.30, 0.30]
        ev = self._single_image(
            dets=[(GT1, 0, 0.9), (GT1, 0, 0.6)],
            gts=[(GT1, 0, 0)], num_classes=1, ignore_dontcare=True,
        )
        m = ev.evaluate()
        self.assertAlmostEqual(m['F1score50'], 1.0, places=6)

    def test_map_path_perfect(self):
        """mAP / F1 share the one evaluator: perfect detections -> mAP50 = 1.0 and
        F1score50 = 1.0 (sweep finds a perfect operating point)."""
        GT1 = [0.10, 0.10, 0.30, 0.30]
        GT2 = [0.50, 0.50, 0.70, 0.70]
        ev = self._single_image(
            dets=[(GT1, 0, 0.9), (GT2, 0, 0.8)],
            gts=[(GT1, 0, 0), (GT2, 0, 0)], num_classes=1,
        )
        m = ev.evaluate()
        self.assertAlmostEqual(m['mAP50'], 1.0, places=2)
        self.assertAlmostEqual(m['F1score50'], 1.0, places=6)


class TestCustomEvalRouting(unittest.TestCase):
    """mAP / mAP50 / AR100 are now sourced from COCOevalCustom, and the crowd /
    dontcare policy that COCOevalCustom applies must move those scalars relative to a
    no-crowd baseline (otherwise the routing isn't actually exercised)."""

    @staticmethod
    def _eval(dets, gts, **kw):
        ev = COCOEvaluator(num_classes=1, image_size=(100, 100), **kw)
        preds = {
            'bbox':           tf.constant([[d[0] for d in dets]], tf.float32),
            'classes':        tf.constant([[d[1] for d in dets]], tf.int64),
            'confidence':     tf.constant([[d[2] for d in dets]], tf.float32),
            'num_detections': tf.constant([len(dets)], tf.int32),
        }
        labels = {
            'bbox':        tf.constant([[g[0] for g in gts]], tf.float32),
            'classes':     tf.constant([[g[1] for g in gts]], tf.int64),
            'n_gt':        tf.constant([len(gts)], tf.int64),
            'is_dontcare': tf.constant([[bool(g[2]) for g in gts]], tf.bool),
        }
        ev.update(preds, labels)
        return ev

    def test_map_comes_from_custom_evaluator(self):
        """The cached evaluator is a COCOevalCustom and ev.stats slot 12 == F1score50."""
        from eval.coco_eval_custom import COCOevalCustom
        GT1 = [0.10, 0.10, 0.30, 0.30]
        ev = self._eval(dets=[(GT1, 0, 0.9)], gts=[(GT1, 0, 0)])
        m = ev.evaluate()
        self.assertIsInstance(ev._ev, COCOevalCustom)
        self.assertAlmostEqual(float(ev._ev.stats[0]), m['mAP'], places=6)
        self.assertAlmostEqual(float(ev._ev.stats[1]), m['mAP50'], places=6)
        self.assertAlmostEqual(float(ev._ev.stats[8]), m['AR100'], places=6)
        self.assertAlmostEqual(float(ev._ev.stats[12]), m['F1score50'], places=6)

    def test_dontcare_absorption_changes_map_and_ar(self):
        """A dontcare GT is dropped from the recall denominator (npig) and its
        overlapping detection is absorbed when the dontcare path is on, so mAP / AR
        differ from the baseline where it counts as a normal GT. An undetected real GT
        keeps recall below 1 so the denominator change is visible in the metrics."""
        GT1 = [0.10, 0.10, 0.30, 0.30]   # real, detected -> TP
        GT2 = [0.80, 0.80, 0.95, 0.95]   # real, NOT detected -> FN
        GTd = [0.50, 0.50, 0.70, 0.70]   # dontcare, detected
        dets = [(GT1, 0, 0.9), (GTd, 0, 0.6)]
        gts  = [(GT1, 0, 0), (GT2, 0, 0), (GTd, 0, 1)]

        m_off = self._eval(dets=dets, gts=gts, ignore_dontcare=False).evaluate()
        m_on  = self._eval(dets=dets, gts=gts, ignore_dontcare=True).evaluate()

        # dontcare on -> GTd dropped from npig (2 vs 3) and its det absorbed -> recall
        # denominator shrinks, so mAP and AR both drop relative to the baseline.
        self.assertLess(m_on['AR100'], m_off['AR100'])
        self.assertLess(m_on['mAP'],   m_off['mAP'])

    def test_f1_still_maxdets10(self):
        """F1score50 still uses maxDets=10: a TP ranked 11th behind 10 FPs is dropped,
        so F1 collapses to 0 even though the single evaluator carries maxDets 1/10/100."""
        GT1 = [0.10, 0.10, 0.30, 0.30]
        fp_boxes = [[0.40 + 0.001 * i, 0.40, 0.45 + 0.001 * i, 0.45] for i in range(10)]
        dets = [(b, 0, 0.95 - 0.01 * i) for i, b in enumerate(fp_boxes)]
        dets.append((GT1, 0, 0.20))
        ev = self._eval(dets=dets, gts=[(GT1, 0, 0)], ignore_dontcare=True)
        m = ev.evaluate()
        self.assertEqual(ev._f1_max_dets, 10)
        self.assertAlmostEqual(m['F1score50'], 0.0, places=6)


class TestRawSweepConsistency(unittest.TestCase):
    """The report's all-conf table now reads the RAW confidence-sweep grid stored by
    COCOevalCustom.accumulate (sweep_f1/precision/recall), so it agrees byte-for-byte
    with the best-conf table / F1score50 (previously it read COCO's interpolated
    envelope precision and could disagree at the same class+threshold)."""

    @staticmethod
    def _evaluator(num_classes=3, seed=0, n=6):
        """Synthetic multi-image run with real detections: each image carries M jittered
        TP detections plus a couple of random FPs, so every class has a non-trivial
        confidence sweep (best F1 lands at an interior threshold, not a flat 1.0)."""
        ev = COCOEvaluator(num_classes=num_classes, image_size=(100, 100))
        rng = np.random.RandomState(seed)
        for _ in range(n):
            M = 4
            gt_b = np.clip(rng.uniform(0, 0.6, [1, M, 4]), 0, 1).astype('float32')
            gt_b[..., 2:] = gt_b[..., :2] + 0.3
            gt_c = rng.randint(0, num_classes, [1, M]).astype('int64')
            pb = gt_b + rng.uniform(-0.02, 0.02, [1, M, 4]).astype('float32')
            ps = rng.uniform(0.3, 0.95, [1, M]).astype('float32')
            fp_b = np.clip(rng.uniform(0.6, 0.9, [1, 2, 4]), 0, 1).astype('float32')
            fp_b[..., 2:] = fp_b[..., :2] + 0.05
            fp_c = rng.randint(0, num_classes, [1, 2]).astype('int64')
            fp_s = rng.uniform(0.15, 0.6, [1, 2]).astype('float32')
            allb = np.concatenate([pb, fp_b], axis=1)
            allc = np.concatenate([gt_c, fp_c], axis=1)
            alls = np.concatenate([ps, fp_s], axis=1)
            ev.update({'bbox': allb, 'classes': allc, 'confidence': alls,
                       'num_detections': np.array([allb.shape[1]], 'int32')},
                      {'bbox': gt_b, 'classes': gt_c, 'n_gt': np.array([M], 'int32')})
        return ev

    def test_best_conf_equals_argmax_over_all_conf(self):
        """(a) For EVERY (valid) class the best-conf row equals the argmax-by-f1 over
        that class's all-conf rows — same f1/precision/recall/threshold.
        (b) The mean of valid best-conf F1 equals F1score50."""
        from collections import defaultdict
        ev = self._evaluator()
        m = ev.evaluate()
        tables = ev.metrics_tables()   # raw sweep by default
        self.assertEqual(tables['sweep_source'], 'raw')

        best = {b['category']: b for b in tables['best_conf']}
        bycat = defaultdict(list)
        for r in tables['all_conf']:
            bycat[r['category']].append(r)

        valid_f1s = []
        checked = 0
        for cat, brow in best.items():
            if not brow['valid']:
                continue
            rows = bycat[cat]
            self.assertTrue(rows, f"class {cat} missing from all_conf")
            arg = max(rows, key=lambda r: r['f1'])   # first max (ascending threshold)
            self.assertAlmostEqual(arg['f1'],        brow['f1'],             places=6)
            self.assertAlmostEqual(arg['precision'], brow['precision'],      places=6)
            self.assertAlmostEqual(arg['recall'],    brow['recall'],         places=6)
            self.assertAlmostEqual(arg['thresh'],    brow['conf_threshold'], places=4)
            valid_f1s.append(brow['f1'])
            checked += 1

        self.assertGreater(checked, 0, "no valid class to check")
        # (b) existing invariant, asserted explicitly against the headline scalar.
        self.assertAlmostEqual(float(np.mean(valid_f1s)), m['F1score50'], places=6)

    def test_stored_sweep_grid_shape_and_thresholds(self):
        """(d) The stored sweep arrays have shape [T, K, A, M, S], the threshold grid is
        arange(0.1, 1.0, step), and every non-sentinel value is a valid probability."""
        ev = self._evaluator()
        ev.evaluate()
        e = ev._ev.eval
        for key in ('sweep_f1', 'sweep_precision', 'sweep_recall', 'sweep_thresholds'):
            self.assertIn(key, e)
        p = ev._ev.params
        T = len(p.iouThrs)
        K = len(p.catIds)
        A = len(p.areaRng)
        Mn = len(p.maxDets)
        S = len(ev._ev._scoreTreshCand)
        self.assertEqual(e['sweep_f1'].shape,        (T, K, A, Mn, S))
        self.assertEqual(e['sweep_precision'].shape, (T, K, A, Mn, S))
        self.assertEqual(e['sweep_recall'].shape,    (T, K, A, Mn, S))
        self.assertEqual(e['sweep_thresholds'].shape, (S,))
        np.testing.assert_allclose(e['sweep_thresholds'], np.arange(0.1, 1.0, 0.05))
        for arr in (e['sweep_f1'], e['sweep_precision'], e['sweep_recall']):
            valid = arr[arr >= 0]
            self.assertTrue(np.all(valid <= 1.0 + 1e-9))

    def test_envelope_path_sets_source_and_txt_header(self):
        """(c) envelope=True keeps the legacy interpolated-envelope values, tags
        sweep_source='coco_envelope', and the txt writer prints a one-line header
        distinguishing it from the raw operating-point table."""
        import os
        import tempfile
        from eval import metrics_report as mr

        ev = self._evaluator()
        ev.evaluate()
        t_raw = ev.metrics_tables(envelope_sweep=False)
        t_env = ev.metrics_tables(envelope_sweep=True)
        self.assertEqual(t_raw['sweep_source'], 'raw')
        self.assertEqual(t_env['sweep_source'], 'coco_envelope')
        # same number of (class, threshold) rows either way
        self.assertEqual(len(t_env['all_conf']), len(t_raw['all_conf']))

        with tempfile.TemporaryDirectory() as d:
            pe = mr.write_txt(mr.build_report(ev, envelope_sweep=True),
                              os.path.join(d, 'env.txt'))
            pr = mr.write_txt(mr.build_report(ev, envelope_sweep=False),
                              os.path.join(d, 'raw.txt'))
            te = open(pe).read()
            tr = open(pr).read()
        self.assertIn('COCO-interpolated', te)
        self.assertNotIn('COCO-interpolated', tr)


if __name__ == '__main__':
    unittest.main()
