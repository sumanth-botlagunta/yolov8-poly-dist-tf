"""Tests for DistanceEvaluator.

Validates:
    - Known MAE and RMSE values are computed correctly.
    - Invalid sentinel values are excluded.
    - Empty accumulation returns zeros without crashing.
    - reset() clears state.
"""

import math
import unittest
import numpy as np

from eval.distance_metrics import DistanceEvaluator, INVALID_SENTINEL


class TestDistanceEvaluator(unittest.TestCase):

    def test_known_mae_rmse(self):
        """Single pair: pred=log(2.0), gt=log(1.0) → MAE=RMSE=1.0 in meter space."""
        ev = DistanceEvaluator()
        ev.update(
            pred_log_dist=np.array([math.log(2.0)]),
            gt_log_dist=  np.array([math.log(1.0)]),
        )
        m = ev.evaluate()
        self.assertAlmostEqual(m['dist_mae'],  1.0, places=5)
        self.assertAlmostEqual(m['dist_rmse'], 1.0, places=5)

        # New metrics: pred=2.0m, gt=1.0m → absrel = |2-1|/1 = 1.0
        # gt=1.0 < 5.0 → near bucket; far bucket is empty → 0.0
        self.assertIn('dist_absrel', m)
        self.assertAlmostEqual(m['dist_absrel'], 1.0, places=5)

        self.assertIn('dist_abs_near', m)
        self.assertIn('dist_absrel_near', m)
        self.assertIn('dist_abs_far', m)
        self.assertIn('dist_absrel_far', m)

        # gt=1.0m is near (<5m): near MAE=1.0, near absrel=1.0
        self.assertAlmostEqual(m['dist_abs_near'],    1.0, places=5)
        self.assertAlmostEqual(m['dist_absrel_near'], 1.0, places=5)
        # No far samples → 0.0
        self.assertAlmostEqual(m['dist_abs_far'],    0.0, places=7)
        self.assertAlmostEqual(m['dist_absrel_far'], 0.0, places=7)

    def test_sentinel_excluded(self):
        """Entries with gt == INVALID_SENTINEL must not contribute to MAE."""
        ev = DistanceEvaluator()
        ev.update(
            pred_log_dist=np.array([math.log(2.0), 99.0]),
            gt_log_dist=  np.array([math.log(1.0), INVALID_SENTINEL]),
        )
        m = ev.evaluate()
        # Only first pair is valid
        self.assertAlmostEqual(m['dist_mae'], 1.0, places=5)

    def test_all_invalid_returns_zeros(self):
        """All-sentinel GT should return zeros without crashing."""
        ev = DistanceEvaluator()
        ev.update(
            pred_log_dist=np.array([1.0, 2.0]),
            gt_log_dist=  np.full(2, INVALID_SENTINEL),
        )
        m = ev.evaluate()
        self.assertAlmostEqual(m['dist_mae'],       0.0, places=7)
        self.assertAlmostEqual(m['dist_rmse'],      0.0, places=7)
        self.assertAlmostEqual(m['dist_absrel'],     0.0, places=7)
        self.assertAlmostEqual(m['dist_abs_near'],   0.0, places=7)
        self.assertAlmostEqual(m['dist_absrel_near'], 0.0, places=7)
        self.assertAlmostEqual(m['dist_abs_far'],    0.0, places=7)
        self.assertAlmostEqual(m['dist_absrel_far'], 0.0, places=7)

    def test_empty_evaluate_returns_zeros(self):
        """evaluate() on fresh evaluator must return zeros."""
        ev = DistanceEvaluator()
        m  = ev.evaluate()
        self.assertAlmostEqual(m['dist_mae'],       0.0, places=7)
        self.assertAlmostEqual(m['dist_rmse'],      0.0, places=7)
        self.assertAlmostEqual(m['dist_absrel'],     0.0, places=7)
        self.assertAlmostEqual(m['dist_abs_near'],   0.0, places=7)
        self.assertAlmostEqual(m['dist_absrel_near'], 0.0, places=7)
        self.assertAlmostEqual(m['dist_abs_far'],    0.0, places=7)
        self.assertAlmostEqual(m['dist_absrel_far'], 0.0, places=7)

    def test_reset_clears_state(self):
        """After reset(), evaluate returns zeros regardless of prior updates."""
        ev = DistanceEvaluator()
        ev.update(np.array([1.0]), np.array([0.5]))
        ev.reset()
        m = ev.evaluate()
        self.assertAlmostEqual(m['dist_mae'], 0.0, places=7)

    def test_multi_batch_accumulation(self):
        """Two calls to update() accumulate correctly."""
        ev = DistanceEvaluator()
        # Both predictions 1 meter above GT
        ev.update(np.array([math.log(2.0)]), np.array([math.log(1.0)]))
        ev.update(np.array([math.log(3.0)]), np.array([math.log(2.0)]))
        m = ev.evaluate()
        # Errors in meter space: |2-1|=1, |3-2|=1 → MAE=1.0
        self.assertAlmostEqual(m['dist_mae'], 1.0, places=4)

    def test_rmse_ge_mae(self):
        """RMSE must always be >= MAE (Cauchy-Schwarz)."""
        ev = DistanceEvaluator()
        ev.update(
            pred_log_dist=np.log([1.5, 3.0, 0.8]),
            gt_log_dist=  np.log([1.0, 1.0, 1.0]),
        )
        m = ev.evaluate()
        self.assertGreaterEqual(m['dist_rmse'], m['dist_mae'] - 1e-6)

    def test_caller_must_pass_log_not_metres(self):
        """Contract guard: the evaluator expects LOG-space inputs and exps once.

        The detection generator emits distance in METRES (already exp'd). Callers
        (train/task.py, tools/eval.py) must np.log() it before update(); passing
        metres directly double-exponentiates and yields a wildly wrong error.
        """
        pred_metres = np.array([2.0])           # 2 m predicted
        gt_log      = np.array([math.log(2.0)])  # GT is also 2 m

        # CORRECT caller behaviour: log the metre prediction → perfect match.
        ev_ok = DistanceEvaluator()
        ev_ok.update(np.log(pred_metres), gt_log)
        self.assertAlmostEqual(ev_ok.evaluate()['dist_mae'], 0.0, places=5)

        # BUGGY behaviour (metres treated as log): exp(2)=7.39 vs 2 → large error.
        ev_bug = DistanceEvaluator()
        ev_bug.update(pred_metres, gt_log)
        self.assertGreater(ev_bug.evaluate()['dist_mae'], 1.0)


if __name__ == '__main__':
    unittest.main()
