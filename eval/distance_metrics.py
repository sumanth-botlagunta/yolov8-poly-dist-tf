"""Distance estimation evaluator.

Matches predicted distances to GT distances for valid samples (gt_dist > sentinel)
and computes MAE, RMSE, relative MAE, and near/far splits in metric (meter) space.

Classes:
    DistanceEvaluator: Accumulates (pred, gt) pairs, computes metrics.
"""

import logging
from typing import Dict

import numpy as np

log = logging.getLogger(__name__)

INVALID_SENTINEL = -10.0
_NEAR_FAR_THRESHOLD = 5.0   # metres


class DistanceEvaluator:
    """Accumulates log-scale distance predictions and computes metric-space errors.

    Predictions and GT values are stored in log scale.  evaluate() exponentiates
    both before computing errors so all metrics are reported in meters.

    GT samples with value <= INVALID_SENTINEL are excluded entirely.

    Metrics returned by evaluate():
        dist_mae:         MAE across all valid samples
        dist_rmse:        RMSE across all valid samples
        dist_absrel:      Relative MAE (|pred-gt|/gt) across all valid samples
        dist_abs_near:    MAE for gt < 5m
        dist_absrel_near: Relative MAE for gt < 5m
        dist_abs_far:     MAE for gt >= 5m
        dist_absrel_far:  Relative MAE for gt >= 5m

    Usage::

        ev = DistanceEvaluator()
        for batch in val_ds:
            ev.update(pred_log_dist, gt_log_dist)
        metrics = ev.evaluate()
        ev.reset()
    """

    def __init__(self):
        self._pred: list = []
        self._gt:   list = []

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(self, pred_log_dist: np.ndarray, gt_log_dist: np.ndarray) -> None:
        """Accumulate one batch of matched prediction/GT pairs.

        Args:
            pred_log_dist: Predicted log-scale distances, shape [N] or [B, N].
                           Already matched to valid GT objects (caller's responsibility).
            gt_log_dist:   GT log-scale distances, same shape as pred_log_dist.
                           Entries with value <= INVALID_SENTINEL are skipped.
        """
        pred = np.asarray(pred_log_dist, dtype=np.float32).ravel()
        gt   = np.asarray(gt_log_dist,   dtype=np.float32).ravel()

        valid = gt > INVALID_SENTINEL
        if valid.any():
            self._pred.append(pred[valid])
            self._gt.append(gt[valid])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Compute distance metrics in meter space.

        Returns:
            Dict with keys: dist_mae, dist_rmse, dist_absrel,
            dist_abs_near, dist_absrel_near, dist_abs_far, dist_absrel_far.
            All values are 0.0 when no valid samples have been accumulated.
        """
        _zero = {
            'dist_mae': 0.0, 'dist_rmse': 0.0, 'dist_absrel': 0.0,
            'dist_abs_near': 0.0, 'dist_absrel_near': 0.0,
            'dist_abs_far':  0.0, 'dist_absrel_far':  0.0,
        }
        if not self._pred:
            log.debug("DistanceEvaluator.evaluate() called with no valid samples.")
            return _zero

        pred_all = np.exp(np.concatenate(self._pred))
        gt_all   = np.exp(np.concatenate(self._gt))

        abs_err = np.abs(pred_all - gt_all)
        rel_err = abs_err / gt_all

        mae    = float(np.mean(abs_err))
        rmse   = float(np.sqrt(np.mean((pred_all - gt_all) ** 2)))
        absrel = float(np.mean(rel_err))

        def _split(mask):
            if not mask.any():
                return 0.0, 0.0
            return float(np.mean(abs_err[mask])), float(np.mean(rel_err[mask]))

        near_mask = gt_all < _NEAR_FAR_THRESHOLD
        far_mask  = ~near_mask

        abs_near, absrel_near = _split(near_mask)
        abs_far,  absrel_far  = _split(far_mask)

        return {
            'dist_mae':         mae,
            'dist_rmse':        rmse,
            'dist_absrel':      absrel,
            'dist_abs_near':    abs_near,
            'dist_absrel_near': absrel_near,
            'dist_abs_far':     abs_far,
            'dist_absrel_far':  absrel_far,
        }

    def reset(self) -> None:
        """Clear accumulated data."""
        self._pred.clear()
        self._gt.clear()
