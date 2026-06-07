"""Distance estimation evaluator.

Matches predicted distances to GT distances for valid samples (gt_dist > sentinel)
and computes MAE and RMSE in metric (meter) space.

Classes:
    DistanceEvaluator: Accumulates (pred, gt) pairs, computes MAE/RMSE.
"""

import logging
from typing import Dict

import numpy as np

log = logging.getLogger(__name__)

INVALID_SENTINEL = -10.0


class DistanceEvaluator:
    """Accumulates log-scale distance predictions and computes metric-space errors.

    Predictions and GT values are stored in log scale.  evaluate() exponentiates
    both before computing MAE and RMSE so errors are reported in meters.

    GT samples with value <= INVALID_SENTINEL are excluded entirely.

    Usage::

        ev = DistanceEvaluator()
        for batch in val_ds:
            ev.update(pred_log_dist, gt_log_dist)
        metrics = ev.evaluate()   # {'dist_mae': ..., 'dist_rmse': ...}
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
        """Compute MAE and RMSE in meter space.

        Returns:
            Dict with keys 'dist_mae' and 'dist_rmse'.
        """
        if not self._pred:
            log.debug("DistanceEvaluator.evaluate() called with no valid samples.")
            return {'dist_mae': 0.0, 'dist_rmse': 0.0}

        pred_all = np.exp(np.concatenate(self._pred))
        gt_all   = np.exp(np.concatenate(self._gt))

        diff = pred_all - gt_all
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        return {'dist_mae': mae, 'dist_rmse': rmse}

    def reset(self) -> None:
        """Clear accumulated data."""
        self._pred.clear()
        self._gt.clear()
