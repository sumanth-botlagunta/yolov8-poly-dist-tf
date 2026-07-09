"""Tests for the fixed-count epoch math in YoloV8Trainer.

The training stream is infinite (the detection sources repeat), so epoch length
is enforced by the trainer: every epoch is exactly ``steps_per_loop`` steps and
epoch k ends at global step ``k * steps_per_loop``. After a mid-epoch resume the
first epoch runs only the remainder to the next boundary.
"""

import unittest

from train.trainer import YoloV8Trainer


class TestStepsForEpoch(unittest.TestCase):
    SPL = 2118  # 271,166 images // 128 batch

    def test_fresh_start_runs_full_epoch(self):
        self.assertEqual(YoloV8Trainer._steps_for_epoch(0, self.SPL, 0), self.SPL)

    def test_epoch_boundary_runs_full_epoch(self):
        for k in (1, 2, 299):
            self.assertEqual(
                YoloV8Trainer._steps_for_epoch(k * self.SPL, self.SPL, k), self.SPL
            )

    def test_mid_epoch_resume_runs_remainder_to_boundary(self):
        # SIGTERM at step 5000 (mid 0-based epoch 2); resume must stop at 3*2118.
        start = 5000
        steps = YoloV8Trainer._steps_for_epoch(start, self.SPL, 2)
        self.assertEqual(start + steps, 3 * self.SPL)

    def test_one_step_before_boundary(self):
        start = self.SPL - 1
        self.assertEqual(YoloV8Trainer._steps_for_epoch(start, self.SPL, 0), 1)

    def test_trained_but_unvalidated_epoch_runs_zero_steps(self):
        """Death during validation: the boundary checkpoint records the epoch as
        not completed; the resume runs 0 training steps for it and goes straight
        to the pending validation (no checkpoint left unevaluated)."""
        self.assertEqual(YoloV8Trainer._steps_for_epoch(3 * self.SPL, self.SPL, 2), 0)

    def test_boundaries_stay_aligned_over_many_epochs(self):
        """Walking epoch by epoch always lands on exact multiples of SPL."""
        step = 1337  # arbitrary mid-epoch resume point (0-based epoch 0)
        for epoch in range(5):
            step += YoloV8Trainer._steps_for_epoch(step, self.SPL, epoch)
            self.assertEqual(step % self.SPL, 0)
            self.assertEqual(step, (epoch + 1) * self.SPL)


if __name__ == "__main__":
    unittest.main()
