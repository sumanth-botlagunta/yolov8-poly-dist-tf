"""Tests for the fixed-count epoch math in YoloV8Trainer.

The training stream is infinite (the detection sources repeat), so epoch length
is enforced by the trainer: every epoch is exactly ``steps_per_loop`` steps and
epoch k ends at global step ``k * steps_per_loop``. After a mid-epoch resume the
first epoch runs only the remainder to the next boundary.
"""

import unittest

from train.trainer import YoloV8Trainer


class TestStepsForEpoch(unittest.TestCase):
    SPL = 2388  # 305,780 images // 128 batch

    def test_fresh_start_runs_full_epoch(self):
        self.assertEqual(YoloV8Trainer._steps_for_epoch(0, self.SPL), self.SPL)

    def test_epoch_boundary_runs_full_epoch(self):
        for k in (1, 2, 299):
            self.assertEqual(
                YoloV8Trainer._steps_for_epoch(k * self.SPL, self.SPL), self.SPL
            )

    def test_mid_epoch_resume_runs_remainder_to_boundary(self):
        # SIGTERM at step 5000 (mid epoch 3); resume must stop at 3*2388 = 7164.
        start = 5000
        steps = YoloV8Trainer._steps_for_epoch(start, self.SPL)
        self.assertEqual(start + steps, 3 * self.SPL)

    def test_one_step_before_boundary(self):
        start = self.SPL - 1
        self.assertEqual(YoloV8Trainer._steps_for_epoch(start, self.SPL), 1)

    def test_boundaries_stay_aligned_over_many_epochs(self):
        """Walking epoch by epoch always lands on exact multiples of SPL."""
        step = 1337  # arbitrary mid-epoch resume point
        for _ in range(5):
            step += YoloV8Trainer._steps_for_epoch(step, self.SPL)
            self.assertEqual(step % self.SPL, 0)


if __name__ == "__main__":
    unittest.main()
