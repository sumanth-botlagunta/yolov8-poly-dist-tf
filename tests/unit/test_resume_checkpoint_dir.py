"""Resume-checkpoint directory arbitration (interruption saves live in resume/).

Interruption saves (SIGTERM/preemption/supervisor restarts) go to
``<output_dir>/resume/`` so the main directory stays a clean sequence of
epoch-boundary checkpoints. ``_auto_resume`` restores whichever of the two
directories holds the HIGHEST global step:

  * boundary at 4236, interruption at 4500  → resume/ wins (continue mid-epoch)
  * later boundary at 6354 lands            → main wins (stale resume ignored —
    "used once" semantics with no extra bookkeeping)
"""

import unittest

from train.trainer import YoloV8Trainer


class TestCheckpointStepParsing(unittest.TestCase):
    def test_parses_step_suffix(self):
        self.assertEqual(YoloV8Trainer._checkpoint_step('/a/b/ckpt-4236'), 4236)
        self.assertEqual(YoloV8Trainer._checkpoint_step('/a/resume/ckpt-4500'), 4500)

    def test_unparseable_is_minus_one(self):
        self.assertEqual(YoloV8Trainer._checkpoint_step('/a/b/checkpoint'), -1)
        self.assertEqual(YoloV8Trainer._checkpoint_step('/a/b/ckpt-final'), -1)


class TestPickLatestCheckpoint(unittest.TestCase):
    def test_interruption_newer_than_boundary_wins(self):
        picked = YoloV8Trainer._pick_latest_checkpoint(
            ['/run/ckpt-4236', '/run/resume/ckpt-4500'])
        self.assertEqual(picked, '/run/resume/ckpt-4500')

    def test_stale_resume_is_superseded_by_newer_boundary(self):
        picked = YoloV8Trainer._pick_latest_checkpoint(
            ['/run/ckpt-6354', '/run/resume/ckpt-4500'])
        self.assertEqual(picked, '/run/ckpt-6354')

    def test_handles_missing_candidates(self):
        self.assertEqual(
            YoloV8Trainer._pick_latest_checkpoint([None, '/run/resume/ckpt-10']),
            '/run/resume/ckpt-10')
        self.assertIsNone(YoloV8Trainer._pick_latest_checkpoint([None, None]))
        self.assertIsNone(YoloV8Trainer._pick_latest_checkpoint([]))

    def test_equal_steps_prefer_either_deterministically(self):
        # Same step in both dirs (interruption exactly at a boundary) — both
        # contain identical state; any deterministic pick is fine.
        picked = YoloV8Trainer._pick_latest_checkpoint(
            ['/run/ckpt-4236', '/run/resume/ckpt-4236'])
        self.assertIn(picked, ['/run/ckpt-4236', '/run/resume/ckpt-4236'])


if __name__ == '__main__':
    unittest.main()
