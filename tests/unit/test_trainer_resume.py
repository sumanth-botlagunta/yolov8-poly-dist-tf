"""Pins checkpoint/resume epoch accounting of YoloV8Trainer.

Regression test for the boundary-checkpoint bug: the final step of an epoch
fires the in-loop periodic save (checkpoint_interval == steps_per_loop in all
tier configs) BEFORE `_epoch_var.assign(epoch + 1)`, and the corrected
epoch-end save is then skipped by the `last_saved_step` guard. Every boundary
checkpoint therefore persisted `completed_epochs` one too low, so any resume
from one re-ran a full extra epoch (and re-launching a finished run trained one
extra epoch past decay_steps).

Uses a stub task (no real data / model graph needed) driving the REAL trainer
loop, checkpoint plumbing, and EMA optimizer.
"""

import numpy as np
import pytest
import tensorflow as tf

from configs.model_config import ExperimentConfig
from optimizers.ema import ExponentialMovingAverage
from optimizers.sgd_warmup import SGDTorch
from train.trainer import YoloV8Trainer

_SPL = 3      # steps per epoch
_EPOCHS = 2


class _StubTask:
    """Minimal task: infinite synthetic train stream, 1-batch val, no-op steps."""

    def __init__(self, config):
        self._config = config
        self._model = None
        self._loss_fn = None

    def build_model(self):
        self._model = tf.keras.Sequential(
            [tf.keras.layers.Dense(1, input_shape=(1,))]
        )
        return self._model

    def initialize(self, model):
        pass

    def apply_freezing(self, model):
        pass

    def prepare_grad_accumulation(self, model):
        pass

    def build_losses(self):
        return None

    def build_optimizer(self):
        lr_fn = tf.keras.optimizers.schedules.CosineDecay(0.01, 100)
        sgd = SGDTorch(
            lr_fn=lr_fn, momentum=0.9, momentum_start=0.8,
            nesterov=True, weight_decay=0.0, warmup_steps=2,
        )
        return ExponentialMovingAverage(
            optimizer=sgd, model=self._model, average_decay=0.9999,
            dynamic_decay=True,
        )

    def build_inputs(self, data_cfg):
        element = (
            tf.zeros([1, 1], tf.float32),
            {'n_gt': tf.zeros([1], tf.int64)},
        )
        ds = tf.data.Dataset.from_tensors(element)
        if getattr(data_cfg, 'is_training', True):
            return ds.repeat()
        return ds  # 1-batch validation set

    def train_step(self, inputs, model, optimizer):
        return {'total_loss': tf.constant(0.0)}

    def validation_step(self, inputs, model):
        return {'predictions': {}}

    def aggregate_logs(self, logs, step_out):
        return logs

    def reduce_aggregated_logs(self, logs, global_step=None):
        return {}


def _make_config():
    cfg = ExperimentConfig()
    cfg.trainer.steps_per_loop = _SPL
    cfg.trainer.checkpoint_interval = _SPL   # == steps_per_loop, like all tiers
    cfg.trainer.train_epochs = _EPOCHS
    cfg.trainer.max_to_keep = 5
    cfg.task.summary_types = 'scalar'        # no image summaries in the stub
    # The dataclass default leaves is_training=True on validation_data too; the
    # stub keys "infinite vs 1-batch" off this flag, so set it explicitly or
    # _run_validation would iterate an infinite stream and hang.
    cfg.task.train_data.is_training = True
    cfg.task.validation_data.is_training = False
    return cfg


def _make_trainer(out_dir, cfg):
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
    return YoloV8Trainer(
        task=_StubTask(cfg), config=cfg, output_dir=str(out_dir),
        strategy=strategy,
    )


class TestBoundaryCheckpointEpochAccounting:

    @pytest.fixture(scope='class')
    def trained_dir(self, tmp_path_factory):
        out = tmp_path_factory.mktemp('trainer_resume')
        cfg = _make_config()
        trainer = _make_trainer(out, cfg)
        trainer.train(_EPOCHS)
        assert int(trainer._global_step) == _SPL * _EPOCHS
        return out

    def test_boundary_checkpoint_persists_completed_epoch(self, trained_dir):
        """The checkpoint written at the final step of epoch k must carry
        completed_epochs == k (it used to carry k-1)."""
        latest = tf.train.latest_checkpoint(str(trained_dir))
        assert latest is not None
        step_var = tf.Variable(0, dtype=tf.int64)
        epoch_var = tf.Variable(0, dtype=tf.int64)
        ck = tf.train.Checkpoint(global_step=step_var, completed_epochs=epoch_var)
        ck.restore(latest).expect_partial()
        assert int(step_var) == _SPL * _EPOCHS
        assert int(epoch_var) == _EPOCHS, (
            f"boundary checkpoint persisted completed_epochs={int(epoch_var)} "
            f"for global_step={int(step_var)} — a resume would re-run a full epoch"
        )

    def test_relaunching_finished_run_is_a_noop(self, trained_dir):
        """Resuming a run that already finished its epochs must run 0 steps."""
        cfg = _make_config()
        trainer2 = _make_trainer(trained_dir, cfg)
        trainer2.train(_EPOCHS)
        assert int(trainer2._global_step) == _SPL * _EPOCHS, (
            "resume from a finished run trained extra steps "
            f"(global_step={int(trainer2._global_step)}, expected {_SPL * _EPOCHS})"
        )


class TestMidEpochCheckpointUnaffected:

    def test_mid_epoch_save_keeps_prior_epoch_count(self, tmp_path):
        """_sync_completed_epochs must be a no-op off the epoch boundary."""
        cfg = _make_config()
        trainer = _make_trainer(tmp_path, cfg)
        trainer._epoch_var.assign(1)
        trainer._sync_completed_epochs(python_step=_SPL + 2, steps_per_loop=_SPL)
        assert int(trainer._epoch_var) == 1
        trainer._sync_completed_epochs(python_step=2 * _SPL, steps_per_loop=_SPL)
        assert int(trainer._epoch_var) == 2
        # Data-driven mode (spl == 0) never syncs.
        trainer._epoch_var.assign(1)
        trainer._sync_completed_epochs(python_step=6, steps_per_loop=0)
        assert int(trainer._epoch_var) == 1
