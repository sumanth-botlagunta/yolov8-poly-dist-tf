"""Tests for gradient accumulation (trainer.grad_accum_steps / YoloV8Task)."""

import types

import tensorflow as tf

import train.task as T


class _FakeOpt:
    def __init__(self):
        self.iterations = tf.Variable(0, dtype=tf.int64)
        self.applied = []

    def apply_gradients(self, gv, clip_norm=None):
        self.applied.append([None if g is None else g.numpy().copy() for g, _ in gv])
        self.iterations.assign_add(1)


def _stub(n_vars=1):
    task = T.YoloV8Task.__new__(T.YoloV8Task)
    vs = [tf.Variable([0.0, 0.0]) for _ in range(n_vars)]
    task._grad_accumulators = [tf.Variable(tf.zeros_like(v), trainable=False) for v in vs]
    task._accum_counter = tf.Variable(0, dtype=tf.int64)
    return task, types.SimpleNamespace(trainable_variables=vs)


def test_prepare_none_when_disabled():
    task = T.YoloV8Task.__new__(T.YoloV8Task)
    task._config = types.SimpleNamespace(trainer=types.SimpleNamespace(grad_accum_steps=1))
    m = types.SimpleNamespace(trainable_variables=[tf.Variable([1.0])])
    task.prepare_grad_accumulation(m)
    assert task._grad_accumulators is None and task._accum_counter is None


def test_prepare_creates_accumulators():
    task = T.YoloV8Task.__new__(T.YoloV8Task)
    task._config = types.SimpleNamespace(trainer=types.SimpleNamespace(grad_accum_steps=4))
    m = types.SimpleNamespace(trainable_variables=[tf.Variable([1.0]), tf.Variable([2.0, 3.0])])
    task.prepare_grad_accumulation(m)
    assert len(task._grad_accumulators) == 2
    assert all(float(tf.reduce_sum(a)) == 0.0 for a in task._grad_accumulators)


def test_applies_mean_every_n_steps():
    task, model = _stub()
    opt = _FakeOpt()
    # N=2: first micro-batch accumulates only
    task._accumulate_and_maybe_apply([tf.constant([2.0, 2.0])], model, opt, 0.0, 2)
    assert len(opt.applied) == 0 and int(task._accum_counter) == 1
    assert list(task._grad_accumulators[0].numpy()) == [2.0, 2.0]
    # second micro-batch triggers apply of the MEAN ([2,2]+[4,4])/2 = [3,3], zeros accum
    task._accumulate_and_maybe_apply([tf.constant([4.0, 4.0])], model, opt, 0.0, 2)
    assert len(opt.applied) == 1 and list(opt.applied[0][0]) == [3.0, 3.0]
    assert list(task._grad_accumulators[0].numpy()) == [0.0, 0.0]
    assert int(opt.iterations) == 1     # one optimizer update per 2 micro-batches


def test_none_grad_preserved_through_apply():
    task, model = _stub(n_vars=2)   # two vars; second gets a None grad
    opt = _FakeOpt()
    for _ in range(2):              # N=2
        task._accumulate_and_maybe_apply(
            [tf.constant([1.0, 1.0]), None], model, opt, 0.0, 2)
    assert len(opt.applied) == 1
    g0, g1 = opt.applied[0]
    assert list(g0) == [1.0, 1.0] and g1 is None   # None grad stays None (skipped, like N=1)
