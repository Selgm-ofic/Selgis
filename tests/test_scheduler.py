"""Tests for selgis.scheduler."""

import pytest
import torch
import torch.nn as nn

from selgis.config import SelgisConfig
from selgis.scheduler import DEFAULT_NUM_TRAINING_STEPS, SmartScheduler


@pytest.fixture
def scheduler_setup():
    """Optimizer and config for scheduler tests."""
    model = nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = SelgisConfig(
        warmup_epochs=2,
        max_epochs=10,
        min_lr=1e-7,
        scheduler_type="cosine",
    )
    return opt, config


def test_scheduler_step_epoch_warmup(scheduler_setup):
    """LR increases during warmup."""
    opt, config = scheduler_setup
    sched = SmartScheduler(opt, 1e-3, config)
    lr0 = sched.step_epoch(0)
    lr1 = sched.step_epoch(1)
    assert lr0 < lr1
    assert lr1 <= 1e-3


def test_scheduler_step_epoch_after_warmup(scheduler_setup):
    """LR decreases after warmup for cosine."""
    opt, config = scheduler_setup
    sched = SmartScheduler(opt, 1e-3, config)
    sched.step_epoch(2)
    lr_mid = sched.step_epoch(5)
    lr_end = sched.step_epoch(9)
    assert lr_mid >= lr_end
    assert lr_end >= config.min_lr


def test_scheduler_reduce_lr(scheduler_setup):
    """reduce_lr multiplies current LR by factor."""
    opt, config = scheduler_setup
    sched = SmartScheduler(opt, 1e-3, config)
    sched.step_epoch(0)
    current = sched.get_lr()
    new_lr = sched.reduce_lr(factor=0.5)
    assert new_lr == current * 0.5


def test_scheduler_surge_lr(scheduler_setup):
    """surge_lr increases LR up to initial."""
    opt, config = scheduler_setup
    sched = SmartScheduler(opt, 1e-3, config)
    sched.step_epoch(5)
    new_lr = sched.surge_lr(factor=3.0)
    assert new_lr >= sched.get_lr() or new_lr == 1e-3


def test_scheduler_default_num_training_steps_constant():
    """DEFAULT_NUM_TRAINING_STEPS is defined."""
    assert DEFAULT_NUM_TRAINING_STEPS == 10_000
