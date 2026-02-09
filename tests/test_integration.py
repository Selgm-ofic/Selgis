"""Integration tests: rollback, LR finder state restore, step-based scheduler."""

import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from selgis import Trainer, SelgisConfig, LRFinder
from selgis.scheduler import SmartScheduler


def test_rollback_after_nan_restores_weights(device):
    """
    When NaN loss triggers rollback, model weights are restored from last good state
    and training can continue.
    """
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    ).to(device)
    X = torch.randn(32, 4, device=device)
    y = (X.sum(dim=1) > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)

    config = SelgisConfig(
        max_epochs=2,
        lr_finder_enabled=False,
        nan_recovery=True,
        state_storage="memory",
        state_update_interval=1,  # save every step so last_good is end of epoch 1
        device="cpu",
    )
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        eval_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
    )
    trainer._train_epoch()
    state_after_epoch1 = {
        n: p.data.cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    # Next epoch: first batch returns NaN -> rollback to last good (end of epoch 1)
    original_forward = trainer._forward
    nan_returned = [False]

    def forward_nan_on_first_call(batch):
        if not nan_returned[0]:
            nan_returned[0] = True
            t = torch.tensor(float("nan"), device=trainer.device, requires_grad=True)
            return t, t
        return original_forward(batch)
    trainer._forward = forward_nan_on_first_call

    trainer._train_epoch()
    state_after_nan = {
        n: p.data.cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    for name in state_after_epoch1:
        assert name in state_after_nan
        assert torch.allclose(state_after_epoch1[name], state_after_nan[name]), (
            f"Rollback should restore {name}"
        )


def test_lr_finder_restores_model_state(device):
    """LRFinder restores model (and optimizer) state after find(); params match before/after."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    X = torch.randn(64, 4, device=device)
    y = (X.sum(dim=1) > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)

    state_before = copy.deepcopy(model.state_dict())
    opt_state_before = copy.deepcopy(opt.state_dict())

    finder = LRFinder(model, opt, nn.CrossEntropyLoss(), device, trainable_only=False)
    lr = finder.find(loader, start_lr=1e-6, end_lr=0.1, num_steps=20)

    state_after = model.state_dict()
    opt_state_after = opt.state_dict()

    assert isinstance(lr, float) and lr > 0
    for key in state_before:
        assert torch.allclose(state_before[key].cpu(), state_after[key].cpu()), (
            f"LR finder should restore param {key}"
        )
    assert list(opt_state_before.keys()) == list(opt_state_after.keys())


def test_scheduler_step_called_when_warmup_ratio_set(device):
    """When warmup_ratio > 0, scheduler.step() is called each optimizer step (step-based)."""
    model = nn.Linear(4, 2).to(device)
    X = torch.randn(24, 4, device=device)
    y = torch.zeros(24, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True)

    config = SelgisConfig(
        max_epochs=1,
        lr_finder_enabled=False,
        warmup_ratio=0.1,
        warmup_epochs=0,
        device="cpu",
    )
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
    )
    assert hasattr(trainer.scheduler, "step")
    initial_step = trainer.scheduler._step
    trainer._train_epoch()
    # Should have stepped at least once per batch (6 batches -> 6 steps)
    assert trainer.scheduler._step > initial_step


def test_scheduler_state_dict_roundtrip(device, base_config):
    """SmartScheduler state_dict/load_state_dict roundtrip preserves _step and _epoch."""
    model = nn.Linear(2, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = SelgisConfig(warmup_ratio=0.1, max_epochs=10)
    sched = SmartScheduler(opt, 1e-3, config, num_training_steps=100)
    sched.step()
    sched.step()
    state = sched.state_dict()
    assert state["_step"] == 2
    sched2 = SmartScheduler(opt, 1e-3, config, num_training_steps=100)
    sched2.load_state_dict(state)
    assert sched2._step == 2 and sched2._epoch == sched._epoch
