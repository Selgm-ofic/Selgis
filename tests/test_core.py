"""Tests for selgis.core."""

import pytest
import torch
import torch.nn as nn

from selgis.config import SelgisConfig
from selgis.core import SelgisConfig
from selgis.scheduler import SmartScheduler


@pytest.fixture
def core_components(device, small_model, base_config):
    """Model, optimizer, scheduler, config for SelgisConfig."""
    base_config.state_storage = "memory"
    opt = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
    sched = SmartScheduler(opt, 1e-3, base_config)
    return small_model, opt, sched, base_config


def test_core_check_loss_accepts_finite(core_components, device):
    """check_loss returns True for finite loss."""
    model, opt, sched, config = core_components
    core = SelgisConfig(model, opt, sched, config, device)
    loss = torch.tensor(0.5, device=device)
    assert core.check_loss(loss) is True


def test_core_check_loss_nan_rollback(core_components, device):
    """check_loss triggers rollback on NaN when nan_recovery True."""
    model, opt, sched, config = core_components
    config.nan_recovery = True
    core = SelgisConfig(model, opt, sched, config, device)
    loss = torch.tensor(float("nan"), device=device)
    assert core.check_loss(loss) is False


def test_core_check_loss_inf_rollback(core_components, device):
    """check_loss triggers rollback on Inf when nan_recovery True."""
    model, opt, sched, config = core_components
    config.nan_recovery = True
    core = SelgisConfig(model, opt, sched, config, device)
    loss = torch.tensor(float("inf"), device=device)
    assert core.check_loss(loss) is False


def test_core_check_loss_nan_recovery_disabled(core_components, device):
    """When nan_recovery False, check_loss does not rollback on NaN."""
    model, opt, sched, config = core_components
    config.nan_recovery = False
    core = SelgisConfig(model, opt, sched, config, device)
    loss = torch.tensor(float("nan"), device=device)
    assert core.check_loss(loss) is True


def test_core_load_trainable_state_warns_missing_in_model(core_components, device):
    """Loading state with extra keys warns."""
    model, opt, sched, config = core_components
    core = SelgisConfig(model, opt, sched, config, device)
    with pytest.warns(UserWarning, match="state has keys not in model"):
        core._load_trainable_state({"nonexistent.layer.weight": torch.zeros(1)})


def test_core_best_metric_loss_properties(core_components, device):
    """best_metric and best_loss are readable."""
    model, opt, sched, config = core_components
    core = SelgisConfig(model, opt, sched, config, device)
    assert core.best_metric == float("-inf")
    assert core.best_loss == float("inf")
    assert core.trainable_params_count > 0
    assert core.total_params_count >= core.trainable_params_count
