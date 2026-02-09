"""Tests for selgis.trainer (minimal integration)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from selgis import Trainer, SelgisConfig


def test_trainer_run_one_epoch(device, base_config):
    """Trainer runs one epoch without error."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    ).to(device)
    X = torch.randn(32, 4, device=device)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    config = SelgisConfig(
        max_epochs=1,
        lr_finder_enabled=False,
        gradient_accumulation_steps=1,
        device="cpu",
    )
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
    )
    metrics = trainer.train()
    assert "train_loss" in metrics
    assert isinstance(metrics["train_loss"], float)


def test_trainer_gradient_accumulation(device, base_config):
    """Trainer respects gradient_accumulation_steps."""
    model = nn.Linear(2, 2).to(device)
    X = torch.randn(16, 2, device=device)
    y = torch.zeros(16, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True)
    config = SelgisConfig(
        max_epochs=1,
        lr_finder_enabled=False,
        gradient_accumulation_steps=2,
        device="cpu",
    )
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
    )
    metrics = trainer.train()
    assert "train_loss" in metrics


def test_load_model_weights_only(device, base_config, tmp_path):
    """load_model with weights_only=True loads state (safe path)."""
    model = nn.Linear(2, 2).to(device)
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    config = SelgisConfig(max_epochs=1, lr_finder_enabled=False, device="cpu")
    loader = DataLoader(
        TensorDataset(torch.randn(8, 2), torch.zeros(8, dtype=torch.long)),
        batch_size=4,
    )
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
    )
    trainer.load_model(str(path), weights_only=True)
