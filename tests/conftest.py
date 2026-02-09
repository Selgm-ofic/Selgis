"""Pytest fixtures for SELGIS tests."""

import pytest
import torch
import torch.nn as nn

from selgis.config import SelgisConfig, TransformerConfig


@pytest.fixture
def device():
    """Use CPU for fast tests."""
    return torch.device("cpu")


@pytest.fixture
def small_model(device):
    """Small MLP for testing."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    return model.to(device)


@pytest.fixture
def base_config():
    """Minimal SelgisConfig for tests."""
    return SelgisConfig(
        max_epochs=2,
        batch_size=4,
        lr_finder_enabled=False,
        nan_recovery=True,
        warmup_epochs=0,
        patience=2,
    )


@pytest.fixture
def transformer_config():
    """Minimal TransformerConfig for tests."""
    return TransformerConfig(
        model_name_or_path="",
        max_epochs=1,
        lr_finder_enabled=False,
    )
