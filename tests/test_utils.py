"""Tests for selgis.utils."""

import pytest
import torch
import torch.nn as nn

from selgis.utils import (
    count_parameters,
    format_params,
    get_optimizer_grouped_params,
    is_dict_like,
    move_to_device,
    to_dict,
    unpack_batch,
)


def test_count_parameters_trainable():
    """count_parameters with trainable_only=True counts only trainable."""
    model = nn.Linear(10, 5)
    total = 10 * 5 + 5
    assert count_parameters(model, trainable_only=True) == total
    assert count_parameters(model, trainable_only=False) == total
    for p in model.parameters():
        p.requires_grad = False
    assert count_parameters(model, trainable_only=True) == 0
    assert count_parameters(model, trainable_only=False) == total


def test_format_params():
    """format_params returns human-readable strings."""
    assert "M" in format_params(1_200_000)
    assert "K" in format_params(1000)
    assert format_params(42) == "42"


def test_is_dict_like():
    """is_dict_like recognizes dict and Mapping."""
    assert is_dict_like({}) is True
    assert is_dict_like({"a": 1}) is True
    assert is_dict_like([]) is False
    assert is_dict_like(1) is False


def test_to_dict():
    """to_dict converts dict-like to dict."""
    assert to_dict({"a": 1}) == {"a": 1}
    d = to_dict({"x": 2})
    assert isinstance(d, dict) and d["x"] == 2


def test_move_to_device_cpu():
    """move_to_device moves tensors to device."""
    t = torch.randn(2, 2)
    out = move_to_device(t, torch.device("cpu"))
    assert out.device.type == "cpu"


def test_move_to_device_dict():
    """move_to_device recurses into dict."""
    batch = {"a": torch.randn(1), "b": torch.randn(2)}
    out = move_to_device(batch, torch.device("cpu"))
    assert out["a"].device.type == "cpu"
    assert out["b"].device.type == "cpu"


def test_unpack_batch_dict_labels():
    """unpack_batch returns (batch, labels) for dict with labels."""
    batch = {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor(0)}
    inputs, labels = unpack_batch(batch)
    assert labels is not None
    assert labels.item() == 0


def test_unpack_batch_tuple():
    """unpack_batch returns (x, y) for (x, y) tuple."""
    x, y = torch.randn(2, 2), torch.tensor([0, 1])
    inputs, labels = unpack_batch((x, y))
    assert inputs is x
    assert labels is y


def test_get_optimizer_grouped_params():
    """get_optimizer_grouped_params returns decay and no_decay groups."""
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    groups = get_optimizer_grouped_params(model, weight_decay=0.01)
    assert len(groups) == 2
    decay = [g for g in groups if g["weight_decay"] > 0]
    no_decay = [g for g in groups if g["weight_decay"] == 0]
    assert len(decay) == 1 and len(no_decay) == 1
