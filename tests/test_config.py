"""Tests for selgis.config."""

import pytest

from selgis.config import SelgisConfig, TransformerConfig


def test_selgis_config_defaults():
    """SelgisConfig has expected defaults."""
    cfg = SelgisConfig()
    assert cfg.batch_size == 32
    assert cfg.max_epochs == 100
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.patience == 5
    assert cfg.nan_recovery is True
    assert cfg.lr_finder_enabled is True
    assert cfg.state_storage == "disk"


def test_selgis_config_fp16_bf16_mutually_exclusive():
    """fp16 and bf16 cannot both be True."""
    with pytest.raises(ValueError, match="fp16 and bf16"):
        SelgisConfig(fp16=True, bf16=True)


def test_selgis_config_warmup_single_mode():
    """Only one of warmup_epochs or warmup_ratio can be non-zero."""
    with pytest.raises(ValueError, match="warmup_epochs or warmup_ratio"):
        SelgisConfig(warmup_epochs=2, warmup_ratio=0.1)


def test_transformer_config_inherits():
    """TransformerConfig extends SelgisConfig."""
    cfg = TransformerConfig(learning_rate=1e-4)
    assert cfg.learning_rate == 1e-4
    assert cfg.batch_size == 32
    assert cfg.problem_type == "single_label_classification"


def test_transformer_config_post_init():
    """TransformerConfig runs parent __post_init__."""
    with pytest.raises(ValueError, match="fp16 and bf16"):
        TransformerConfig(fp16=True, bf16=True)


def test_lr_finder_trainable_only_option():
    """lr_finder_trainable_only is available in config."""
    cfg = SelgisConfig(lr_finder_trainable_only=True)
    assert cfg.lr_finder_trainable_only is True
