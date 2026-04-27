"""
selgis/datasets — Universal data loader factory.

Supports:
- Text (JSONL, TXT, CSV, HuggingFace datasets)
- Images (folders, CSV, WebDataset)
- Multimodal data (text + images)
- Custom user datasets
- Streaming for large data (>100GB)

Architecture:
- BaseDataset — unified interface for all datasets
- Specific implementations for each data type
- Factory for creation via configuration
"""

from selgis.datasets.base import BaseDataset, StreamingDataset
from selgis.datasets.config import DatasetConfig
from selgis.datasets.custom import CustomDataset
from selgis.datasets.factory import create_dataloaders, create_dataset, prepare_data_for_trainer
from selgis.datasets.image import ImageDataset
from selgis.datasets.multimodal import MultimodalDataset
from selgis.datasets.streaming import StreamingTextDataset
from selgis.datasets.text import HFTextDataset, TextDataset

__all__ = [
    # Base classes
    "BaseDataset",
    "StreamingDataset",

    # Configuration
    "DatasetConfig",

    # Text datasets
    "TextDataset",
    "HFTextDataset",

    # Image datasets
    "ImageDataset",

    # Multimodal datasets
    "MultimodalDataset",

    # Streaming datasets
    "StreamingTextDataset",

    # Custom datasets
    "CustomDataset",

    # Factory
    "create_dataset",
    "create_dataloaders",
    "prepare_data_for_trainer",
]
