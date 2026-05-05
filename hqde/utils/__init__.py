"""
Utility modules for HQDE framework.

This module provides various utility functions for performance monitoring,
data preprocessing, visualization, and system configuration.
"""

from .performance_monitor import PerformanceMonitor, SystemMetrics
from .data_utils import DataLoader, DataLoaderConfig, DataPreprocessor
from .visualization import HQDEVisualizer
from .config_manager import ConfigManager
from .training_presets import make_cifar_training_config

# Import text/transformer utilities
try:
    from .text_data_utils import (
        SimpleTokenizer,
        TextClassificationDataset,
        CBTDataset,
        TextDataLoader,
        TextDataConfig,
        create_cbt_sample_data,
        preprocess_cbt_text
    )
    from .transformer_presets import (
        make_transformer_training_config,
        make_cbt_training_config,
        make_lightweight_transformer_config,
        make_large_transformer_config
    )
    TEXT_UTILS_AVAILABLE = True
except ImportError:
    SimpleTokenizer = None
    TextClassificationDataset = None
    CBTDataset = None
    TextDataLoader = None
    TextDataConfig = None
    create_cbt_sample_data = None
    preprocess_cbt_text = None
    make_transformer_training_config = None
    make_cbt_training_config = None
    make_lightweight_transformer_config = None
    make_large_transformer_config = None
    TEXT_UTILS_AVAILABLE = False

__all__ = [
    'PerformanceMonitor',
    'SystemMetrics',
    'DataLoader',
    'DataLoaderConfig',
    'DataPreprocessor',
    'HQDEVisualizer',
    'ConfigManager',
    'make_cifar_training_config',
    # Text/Transformer utilities
    'SimpleTokenizer',
    'TextClassificationDataset',
    'CBTDataset',
    'TextDataLoader',
    'TextDataConfig',
    'create_cbt_sample_data',
    'preprocess_cbt_text',
    'make_transformer_training_config',
    'make_cbt_training_config',
    'make_lightweight_transformer_config',
    'make_large_transformer_config',
]
