"""
HQDE: Hierarchical Quantum-Distributed Ensemble Learning Framework

A comprehensive framework for distributed ensemble learning with quantum-inspired
algorithms, adaptive quantization, and efficient hierarchical aggregation.
"""

__version__ = "0.1.11"
__author__ = "HQDE Team"

# Core components
from .core.hqde_system import HQDESystem, create_hqde_system
from .core.hqde_system import (
    AdaptiveQuantizer,
    QuantumInspiredAggregator,
    DistributedEnsembleManager
)
from .models import SmallImageResNet18

# Transformer models (optional)
try:
    from .models import (
        TransformerTextClassifier,
        LightweightTransformerClassifier,
        CBTTransformerClassifier,
        SmallTransformerClassifier,
        StandardTransformerClassifier
    )
except ImportError:
    TransformerTextClassifier = None
    LightweightTransformerClassifier = None
    CBTTransformerClassifier = None
    SmallTransformerClassifier = None
    StandardTransformerClassifier = None

# Quantum-inspired components
from .quantum import (
    QuantumEnsembleAggregator,
    QuantumNoiseGenerator,
    QuantumEnsembleOptimizer
)

# Distributed components are optional because they require Ray.
try:
    from .distributed import (
        MapReduceEnsembleManager,
        HierarchicalAggregator,
        ByzantineFaultTolerantAggregator,
        DynamicLoadBalancer
    )
except ImportError:
    MapReduceEnsembleManager = None
    HierarchicalAggregator = None
    ByzantineFaultTolerantAggregator = None
    DynamicLoadBalancer = None

# Utilities
from .utils import (
    PerformanceMonitor,
    SystemMetrics,
    DataLoader,
    DataLoaderConfig,
    DataPreprocessor,
    make_cifar_training_config,
)

# Text/Transformer utilities (optional)
try:
    from .utils import (
        SimpleTokenizer,
        TextClassificationDataset,
        CBTDataset,
        TextDataLoader,
        TextDataConfig,
        create_cbt_sample_data,
        preprocess_cbt_text,
        make_transformer_training_config,
        make_cbt_training_config,
        make_lightweight_transformer_config,
        make_large_transformer_config
    )
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

__all__ = [
    # Core
    'HQDESystem',
    'create_hqde_system',
    'AdaptiveQuantizer',
    'QuantumInspiredAggregator',
    'DistributedEnsembleManager',
    'SmallImageResNet18',

    # Transformer models
    'TransformerTextClassifier',
    'LightweightTransformerClassifier',
    'CBTTransformerClassifier',
    'SmallTransformerClassifier',
    'StandardTransformerClassifier',

    # Quantum
    'QuantumEnsembleAggregator',
    'QuantumNoiseGenerator',
    'QuantumEnsembleOptimizer',

    # Distributed
    'MapReduceEnsembleManager',
    'HierarchicalAggregator',
    'ByzantineFaultTolerantAggregator',
    'DynamicLoadBalancer',

    # Utils
    'PerformanceMonitor',
    'SystemMetrics',
    'DataLoader',
    'DataLoaderConfig',
    'DataPreprocessor',
    'make_cifar_training_config',
    
    # Text/Transformer utils
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
