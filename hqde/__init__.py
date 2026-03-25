"""
HQDE: Hierarchical Quantum-Distributed Ensemble Learning Framework

A comprehensive framework for distributed ensemble learning with quantum-inspired
algorithms, adaptive quantization, and efficient hierarchical aggregation.
"""

__version__ = "0.1.10"
__author__ = "HQDE Team"

# Core components
from .core.hqde_system import HQDESystem, create_hqde_system
from .core.hqde_system import (
    AdaptiveQuantizer,
    QuantumInspiredAggregator,
    DistributedEnsembleManager
)
from .models import SmallImageResNet18

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

__all__ = [
    # Core
    'HQDESystem',
    'create_hqde_system',
    'AdaptiveQuantizer',
    'QuantumInspiredAggregator',
    'DistributedEnsembleManager',
    'SmallImageResNet18',

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
]
