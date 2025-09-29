"""
HQDE: Hierarchical Quantum-Distributed Ensemble Learning Framework

A comprehensive framework for distributed ensemble learning with quantum-inspired
algorithms, adaptive quantization, and efficient hierarchical aggregation.
"""

__version__ = "0.1.1"
__author__ = "HQDE Team"

# Core components
from .core.hqde_system import HQDESystem, create_hqde_system
from .core.hqde_system import (
    AdaptiveQuantizer,
    QuantumInspiredAggregator,
    DistributedEnsembleManager
)

# Quantum-inspired components
from .quantum import (
    QuantumEnsembleAggregator,
    QuantumNoiseGenerator,
    QuantumEnsembleOptimizer
)

# Distributed components
from .distributed import (
    MapReduceEnsembleManager,
    HierarchicalAggregator,
    ByzantineFaultTolerantAggregator,
    DynamicLoadBalancer
)

# Utilities
from .utils import (
    PerformanceMonitor,
    SystemMetrics
)

__all__ = [
    # Core
    'HQDESystem',
    'create_hqde_system',
    'AdaptiveQuantizer',
    'QuantumInspiredAggregator',
    'DistributedEnsembleManager',

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
    'SystemMetrics'
]