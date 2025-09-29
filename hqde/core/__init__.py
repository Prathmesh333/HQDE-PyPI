"""
Core HQDE system components.

This module contains the main HQDE system implementation including
adaptive quantization, quantum-inspired aggregation, and distributed
ensemble management.
"""

from .hqde_system import (
    HQDESystem,
    AdaptiveQuantizer,
    QuantumInspiredAggregator,
    DistributedEnsembleManager,
    create_hqde_system
)

__all__ = [
    'HQDESystem',
    'AdaptiveQuantizer',
    'QuantumInspiredAggregator',
    'DistributedEnsembleManager',
    'create_hqde_system'
]