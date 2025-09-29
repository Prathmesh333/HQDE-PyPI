"""
Quantum-inspired components for HQDE framework.

This module provides quantum-inspired algorithms for ensemble learning,
including quantum noise injection, entanglement simulation, and quantum
annealing approaches for ensemble optimization.
"""

from .quantum_aggregator import QuantumEnsembleAggregator
from .quantum_noise import QuantumNoiseGenerator
from .quantum_optimization import QuantumEnsembleOptimizer

__all__ = [
    'QuantumEnsembleAggregator',
    'QuantumNoiseGenerator',
    'QuantumEnsembleOptimizer'
]