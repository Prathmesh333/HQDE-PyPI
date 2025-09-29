"""
Distributed computing components for HQDE framework.

This module provides distributed ensemble management, hierarchical aggregation,
and MapReduce-inspired weight management for scalable ensemble learning.
"""

from .mapreduce_ensemble import MapReduceEnsembleManager
from .hierarchical_aggregator import HierarchicalAggregator
from .fault_tolerance import ByzantineFaultTolerantAggregator
from .load_balancer import DynamicLoadBalancer

__all__ = [
    'MapReduceEnsembleManager',
    'HierarchicalAggregator',
    'ByzantineFaultTolerantAggregator',
    'DynamicLoadBalancer'
]