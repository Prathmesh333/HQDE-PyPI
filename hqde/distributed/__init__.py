"""
Distributed computing components for HQDE framework.

This module provides distributed ensemble management, hierarchical aggregation,
and MapReduce-inspired weight management for scalable ensemble learning.
"""

try:
    from .mapreduce_ensemble import MapReduceEnsembleManager
    from .hierarchical_aggregator import HierarchicalAggregator
    from .fault_tolerance import ByzantineFaultTolerantAggregator
    from .load_balancer import DynamicLoadBalancer
except ImportError:
    MapReduceEnsembleManager = None
    HierarchicalAggregator = None
    ByzantineFaultTolerantAggregator = None
    DynamicLoadBalancer = None

__all__ = [
    'MapReduceEnsembleManager',
    'HierarchicalAggregator',
    'ByzantineFaultTolerantAggregator',
    'DynamicLoadBalancer'
]
