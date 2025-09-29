"""
Hierarchical aggregation module for HQDE framework.

This module implements tree-structured aggregation with adaptive topology
optimization and communication-efficient ensemble weight combination.
"""

import torch
import ray
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math
import time
import logging
from collections import defaultdict


@ray.remote
class AggregationNode:
    """Individual node in the hierarchical aggregation tree."""

    def __init__(self, node_id: str, level: int, max_children: int = 4):
        self.node_id = node_id
        self.level = level
        self.max_children = max_children
        self.children = []
        self.parent = None
        self.local_weights = {}
        self.aggregated_weights = {}
        self.performance_metrics = {
            'processing_time': 0.0,
            'communication_time': 0.0,
            'data_size_processed': 0
        }

    def add_child(self, child_node_id: str):
        """Add a child node to this aggregation node."""
        if len(self.children) < self.max_children:
            self.children.append(child_node_id)
            return True
        return False

    def set_parent(self, parent_node_id: str):
        """Set the parent node for this aggregation node."""
        self.parent = parent_node_id

    def receive_weights(self, source_id: str, weights: Dict[str, torch.Tensor], metadata: Dict[str, Any] = None):
        """Receive weights from a child node or ensemble member."""
        self.local_weights[source_id] = {
            'weights': weights,
            'metadata': metadata or {},
            'timestamp': time.time()
        }

    def aggregate_local_weights(self, aggregation_method: str = "weighted_mean") -> Dict[str, torch.Tensor]:
        """Aggregate weights from all sources at this node."""
        if not self.local_weights:
            return {}

        start_time = time.time()

        # Collect all weights by parameter name
        param_weights = defaultdict(list)
        param_metadata = defaultdict(list)

        for source_id, data in self.local_weights.items():
            weights = data['weights']
            metadata = data['metadata']

            for param_name, weight_tensor in weights.items():
                param_weights[param_name].append(weight_tensor)
                param_metadata[param_name].append(metadata.get('importance_score', 1.0))

        # Aggregate each parameter
        aggregated = {}
        for param_name, weight_list in param_weights.items():
            if aggregation_method == "weighted_mean":
                importance_scores = param_metadata[param_name]
                weights_tensor = torch.tensor(importance_scores, dtype=torch.float32)
                weights_normalized = torch.softmax(weights_tensor, dim=0)

                aggregated_param = torch.zeros_like(weight_list[0])
                for weight, norm_weight in zip(weight_list, weights_normalized):
                    aggregated_param += norm_weight * weight

                aggregated[param_name] = aggregated_param

            elif aggregation_method == "median":
                stacked_weights = torch.stack(weight_list)
                aggregated[param_name] = torch.median(stacked_weights, dim=0)[0]

            else:  # default to mean
                stacked_weights = torch.stack(weight_list)
                aggregated[param_name] = torch.mean(stacked_weights, dim=0)

        self.aggregated_weights = aggregated
        self.performance_metrics['processing_time'] = time.time() - start_time

        return aggregated

    def get_aggregated_weights(self) -> Dict[str, torch.Tensor]:
        """Get the aggregated weights from this node."""
        return self.aggregated_weights.copy()

    def clear_local_weights(self):
        """Clear local weights to free memory."""
        self.local_weights.clear()

    def get_node_info(self) -> Dict[str, Any]:
        """Get information about this node."""
        return {
            'node_id': self.node_id,
            'level': self.level,
            'num_children': len(self.children),
            'children': self.children,
            'parent': self.parent,
            'num_local_weights': len(self.local_weights),
            'performance_metrics': self.performance_metrics
        }


class HierarchicalAggregator:
    """Hierarchical aggregation system for distributed ensemble learning."""

    def __init__(self,
                 num_ensemble_members: int,
                 tree_branching_factor: int = 4,
                 adaptive_topology: bool = True):
        """
        Initialize hierarchical aggregator.

        Args:
            num_ensemble_members: Number of ensemble members
            tree_branching_factor: Maximum children per node in aggregation tree
            adaptive_topology: Whether to use adaptive topology optimization
        """
        self.num_ensemble_members = num_ensemble_members
        self.tree_branching_factor = tree_branching_factor
        self.adaptive_topology = adaptive_topology

        self.nodes = {}
        self.tree_structure = {}
        self.root_node_id = None

        # Performance monitoring
        self.aggregation_metrics = {
            'total_aggregation_time': 0.0,
            'communication_overhead': 0.0,
            'tree_depth': 0,
            'total_nodes': 0
        }

        self._build_aggregation_tree()

    def _build_aggregation_tree(self):
        """Build the hierarchical aggregation tree."""
        if self.num_ensemble_members <= 0:
            return

        # Calculate tree structure
        num_leaves = self.num_ensemble_members
        tree_levels = []
        current_level_size = num_leaves

        # Build tree bottom-up
        level = 0
        while current_level_size > 1:
            tree_levels.append(current_level_size)
            current_level_size = math.ceil(current_level_size / self.tree_branching_factor)
            level += 1

        tree_levels.append(1)  # Root node
        tree_levels.reverse()  # Reverse to get top-down structure

        self.aggregation_metrics['tree_depth'] = len(tree_levels) - 1
        self.aggregation_metrics['total_nodes'] = sum(tree_levels)

        # Create nodes for each level
        node_counter = 0
        level_nodes = {}

        for level_idx, num_nodes in enumerate(tree_levels):
            level_nodes[level_idx] = []

            for node_idx in range(num_nodes):
                node_id = f"agg_node_{level_idx}_{node_idx}"
                node = AggregationNode.remote(node_id, level_idx, self.tree_branching_factor)
                self.nodes[node_id] = node
                level_nodes[level_idx].append(node_id)
                node_counter += 1

        # Set up parent-child relationships
        for level_idx in range(len(tree_levels) - 1):
            parent_level = level_idx
            child_level = level_idx + 1

            parent_nodes = level_nodes[parent_level]
            child_nodes = level_nodes[child_level]

            for child_idx, child_node_id in enumerate(child_nodes):
                parent_idx = child_idx // self.tree_branching_factor
                if parent_idx < len(parent_nodes):
                    parent_node_id = parent_nodes[parent_idx]

                    # Set parent-child relationship
                    ray.get(self.nodes[parent_node_id].add_child.remote(child_node_id))
                    ray.get(self.nodes[child_node_id].set_parent.remote(parent_node_id))

        # Set root node
        if tree_levels:
            self.root_node_id = level_nodes[0][0]

        self.tree_structure = level_nodes

    def aggregate_ensemble_weights(self,
                                 ensemble_weights: List[Dict[str, torch.Tensor]],
                                 ensemble_metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform hierarchical aggregation of ensemble weights.

        Args:
            ensemble_weights: List of weight dictionaries from ensemble members
            ensemble_metadata: Optional metadata for each ensemble member

        Returns:
            Hierarchically aggregated weights
        """
        if len(ensemble_weights) != self.num_ensemble_members:
            raise ValueError(f"Expected {self.num_ensemble_members} ensemble members, got {len(ensemble_weights)}")

        start_time = time.time()

        # Distribute weights to leaf nodes
        self._distribute_weights_to_leaves(ensemble_weights, ensemble_metadata)

        # Perform bottom-up aggregation
        aggregated_weights = self._perform_bottom_up_aggregation()

        # Update performance metrics
        self.aggregation_metrics['total_aggregation_time'] = time.time() - start_time

        return aggregated_weights

    def _distribute_weights_to_leaves(self,
                                    ensemble_weights: List[Dict[str, torch.Tensor]],
                                    ensemble_metadata: Optional[List[Dict[str, Any]]]):
        """Distribute ensemble weights to leaf nodes."""
        if not ensemble_metadata:
            ensemble_metadata = [{}] * len(ensemble_weights)

        # Get leaf nodes (highest level in tree_structure)
        max_level = max(self.tree_structure.keys())
        leaf_nodes = self.tree_structure[max_level]

        # Distribute weights to leaf nodes
        distribution_futures = []
        for i, (weights, metadata) in enumerate(zip(ensemble_weights, ensemble_metadata)):
            leaf_node_idx = i % len(leaf_nodes)
            leaf_node_id = leaf_nodes[leaf_node_idx]
            leaf_node = self.nodes[leaf_node_id]

            source_id = f"ensemble_member_{i}"
            future = leaf_node.receive_weights.remote(source_id, weights, metadata)
            distribution_futures.append(future)

        # Wait for all distributions to complete
        ray.get(distribution_futures)

    def _perform_bottom_up_aggregation(self) -> Dict[str, torch.Tensor]:
        """Perform bottom-up aggregation through the tree."""
        # Process levels from bottom to top
        for level in sorted(self.tree_structure.keys(), reverse=True):
            level_nodes = self.tree_structure[level]

            # Aggregate at each node in this level
            aggregation_futures = []
            for node_id in level_nodes:
                node = self.nodes[node_id]
                future = node.aggregate_local_weights.remote("weighted_mean")
                aggregation_futures.append((node_id, future))

            # Wait for aggregations to complete
            level_results = {}
            for node_id, future in aggregation_futures:
                aggregated_weights = ray.get(future)
                level_results[node_id] = aggregated_weights

            # If not at root level, send results to parent nodes
            if level > 0:
                parent_transmission_futures = []
                for node_id in level_nodes:
                    if node_id in level_results:
                        node = self.nodes[node_id]
                        parent_info = ray.get(node.get_node_info.remote())
                        parent_id = parent_info['parent']

                        if parent_id and parent_id in self.nodes:
                            parent_node = self.nodes[parent_id]
                            weights = level_results[node_id]
                            metadata = {'source_level': level, 'source_node': node_id}

                            future = parent_node.receive_weights.remote(node_id, weights, metadata)
                            parent_transmission_futures.append(future)

                # Wait for all transmissions to complete
                ray.get(parent_transmission_futures)

            # Clear local weights to free memory (except for root)
            if level > 0:
                clear_futures = []
                for node_id in level_nodes:
                    node = self.nodes[node_id]
                    future = node.clear_local_weights.remote()
                    clear_futures.append(future)
                ray.get(clear_futures)

        # Get final result from root node
        if self.root_node_id:
            root_node = self.nodes[self.root_node_id]
            final_weights = ray.get(root_node.get_aggregated_weights.remote())
            return final_weights
        else:
            return {}

    def optimize_tree_topology(self,
                              node_performance_data: Dict[str, Dict[str, float]],
                              network_topology: Optional[Dict[str, Any]] = None):
        """
        Optimize tree topology based on node performance and network characteristics.

        Args:
            node_performance_data: Performance metrics for each node
            network_topology: Network topology information
        """
        if not self.adaptive_topology:
            return

        # Analyze current performance
        bottleneck_nodes = []
        for node_id, performance in node_performance_data.items():
            processing_time = performance.get('processing_time', 0)
            communication_time = performance.get('communication_time', 0)

            # Identify bottlenecks
            if processing_time > 1.0 or communication_time > 0.5:  # Thresholds
                bottleneck_nodes.append(node_id)

        # Rebalance tree if bottlenecks detected
        if bottleneck_nodes:
            self._rebalance_tree(bottleneck_nodes, node_performance_data)

    def _rebalance_tree(self,
                       bottleneck_nodes: List[str],
                       performance_data: Dict[str, Dict[str, float]]):
        """Rebalance tree structure to address bottlenecks."""
        # This is a simplified rebalancing strategy
        # In practice, this would involve more sophisticated optimization

        for bottleneck_node_id in bottleneck_nodes:
            if bottleneck_node_id in self.nodes:
                node_info = ray.get(self.nodes[bottleneck_node_id].get_node_info.remote())

                # If node has too many children, redistribute them
                if len(node_info['children']) > self.tree_branching_factor:
                    # Create additional intermediate nodes
                    self._split_overloaded_node(bottleneck_node_id, node_info)

    def _split_overloaded_node(self, node_id: str, node_info: Dict[str, Any]):
        """Split an overloaded node by creating intermediate nodes."""
        # This would involve creating new intermediate nodes and
        # redistributing children - simplified implementation
        logging.info(f"Would split overloaded node {node_id} with {len(node_info['children'])} children")

    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hierarchical aggregation process."""
        # Collect node statistics
        node_stat_futures = []
        for node_id, node in self.nodes.items():
            future = node.get_node_info.remote()
            node_stat_futures.append((node_id, future))

        node_statistics = {}
        for node_id, future in node_stat_futures:
            node_info = ray.get(future)
            node_statistics[node_id] = node_info

        return {
            'aggregation_metrics': self.aggregation_metrics,
            'tree_structure': self.tree_structure,
            'node_statistics': node_statistics,
            'num_ensemble_members': self.num_ensemble_members,
            'tree_branching_factor': self.tree_branching_factor
        }

    def cleanup(self):
        """Cleanup hierarchical aggregation resources."""
        # Ray will automatically clean up remote actors
        self.nodes.clear()
        self.tree_structure.clear()