"""
Fault tolerance module for HQDE framework.

This module implements Byzantine fault tolerance, checkpointing, and recovery
mechanisms for robust distributed ensemble learning.
"""

import torch
import ray
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import pickle
import hashlib
import logging
from collections import defaultdict


class ByzantineFaultTolerantAggregator:
    """Byzantine fault-tolerant aggregator for ensemble weights."""

    def __init__(self,
                 byzantine_threshold: float = 0.33,
                 outlier_detection_method: str = "median_absolute_deviation",
                 min_reliable_sources: int = 3):
        """
        Initialize Byzantine fault-tolerant aggregator.

        Args:
            byzantine_threshold: Maximum fraction of Byzantine nodes to tolerate
            outlier_detection_method: Method for detecting outliers
            min_reliable_sources: Minimum number of reliable sources required
        """
        self.byzantine_threshold = byzantine_threshold
        self.outlier_detection_method = outlier_detection_method
        self.min_reliable_sources = min_reliable_sources
        self.source_reliability_scores = defaultdict(float)
        self.detection_history = defaultdict(list)

    def robust_aggregation(self,
                          weight_updates: List[Dict[str, torch.Tensor]],
                          source_ids: List[str],
                          confidence_scores: Optional[List[float]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Perform Byzantine fault-tolerant aggregation.

        Args:
            weight_updates: List of weight updates from different sources
            source_ids: Identifiers for each source
            confidence_scores: Optional confidence scores for each source

        Returns:
            Tuple of (aggregated_weights, fault_tolerance_metrics)
        """
        if len(weight_updates) != len(source_ids):
            raise ValueError("Number of weight updates must match number of source IDs")

        if len(weight_updates) < self.min_reliable_sources:
            raise ValueError(f"Need at least {self.min_reliable_sources} sources for fault tolerance")

        # Filter out potentially corrupted updates
        reliable_updates, reliable_sources, fault_metrics = self._detect_and_filter_byzantines(
            weight_updates, source_ids, confidence_scores
        )

        # Perform robust aggregation on reliable updates
        if len(reliable_updates) >= self.min_reliable_sources:
            aggregated_weights = self._geometric_median_aggregation(reliable_updates)
        else:
            # Fallback to simple median if not enough reliable sources
            aggregated_weights = self._median_aggregation(weight_updates)
            fault_metrics['fallback_used'] = True

        # Update source reliability scores
        self._update_reliability_scores(source_ids, fault_metrics['byzantine_sources'])

        return aggregated_weights, fault_metrics

    def _detect_and_filter_byzantines(self,
                                    weight_updates: List[Dict[str, torch.Tensor]],
                                    source_ids: List[str],
                                    confidence_scores: Optional[List[float]]) -> Tuple[List[Dict[str, torch.Tensor]], List[str], Dict[str, Any]]:
        """Detect and filter out Byzantine sources."""
        num_sources = len(weight_updates)
        max_byzantines = int(num_sources * self.byzantine_threshold)

        byzantine_scores = []
        fault_metrics = {
            'byzantine_sources': [],
            'outlier_scores': {},
            'detection_method': self.outlier_detection_method,
            'fallback_used': False
        }

        # Calculate outlier scores for each source
        for i, (update, source_id) in enumerate(zip(weight_updates, source_ids)):
            outlier_score = self._calculate_outlier_score(update, weight_updates, i)
            byzantine_scores.append(outlier_score)
            fault_metrics['outlier_scores'][source_id] = outlier_score

        # Identify Byzantine sources
        byzantine_indices = []
        if max_byzantines > 0:
            # Sort by outlier score and mark worst ones as Byzantine
            sorted_indices = sorted(range(num_sources), key=lambda i: byzantine_scores[i], reverse=True)
            byzantine_indices = sorted_indices[:max_byzantines]

            # Additional filtering based on reliability history
            for idx in sorted_indices:
                source_id = source_ids[idx]
                if (self.source_reliability_scores[source_id] < 0.3 and
                    byzantine_scores[idx] > np.median(byzantine_scores) + np.std(byzantine_scores)):
                    if idx not in byzantine_indices:
                        byzantine_indices.append(idx)

        # Filter out Byzantine sources
        reliable_updates = []
        reliable_sources = []

        for i, (update, source_id) in enumerate(zip(weight_updates, source_ids)):
            if i not in byzantine_indices:
                reliable_updates.append(update)
                reliable_sources.append(source_id)
            else:
                fault_metrics['byzantine_sources'].append(source_id)

        return reliable_updates, reliable_sources, fault_metrics

    def _calculate_outlier_score(self,
                                target_update: Dict[str, torch.Tensor],
                                all_updates: List[Dict[str, torch.Tensor]],
                                target_index: int) -> float:
        """Calculate outlier score for a target update."""
        if self.outlier_detection_method == "median_absolute_deviation":
            return self._mad_outlier_score(target_update, all_updates, target_index)
        elif self.outlier_detection_method == "cosine_similarity":
            return self._cosine_similarity_outlier_score(target_update, all_updates, target_index)
        else:
            return self._euclidean_distance_outlier_score(target_update, all_updates, target_index)

    def _mad_outlier_score(self,
                          target_update: Dict[str, torch.Tensor],
                          all_updates: List[Dict[str, torch.Tensor]],
                          target_index: int) -> float:
        """Calculate outlier score using Median Absolute Deviation."""
        total_mad_score = 0.0
        param_count = 0

        for param_name in target_update.keys():
            # Collect parameter values from all updates
            param_values = []
            target_value = target_update[param_name].flatten()

            for i, update in enumerate(all_updates):
                if param_name in update and i != target_index:
                    param_values.append(update[param_name].flatten())

            if len(param_values) < 2:
                continue

            # Calculate median and MAD
            stacked_values = torch.stack(param_values)
            median_value = torch.median(stacked_values, dim=0)[0]

            absolute_deviations = torch.abs(stacked_values - median_value.unsqueeze(0))
            mad = torch.median(absolute_deviations, dim=0)[0]

            # Calculate MAD score for target
            target_deviation = torch.abs(target_value - median_value)
            mad_score = torch.mean(target_deviation / (mad + 1e-8)).item()

            total_mad_score += mad_score
            param_count += 1

        return total_mad_score / max(param_count, 1)

    def _cosine_similarity_outlier_score(self,
                                       target_update: Dict[str, torch.Tensor],
                                       all_updates: List[Dict[str, torch.Tensor]],
                                       target_index: int) -> float:
        """Calculate outlier score using cosine similarity."""
        similarities = []

        # Flatten target update
        target_flat = torch.cat([param.flatten() for param in target_update.values()])

        for i, update in enumerate(all_updates):
            if i != target_index:
                # Flatten comparison update
                try:
                    update_flat = torch.cat([update[param_name].flatten()
                                           for param_name in target_update.keys()
                                           if param_name in update])

                    if len(update_flat) == len(target_flat):
                        similarity = torch.cosine_similarity(target_flat, update_flat, dim=0)
                        similarities.append(similarity.item())
                except:
                    continue

        if not similarities:
            return 0.0

        # Lower similarity means higher outlier score
        avg_similarity = np.mean(similarities)
        return 1.0 - max(0.0, avg_similarity)

    def _euclidean_distance_outlier_score(self,
                                        target_update: Dict[str, torch.Tensor],
                                        all_updates: List[Dict[str, torch.Tensor]],
                                        target_index: int) -> float:
        """Calculate outlier score using Euclidean distance."""
        distances = []

        # Flatten target update
        target_flat = torch.cat([param.flatten() for param in target_update.values()])

        for i, update in enumerate(all_updates):
            if i != target_index:
                try:
                    update_flat = torch.cat([update[param_name].flatten()
                                           for param_name in target_update.keys()
                                           if param_name in update])

                    if len(update_flat) == len(target_flat):
                        distance = torch.norm(target_flat - update_flat).item()
                        distances.append(distance)
                except:
                    continue

        if not distances:
            return 0.0

        # Normalize by median distance
        median_distance = np.median(distances)
        avg_distance = np.mean(distances)

        return avg_distance / (median_distance + 1e-8)

    def _geometric_median_aggregation(self, weight_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate weights using geometric median for robustness."""
        if len(weight_updates) == 1:
            return weight_updates[0].copy()

        aggregated_weights = {}

        for param_name in weight_updates[0].keys():
            # Collect parameter tensors
            param_tensors = []
            for update in weight_updates:
                if param_name in update:
                    param_tensors.append(update[param_name])

            if len(param_tensors) < 2:
                aggregated_weights[param_name] = param_tensors[0].clone()
                continue

            # Calculate geometric median using iterative algorithm
            geometric_median = self._calculate_geometric_median(param_tensors)
            aggregated_weights[param_name] = geometric_median

        return aggregated_weights

    def _calculate_geometric_median(self, tensors: List[torch.Tensor], max_iterations: int = 100) -> torch.Tensor:
        """Calculate geometric median of tensor list."""
        if len(tensors) == 1:
            return tensors[0].clone()

        # Initialize with arithmetic mean
        current_median = torch.stack(tensors).mean(dim=0)

        for iteration in range(max_iterations):
            # Calculate weights based on distances
            distances = []
            for tensor in tensors:
                dist = torch.norm(tensor - current_median)
                distances.append(max(dist.item(), 1e-8))  # Avoid division by zero

            # Update median using weighted average
            weights = [1.0 / dist for dist in distances]
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

            new_median = torch.zeros_like(current_median)
            for tensor, weight in zip(tensors, weights):
                new_median += weight * tensor

            # Check convergence
            if torch.norm(new_median - current_median) < 1e-6:
                break

            current_median = new_median

        return current_median

    def _median_aggregation(self, weight_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Simple median aggregation as fallback."""
        aggregated_weights = {}

        for param_name in weight_updates[0].keys():
            param_tensors = []
            for update in weight_updates:
                if param_name in update:
                    param_tensors.append(update[param_name])

            if param_tensors:
                stacked_tensors = torch.stack(param_tensors)
                aggregated_weights[param_name] = torch.median(stacked_tensors, dim=0)[0]

        return aggregated_weights

    def _update_reliability_scores(self, source_ids: List[str], byzantine_sources: List[str]):
        """Update reliability scores for sources."""
        for source_id in source_ids:
            if source_id in byzantine_sources:
                # Decrease reliability for Byzantine sources
                self.source_reliability_scores[source_id] = max(
                    0.0, self.source_reliability_scores[source_id] - 0.1
                )
                self.detection_history[source_id].append(('byzantine', time.time()))
            else:
                # Increase reliability for honest sources
                self.source_reliability_scores[source_id] = min(
                    1.0, self.source_reliability_scores[source_id] + 0.05
                )
                self.detection_history[source_id].append(('honest', time.time()))

            # Keep only recent history
            if len(self.detection_history[source_id]) > 100:
                self.detection_history[source_id] = self.detection_history[source_id][-100:]

    def get_reliability_statistics(self) -> Dict[str, Any]:
        """Get reliability statistics for all sources."""
        return {
            'source_reliability_scores': dict(self.source_reliability_scores),
            'detection_history_summary': {
                source_id: {
                    'total_detections': len(history),
                    'byzantine_count': sum(1 for event, _ in history if event == 'byzantine'),
                    'honest_count': sum(1 for event, _ in history if event == 'honest')
                }
                for source_id, history in self.detection_history.items()
            },
            'byzantine_threshold': self.byzantine_threshold,
            'detection_method': self.outlier_detection_method
        }