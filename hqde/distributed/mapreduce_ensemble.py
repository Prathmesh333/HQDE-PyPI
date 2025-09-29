"""
MapReduce-inspired ensemble management for HQDE framework.

This module implements distributed key-value storage and MapReduce patterns
for efficient ensemble weight management and aggregation.
"""

import torch
import ray
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import hashlib
import pickle
import time
from collections import defaultdict
import logging


@ray.remote
class DistributedWeightStore:
    """Distributed key-value store for ensemble weights."""

    def __init__(self, store_id: str, replication_factor: int = 3):
        self.store_id = store_id
        self.replication_factor = replication_factor
        self.weights = {}
        self.metadata = {}
        self.access_count = defaultdict(int)
        self.last_access = {}

    def put_weight(self, key: str, weight_tensor: torch.Tensor, metadata: Dict[str, Any] = None):
        """Store a weight tensor with optional metadata."""
        self.weights[key] = weight_tensor.cpu()  # Store on CPU to save GPU memory
        self.metadata[key] = metadata or {}
        self.access_count[key] = 0
        self.last_access[key] = time.time()

    def get_weight(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a weight tensor."""
        if key in self.weights:
            self.access_count[key] += 1
            self.last_access[key] = time.time()
            return self.weights[key]
        return None

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a weight tensor."""
        return self.metadata.get(key, None)

    def list_keys(self) -> List[str]:
        """List all stored weight keys."""
        return list(self.weights.keys())

    def delete_weight(self, key: str) -> bool:
        """Delete a weight tensor."""
        if key in self.weights:
            del self.weights[key]
            del self.metadata[key]
            if key in self.access_count:
                del self.access_count[key]
            if key in self.last_access:
                del self.last_access[key]
            return True
        return False

    def get_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        return {
            'store_id': self.store_id,
            'num_weights': len(self.weights),
            'total_memory_mb': sum(w.numel() * w.element_size() for w in self.weights.values()) / (1024 * 1024),
            'access_counts': dict(self.access_count),
            'last_access_times': dict(self.last_access)
        }


@ray.remote
class MapperWorker:
    """Worker node that performs map operations on ensemble data."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.local_cache = {}

    def map_operation(self,
                     data_partition: List[Any],
                     map_function: Callable,
                     context: Dict[str, Any] = None) -> List[Tuple[str, Any]]:
        """Perform map operation on data partition."""
        results = []
        for item in data_partition:
            try:
                mapped_result = map_function(item, context or {})
                # Create key-value pairs
                if isinstance(mapped_result, (list, tuple)):
                    results.extend(mapped_result)
                else:
                    results.append(mapped_result)
            except Exception as e:
                logging.error(f"Map operation failed for item {item}: {e}")
                continue

        return results

    def cache_intermediate_result(self, key: str, value: Any):
        """Cache intermediate results locally."""
        self.local_cache[key] = value

    def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve cached intermediate result."""
        return self.local_cache.get(key, None)


@ray.remote
class ReducerWorker:
    """Worker node that performs reduce operations on mapped data."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id

    def reduce_operation(self,
                        key: str,
                        values: List[Any],
                        reduce_function: Callable,
                        context: Dict[str, Any] = None) -> Tuple[str, Any]:
        """Perform reduce operation on values for a given key."""
        try:
            reduced_result = reduce_function(key, values, context or {})
            return key, reduced_result
        except Exception as e:
            logging.error(f"Reduce operation failed for key {key}: {e}")
            return key, None

    def aggregate_weights(self,
                         weight_list: List[torch.Tensor],
                         aggregation_method: str = "mean") -> torch.Tensor:
        """Aggregate a list of weight tensors."""
        if not weight_list:
            return torch.empty(0)

        if aggregation_method == "mean":
            return torch.stack(weight_list).mean(dim=0)
        elif aggregation_method == "weighted_mean":
            # Assume equal weights for now
            return torch.stack(weight_list).mean(dim=0)
        elif aggregation_method == "median":
            return torch.stack(weight_list).median(dim=0)[0]
        else:
            return torch.stack(weight_list).mean(dim=0)


class MapReduceEnsembleManager:
    """MapReduce-inspired ensemble manager for distributed weight aggregation."""

    def __init__(self,
                 num_stores: int = 3,
                 num_mappers: int = 4,
                 num_reducers: int = 2,
                 replication_factor: int = 3):
        """
        Initialize MapReduce ensemble manager.

        Args:
            num_stores: Number of distributed weight stores
            num_mappers: Number of mapper workers
            num_reducers: Number of reducer workers
            replication_factor: Replication factor for fault tolerance
        """
        self.num_stores = num_stores
        self.num_mappers = num_mappers
        self.num_reducers = num_reducers
        self.replication_factor = replication_factor

        # Initialize distributed components
        self.weight_stores = []
        self.mappers = []
        self.reducers = []

        self._initialize_workers()

    def _initialize_workers(self):
        """Initialize all distributed workers."""
        # Initialize weight stores
        for i in range(self.num_stores):
            store = DistributedWeightStore.remote(f"store_{i}", self.replication_factor)
            self.weight_stores.append(store)

        # Initialize mapper workers
        for i in range(self.num_mappers):
            mapper = MapperWorker.remote(f"mapper_{i}")
            self.mappers.append(mapper)

        # Initialize reducer workers
        for i in range(self.num_reducers):
            reducer = ReducerWorker.remote(f"reducer_{i}")
            self.reducers.append(reducer)

    def _hash_key_to_store(self, key: str) -> int:
        """Hash a key to determine which store it belongs to."""
        hash_obj = hashlib.md5(key.encode())
        return int(hash_obj.hexdigest(), 16) % self.num_stores

    def store_ensemble_weights(self,
                             ensemble_weights: Dict[str, torch.Tensor],
                             metadata: Dict[str, Any] = None) -> bool:
        """Store ensemble weights across distributed stores."""
        storage_futures = []

        for weight_key, weight_tensor in ensemble_weights.items():
            # Determine primary store
            primary_store_idx = self._hash_key_to_store(weight_key)

            # Store in primary store
            primary_store = self.weight_stores[primary_store_idx]
            future = primary_store.put_weight.remote(weight_key, weight_tensor, metadata)
            storage_futures.append(future)

            # Replicate to other stores for fault tolerance
            for replica in range(1, min(self.replication_factor, self.num_stores)):
                replica_store_idx = (primary_store_idx + replica) % self.num_stores
                replica_store = self.weight_stores[replica_store_idx]
                replica_key = f"{weight_key}_replica_{replica}"
                replica_future = replica_store.put_weight.remote(replica_key, weight_tensor, metadata)
                storage_futures.append(replica_future)

        # Wait for all storage operations to complete
        ray.get(storage_futures)
        return True

    def retrieve_ensemble_weights(self, weight_keys: List[str]) -> Dict[str, torch.Tensor]:
        """Retrieve ensemble weights from distributed stores."""
        retrieval_futures = {}

        for weight_key in weight_keys:
            store_idx = self._hash_key_to_store(weight_key)
            store = self.weight_stores[store_idx]
            future = store.get_weight.remote(weight_key)
            retrieval_futures[weight_key] = future

        # Collect results
        retrieved_weights = {}
        for weight_key, future in retrieval_futures.items():
            weight_tensor = ray.get(future)
            if weight_tensor is not None:
                retrieved_weights[weight_key] = weight_tensor

        return retrieved_weights

    def mapreduce_ensemble_aggregation(self,
                                     ensemble_data: List[Dict[str, Any]],
                                     aggregation_strategy: str = "hierarchical") -> Dict[str, torch.Tensor]:
        """
        Perform MapReduce-style ensemble aggregation.

        Args:
            ensemble_data: List of ensemble member data
            aggregation_strategy: Strategy for aggregation ("hierarchical", "flat")

        Returns:
            Aggregated ensemble weights
        """
        # Map phase: distribute data processing
        map_results = self._map_phase(ensemble_data)

        # Shuffle phase: group by keys
        grouped_data = self._shuffle_phase(map_results)

        # Reduce phase: aggregate grouped data
        aggregated_weights = self._reduce_phase(grouped_data, aggregation_strategy)

        return aggregated_weights

    def _map_phase(self, ensemble_data: List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
        """Map phase: process ensemble data in parallel."""
        # Partition data across mappers
        partitions = [[] for _ in range(self.num_mappers)]
        for i, data_item in enumerate(ensemble_data):
            partition_idx = i % self.num_mappers
            partitions[partition_idx].append(data_item)

        # Define map function
        def ensemble_map_function(item: Dict[str, Any], context: Dict[str, Any]) -> List[Tuple[str, Any]]:
            """Map function to extract weight information."""
            results = []
            if 'weights' in item:
                for param_name, weight_tensor in item['weights'].items():
                    results.append((param_name, {
                        'weight': weight_tensor,
                        'source_id': item.get('source_id', 'unknown'),
                        'accuracy': item.get('accuracy', 0.0),
                        'timestamp': item.get('timestamp', time.time())
                    }))
            return results

        # Execute map operations
        map_futures = []
        for i, partition in enumerate(partitions):
            if partition:  # Only process non-empty partitions
                mapper = self.mappers[i]
                future = mapper.map_operation.remote(partition, ensemble_map_function)
                map_futures.append(future)

        # Collect map results
        all_map_results = []
        for future in map_futures:
            partition_results = ray.get(future)
            all_map_results.extend(partition_results)

        return all_map_results

    def _shuffle_phase(self, map_results: List[Tuple[str, Any]]) -> Dict[str, List[Any]]:
        """Shuffle phase: group map results by key."""
        grouped_data = defaultdict(list)

        for key, value in map_results:
            grouped_data[key].append(value)

        return dict(grouped_data)

    def _reduce_phase(self,
                     grouped_data: Dict[str, List[Any]],
                     aggregation_strategy: str) -> Dict[str, torch.Tensor]:
        """Reduce phase: aggregate grouped data."""
        # Partition keys across reducers
        keys = list(grouped_data.keys())
        key_partitions = [[] for _ in range(self.num_reducers)]

        for i, key in enumerate(keys):
            partition_idx = i % self.num_reducers
            key_partitions[partition_idx].append(key)

        # Define reduce function
        def ensemble_reduce_function(key: str, values: List[Any], context: Dict[str, Any]) -> torch.Tensor:
            """Reduce function to aggregate weights."""
            weight_tensors = [v['weight'] for v in values if 'weight' in v]

            if not weight_tensors:
                return torch.empty(0)

            if aggregation_strategy == "hierarchical":
                # Weight by accuracy
                accuracies = [v.get('accuracy', 1.0) for v in values]
                accuracy_weights = torch.softmax(torch.tensor(accuracies), dim=0)

                weighted_sum = torch.zeros_like(weight_tensors[0])
                for weight, acc_weight in zip(weight_tensors, accuracy_weights):
                    weighted_sum += acc_weight * weight

                return weighted_sum
            else:
                # Simple averaging
                return torch.stack(weight_tensors).mean(dim=0)

        # Execute reduce operations
        reduce_futures = {}
        for i, key_partition in enumerate(key_partitions):
            if key_partition:  # Only process non-empty partitions
                reducer = self.reducers[i]
                for key in key_partition:
                    values = grouped_data[key]
                    future = reducer.reduce_operation.remote(key, values, ensemble_reduce_function)
                    reduce_futures[key] = future

        # Collect reduce results
        aggregated_weights = {}
        for key, future in reduce_futures.items():
            result_key, aggregated_weight = ray.get(future)
            if aggregated_weight is not None:
                aggregated_weights[result_key] = aggregated_weight

        return aggregated_weights

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get statistics about the distributed cluster."""
        # Get store statistics
        store_stat_futures = [store.get_store_stats.remote() for store in self.weight_stores]
        store_stats = ray.get(store_stat_futures)

        total_weights = sum(stats['num_weights'] for stats in store_stats)
        total_memory = sum(stats['total_memory_mb'] for stats in store_stats)

        return {
            'num_stores': self.num_stores,
            'num_mappers': self.num_mappers,
            'num_reducers': self.num_reducers,
            'total_stored_weights': total_weights,
            'total_memory_usage_mb': total_memory,
            'store_statistics': store_stats
        }

    def cleanup(self):
        """Cleanup distributed resources."""
        # Ray will automatically clean up remote actors
        pass