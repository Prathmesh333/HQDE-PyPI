"""
Dynamic load balancing module for HQDE framework.

This module implements intelligent workload distribution, performance monitoring,
and adaptive task scheduling for optimal resource utilization.
"""

import torch
import ray
import numpy as np
import psutil
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import threading
import logging
from collections import defaultdict, deque


@ray.remote
class WorkerNode:
    """Individual worker node with performance monitoring."""

    def __init__(self, node_id: str, capabilities: Dict[str, Any]):
        self.node_id = node_id
        self.capabilities = capabilities
        self.current_load = 0.0
        self.task_queue = deque()
        self.performance_history = deque(maxlen=100)
        self.is_active = True

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics for this node."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0.0,
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
            'current_task_load': self.current_load
        }

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task and return results with performance metrics."""
        start_time = time.time()
        task_id = task_data.get('task_id', 'unknown')

        try:
            # Simulate task execution
            self.current_load += task_data.get('estimated_load', 0.1)

            # Process the task based on type
            task_type = task_data.get('type', 'default')
            if task_type == 'weight_aggregation':
                result = self._execute_weight_aggregation(task_data)
            elif task_type == 'quantization':
                result = self._execute_quantization(task_data)
            elif task_type == 'ensemble_training':
                result = self._execute_ensemble_training(task_data)
            else:
                result = {'status': 'completed', 'data': task_data.get('data', {})}

            execution_time = time.time() - start_time

            # Update performance history
            self.performance_history.append({
                'task_id': task_id,
                'execution_time': execution_time,
                'task_type': task_type,
                'success': True,
                'timestamp': time.time()
            })

            self.current_load = max(0.0, self.current_load - task_data.get('estimated_load', 0.1))

            return {
                'node_id': self.node_id,
                'task_id': task_id,
                'result': result,
                'execution_time': execution_time,
                'status': 'success'
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_history.append({
                'task_id': task_id,
                'execution_time': execution_time,
                'task_type': task_type,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            })

            self.current_load = max(0.0, self.current_load - task_data.get('estimated_load', 0.1))

            return {
                'node_id': self.node_id,
                'task_id': task_id,
                'result': None,
                'execution_time': execution_time,
                'status': 'error',
                'error': str(e)
            }

    def _execute_weight_aggregation(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute weight aggregation task."""
        weights = task_data.get('weights', [])
        if not weights:
            return {'aggregated_weights': {}}

        # Simple aggregation for demonstration
        aggregated = {}
        for param_name in weights[0].keys():
            param_tensors = [w[param_name] for w in weights if param_name in w]
            if param_tensors:
                aggregated[param_name] = torch.stack(param_tensors).mean(dim=0)

        return {'aggregated_weights': aggregated}

    def _execute_quantization(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantization task."""
        weights = task_data.get('weights', {})
        quantized_weights = {}

        for param_name, weight_tensor in weights.items():
            # Simple quantization simulation
            quantized_weights[param_name] = torch.round(weight_tensor * 255) / 255

        return {'quantized_weights': quantized_weights}

    def _execute_ensemble_training(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ensemble training task."""
        # Simulate training step
        training_time = np.random.uniform(0.1, 1.0)
        time.sleep(training_time / 10)  # Scaled down for simulation

        return {
            'training_loss': np.random.uniform(0.1, 2.0),
            'accuracy': np.random.uniform(0.7, 0.95),
            'training_time': training_time
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this node."""
        if not self.performance_history:
            return {
                'node_id': self.node_id,
                'avg_execution_time': 0.0,
                'success_rate': 1.0,
                'total_tasks': 0,
                'current_load': self.current_load
            }

        successful_tasks = [task for task in self.performance_history if task['success']]
        total_tasks = len(self.performance_history)

        avg_execution_time = np.mean([task['execution_time'] for task in self.performance_history])
        success_rate = len(successful_tasks) / total_tasks if total_tasks > 0 else 1.0

        return {
            'node_id': self.node_id,
            'avg_execution_time': avg_execution_time,
            'success_rate': success_rate,
            'total_tasks': total_tasks,
            'current_load': self.current_load,
            'capabilities': self.capabilities
        }

    def set_active_status(self, is_active: bool):
        """Set the active status of this node."""
        self.is_active = is_active

    def get_active_status(self) -> bool:
        """Get the active status of this node."""
        return self.is_active


class DynamicLoadBalancer:
    """Dynamic load balancer for HQDE distributed ensemble learning."""

    def __init__(self,
                 balancing_strategy: str = "adaptive",
                 monitoring_interval: float = 5.0,
                 load_threshold: float = 0.8):
        """
        Initialize dynamic load balancer.

        Args:
            balancing_strategy: Load balancing strategy ("round_robin", "least_loaded", "adaptive")
            monitoring_interval: Interval for monitoring node performance
            load_threshold: Threshold for triggering load redistribution
        """
        self.balancing_strategy = balancing_strategy
        self.monitoring_interval = monitoring_interval
        self.load_threshold = load_threshold

        self.worker_nodes = {}
        self.task_queue = deque()
        self.task_history = defaultdict(list)
        self.performance_predictor = PerformancePredictor()

        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None

        # Load balancing metrics
        self.balancing_metrics = {
            'total_tasks_scheduled': 0,
            'load_redistributions': 0,
            'average_response_time': 0.0,
            'node_utilization': {}
        }

    def add_worker_node(self, node_id: str, capabilities: Dict[str, Any]):
        """Add a worker node to the load balancer."""
        worker_node = WorkerNode.remote(node_id, capabilities)
        self.worker_nodes[node_id] = worker_node
        self.balancing_metrics['node_utilization'][node_id] = 0.0

    def remove_worker_node(self, node_id: str):
        """Remove a worker node from the load balancer."""
        if node_id in self.worker_nodes:
            # Set node as inactive first
            ray.get(self.worker_nodes[node_id].set_active_status.remote(False))
            del self.worker_nodes[node_id]
            if node_id in self.balancing_metrics['node_utilization']:
                del self.balancing_metrics['node_utilization'][node_id]

    def schedule_task(self, task_data: Dict[str, Any]) -> str:
        """Schedule a task for execution."""
        if not self.worker_nodes:
            raise RuntimeError("No worker nodes available")

        # Select best node for the task
        selected_node_id = self._select_node_for_task(task_data)

        if selected_node_id is None:
            raise RuntimeError("No suitable node found for task")

        # Schedule task on selected node
        task_future = self.worker_nodes[selected_node_id].execute_task.remote(task_data)

        # Track task
        task_id = task_data.get('task_id', f"task_{int(time.time())}")
        self.task_history[selected_node_id].append({
            'task_id': task_id,
            'future': task_future,
            'scheduled_time': time.time(),
            'task_data': task_data
        })

        self.balancing_metrics['total_tasks_scheduled'] += 1

        return task_id

    def _select_node_for_task(self, task_data: Dict[str, Any]) -> Optional[str]:
        """Select the best node for executing a task."""
        if self.balancing_strategy == "round_robin":
            return self._round_robin_selection()
        elif self.balancing_strategy == "least_loaded":
            return self._least_loaded_selection()
        elif self.balancing_strategy == "adaptive":
            return self._adaptive_selection(task_data)
        else:
            return self._round_robin_selection()

    def _round_robin_selection(self) -> Optional[str]:
        """Simple round-robin node selection."""
        active_nodes = list(self.worker_nodes.keys())
        if not active_nodes:
            return None

        # Simple counter-based round robin
        selection_index = self.balancing_metrics['total_tasks_scheduled'] % len(active_nodes)
        return active_nodes[selection_index]

    def _least_loaded_selection(self) -> Optional[str]:
        """Select the least loaded node."""
        if not self.worker_nodes:
            return None

        # Get current load for all nodes
        load_futures = {
            node_id: node.get_system_metrics.remote()
            for node_id, node in self.worker_nodes.items()
        }

        node_loads = {}
        for node_id, future in load_futures.items():
            try:
                metrics = ray.get(future)
                node_loads[node_id] = metrics.get('current_task_load', 0.0)
            except:
                node_loads[node_id] = float('inf')  # Exclude failed nodes

        # Select node with minimum load
        return min(node_loads.keys(), key=lambda x: node_loads[x])

    def _adaptive_selection(self, task_data: Dict[str, Any]) -> Optional[str]:
        """Adaptive node selection based on task requirements and node capabilities."""
        if not self.worker_nodes:
            return None

        # Get performance stats for all nodes
        perf_futures = {
            node_id: node.get_performance_stats.remote()
            for node_id, node in self.worker_nodes.items()
        }

        node_scores = {}
        for node_id, future in perf_futures.items():
            try:
                stats = ray.get(future)

                # Calculate suitability score
                score = self._calculate_node_suitability_score(task_data, stats)
                node_scores[node_id] = score
            except:
                node_scores[node_id] = 0.0  # Exclude failed nodes

        if not node_scores:
            return None

        # Select node with highest suitability score
        return max(node_scores.keys(), key=lambda x: node_scores[x])

    def _calculate_node_suitability_score(self,
                                        task_data: Dict[str, Any],
                                        node_stats: Dict[str, Any]) -> float:
        """Calculate suitability score for a node given a task."""
        score = 0.0

        # Factor in success rate
        success_rate = node_stats.get('success_rate', 1.0)
        score += success_rate * 0.4

        # Factor in current load (lower is better)
        current_load = node_stats.get('current_load', 0.0)
        load_factor = max(0.0, 1.0 - current_load)
        score += load_factor * 0.3

        # Factor in average execution time (lower is better)
        avg_time = node_stats.get('avg_execution_time', 1.0)
        time_factor = max(0.0, 1.0 - min(avg_time / 10.0, 1.0))  # Normalize to 10 seconds max
        score += time_factor * 0.2

        # Factor in capabilities match
        task_requirements = task_data.get('requirements', {})
        node_capabilities = node_stats.get('capabilities', {})
        capability_match = self._calculate_capability_match(task_requirements, node_capabilities)
        score += capability_match * 0.1

        return score

    def _calculate_capability_match(self,
                                  requirements: Dict[str, Any],
                                  capabilities: Dict[str, Any]) -> float:
        """Calculate how well node capabilities match task requirements."""
        if not requirements:
            return 1.0

        matches = 0
        total_requirements = 0

        for req_key, req_value in requirements.items():
            total_requirements += 1
            if req_key in capabilities:
                cap_value = capabilities[req_key]
                if isinstance(req_value, (int, float)) and isinstance(cap_value, (int, float)):
                    if cap_value >= req_value:
                        matches += 1
                elif req_value == cap_value:
                    matches += 1

        return matches / total_requirements if total_requirements > 0 else 1.0

    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_performance_metrics()
                self._check_load_balance()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")

    def _collect_performance_metrics(self):
        """Collect performance metrics from all nodes."""
        metric_futures = {
            node_id: node.get_system_metrics.remote()
            for node_id, node in self.worker_nodes.items()
        }

        for node_id, future in metric_futures.items():
            try:
                metrics = ray.get(future)
                current_load = metrics.get('current_task_load', 0.0)
                self.balancing_metrics['node_utilization'][node_id] = current_load
            except:
                # Node might be unavailable
                pass

    def _check_load_balance(self):
        """Check if load rebalancing is needed."""
        node_loads = list(self.balancing_metrics['node_utilization'].values())

        if len(node_loads) < 2:
            return

        max_load = max(node_loads)
        min_load = min(node_loads)
        load_imbalance = max_load - min_load

        if load_imbalance > self.load_threshold:
            self._rebalance_load()

    def _rebalance_load(self):
        """Perform load rebalancing."""
        # This is a simplified rebalancing strategy
        # In practice, this would involve more sophisticated task migration
        self.balancing_metrics['load_redistributions'] += 1
        logging.info("Load rebalancing triggered")

    def get_balancing_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        # Calculate average response time
        all_tasks = []
        for node_tasks in self.task_history.values():
            all_tasks.extend(node_tasks)

        if all_tasks:
            # This is simplified - in practice, you'd track completion times
            self.balancing_metrics['average_response_time'] = 1.0  # Placeholder

        return {
            'balancing_metrics': self.balancing_metrics.copy(),
            'active_nodes': len(self.worker_nodes),
            'balancing_strategy': self.balancing_strategy,
            'monitoring_interval': self.monitoring_interval,
            'load_threshold': self.load_threshold
        }

    def cleanup(self):
        """Cleanup load balancer resources."""
        self.stop_monitoring()
        # Ray will automatically clean up remote actors


class PerformancePredictor:
    """Simple performance predictor for task scheduling."""

    def __init__(self):
        self.task_performance_history = defaultdict(list)

    def predict_completion_time(self,
                              task_features: Dict[str, Any],
                              node_features: Dict[str, Any]) -> float:
        """Predict task completion time based on features."""
        # Simplified prediction based on task type and node performance
        task_type = task_features.get('type', 'default')
        node_avg_time = node_features.get('avg_execution_time', 1.0)

        # Task type multipliers
        type_multipliers = {
            'weight_aggregation': 0.5,
            'quantization': 0.8,
            'ensemble_training': 2.0,
            'default': 1.0
        }

        base_time = type_multipliers.get(task_type, 1.0)
        predicted_time = base_time * node_avg_time

        return predicted_time

    def update_performance_history(self,
                                 task_features: Dict[str, Any],
                                 actual_time: float):
        """Update performance history with actual completion time."""
        task_type = task_features.get('type', 'default')
        self.task_performance_history[task_type].append(actual_time)

        # Keep only recent history
        if len(self.task_performance_history[task_type]) > 100:
            self.task_performance_history[task_type] = \
                self.task_performance_history[task_type][-100:]