"""
Performance monitoring utilities for HQDE framework.

This module provides comprehensive performance monitoring, metrics collection,
and system resource tracking for distributed ensemble learning.
"""

import torch
import psutil
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import logging
import json


class SystemMetrics:
    """Container for system performance metrics."""

    def __init__(self):
        self.timestamp = time.time()
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_used_gb = 0.0
        self.disk_io_read_mb = 0.0
        self.disk_io_write_mb = 0.0
        self.network_sent_mb = 0.0
        self.network_recv_mb = 0.0
        self.gpu_memory_used_gb = 0.0
        self.gpu_utilization = 0.0
        self.load_average = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb,
            'gpu_memory_used_gb': self.gpu_memory_used_gb,
            'gpu_utilization': self.gpu_utilization,
            'load_average': self.load_average
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SystemMetrics':
        """Create SystemMetrics from dictionary."""
        metrics = cls()
        for key, value in data.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics


class PerformanceMonitor:
    """Comprehensive performance monitor for HQDE systems."""

    def __init__(self,
                 monitoring_interval: float = 1.0,
                 history_size: int = 1000,
                 enable_gpu_monitoring: bool = True):
        """
        Initialize performance monitor.

        Args:
            monitoring_interval: Interval between metric collections (seconds)
            history_size: Maximum number of historical metrics to keep
            enable_gpu_monitoring: Whether to monitor GPU metrics
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring

        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.ensemble_metrics = defaultdict(list)
        self.training_metrics = defaultdict(list)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None

        # Performance baselines
        self.baseline_metrics = None
        self.performance_alerts = []

        # Event tracking
        self.events = []
        self.custom_metrics = defaultdict(list)

        # Initialize baseline
        self._establish_baseline()

    def _establish_baseline(self):
        """Establish performance baseline."""
        baseline_samples = []
        for _ in range(10):
            metrics = self._collect_system_metrics()
            baseline_samples.append(metrics)
            time.sleep(0.1)

        if baseline_samples:
            self.baseline_metrics = self._calculate_average_metrics(baseline_samples)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics()

        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_gb = memory.used / (1024**3)

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.disk_io_read_mb = disk_io.read_bytes / (1024**2)
                metrics.disk_io_write_mb = disk_io.write_bytes / (1024**2)

            # Network metrics
            network_io = psutil.net_io_counters()
            if network_io:
                metrics.network_sent_mb = network_io.bytes_sent / (1024**2)
                metrics.network_recv_mb = network_io.bytes_recv / (1024**2)

            # Load average (Unix-like systems)
            if hasattr(psutil, 'getloadavg'):
                metrics.load_average = psutil.getloadavg()[0]

            # GPU metrics
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                try:
                    metrics.gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                    # GPU utilization would require nvidia-ml-py or similar
                    metrics.gpu_utilization = 0.0  # Placeholder
                except:
                    pass

        except Exception as e:
            logging.warning(f"Error collecting system metrics: {e}")

        return metrics

    def _calculate_average_metrics(self, metrics_list: List[SystemMetrics]) -> SystemMetrics:
        """Calculate average of multiple SystemMetrics."""
        if not metrics_list:
            return SystemMetrics()

        avg_metrics = SystemMetrics()
        fields = ['cpu_percent', 'memory_percent', 'memory_used_gb',
                 'disk_io_read_mb', 'disk_io_write_mb', 'network_sent_mb',
                 'network_recv_mb', 'gpu_memory_used_gb', 'gpu_utilization',
                 'load_average']

        for field in fields:
            values = [getattr(m, field) for m in metrics_list]
            setattr(avg_metrics, field, np.mean(values))

        return avg_metrics

    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logging.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

        logging.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)

                # Check for performance alerts
                self._check_performance_alerts(metrics)

                time.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")

    def _check_performance_alerts(self, metrics: SystemMetrics):
        """Check for performance alerts based on current metrics."""
        alerts = []

        # High CPU usage alert
        if metrics.cpu_percent > 90:
            alerts.append({
                'type': 'high_cpu',
                'message': f'High CPU usage: {metrics.cpu_percent:.1f}%',
                'timestamp': metrics.timestamp,
                'severity': 'warning'
            })

        # High memory usage alert
        if metrics.memory_percent > 85:
            alerts.append({
                'type': 'high_memory',
                'message': f'High memory usage: {metrics.memory_percent:.1f}%',
                'timestamp': metrics.timestamp,
                'severity': 'warning'
            })

        # GPU memory alert
        if metrics.gpu_memory_used_gb > 8:  # Assuming 8GB+ is high usage
            alerts.append({
                'type': 'high_gpu_memory',
                'message': f'High GPU memory usage: {metrics.gpu_memory_used_gb:.1f}GB',
                'timestamp': metrics.timestamp,
                'severity': 'warning'
            })

        # Add alerts to history
        self.performance_alerts.extend(alerts)

        # Keep only recent alerts
        cutoff_time = time.time() - 3600  # Last hour
        self.performance_alerts = [
            alert for alert in self.performance_alerts
            if alert['timestamp'] > cutoff_time
        ]

    def record_ensemble_metric(self,
                             metric_name: str,
                             value: float,
                             ensemble_id: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Record an ensemble-specific metric."""
        metric_data = {
            'timestamp': time.time(),
            'value': value,
            'ensemble_id': ensemble_id,
            'metadata': metadata or {}
        }

        self.ensemble_metrics[metric_name].append(metric_data)

        # Keep only recent metrics
        if len(self.ensemble_metrics[metric_name]) > self.history_size:
            self.ensemble_metrics[metric_name] = \
                self.ensemble_metrics[metric_name][-self.history_size:]

    def record_training_metric(self,
                             metric_name: str,
                             value: float,
                             epoch: Optional[int] = None,
                             batch: Optional[int] = None):
        """Record a training-specific metric."""
        metric_data = {
            'timestamp': time.time(),
            'value': value,
            'epoch': epoch,
            'batch': batch
        }

        self.training_metrics[metric_name].append(metric_data)

        # Keep only recent metrics
        if len(self.training_metrics[metric_name]) > self.history_size:
            self.training_metrics[metric_name] = \
                self.training_metrics[metric_name][-self.history_size:]

    def record_custom_metric(self,
                           metric_name: str,
                           value: Any,
                           tags: Optional[Dict[str, str]] = None):
        """Record a custom metric."""
        metric_data = {
            'timestamp': time.time(),
            'value': value,
            'tags': tags or {}
        }

        self.custom_metrics[metric_name].append(metric_data)

        # Keep only recent metrics
        if len(self.custom_metrics[metric_name]) > self.history_size:
            self.custom_metrics[metric_name] = \
                self.custom_metrics[metric_name][-self.history_size:]

    def record_event(self,
                    event_type: str,
                    description: str,
                    metadata: Optional[Dict[str, Any]] = None):
        """Record a system event."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'description': description,
            'metadata': metadata or {}
        }

        self.events.append(event)

        # Keep only recent events
        if len(self.events) > self.history_size:
            self.events = self.events[-self.history_size:]

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_metrics_summary(self, window_minutes: float = 60) -> Dict[str, Any]:
        """Get summary statistics for metrics within a time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        if not recent_metrics:
            return {}

        summary = {}
        fields = ['cpu_percent', 'memory_percent', 'memory_used_gb',
                 'gpu_memory_used_gb', 'load_average']

        for field in fields:
            values = [getattr(m, field) for m in recent_metrics]
            summary[field] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'current': values[-1] if values else 0
            }

        return summary

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_metrics = self.get_current_metrics()
        metrics_summary = self.get_metrics_summary()

        # Calculate performance compared to baseline
        performance_comparison = {}
        if self.baseline_metrics and current_metrics:
            fields = ['cpu_percent', 'memory_percent', 'gpu_memory_used_gb']
            for field in fields:
                baseline_val = getattr(self.baseline_metrics, field)
                current_val = getattr(current_metrics, field)
                if baseline_val > 0:
                    change_percent = ((current_val - baseline_val) / baseline_val) * 100
                    performance_comparison[field] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'change_percent': change_percent
                    }

        return {
            'current_metrics': current_metrics.to_dict() if current_metrics else {},
            'metrics_summary': metrics_summary,
            'performance_comparison': performance_comparison,
            'recent_alerts': self.performance_alerts[-10:],  # Last 10 alerts
            'ensemble_metrics_count': {k: len(v) for k, v in self.ensemble_metrics.items()},
            'training_metrics_count': {k: len(v) for k, v in self.training_metrics.items()},
            'recent_events': self.events[-10:],  # Last 10 events
            'monitoring_status': self.is_monitoring
        }

    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file."""
        if format.lower() == 'json':
            self._export_json(filepath)
        elif format.lower() == 'csv':
            self._export_csv(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, filepath: str):
        """Export metrics to JSON file."""
        export_data = {
            'system_metrics': [m.to_dict() for m in self.metrics_history],
            'ensemble_metrics': dict(self.ensemble_metrics),
            'training_metrics': dict(self.training_metrics),
            'custom_metrics': dict(self.custom_metrics),
            'events': self.events,
            'performance_alerts': self.performance_alerts
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

    def _export_csv(self, filepath: str):
        """Export system metrics to CSV file."""
        import csv

        with open(filepath, 'w', newline='') as f:
            if not self.metrics_history:
                return

            fieldnames = self.metrics_history[0].to_dict().keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for metrics in self.metrics_history:
                writer.writerow(metrics.to_dict())

    def cleanup(self):
        """Cleanup monitoring resources."""
        self.stop_monitoring()
        self.metrics_history.clear()
        self.ensemble_metrics.clear()
        self.training_metrics.clear()
        self.custom_metrics.clear()
        self.events.clear()
        self.performance_alerts.clear()


class TimingContext:
    """Context manager for measuring execution time."""

    def __init__(self, monitor: PerformanceMonitor, metric_name: str, **kwargs):
        self.monitor = monitor
        self.metric_name = metric_name
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            self.monitor.record_custom_metric(
                self.metric_name,
                execution_time,
                tags={'unit': 'seconds', **self.kwargs}
            )


def monitor_performance(monitor: PerformanceMonitor, metric_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with TimingContext(monitor, f"{metric_name}_execution_time"):
                return func(*args, **kwargs)
        return wrapper
    return decorator