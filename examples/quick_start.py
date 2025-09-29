"""
HQDE Framework Quick Start Example

This example demonstrates how to use the HQDE framework for distributed
ensemble learning with quantum-inspired algorithms and adaptive quantization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import logging
import time

# Import HQDE components
from hqde import (
    HQDESystem,
    create_hqde_system,
    PerformanceMonitor
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple neural network model for demonstration."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DummyDataLoader:
    """Dummy data loader for demonstration purposes."""

    def __init__(self, batch_size: int = 32, num_batches: int = 10):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        # Generate dummy data (MNIST-like)
        data = torch.randn(self.batch_size, 1, 28, 28)
        labels = torch.randint(0, 10, (self.batch_size,))

        self.current_batch += 1
        return data, labels

    def __len__(self):
        return self.num_batches


def demonstrate_basic_hqde():
    """Demonstrate basic HQDE functionality."""
    logger.info("=== HQDE Framework Quick Start Demo ===")

    # Initialize performance monitor
    monitor = PerformanceMonitor(monitoring_interval=1.0)
    monitor.start_monitoring()

    try:
        # Model configuration
        model_kwargs = {
            'input_size': 784,
            'hidden_size': 128,
            'num_classes': 10
        }

        # HQDE system configuration
        quantization_config = {
            'base_bits': 8,
            'min_bits': 4,
            'max_bits': 16
        }

        aggregation_config = {
            'noise_scale': 0.01,
            'exploration_factor': 0.1
        }

        # Create HQDE system
        logger.info("Creating HQDE system with 4 workers...")
        hqde_system = create_hqde_system(
            model_class=SimpleModel,
            model_kwargs=model_kwargs,
            num_workers=4,
            quantization_config=quantization_config,
            aggregation_config=aggregation_config
        )

        # Create dummy data loaders
        train_loader = DummyDataLoader(batch_size=32, num_batches=20)
        val_loader = DummyDataLoader(batch_size=32, num_batches=5)

        # Record start event
        monitor.record_event(
            'training_start',
            'Started HQDE ensemble training',
            {'num_workers': 4, 'model_type': 'SimpleModel'}
        )

        # Train the ensemble
        logger.info("Starting HQDE ensemble training...")
        start_time = time.time()

        training_metrics = hqde_system.train(
            data_loader=train_loader,
            num_epochs=5,
            validation_loader=val_loader
        )

        training_time = time.time() - start_time

        # Record training metrics
        monitor.record_training_metric('training_time', training_time)
        monitor.record_training_metric('memory_usage', training_metrics.get('memory_usage', 0))

        # Log results
        logger.info("Training completed!")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Memory usage: {training_metrics.get('memory_usage', 0):.2f} MB")

        # Demonstrate prediction
        logger.info("Testing ensemble predictions...")
        predictions = hqde_system.predict(val_loader)
        logger.info(f"Prediction shape: {predictions.shape}")

        # Get performance metrics
        performance_metrics = hqde_system.get_performance_metrics()
        logger.info("Performance Metrics:")
        for metric, value in performance_metrics.items():
            logger.info(f"  {metric}: {value}")

        # Demonstrate model saving
        model_path = "hqde_model_demo.pth"
        logger.info(f"Saving model to {model_path}...")
        hqde_system.save_model(model_path)

        # Record completion event
        monitor.record_event(
            'training_complete',
            'HQDE ensemble training completed successfully',
            {'final_metrics': performance_metrics}
        )

    except Exception as e:
        logger.error(f"Error during HQDE demo: {e}")
        monitor.record_event('training_error', f'Training failed: {str(e)}')
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        hqde_system.cleanup()

        # Stop monitoring and generate report
        monitor.stop_monitoring()
        performance_report = monitor.get_performance_report()

        logger.info("Performance Report Summary:")
        current_metrics = performance_report.get('current_metrics', {})
        logger.info(f"  CPU Usage: {current_metrics.get('cpu_percent', 0):.1f}%")
        logger.info(f"  Memory Usage: {current_metrics.get('memory_percent', 0):.1f}%")
        logger.info(f"  GPU Memory: {current_metrics.get('gpu_memory_used_gb', 0):.2f} GB")

        # Export performance data
        logger.info("Exporting performance data...")
        monitor.export_metrics("hqde_performance_demo.json", format="json")


def demonstrate_advanced_features():
    """Demonstrate advanced HQDE features."""
    logger.info("\n=== Advanced HQDE Features Demo ===")

    # Import additional components for advanced demo
    from hqde.quantum import QuantumEnsembleAggregator, QuantumNoiseGenerator
    from hqde.distributed import HierarchicalAggregator, ByzantineFaultTolerantAggregator

    # Quantum aggregation demo
    logger.info("Demonstrating quantum-inspired aggregation...")
    quantum_aggregator = QuantumEnsembleAggregator(
        num_ensembles=3,
        entanglement_strength=0.1,
        quantum_noise_scale=0.01
    )

    # Create dummy ensemble predictions
    dummy_predictions = [
        torch.randn(10, 5) for _ in range(3)
    ]

    # Test different aggregation methods
    methods = ["superposition", "entanglement", "voting"]
    for method in methods:
        dummy_uncertainties = [torch.abs(torch.randn(10, 5)) for _ in range(3)]
        aggregated, metrics = quantum_aggregator.adaptive_quantum_aggregation(
            dummy_predictions,
            dummy_uncertainties,
            aggregation_mode=method
        )
        logger.info(f"  {method.capitalize()} aggregation - Shape: {aggregated.shape}, "
                   f"Diversity: {metrics['ensemble_diversity']:.3f}")

    # Quantum noise generation demo
    logger.info("Demonstrating quantum noise generation...")
    noise_generator = QuantumNoiseGenerator(noise_scale=0.01)

    dummy_weights = torch.randn(100, 50)
    quantum_noise = noise_generator.generate_exploration_noise(dummy_weights)
    logger.info(f"  Generated quantum noise shape: {quantum_noise.shape}")

    # Hierarchical aggregation demo
    logger.info("Demonstrating hierarchical aggregation...")
    hierarchical_agg = HierarchicalAggregator(
        num_ensemble_members=8,
        tree_branching_factor=4
    )

    dummy_ensemble_weights = [
        {'layer1': torch.randn(10, 5), 'layer2': torch.randn(5, 1)}
        for _ in range(8)
    ]

    aggregated_weights = hierarchical_agg.aggregate_ensemble_weights(dummy_ensemble_weights)
    logger.info(f"  Hierarchical aggregation completed - Parameters: {list(aggregated_weights.keys())}")

    # Byzantine fault tolerance demo
    logger.info("Demonstrating Byzantine fault tolerance...")
    bft_aggregator = ByzantineFaultTolerantAggregator(byzantine_threshold=0.33)

    # Create some "corrupted" weights
    honest_weights = [{'param': torch.randn(5, 5)} for _ in range(6)]
    byzantine_weights = [{'param': torch.randn(5, 5) * 10} for _ in range(2)]  # Outliers
    all_weights = honest_weights + byzantine_weights
    source_ids = [f"node_{i}" for i in range(8)]

    robust_weights, fault_metrics = bft_aggregator.robust_aggregation(
        all_weights, source_ids
    )

    logger.info(f"  Byzantine detection - Detected sources: {fault_metrics['byzantine_sources']}")
    logger.info(f"  Robust aggregation completed - Shape: {robust_weights['param'].shape}")

    # Cleanup
    hierarchical_agg.cleanup()


def main():
    """Main demo function."""
    try:
        # Run basic demo
        demonstrate_basic_hqde()

        # Run advanced features demo
        demonstrate_advanced_features()

        logger.info("\n=== HQDE Demo Completed Successfully! ===")
        logger.info("Check the generated files:")
        logger.info("  - hqde_model_demo.pth (saved model)")
        logger.info("  - hqde_performance_demo.json (performance metrics)")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()