"""
HQDE Framework CIFAR-10 benchmark-style example.

This example mirrors the stronger v4 notebook recipe more closely:
- small-image ResNet-18 backbone
- SGD + warmup + cosine via `make_cifar_training_config`
- epoch-level validation metrics from the framework
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset

from hqde import (
    DataLoader as HQDEDataLoader,
    DataLoaderConfig,
    DataPreprocessor,
    PerformanceMonitor,
    SmallImageResNet18,
    create_hqde_system,
    make_cifar_training_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CIFAR10DataManager:
    """Prepare CIFAR-10 loaders for full-data or subset experiments."""

    def __init__(
        self,
        data_dir: str = "./data",
        subset_size: Optional[int] = None,
        batch_size: int = 128,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.transform_train = DataPreprocessor.cifar10_transforms(is_training=True)
        self.transform_test = DataPreprocessor.cifar10_transforms(is_training=False)
        self.loader_config = DataLoaderConfig(
            batch_size=batch_size,
            num_workers=HQDEDataLoader.recommended_num_workers(max_workers=4),
            pin_memory=True,
        )

    def load_data(self):
        """Load CIFAR-10 and optionally down-sample to a balanced subset."""
        logger.info("Loading CIFAR-10 dataset")

        train_dataset_full = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform_train,
        )
        test_dataset_full = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform_test,
        )

        train_dataset = self._maybe_subset_dataset(train_dataset_full, self.subset_size)
        test_subset_size = None if self.subset_size is None else min(2000, max(self.subset_size // 5, 1000))
        test_dataset = self._maybe_subset_dataset(test_dataset_full, test_subset_size)

        train_loader = HQDEDataLoader.create(
            train_dataset,
            batch_size=self.loader_config.batch_size,
            is_training=True,
            num_workers=self.loader_config.num_workers,
            pin_memory=self.loader_config.pin_memory,
            persistent_workers=self.loader_config.persistent_workers,
            prefetch_factor=self.loader_config.prefetch_factor,
        )
        test_loader = HQDEDataLoader.create(
            test_dataset,
            batch_size=self.loader_config.batch_size,
            is_training=False,
            num_workers=self.loader_config.num_workers,
            pin_memory=self.loader_config.pin_memory,
            persistent_workers=self.loader_config.persistent_workers,
            prefetch_factor=self.loader_config.prefetch_factor,
            drop_last=False,
        )

        logger.info("Training samples: %s", len(train_dataset))
        logger.info("Validation samples: %s", len(test_dataset))
        logger.info("Batch size: %s | DataLoader workers: %s", self.batch_size, self.loader_config.num_workers)
        return train_loader, test_loader

    def _maybe_subset_dataset(self, dataset, subset_size: Optional[int]):
        if subset_size is None or subset_size >= len(dataset):
            return dataset

        labels = np.asarray(dataset.targets, dtype=np.int64)
        num_classes = int(labels.max()) + 1
        samples_per_class = max(int(subset_size) // max(num_classes, 1), 1)
        subset_indices = []

        for class_idx in range(num_classes):
            class_indices = np.where(labels == class_idx)[0]
            take = min(samples_per_class, len(class_indices))
            selected_indices = self.rng.choice(class_indices, size=take, replace=False)
            subset_indices.extend(int(index) for index in selected_indices)

        self.rng.shuffle(subset_indices)
        return Subset(dataset, subset_indices)


class CIFAR10HQDETrainer:
    """Train HQDE on CIFAR-10 with the notebook-aligned recipe."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.performance_monitor = PerformanceMonitor(monitoring_interval=2.0)
        self.model_kwargs = {"num_classes": 10}
        self.quantization_config = {
            "base_bits": 12,
            "min_bits": 8,
            "max_bits": 16,
        }
        self.aggregation_config = {
            "noise_scale": 0.001,
            "exploration_factor": 0.05,
        }
        self.training_config = make_cifar_training_config(
            ensemble_mode="independent",
            batch_assignment="replicate",
            prediction_aggregation="mean",
        )

    def train_and_evaluate(self, train_loader, test_loader, num_epochs: int = 20):
        """Train HQDE and report final validation metrics."""
        logger.info("=== HQDE CIFAR-10 Training and Evaluation ===")
        hqde_system = None
        self.performance_monitor.start_monitoring()

        try:
            logger.info("Creating HQDE system with %s workers", self.num_workers)
            hqde_system = create_hqde_system(
                model_class=SmallImageResNet18,
                model_kwargs=self.model_kwargs,
                num_workers=self.num_workers,
                quantization_config=self.quantization_config,
                aggregation_config=self.aggregation_config,
                training_config=self.training_config,
            )

            self.performance_monitor.record_event(
                "cifar10_training_start",
                "Started HQDE training on CIFAR-10",
                {
                    "num_workers": self.num_workers,
                    "train_samples": len(train_loader.dataset),
                    "validation_samples": len(test_loader.dataset),
                    "num_epochs": num_epochs,
                },
            )

            logger.info("Starting notebook-style HQDE training")
            start_time = time.time()
            training_metrics = hqde_system.train(
                train_loader,
                num_epochs=num_epochs,
                validation_loader=test_loader,
            )
            training_time = time.time() - start_time

            for epoch_metrics in training_metrics.get("epoch_history", []):
                self.performance_monitor.record_training_metric(
                    "epoch_loss",
                    epoch_metrics["loss"],
                    epoch=epoch_metrics["epoch"],
                )
                self.performance_monitor.record_training_metric(
                    "epoch_accuracy",
                    epoch_metrics["accuracy"],
                    epoch=epoch_metrics["epoch"],
                )
                if "val_loss" in epoch_metrics:
                    self.performance_monitor.record_training_metric(
                        "val_loss",
                        epoch_metrics["val_loss"],
                        epoch=epoch_metrics["epoch"],
                    )
                if "val_accuracy" in epoch_metrics:
                    self.performance_monitor.record_training_metric(
                        "val_accuracy",
                        epoch_metrics["val_accuracy"],
                        epoch=epoch_metrics["epoch"],
                    )

            eval_metrics = hqde_system.evaluate(test_loader)

            logger.info("=== Training Results ===")
            logger.info("Training time: %.2f seconds", training_time)
            logger.info("Final validation accuracy: %.4f", eval_metrics["accuracy"])
            logger.info("Final validation loss: %.4f", eval_metrics["loss"])

            self.performance_monitor.record_training_metric("training_time", training_time)
            self.performance_monitor.record_training_metric("test_accuracy", eval_metrics["accuracy"])
            self.performance_monitor.record_training_metric("test_loss", eval_metrics["loss"])

            hqde_metrics = hqde_system.get_performance_metrics()
            model_path = "hqde_cifar10_model.pth"
            logger.info("Saving trained model to %s", model_path)
            hqde_system.save_model(model_path)

            self.performance_monitor.record_event(
                "cifar10_training_complete",
                "HQDE CIFAR-10 training completed successfully",
                {
                    "final_accuracy": eval_metrics["accuracy"],
                    "training_time": training_time,
                },
            )

            return {
                "training_time": training_time,
                "test_accuracy": eval_metrics["accuracy"],
                "test_loss": eval_metrics["loss"],
                "hqde_metrics": hqde_metrics,
            }
        except Exception as exc:
            logger.error("Error during HQDE CIFAR-10 training: %s", exc)
            self.performance_monitor.record_event(
                "cifar10_training_error",
                f"Training failed: {exc}",
            )
            raise
        finally:
            logger.info("Cleaning up resources")
            if hqde_system is not None:
                hqde_system.cleanup()
            self.performance_monitor.stop_monitoring()
            self._generate_performance_report()

    def _generate_performance_report(self):
        performance_report = self.performance_monitor.get_performance_report()
        current_metrics = performance_report.get("current_metrics", {})

        logger.info("=== Performance Report ===")
        logger.info("CPU Usage: %.1f%%", current_metrics.get("cpu_percent", 0.0))
        logger.info("Memory Usage: %.1f%%", current_metrics.get("memory_percent", 0.0))
        logger.info("GPU Memory: %.2f GB", current_metrics.get("gpu_memory_used_gb", 0.0))
        self.performance_monitor.export_metrics("hqde_cifar10_performance.json", format="json")
        self.performance_monitor.cleanup()


def main():
    """Run the benchmark-style CIFAR-10 example."""
    logger.info("Starting HQDE CIFAR-10 example")
    torch.manual_seed(42)
    np.random.seed(42)

    data_manager = CIFAR10DataManager(
        subset_size=None,
        batch_size=128,
        seed=42,
    )
    train_loader, test_loader = data_manager.load_data()

    trainer = CIFAR10HQDETrainer(num_workers=4)
    results = trainer.train_and_evaluate(train_loader, test_loader, num_epochs=20)

    logger.info("=== Final Results ===")
    logger.info("Validation Accuracy: %.4f (%.2f%%)", results["test_accuracy"], results["test_accuracy"] * 100.0)
    logger.info("Training Time: %.2f seconds", results["training_time"])
    logger.info("Generated files:")
    logger.info("  - hqde_cifar10_model.pth")
    logger.info("  - hqde_cifar10_performance.json")


if __name__ == "__main__":
    main()
