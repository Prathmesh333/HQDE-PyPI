"""
Single-node A100 experiment utilities for HQDE paper benchmarks.

This module implements a Ray-based scheduler with:
- logical workers: dataset shards / client identities
- GPU execution workers: long-lived trainers sharing one GPU

The design is intended for Colab-style environments where a single A100 is
available and spawning many full CUDA actors would cause contention.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from hqde import create_hqde_system
from hqde.core.hqde_system import AdaptiveQuantizer, QuantumInspiredAggregator

try:
    import ray
except ImportError:  # pragma: no cover - example utility
    ray = None


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


@dataclass
class DatasetSpec:
    name: str
    data_root: str = "./data"
    subset_size: Optional[int] = None
    train_batch_size: int = 128
    eval_batch_size: int = 256
    actor_dataloader_workers: int = 2


@dataclass
class ScheduledHQDEConfig:
    logical_workers: int = 12
    active_gpu_executors: int = 4
    gpu_fraction_per_executor: float = 0.25
    local_epochs: int = 1
    global_rounds: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    use_amp: bool = True
    compile_model: bool = False
    compile_mode: str = "default"
    aggregation_mode: str = "efficiency_weighted"
    use_quantization: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    partition_mode: str = "dirichlet"
    dirichlet_alpha: float = 0.5
    seed: int = 42
    num_cpus: int = 12


class SmallImageResNet18(nn.Module):
    """ResNet-18 adapted for 32x32 image classification."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.0):
        super().__init__()
        model = torchvision.models.resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        if dropout_rate > 0:
            model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(model.fc.in_features, num_classes))
        self.model = model

    def forward(self, x):
        return self.model(x)


def seed_everything(seed: int = 42):
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_transforms(name: str, is_training: bool):
    dataset_name = str(name).lower()
    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    elif dataset_name == "svhn":
        mean, std = SVHN_MEAN, SVHN_STD
    else:
        raise ValueError(f"Unsupported dataset '{name}'.")

    if is_training:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _load_dataset(dataset_name: str, data_root: str, train: bool, subset_size: Optional[int], seed: int):
    transform = _dataset_transforms(dataset_name, is_training=train)
    name = str(dataset_name).lower()
    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_root, train=train, download=True, transform=transform)
        num_classes = 10
    elif name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root=data_root, train=train, download=True, transform=transform)
        num_classes = 100
    elif name == "svhn":
        split = "train" if train else "test"
        dataset = torchvision.datasets.SVHN(root=data_root, split=split, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    if subset_size is not None and subset_size < len(dataset):
        rng = np.random.default_rng(seed + (0 if train else 1))
        indices = rng.choice(len(dataset), size=int(subset_size), replace=False).tolist()
        dataset = Subset(dataset, indices)

    return dataset, num_classes


def _extract_labels(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent_labels = _extract_labels(dataset.dataset)
        return parent_labels[np.asarray(dataset.indices)]

    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise ValueError("Dataset does not expose targets or labels.")
    return np.asarray(labels, dtype=np.int64)


def create_data_loaders(dataset_spec: DatasetSpec, seed: int = 42):
    """Create driver-side train and eval loaders."""
    train_dataset, num_classes = _load_dataset(
        dataset_spec.name,
        dataset_spec.data_root,
        train=True,
        subset_size=dataset_spec.subset_size,
        seed=seed,
    )
    test_dataset, _ = _load_dataset(
        dataset_spec.name,
        dataset_spec.data_root,
        train=False,
        subset_size=dataset_spec.subset_size,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_spec.train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_spec.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader, num_classes


def partition_indices(labels: np.ndarray, num_workers: int, mode: str = "dirichlet", alpha: float = 0.5, seed: int = 42):
    """Partition labels into logical worker shards."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    mode = str(mode).lower()

    if mode == "iid":
        indices = rng.permutation(len(labels))
        splits = np.array_split(indices, num_workers)
        return [split.tolist() for split in splits if len(split) > 0]

    if mode != "dirichlet":
        raise ValueError(f"Unsupported partition mode '{mode}'.")

    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == class_id)[0] for class_id in range(num_classes)]
    worker_indices: List[List[int]] = [[] for _ in range(num_workers)]

    for class_id in range(num_classes):
        class_id_indices = np.array(class_indices[class_id], copy=True)
        rng.shuffle(class_id_indices)
        proportions = rng.dirichlet(np.full(num_workers, alpha, dtype=np.float64))
        cut_points = (np.cumsum(proportions) * len(class_id_indices)).astype(int)[:-1]
        splits = np.split(class_id_indices, cut_points)
        for worker_id, split in enumerate(splits):
            worker_indices[worker_id].extend(split.tolist())

    for indices in worker_indices:
        rng.shuffle(indices)
    return worker_indices


class LogicalShardWorker:
    """CPU-side Ray actor storing shard assignments and metrics."""

    def __init__(self, worker_id: int, shard_indices: Sequence[int]):
        self.worker_id = int(worker_id)
        self.shard_indices = list(int(index) for index in shard_indices)
        self.history: List[Dict[str, Any]] = []

    def get_assignment(self) -> Dict[str, Any]:
        return {"worker_id": self.worker_id, "indices": self.shard_indices}

    def record_round(self, metrics: Dict[str, Any]):
        self.history.append(dict(metrics))
        return True

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.history)


class GPUExecutionWorker:
    """GPU-bound Ray actor that trains one logical worker shard at a time."""

    def __init__(self, dataset_name: str, data_root: str, subset_size: Optional[int], seed: int, actor_dataloader_workers: int = 2):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.seed = int(seed)
        self.actor_dataloader_workers = int(actor_dataloader_workers)
        self.train_dataset, self.num_classes = _load_dataset(
            dataset_name,
            data_root,
            train=True,
            subset_size=subset_size,
            seed=seed,
        )
        self.test_dataset, _ = _load_dataset(
            dataset_name,
            data_root,
            train=False,
            subset_size=subset_size,
            seed=seed,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SmallImageResNet18(num_classes=self.num_classes)
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

    def _create_loader(self, indices: Sequence[int], batch_size: int, shuffle: bool) -> DataLoader:
        subset = Subset(self.train_dataset, list(indices))
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.actor_dataloader_workers,
            pin_memory=True,
            persistent_workers=self.actor_dataloader_workers > 0,
        )

    def train_local(
        self,
        global_state: Dict[str, torch.Tensor],
        shard_indices: Sequence[int],
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        label_smoothing: float,
        use_amp: bool,
        compile_model: bool,
        compile_mode: str,
    ) -> Dict[str, Any]:
        self.model = SmallImageResNet18(num_classes=self.num_classes)
        self.model.load_state_dict(global_state, strict=True)
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode=str(compile_mode))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and self.device.type == "cuda"))
        loader = self._create_loader(shard_indices, batch_size=batch_size, shuffle=True)

        total_samples = 0
        total_correct = 0
        total_loss = 0.0
        self.model.train()

        for _ in range(int(local_epochs)):
            for data_batch, targets in loader:
                data_batch = data_batch.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.device.type == "cuda" and data_batch.ndim == 4:
                    data_batch = data_batch.contiguous(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    outputs = self.model(data_batch)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    predictions = outputs.argmax(dim=1)
                    batch_samples = int(targets.size(0))
                    total_samples += batch_samples
                    total_correct += int((predictions == targets).sum().item())
                    total_loss += float(loss.item()) * batch_samples

        avg_loss = total_loss / max(total_samples, 1)
        avg_accuracy = total_correct / max(total_samples, 1)
        efficiency_score = 0.9 + 0.1 * max(avg_accuracy, 1.0 / (1.0 + avg_loss))

        return {
            "worker_state": {
                name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()
            },
            "num_samples": total_samples,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "efficiency_score": efficiency_score,
        }

    def evaluate(self, global_state: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, float]:
        self.model = SmallImageResNet18(num_classes=self.num_classes)
        self.model.load_state_dict(global_state, strict=True)
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

        loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.actor_dataloader_workers,
            pin_memory=True,
            persistent_workers=self.actor_dataloader_workers > 0,
        )
        criterion = nn.CrossEntropyLoss()
        total_samples = 0
        total_correct = 0
        total_loss = 0.0

        with torch.no_grad():
            for data_batch, targets in loader:
                data_batch = data_batch.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.device.type == "cuda" and data_batch.ndim == 4:
                    data_batch = data_batch.contiguous(memory_format=torch.channels_last)
                outputs = self.model(data_batch)
                loss = criterion(outputs, targets)
                predictions = outputs.argmax(dim=1)
                batch_samples = int(targets.size(0))
                total_samples += batch_samples
                total_correct += int((predictions == targets).sum().item())
                total_loss += float(loss.item()) * batch_samples

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }


def build_initial_state(num_classes: int) -> Dict[str, torch.Tensor]:
    """Create a fresh global model state."""
    model = SmallImageResNet18(num_classes=num_classes)
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _weighted_mean(tensors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    denominator = float(sum(weights)) or 1.0
    result = None
    for tensor, weight in zip(tensors, weights):
        contribution = tensor.float() * float(weight / denominator)
        result = contribution if result is None else result + contribution
    return result if result is not None else torch.tensor(0.0)


def aggregate_worker_states(
    worker_results: Sequence[Dict[str, Any]],
    aggregation_mode: str = "efficiency_weighted",
    use_quantization: bool = False,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Aggregate worker states using HQDE-style weighting and optional quantization."""
    if not worker_results:
        raise ValueError("worker_results must not be empty.")

    aggregator = QuantumInspiredAggregator()
    quantizer = AdaptiveQuantizer(**(quantization_config or {})) if use_quantization else None

    aggregated_state: Dict[str, torch.Tensor] = {}
    sample_weights = [float(result["num_samples"]) for result in worker_results]
    efficiency_scores = [
        float(result["efficiency_score"]) * max(float(result["num_samples"]), 1.0)
        for result in worker_results
    ]
    compression_ratios: List[float] = []
    first_state = worker_results[0]["worker_state"]

    for name, reference_tensor in first_state.items():
        state_tensors = [result["worker_state"][name] for result in worker_results]
        if not torch.is_floating_point(reference_tensor):
            aggregated_state[name] = state_tensors[0].clone()
            continue

        prepared_tensors = []
        for tensor in state_tensors:
            prepared = tensor.float()
            if quantizer is not None:
                importance = quantizer.compute_importance_score(prepared)
                prepared, metadata = quantizer.adaptive_quantize(prepared, importance)
                compression_ratios.append(float(metadata.get("compression_ratio", 1.0)))
            prepared_tensors.append(prepared)

        if str(aggregation_mode).lower() == "mean":
            aggregated_tensor = _weighted_mean(prepared_tensors, sample_weights)
        else:
            aggregated_tensor = aggregator.efficiency_weighted_aggregation(prepared_tensors, efficiency_scores).cpu()
        aggregated_state[name] = aggregated_tensor.to(dtype=reference_tensor.dtype)

    return aggregated_state, {
        "compression_ratio": float(np.mean(compression_ratios)) if compression_ratios else 1.0,
        "avg_worker_accuracy": float(np.mean([result["accuracy"] for result in worker_results])),
        "avg_worker_loss": float(np.mean([result["loss"] for result in worker_results])),
    }


def _ensure_ray(config: ScheduledHQDEConfig):
    if ray is None:
        raise ImportError("Ray is required for the scheduled single-node experiment notebook.")
    if not ray.is_initialized():
        available_gpus = 1 if torch.cuda.is_available() else 0
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            num_cpus=int(config.num_cpus),
            num_gpus=available_gpus,
        )


def run_scheduled_hqde_experiment(dataset_spec: DatasetSpec, experiment_config: ScheduledHQDEConfig):
    """
    Run the single-node scheduled HQDE experiment.

    Returns a dictionary with the round history and final evaluation metrics.
    """
    seed_everything(experiment_config.seed)
    _ensure_ray(experiment_config)

    train_dataset, _, _, _, num_classes = create_data_loaders(dataset_spec, seed=experiment_config.seed)
    labels = _extract_labels(train_dataset)
    partitions = partition_indices(
        labels,
        num_workers=experiment_config.logical_workers,
        mode=experiment_config.partition_mode,
        alpha=experiment_config.dirichlet_alpha,
        seed=experiment_config.seed,
    )

    logical_actor_cls = ray.remote(num_cpus=0.1)(LogicalShardWorker)
    gpu_resources = float(experiment_config.gpu_fraction_per_executor) if torch.cuda.is_available() else 0.0
    gpu_actor_cls = ray.remote(num_gpus=gpu_resources)(GPUExecutionWorker)

    logical_workers = [
        logical_actor_cls.remote(worker_id=worker_id, shard_indices=indices)
        for worker_id, indices in enumerate(partitions)
    ]
    gpu_workers = [
        gpu_actor_cls.remote(
            dataset_name=dataset_spec.name,
            data_root=dataset_spec.data_root,
            subset_size=dataset_spec.subset_size,
            seed=experiment_config.seed + worker_id,
            actor_dataloader_workers=dataset_spec.actor_dataloader_workers,
        )
        for worker_id in range(experiment_config.active_gpu_executors)
    ]

    global_state = build_initial_state(num_classes)
    round_history: List[Dict[str, Any]] = []
    logical_assignments = ray.get([worker.get_assignment.remote() for worker in logical_workers])

    experiment_started = time.time()
    for round_idx in range(experiment_config.global_rounds):
        round_started = time.time()
        global_state_ref = ray.put(global_state)
        worker_results: List[Dict[str, Any]] = []

        for wave_start in range(0, len(logical_assignments), experiment_config.active_gpu_executors):
            wave_assignments = logical_assignments[wave_start: wave_start + experiment_config.active_gpu_executors]
            wave_futures = []
            for gpu_slot, assignment in enumerate(wave_assignments):
                wave_futures.append(
                    gpu_workers[gpu_slot].train_local.remote(
                        global_state_ref,
                        assignment["indices"],
                        experiment_config.local_epochs,
                        dataset_spec.train_batch_size,
                        experiment_config.learning_rate,
                        experiment_config.weight_decay,
                        experiment_config.label_smoothing,
                        experiment_config.use_amp,
                        experiment_config.compile_model,
                        experiment_config.compile_mode,
                    )
                )
            wave_results = ray.get(wave_futures)
            worker_results.extend(wave_results)
            for assignment, result in zip(wave_assignments, wave_results):
                logical_workers[assignment["worker_id"]].record_round.remote(
                    {
                        "round": round_idx + 1,
                        "loss": result["loss"],
                        "accuracy": result["accuracy"],
                        "num_samples": result["num_samples"],
                    }
                )

        global_state, aggregation_metrics = aggregate_worker_states(
            worker_results,
            aggregation_mode=experiment_config.aggregation_mode,
            use_quantization=experiment_config.use_quantization,
            quantization_config=experiment_config.quantization_config,
        )
        evaluation = ray.get(gpu_workers[0].evaluate.remote(ray.put(global_state), dataset_spec.eval_batch_size))
        round_history.append(
            {
                "round": round_idx + 1,
                "eval_accuracy": float(evaluation["accuracy"]),
                "eval_loss": float(evaluation["loss"]),
                "avg_worker_accuracy": float(aggregation_metrics["avg_worker_accuracy"]),
                "avg_worker_loss": float(aggregation_metrics["avg_worker_loss"]),
                "compression_ratio": float(aggregation_metrics["compression_ratio"]),
                "round_time_sec": time.time() - round_started,
            }
        )

    histories = ray.get([worker.get_history.remote() for worker in logical_workers])
    return {
        "dataset": dataset_spec.name,
        "num_classes": num_classes,
        "config": asdict(experiment_config),
        "dataset_config": asdict(dataset_spec),
        "final_accuracy": round_history[-1]["eval_accuracy"] if round_history else 0.0,
        "final_loss": round_history[-1]["eval_loss"] if round_history else 0.0,
        "training_time_sec": time.time() - experiment_started,
        "round_history": round_history,
        "logical_worker_histories": histories,
    }


def run_standard_hqde_baseline(
    dataset_spec: DatasetSpec,
    num_epochs: int = 20,
    num_workers: int = 4,
    training_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
):
    """Run the current built-in HQDE trainer as a baseline."""
    seed_everything(seed)
    _, test_dataset, train_loader, test_loader, num_classes = create_data_loaders(dataset_spec, seed=seed)
    system = create_hqde_system(
        model_class=SmallImageResNet18,
        model_kwargs={"num_classes": num_classes},
        num_workers=num_workers,
        training_config=training_config
        or {
            "ensemble_mode": "independent",
            "batch_assignment": "replicate",
            "optimizer": "adamw",
            "use_amp": True,
            "label_smoothing": 0.1,
            "warmup_epochs": 2,
            "prediction_aggregation": "efficiency_weighted",
        },
    )
    try:
        started = time.time()
        metrics = system.train(train_loader, num_epochs=num_epochs)
        predictions = system.predict(test_loader)

        targets = []
        for _, batch_targets in test_loader:
            targets.append(batch_targets)
        target_tensor = torch.cat(targets, dim=0)
        predicted_classes = predictions.argmax(dim=1)
        accuracy = float((predicted_classes == target_tensor).float().mean().item())

        return {
            "dataset": dataset_spec.name,
            "num_workers": num_workers,
            "num_classes": num_classes,
            "final_accuracy": accuracy,
            "training_time_sec": time.time() - started,
            "metrics": copy.deepcopy(metrics),
        }
    finally:
        system.cleanup()


def save_experiment_results(results: Dict[str, Any], output_path: str):
    """Save a result dictionary as JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def summarize_round_history(round_history: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a compact history view suitable for DataFrame display."""
    summary = []
    for item in round_history:
        summary.append(
            {
                "round": item["round"],
                "eval_accuracy": round(item["eval_accuracy"], 4),
                "eval_loss": round(item["eval_loss"], 4),
                "avg_worker_accuracy": round(item["avg_worker_accuracy"], 4),
                "compression_ratio": round(item["compression_ratio"], 3),
                "round_time_sec": round(item["round_time_sec"], 2),
            }
        )
    return summary
