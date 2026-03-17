"""
HQDE (Hierarchical Quantum-Distributed Ensemble Learning) Core System.

This module implements the main HQDE framework with a practical ensemble
training loop, prediction aggregation, and checkpoint handling.
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Try to import optional dependencies for notebook compatibility
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

if not RAY_AVAILABLE:
    LOGGER.warning("Ray is not available. HQDE will run in local worker mode.")
if not PSUTIL_AVAILABLE:
    LOGGER.warning("psutil is not available. Memory monitoring will be disabled.")

DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "learning_rate": 1e-3,
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "min_learning_rate": 1e-6,
    "warmup_epochs": 0,
    "warmup_start_factor": 0.1,
    "label_smoothing": 0.0,
    "gradient_clip_norm": 1.0,
    "use_amp": True,
    "compile_model": False,
    "compile_mode": "default",
    "ensemble_mode": "independent",
    "batch_assignment": "replicate",
    "prediction_aggregation": "efficiency_weighted",
}

VALID_OPTIMIZERS = {"sgd", "adam", "adamw"}
VALID_ENSEMBLE_MODES = {"independent", "fedavg"}
VALID_BATCH_ASSIGNMENTS = {"replicate", "split"}
VALID_PREDICTION_AGGREGATIONS = {"efficiency_weighted", "mean"}
VALID_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune"}


def validate_training_config(training_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize and validate HQDE training configuration.

    Examples:
        True ensemble training:
            {
                "ensemble_mode": "independent",
                "batch_assignment": "replicate",
                "optimizer": "adamw",
                "use_amp": True,
            }

        Epoch-wise FedAvg/local-SGD style training:
            {
                "ensemble_mode": "fedavg",
                "batch_assignment": "split",
                "warmup_epochs": 2,
            }

    Supported keys:
        learning_rate: positive float optimizer learning rate.
        optimizer: one of {"sgd", "adam", "adamw"}.
        weight_decay: non-negative float regularization coefficient.
        min_learning_rate: non-negative cosine-scheduler floor.
        warmup_epochs: non-negative integer number of warmup epochs.
        warmup_start_factor: float in (0, 1] for LinearLR warmup start.
        label_smoothing: float in [0, 1).
        gradient_clip_norm: non-negative float clip threshold, 0 disables clipping.
        use_amp: bool enabling CUDA mixed precision.
        compile_model: bool enabling torch.compile when available.
        compile_mode: one of {"default", "reduce-overhead", "max-autotune"}.
        ensemble_mode: "independent" for diverse ensembles or "fedavg" for epoch-wise averaging.
        batch_assignment: "replicate" for true ensemble training or "split" for local-SGD style batches.
        prediction_aggregation: "efficiency_weighted" or "mean".
    """
    config = dict(DEFAULT_TRAINING_CONFIG)
    config.update(training_config or {})

    config["optimizer"] = str(config["optimizer"]).lower()
    config["ensemble_mode"] = str(config["ensemble_mode"]).lower()
    config["batch_assignment"] = str(config["batch_assignment"]).lower()
    config["prediction_aggregation"] = str(config["prediction_aggregation"]).lower()

    if config["optimizer"] not in VALID_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{config['optimizer']}'. Expected one of {sorted(VALID_OPTIMIZERS)}."
        )
    if config["ensemble_mode"] not in VALID_ENSEMBLE_MODES:
        raise ValueError(
            f"Unsupported ensemble_mode '{config['ensemble_mode']}'. Expected one of {sorted(VALID_ENSEMBLE_MODES)}."
        )
    if config["batch_assignment"] not in VALID_BATCH_ASSIGNMENTS:
        raise ValueError(
            "Unsupported batch_assignment "
            f"'{config['batch_assignment']}'. Expected one of {sorted(VALID_BATCH_ASSIGNMENTS)}."
        )
    if config["prediction_aggregation"] not in VALID_PREDICTION_AGGREGATIONS:
        raise ValueError(
            "Unsupported prediction_aggregation "
            f"'{config['prediction_aggregation']}'. Expected one of {sorted(VALID_PREDICTION_AGGREGATIONS)}."
        )
    config["compile_mode"] = str(config["compile_mode"]).lower()
    if config["compile_mode"] not in VALID_COMPILE_MODES:
        raise ValueError(
            f"Unsupported compile_mode '{config['compile_mode']}'. Expected one of {sorted(VALID_COMPILE_MODES)}."
        )

    numeric_constraints = {
        "learning_rate": (0.0, False),
        "weight_decay": (0.0, True),
        "min_learning_rate": (0.0, True),
        "gradient_clip_norm": (0.0, True),
        "warmup_start_factor": (0.0, False),
    }
    for key, (minimum, inclusive) in numeric_constraints.items():
        value = float(config[key])
        if inclusive:
            if value < minimum:
                raise ValueError(f"{key} must be >= {minimum}, got {value}.")
        elif value <= minimum:
            raise ValueError(f"{key} must be > {minimum}, got {value}.")
        config[key] = value

    config["label_smoothing"] = float(config["label_smoothing"])
    if not 0.0 <= config["label_smoothing"] < 1.0:
        raise ValueError(f"label_smoothing must be in [0.0, 1.0), got {config['label_smoothing']}.")

    config["warmup_epochs"] = int(config["warmup_epochs"])
    if config["warmup_epochs"] < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {config['warmup_epochs']}.")
    if config["warmup_start_factor"] > 1.0:
        raise ValueError(
            f"warmup_start_factor must be <= 1.0, got {config['warmup_start_factor']}."
        )

    config["use_amp"] = bool(config["use_amp"])
    config["compile_model"] = bool(config["compile_model"])
    return config

class AdaptiveQuantizer:
    """Adaptive weight quantization based on real-time importance scoring."""

    def __init__(self, base_bits: int = 8, min_bits: int = 4, max_bits: int = 16):
        self.base_bits = base_bits
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.compression_cache = {}

    def compute_importance_score(self, weights: torch.Tensor, gradients: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute importance scores based on gradient magnitude and weight variance."""
        with torch.no_grad():
            # Weight-based importance
            weight_importance = torch.abs(weights)

            # Gradient-based importance if available
            if gradients is not None:
                grad_importance = torch.abs(gradients)
                combined_importance = 0.7 * weight_importance + 0.3 * grad_importance
            else:
                combined_importance = weight_importance

            # Normalize to [0, 1]
            if combined_importance.numel() > 0:
                min_val = combined_importance.min()
                max_val = combined_importance.max()
                if max_val > min_val:
                    importance = (combined_importance - min_val) / (max_val - min_val)
                else:
                    importance = torch.ones_like(combined_importance) * 0.5
            else:
                importance = torch.ones_like(combined_importance) * 0.5

        return importance

    def adaptive_quantize(self, weights: torch.Tensor, importance_score: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform adaptive quantization based on importance scores."""
        # Determine bits per parameter based on importance
        bits_per_param = self.min_bits + (self.max_bits - self.min_bits) * importance_score
        bits_per_param = torch.clamp(bits_per_param, self.min_bits, self.max_bits).int()

        # For simplicity, use uniform quantization with average bits
        avg_bits = int(bits_per_param.float().mean().item())

        # Quantize weights
        weight_min = weights.min()
        weight_max = weights.max()

        if weight_max > weight_min:
            scale = (weight_max - weight_min) / (2**avg_bits - 1)
            zero_point = weight_min

            quantized = torch.round((weights - zero_point) / scale)
            quantized = torch.clamp(quantized, 0, 2**avg_bits - 1)

            # Dequantize for use
            dequantized = quantized * scale + zero_point
        else:
            dequantized = weights.clone()
            scale = torch.tensor(1.0)
            zero_point = torch.tensor(0.0)

        metadata = {
            'scale': scale,
            'zero_point': zero_point,
            'avg_bits': avg_bits,
            'compression_ratio': 32.0 / avg_bits  # Assuming original is float32
        }

        return dequantized, metadata

class QuantumInspiredAggregator:
    """Quantum-inspired ensemble aggregation with controlled noise injection."""

    def __init__(self, noise_scale: float = 0.01, exploration_factor: float = 0.1):
        self.noise_scale = noise_scale
        self.exploration_factor = exploration_factor

    def quantum_noise_injection(self, weights: torch.Tensor) -> torch.Tensor:
        """Add quantum-inspired noise for exploration."""
        noise = torch.randn_like(weights) * self.noise_scale
        return weights + noise

    def efficiency_weighted_aggregation(self, weight_list: List[torch.Tensor],
                                      efficiency_scores: List[float]) -> torch.Tensor:
        """Aggregate weights using efficiency-based weighting."""
        if not weight_list or not efficiency_scores:
            raise ValueError("Empty weight list or efficiency scores")

        if len(efficiency_scores) != len(weight_list):
            efficiency_scores = [1.0] * len(weight_list)

        reference_device = weight_list[0].device
        stacked = torch.stack([tensor.to(reference_device) for tensor in weight_list], dim=0)
        weights = torch.tensor(efficiency_scores, dtype=torch.float32, device=reference_device)
        weights = torch.softmax(weights, dim=0)
        reshape_dims = (weights.shape[0],) + (1,) * (stacked.dim() - 1)
        return (stacked * weights.view(reshape_dims)).sum(dim=0)

class _EnsembleWorkerBase:
    """Single ensemble member used locally or behind a Ray actor."""

    def __init__(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        worker_id: int = 0,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.15,
        training_config: Optional[Dict[str, Any]] = None,
    ):
        self.training_config = validate_training_config(training_config)

        model_init_params = inspect.signature(model_class.__init__).parameters
        supports_dropout = "dropout_rate" in model_init_params
        worker_model_kwargs = dict(model_kwargs)
        if supports_dropout and "dropout_rate" not in worker_model_kwargs:
            worker_model_kwargs["dropout_rate"] = dropout_rate

        self.model = model_class(**worker_model_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.training_config.get("compile_model", False) and hasattr(torch, "compile"):
            self.model = torch.compile(
                self.model,
                mode=str(self.training_config.get("compile_mode", "default")),
            )

        self.efficiency_score = 1.0
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.learning_rate = learning_rate
        self.worker_id = worker_id

    def setup_training(self, learning_rate: Optional[float] = None, total_epochs: int = 1):
        """Setup optimizer, scheduler, and criterion for training."""
        learning_rate = learning_rate or self.learning_rate
        optimizer_name = str(self.training_config.get("optimizer", "adamw")).lower()
        weight_decay = float(self.training_config.get("weight_decay", 1e-4))

        if optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        warmup_epochs = min(int(self.training_config.get("warmup_epochs", 0)), max(int(total_epochs), 1))
        cosine_epochs = max(int(total_epochs) - warmup_epochs, 1)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=float(self.training_config.get("min_learning_rate", 1e-6)),
        )
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=float(self.training_config.get("warmup_start_factor", 0.1)),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = cosine_scheduler
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(self.training_config.get("label_smoothing", 0.0))
        )

        use_amp = bool(self.training_config.get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
        return True

    def step_scheduler(self) -> float:
        """Step the learning rate scheduler once per epoch."""
        if self.scheduler is not None and self.optimizer is not None:
            self.scheduler.step()
            return float(self.optimizer.param_groups[0]["lr"])
        return float(self.learning_rate)

    def train_step(self, data_batch, targets=None):
        """Run a single training step and return batch metrics."""
        if data_batch is None or targets is None:
            return None
        if self.optimizer is None or self.criterion is None:
            raise RuntimeError("Worker training has not been initialized. Call setup_training first.")

        self.model.train()
        data_batch = data_batch.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        if self.device.type == "cuda" and data_batch.ndim == 4:
            data_batch = data_batch.contiguous(memory_format=torch.channels_last)

        amp_enabled = self.scaler is not None and self.device.type == "cuda"
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if amp_enabled
            else nullcontext()
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast_context:
            outputs = self.model(data_batch)
            loss = self.criterion(outputs, targets)

        clip_norm = float(self.training_config.get("gradient_clip_norm", 1.0))
        if amp_enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            self.optimizer.step()

        with torch.no_grad():
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == targets).float().mean().item()
            loss_value = float(loss.item())
            self.efficiency_score = 0.9 * self.efficiency_score + 0.1 * max(
                accuracy,
                1.0 / (1.0 + loss_value),
            )

        return {
            "loss": loss_value,
            "accuracy": float(accuracy),
            "num_samples": int(targets.size(0)),
        }

    def get_weights(self):
        """Return a CPU copy of the full model state."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def set_weights(self, weights_dict):
        """Load a state dict into the worker model."""
        current_state = self.model.state_dict()
        updated_state = {}
        for name, tensor in current_state.items():
            updated_state[name] = weights_dict.get(name, tensor).to(self.device)
        self.model.load_state_dict(updated_state, strict=False)
        return True

    def get_efficiency_score(self):
        """Return the current worker efficiency score."""
        return float(self.efficiency_score)

    def predict(self, data_batch):
        """Make predictions on data batch."""
        self.model.eval()
        data_batch = data_batch.to(self.device, non_blocking=True)
        if self.device.type == "cuda" and data_batch.ndim == 4:
            data_batch = data_batch.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            outputs = self.model(data_batch)
        return outputs.detach().cpu()

    def get_checkpoint(self):
        """Return a serializable worker checkpoint."""
        return {
            "worker_id": self.worker_id,
            "learning_rate": self.learning_rate,
            "efficiency_score": self.efficiency_score,
            "model_state_dict": self.get_weights(),
        }

    def load_checkpoint(self, checkpoint):
        """Load a worker checkpoint."""
        if "model_state_dict" in checkpoint:
            self.set_weights(checkpoint["model_state_dict"])
        self.efficiency_score = float(checkpoint.get("efficiency_score", self.efficiency_score))
        self.learning_rate = float(checkpoint.get("learning_rate", self.learning_rate))
        return True


if RAY_AVAILABLE:

    @ray.remote
    class RayEnsembleWorker(_EnsembleWorkerBase):
        """Ray actor wrapper around the local ensemble worker implementation."""

        pass

else:
    RayEnsembleWorker = None


class DistributedEnsembleManager:
    """Manage distributed or local ensemble workers."""

    def __init__(
        self,
        num_workers: int = 4,
        training_config: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
    ):
        self.num_workers = num_workers
        self.workers = []
        self.training_config = validate_training_config(training_config)
        self.quantizer = (
            AdaptiveQuantizer(**(quantization_config or {}))
            if self.training_config["ensemble_mode"] == "fedavg"
            else None
        )
        self.aggregator = QuantumInspiredAggregator()
        self.last_compression_ratio = 1.0
        self.use_ray = bool(RAY_AVAILABLE)
        self.logger = logging.getLogger(__name__)

        if self.use_ray and ray is not None and not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _worker_call(self, worker, method_name: str, *args, **kwargs):
        method = getattr(worker, method_name)
        if self.use_ray:
            return method.remote(*args, **kwargs)
        return method(*args, **kwargs)

    def _resolve(self, values):
        if self.use_ray:
            return ray.get(values)
        return values

    def _build_worker_batches(self, data: torch.Tensor, targets: torch.Tensor):
        batch_assignment = str(self.training_config.get("batch_assignment", "replicate")).lower()
        if batch_assignment == "split":
            data_chunks = torch.tensor_split(data, self.num_workers, dim=0)
            target_chunks = torch.tensor_split(targets, self.num_workers, dim=0)
            return [
                (data_chunk, target_chunk)
                for data_chunk, target_chunk in zip(data_chunks, target_chunks)
                if data_chunk.size(0) > 0
            ]
        return [(data, targets) for _ in range(self.num_workers)]

    def create_ensemble_workers(self, model_class, model_kwargs: Dict[str, Any]):
        """Create ensemble workers with small learning-rate and dropout variation."""
        base_lr = float(self.training_config.get("learning_rate", 1e-3))
        lr_multipliers = [1.0, 0.85, 1.15, 0.95]
        dropout_rates = [0.10, 0.12, 0.15, 0.13]

        while len(lr_multipliers) < self.num_workers:
            lr_multipliers.append(1.0)
            dropout_rates.append(0.12)

        self.workers = []
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_per_worker = (gpu_count / self.num_workers) if gpu_count > 0 else 0

        for worker_id in range(self.num_workers):
            worker_kwargs = {
                "model_class": model_class,
                "model_kwargs": dict(model_kwargs),
                "worker_id": worker_id,
                "learning_rate": base_lr * lr_multipliers[worker_id],
                "dropout_rate": dropout_rates[worker_id],
                "training_config": self.training_config,
            }
            if self.use_ray and RayEnsembleWorker is not None:
                worker = RayEnsembleWorker.options(num_gpus=gpu_per_worker).remote(**worker_kwargs)
            else:
                worker = _EnsembleWorkerBase(**worker_kwargs)
            self.workers.append(worker)

    def setup_workers_training(self, learning_rate: Optional[float] = None, total_epochs: int = 1):
        """Setup training for all workers."""
        learning_rate = learning_rate or float(self.training_config.get("learning_rate", 1e-3))
        setup_calls = [
            self._worker_call(worker, "setup_training", learning_rate, total_epochs)
            for worker in self.workers
        ]
        self._resolve(setup_calls)
        self.logger.info("Training setup completed for %s workers", self.num_workers)

    def aggregate_weights(self) -> Dict[str, torch.Tensor]:
        """Aggregate model states from all workers."""
        all_weights = self._resolve([self._worker_call(worker, "get_weights") for worker in self.workers])
        efficiency_scores = self.get_efficiency_scores()
        if not all_weights:
            return {}

        aggregated_weights = {}
        compression_ratios = []
        for param_name in all_weights[0].keys():
            param_tensors = [weights[param_name] for weights in all_weights]
            if self.quantizer is not None:
                quantized_tensors = []
                for tensor in param_tensors:
                    importance = self.quantizer.compute_importance_score(tensor)
                    dequantized, metadata = self.quantizer.adaptive_quantize(tensor, importance)
                    quantized_tensors.append(dequantized)
                    compression_ratios.append(float(metadata.get("compression_ratio", 1.0)))
                param_tensors = quantized_tensors
            aggregated_weights[param_name] = self.aggregator.efficiency_weighted_aggregation(
                param_tensors,
                efficiency_scores,
            )
        self.last_compression_ratio = (
            float(np.mean(compression_ratios)) if compression_ratios else 1.0
        )
        return aggregated_weights

    def broadcast_weights(self, weights: Dict[str, torch.Tensor]):
        """Broadcast weights to every worker."""
        calls = [self._worker_call(worker, "set_weights", weights) for worker in self.workers]
        self._resolve(calls)

    def get_efficiency_scores(self) -> List[float]:
        """Collect efficiency scores for all workers."""
        if not self.workers:
            return []
        return self._resolve([self._worker_call(worker, "get_efficiency_score") for worker in self.workers])

    def get_worker_checkpoints(self):
        """Collect per-worker checkpoints."""
        return self._resolve([self._worker_call(worker, "get_checkpoint") for worker in self.workers])

    def get_quantization_metrics(self) -> Dict[str, float]:
        """Return communication quantization metrics for aggregation-enabled modes."""
        return {"compression_ratio": float(self.last_compression_ratio)}

    def load_worker_checkpoints(self, checkpoints):
        """Restore workers from per-worker checkpoints."""
        if not checkpoints:
            return
        calls = []
        for worker, checkpoint in zip(self.workers, checkpoints):
            calls.append(self._worker_call(worker, "load_checkpoint", checkpoint))
        self._resolve(calls)

    def train_ensemble(self, data_loader, num_epochs: int = 10):
        """Train the ensemble using distributed or local workers."""
        self.setup_workers_training(total_epochs=num_epochs)
        epoch_history = []
        ensemble_mode = str(self.training_config.get("ensemble_mode", "independent")).lower()

        for epoch in range(num_epochs):
            epoch_loss_sum = 0.0
            epoch_accuracy_sum = 0.0
            epoch_samples = 0

            for batch in data_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise ValueError("Training batches must contain at least (data, targets)")

                data, targets = batch[0], batch[1]
                worker_batches = self._build_worker_batches(data, targets)
                active_workers = self.workers[: len(worker_batches)]
                calls = [
                    self._worker_call(worker, "train_step", worker_data, worker_targets)
                    for worker, (worker_data, worker_targets) in zip(active_workers, worker_batches)
                ]
                batch_results = self._resolve(calls)

                for result in batch_results:
                    if not result:
                        continue
                    num_samples = int(result["num_samples"])
                    epoch_loss_sum += float(result["loss"]) * num_samples
                    epoch_accuracy_sum += float(result["accuracy"]) * num_samples
                    epoch_samples += num_samples

            if ensemble_mode == "fedavg":
                aggregated_weights = self.aggregate_weights()
                if aggregated_weights:
                    self.broadcast_weights(aggregated_weights)

            current_lrs = self._resolve([self._worker_call(worker, "step_scheduler") for worker in self.workers])
            avg_lr = float(np.mean(current_lrs)) if current_lrs else float(
                self.training_config.get("learning_rate", 1e-3)
            )
            avg_loss = epoch_loss_sum / epoch_samples if epoch_samples else 0.0
            avg_accuracy = epoch_accuracy_sum / epoch_samples if epoch_samples else 0.0

            epoch_metrics = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "accuracy": avg_accuracy,
                "learning_rate": avg_lr,
            }
            epoch_history.append(epoch_metrics)
            self.logger.info(
                "Epoch %s/%s - loss: %.4f - accuracy: %.4f - lr: %.6f",
                epoch + 1,
                num_epochs,
                avg_loss,
                avg_accuracy,
                avg_lr,
            )

        return epoch_history

    def shutdown(self):
        """Shutdown the distributed ensemble manager."""
        self.workers = []
        if self.use_ray and ray is not None and ray.is_initialized():
            ray.shutdown()

class HQDESystem:
    """Main HQDE (Hierarchical Quantum-Distributed Ensemble Learning) system."""

    def __init__(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        num_workers: int = 4,
        quantization_config: Optional[Dict[str, Any]] = None,
        aggregation_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the HQDE system.

        Args:
            model_class: PyTorch module class used for each ensemble member.
            model_kwargs: Keyword arguments used to initialize `model_class`.
            num_workers: Number of ensemble members to create.
            quantization_config: Optional AdaptiveQuantizer configuration.
            aggregation_config: Optional QuantumInspiredAggregator configuration.
            training_config: Optional training configuration. See
                `validate_training_config` for supported keys and accepted values.
        """
        self.model_class = model_class
        self.model_kwargs = dict(model_kwargs)
        self.num_workers = num_workers
        self.training_config = validate_training_config(training_config)
        self.quantization_config = dict(quantization_config or {}) if quantization_config is not None else None
        self.aggregator = QuantumInspiredAggregator(**(aggregation_config or {}))
        self.ensemble_manager = DistributedEnsembleManager(
            num_workers=num_workers,
            training_config=self.training_config,
            quantization_config=self.quantization_config,
        )
        self.quantizer = self.ensemble_manager.quantizer

        self.metrics: Dict[str, Any] = {
            "training_time": 0.0,
            "communication_overhead": 0.0,
            "memory_usage": 0.0,
            "compression_ratio": 1.0,
            "epoch_history": [],
        }
        self.logger = logging.getLogger(__name__)

    def initialize_ensemble(self):
        """Initialize the ensemble workers."""
        self.logger.info("Initializing HQDE ensemble with %s workers", self.num_workers)
        self.ensemble_manager.create_ensemble_workers(self.model_class, self.model_kwargs)

    def train(self, data_loader, num_epochs: int = 10, validation_loader=None):
        """Train the HQDE ensemble."""
        del validation_loader
        start_time = time.time()
        initial_memory = (
            psutil.Process().memory_info().rss / 1024 / 1024
            if PSUTIL_AVAILABLE and psutil is not None
            else 0
        )

        self.logger.info("Starting HQDE training for %s epochs", num_epochs)
        epoch_history = self.ensemble_manager.train_ensemble(data_loader, num_epochs)

        end_time = time.time()
        final_memory = (
            psutil.Process().memory_info().rss / 1024 / 1024
            if PSUTIL_AVAILABLE and psutil is not None
            else 0
        )

        self.metrics.update(
            {
                "training_time": end_time - start_time,
                "memory_usage": final_memory - initial_memory,
                "epoch_history": epoch_history,
                "compression_ratio": self.ensemble_manager.get_quantization_metrics()["compression_ratio"],
            }
        )
        if epoch_history:
            self.metrics["final_loss"] = epoch_history[-1]["loss"]
            self.metrics["final_accuracy"] = epoch_history[-1]["accuracy"]

        self.logger.info("HQDE training completed in %.2f seconds", self.metrics["training_time"])
        self.logger.info("Memory usage delta: %.2f MB", self.metrics["memory_usage"])
        return self.metrics.copy()

    def predict(self, data_loader):
        """Make predictions using the trained ensemble."""
        if not self.ensemble_manager.workers:
            self.logger.warning("No workers available for prediction")
            return torch.empty(0)

        predictions = []
        aggregation_mode = str(self.training_config.get("prediction_aggregation", "efficiency_weighted")).lower()

        try:
            efficiency_scores = self.ensemble_manager.get_efficiency_scores()
            for batch in data_loader:
                data = batch[0] if isinstance(batch, (list, tuple)) else batch
                prediction_calls = [
                    self.ensemble_manager._worker_call(worker, "predict", data)
                    for worker in self.ensemble_manager.workers
                ]
                worker_predictions = self.ensemble_manager._resolve(prediction_calls)
                if not worker_predictions:
                    continue

                if aggregation_mode == "mean":
                    ensemble_prediction = torch.stack(worker_predictions, dim=0).mean(dim=0)
                else:
                    ensemble_prediction = self.aggregator.efficiency_weighted_aggregation(
                        worker_predictions,
                        efficiency_scores,
                    )
                predictions.append(ensemble_prediction.cpu())
        except Exception as exc:
            self.logger.error("Prediction failed: %s", exc)
            raise RuntimeError("HQDE prediction failed") from exc

        return torch.cat(predictions, dim=0) if predictions else torch.empty(0)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the HQDE system."""
        return self.metrics.copy()

    def save_model(self, filepath: str):
        """Save the full ensemble checkpoint."""
        model_state = {
            "worker_checkpoints": self.ensemble_manager.get_worker_checkpoints(),
            "model_kwargs": self.model_kwargs,
            "metrics": self.metrics,
            "num_workers": self.num_workers,
            "training_config": self.training_config,
            "quantization_config": self.quantization_config,
        }
        torch.save(model_state, filepath)
        self.logger.info("HQDE model saved to %s", filepath)

    def load_model(self, filepath: str):
        """Load a saved ensemble checkpoint."""
        model_state = torch.load(filepath, map_location="cpu")
        self.model_kwargs = dict(model_state["model_kwargs"])
        self.metrics = dict(model_state["metrics"])
        self.num_workers = int(model_state["num_workers"])
        self.training_config = validate_training_config(model_state.get("training_config", {}))
        self.quantization_config = model_state.get("quantization_config")

        self.cleanup()
        self.ensemble_manager = DistributedEnsembleManager(
            num_workers=self.num_workers,
            training_config=self.training_config,
            quantization_config=self.quantization_config,
        )
        self.initialize_ensemble()
        self.quantizer = self.ensemble_manager.quantizer

        if "worker_checkpoints" in model_state:
            self.ensemble_manager.load_worker_checkpoints(model_state["worker_checkpoints"])
        elif "aggregated_weights" in model_state:
            self.ensemble_manager.broadcast_weights(model_state["aggregated_weights"])

        self.logger.info("HQDE model loaded from %s", filepath)

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "ensemble_manager") and self.ensemble_manager is not None:
            self.ensemble_manager.shutdown()

# Factory function for easy instantiation
def create_hqde_system(model_class,
                      model_kwargs: Dict[str, Any],
                      num_workers: int = 4,
                      **kwargs) -> HQDESystem:
    """
    Factory function to create and initialize an HQDE system.

    Args:
        model_class: The model class for ensemble members
        model_kwargs: Model initialization parameters
        num_workers: Number of distributed workers
        **kwargs: Additional configuration parameters

    Returns:
        Initialized HQDESystem instance
    """
    system = HQDESystem(model_class, model_kwargs, num_workers, **kwargs)
    system.initialize_ensemble()
    return system
