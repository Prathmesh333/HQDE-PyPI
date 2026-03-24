"""
HQDE (Hierarchical Quantum-Distributed Ensemble Learning) Core System.

This module implements the main HQDE framework with a practical ensemble
training loop, prediction aggregation, and checkpoint handling.
"""

from __future__ import annotations

import inspect
import logging
import time
from contextlib import nullcontext
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
    "training_aggregation": "sample_weighted",
    "prediction_aggregation": "efficiency_weighted",
    "server_optimizer": "fedadam",
    "server_learning_rate": 1.0,
    "server_beta1": 0.9,
    "server_beta2": 0.99,
    "server_epsilon": 1e-3,
    "federated_normalization": "local_bn",
}

VALID_OPTIMIZERS = {"sgd", "adam", "adamw"}
VALID_ENSEMBLE_MODES = {"independent", "fedavg"}
VALID_BATCH_ASSIGNMENTS = {"replicate", "split"}
VALID_TRAINING_AGGREGATIONS = {"sample_weighted", "mean", "efficiency_weighted"}
VALID_PREDICTION_AGGREGATIONS = {"efficiency_weighted", "mean"}
VALID_SERVER_OPTIMIZERS = {"mean", "fedadam"}
VALID_FEDERATED_NORMALIZATIONS = {"shared", "local_bn"}
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
        training_aggregation: "sample_weighted", "mean", or "efficiency_weighted".
        prediction_aggregation: "efficiency_weighted" or "mean".
        server_optimizer: "mean" or "fedadam" for federated server-side updates.
        server_learning_rate: positive float server update scale.
        server_beta1: float in [0, 1) first-moment coefficient for FedAdam.
        server_beta2: float in [0, 1) second-moment coefficient for FedAdam.
        server_epsilon: positive float numerical stabilizer for FedAdam.
        federated_normalization: "shared" or "local_bn" to preserve per-worker batch norm.
    """
    config = dict(DEFAULT_TRAINING_CONFIG)
    config.update(training_config or {})

    config["optimizer"] = str(config["optimizer"]).lower()
    config["ensemble_mode"] = str(config["ensemble_mode"]).lower()
    config["batch_assignment"] = str(config["batch_assignment"]).lower()
    config["training_aggregation"] = str(config["training_aggregation"]).lower()
    config["prediction_aggregation"] = str(config["prediction_aggregation"]).lower()
    config["server_optimizer"] = str(config["server_optimizer"]).lower()
    config["federated_normalization"] = str(config["federated_normalization"]).lower()

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
    if config["training_aggregation"] not in VALID_TRAINING_AGGREGATIONS:
        raise ValueError(
            "Unsupported training_aggregation "
            f"'{config['training_aggregation']}'. Expected one of {sorted(VALID_TRAINING_AGGREGATIONS)}."
        )
    if config["prediction_aggregation"] not in VALID_PREDICTION_AGGREGATIONS:
        raise ValueError(
            "Unsupported prediction_aggregation "
            f"'{config['prediction_aggregation']}'. Expected one of {sorted(VALID_PREDICTION_AGGREGATIONS)}."
        )
    if config["server_optimizer"] not in VALID_SERVER_OPTIMIZERS:
        raise ValueError(
            "Unsupported server_optimizer "
            f"'{config['server_optimizer']}'. Expected one of {sorted(VALID_SERVER_OPTIMIZERS)}."
        )
    if config["federated_normalization"] not in VALID_FEDERATED_NORMALIZATIONS:
        raise ValueError(
            "Unsupported federated_normalization "
            f"'{config['federated_normalization']}'. Expected one of {sorted(VALID_FEDERATED_NORMALIZATIONS)}."
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
        "server_learning_rate": (0.0, False),
        "server_epsilon": (0.0, False),
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
    config["server_beta1"] = float(config["server_beta1"])
    config["server_beta2"] = float(config["server_beta2"])
    if not 0.0 <= config["server_beta1"] < 1.0:
        raise ValueError(f"server_beta1 must be in [0.0, 1.0), got {config['server_beta1']}.")
    if not 0.0 <= config["server_beta2"] < 1.0:
        raise ValueError(f"server_beta2 must be in [0.0, 1.0), got {config['server_beta2']}.")

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
    """Adaptive delta quantizer for communication-efficient FedAvg."""

    def __init__(
        self,
        base_bits: int = 12,
        min_bits: int = 8,
        max_bits: int = 16,
        block_size: int = 1024,
        warmup_rounds: int = 5,
        min_tensor_elements: int = 2048,
        skip_bias: bool = True,
        skip_norm: bool = True,
        error_feedback: bool = True,
    ):
        self.base_bits = int(base_bits)
        self.min_bits = int(min_bits)
        self.max_bits = int(max_bits)
        self.block_size = max(int(block_size), 1)
        self.warmup_rounds = max(int(warmup_rounds), 0)
        self.min_tensor_elements = max(int(min_tensor_elements), 1)
        self.skip_bias = bool(skip_bias)
        self.skip_norm = bool(skip_norm)
        self.error_feedback = bool(error_feedback)
        if not (self.min_bits <= self.base_bits <= self.max_bits):
            raise ValueError(
                "Quantization bits must satisfy min_bits <= base_bits <= max_bits. "
                f"Got min_bits={self.min_bits}, base_bits={self.base_bits}, max_bits={self.max_bits}."
            )
        self.residual_buffers: Dict[str, torch.Tensor] = {}

    def _buffer_key(self, worker_id: int, param_name: str) -> str:
        return f"{int(worker_id)}::{param_name}"

    def reset_state(self):
        """Reset accumulated quantization residuals."""
        self.residual_buffers = {}

    def get_state(self) -> Dict[str, Any]:
        """Return serializable quantizer state."""
        return {
            "residual_buffers": {
                key: tensor.detach().cpu().clone()
                for key, tensor in self.residual_buffers.items()
            }
        }

    def load_state(self, state: Optional[Dict[str, Any]]):
        """Restore quantizer state from a checkpoint payload."""
        state = state or {}
        self.residual_buffers = {
            str(key): tensor.detach().cpu().clone()
            for key, tensor in state.get("residual_buffers", {}).items()
        }

    def compute_importance_score(
        self,
        weights: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute normalized importance scores for a tensor."""
        with torch.no_grad():
            weight_importance = torch.abs(weights)
            if gradients is not None:
                grad_importance = torch.abs(gradients)
                combined_importance = 0.7 * weight_importance + 0.3 * grad_importance
            else:
                combined_importance = weight_importance

            if combined_importance.numel() == 0:
                return torch.zeros_like(combined_importance)

            min_val = combined_importance.min()
            max_val = combined_importance.max()
            if max_val > min_val:
                return (combined_importance - min_val) / (max_val - min_val)
            return torch.ones_like(combined_importance) * 0.5

    def _scheduled_bits(self, round_index: int, total_rounds: int) -> int:
        """Return the round-level target bitwidth."""
        if round_index <= self.warmup_rounds:
            return 32

        quant_rounds = max(int(total_rounds) - self.warmup_rounds, 1)
        progress = min(max((round_index - self.warmup_rounds - 1) / quant_rounds, 0.0), 1.0)
        if progress < 0.33:
            return self.max_bits
        if progress < 0.66:
            return self.base_bits
        return self.min_bits

    def should_quantize_parameter(
        self,
        param_name: str,
        tensor: torch.Tensor,
        round_index: int,
    ) -> bool:
        """Return whether a tensor should be quantized on the current round."""
        if round_index <= self.warmup_rounds:
            return False
        if not torch.is_floating_point(tensor):
            return False
        if tensor.numel() < self.min_tensor_elements:
            return False

        lowered_name = str(param_name).lower()
        if "num_batches_tracked" in lowered_name:
            return False
        if "running_mean" in lowered_name or "running_var" in lowered_name:
            return False
        if self.skip_bias and lowered_name.endswith(".bias"):
            return False
        if self.skip_norm and any(token in lowered_name for token in ("norm", "bn", "ln", "gn")):
            return False
        return True

    def quantize_delta(
        self,
        param_name: str,
        delta: torch.Tensor,
        worker_id: int,
        round_index: int,
        total_rounds: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize a model delta using blockwise symmetric quantization."""
        delta_tensor = delta.detach().cpu().float()
        original_bytes = float(delta_tensor.numel() * delta_tensor.element_size())

        if not self.should_quantize_parameter(param_name, delta_tensor, round_index):
            return delta_tensor.clone(), {
                "avg_bits": 32.0,
                "compression_ratio": 1.0,
                "original_bytes": original_bytes,
                "transmitted_bytes": original_bytes,
                "quantized": False,
            }

        buffer_key = self._buffer_key(worker_id, param_name)
        residual = self.residual_buffers.get(buffer_key)
        if residual is None or residual.shape != delta_tensor.shape:
            residual = torch.zeros_like(delta_tensor)
        work_tensor = delta_tensor + residual if self.error_feedback else delta_tensor
        importance = self.compute_importance_score(work_tensor)

        flat_tensor = work_tensor.reshape(-1)
        flat_importance = importance.reshape(-1)
        dequantized_flat = torch.empty_like(flat_tensor)
        round_bits = self._scheduled_bits(round_index, total_rounds)

        total_bits = 0.0
        total_scale_bytes = 0.0
        block_bitwidths: List[int] = []

        for start_index in range(0, flat_tensor.numel(), self.block_size):
            end_index = min(start_index + self.block_size, flat_tensor.numel())
            block = flat_tensor[start_index:end_index]
            block_importance = flat_importance[start_index:end_index]
            if block.numel() == 0:
                continue

            mean_importance = float(block_importance.mean().item()) if block_importance.numel() > 0 else 0.5
            low_bits = max(self.min_bits, round_bits - 2)
            high_bits = min(self.max_bits, round_bits + 2)
            block_bits = int(round(low_bits + (high_bits - low_bits) * mean_importance))
            block_bits = max(self.min_bits, min(self.max_bits, block_bits))
            block_bitwidths.append(block_bits)

            max_abs = block.abs().max()
            if max_abs > 0:
                qmax = max(2 ** (block_bits - 1) - 1, 1)
                scale = max_abs / qmax
                quantized = torch.round(block / scale).clamp(-qmax, qmax)
                dequantized = quantized * scale
            else:
                dequantized = block.clone()

            dequantized_flat[start_index:end_index] = dequantized
            total_bits += float(block.numel() * block_bits)
            total_scale_bytes += 4.0

        dequantized_tensor = dequantized_flat.view_as(delta_tensor)
        if self.error_feedback:
            self.residual_buffers[buffer_key] = (work_tensor - dequantized_tensor).detach().cpu()

        transmitted_bytes = (total_bits / 8.0) + total_scale_bytes
        avg_bits = total_bits / max(float(delta_tensor.numel()), 1.0)
        compression_ratio = original_bytes / max(transmitted_bytes, 1.0)
        return dequantized_tensor, {
            "avg_bits": avg_bits,
            "compression_ratio": compression_ratio,
            "original_bytes": original_bytes,
            "transmitted_bytes": transmitted_bytes,
            "quantized": True,
            "block_bits": block_bitwidths,
        }

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
            score_signal = (2.0 * accuracy) + (1.0 / max(1.0 + loss_value, 1e-6))
            self.efficiency_score = 0.8 * self.efficiency_score + 0.2 * score_signal

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

    def get_local_normalization_keys(self, mode: str = "local_bn") -> List[str]:
        """Return state-dict keys that should remain local in federated mode."""
        if str(mode).lower() != "local_bn":
            return []

        state_keys = set(self.model.state_dict().keys())
        local_keys = []
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            prefix = f"{module_name}." if module_name else ""
            for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                key = prefix + suffix
                if key in state_keys:
                    local_keys.append(key)
        return local_keys

    def set_weights(self, weights_dict, preserve_names: Optional[List[str]] = None):
        """Load a state dict into the worker model."""
        current_state = self.model.state_dict()
        preserve_names = set(preserve_names or [])
        updated_state = {}
        for name, tensor in current_state.items():
            if name in preserve_names:
                updated_state[name] = tensor
            else:
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
        self.last_original_bytes = 0.0
        self.last_transmitted_bytes = 0.0
        self.reference_weights: Optional[Dict[str, torch.Tensor]] = None
        self.current_round = 0
        self.total_rounds = 0
        self.round_sample_counts: List[float] = [0.0] * self.num_workers
        self.local_normalization_keys = set()
        self.server_step = 0
        self.server_first_moment: Dict[str, torch.Tensor] = {}
        self.server_second_moment: Dict[str, torch.Tensor] = {}
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

    @staticmethod
    def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in state_dict.items()
        }

    def _uses_local_normalization(self) -> bool:
        return (
            self.training_config["ensemble_mode"] == "fedavg"
            and str(self.training_config.get("federated_normalization", "shared")).lower() == "local_bn"
        )

    def _is_local_normalization_key(self, param_name: str) -> bool:
        return param_name in self.local_normalization_keys

    @staticmethod
    def _weighted_average_tensors(tensor_list: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        if not tensor_list:
            raise ValueError("Cannot aggregate an empty tensor list.")
        reference_device = tensor_list[0].device
        stacked = torch.stack([tensor.to(reference_device) for tensor in tensor_list], dim=0)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=reference_device)
        weight_sum = float(weight_tensor.sum().item())
        if weight_sum <= 0:
            weight_tensor = torch.ones_like(weight_tensor) / max(float(weight_tensor.numel()), 1.0)
        else:
            weight_tensor = weight_tensor / weight_sum
        reshape_dims = (weight_tensor.shape[0],) + (1,) * (stacked.dim() - 1)
        return (stacked * weight_tensor.view(reshape_dims)).sum(dim=0)

    def _resolve_training_weights(self, efficiency_scores: List[float]) -> List[float]:
        aggregation_mode = str(self.training_config.get("training_aggregation", "sample_weighted")).lower()
        if aggregation_mode == "mean":
            return [1.0] * len(self.workers)
        if aggregation_mode == "efficiency_weighted":
            return list(efficiency_scores or [1.0] * len(self.workers))
        sample_counts = [float(count) for count in self.round_sample_counts]
        if len(sample_counts) != len(self.workers) or sum(sample_counts) <= 0:
            return [1.0] * len(self.workers)
        return sample_counts

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

        if self.training_config["ensemble_mode"] == "fedavg":
            lr_multipliers = [1.0] * self.num_workers
            dropout_rates = [0.12] * self.num_workers

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

        if self.training_config["ensemble_mode"] == "fedavg" and self.workers:
            if self._uses_local_normalization():
                self.local_normalization_keys = set(
                    self._resolve(
                        [
                            self._worker_call(
                                self.workers[0],
                                "get_local_normalization_keys",
                                self.training_config.get("federated_normalization", "local_bn"),
                            )
                        ]
                    )[0]
                )
            else:
                self.local_normalization_keys = set()
            initial_weights = self._resolve([self._worker_call(self.workers[0], "get_weights")])[0]
            self.broadcast_weights(initial_weights, preserve_local_norm=False)
            if self.quantizer is not None:
                self.quantizer.reset_state()
            self.current_round = 0
            self.round_sample_counts = [0.0] * self.num_workers
            self.server_step = 0
            self.server_first_moment = {}
            self.server_second_moment = {}

    def setup_workers_training(self, learning_rate: Optional[float] = None, total_epochs: int = 1):
        """Setup training for all workers."""
        learning_rate = learning_rate or float(self.training_config.get("learning_rate", 1e-3))
        self.total_rounds = max(int(total_epochs), 1)
        setup_calls = [
            self._worker_call(worker, "setup_training", learning_rate, total_epochs)
            for worker in self.workers
        ]
        self._resolve(setup_calls)
        self.logger.info("Training setup completed for %s workers", self.num_workers)

    def aggregate_weights(self) -> Dict[str, torch.Tensor]:
        """Aggregate model states from all workers."""
        all_weights = self._resolve([self._worker_call(worker, "get_weights") for worker in self.workers])
        if not all_weights:
            return {}

        efficiency_scores = self.get_efficiency_scores()
        aggregation_weights = self._resolve_training_weights(efficiency_scores)
        reference_weights = self.reference_weights or self._clone_state_dict(all_weights[0])
        aggregated_weights = {}
        total_original_bytes = 0.0
        total_transmitted_bytes = 0.0
        server_optimizer = str(self.training_config.get("server_optimizer", "mean")).lower()
        use_fedadam = server_optimizer == "fedadam"
        if use_fedadam:
            self.server_step += 1

        server_lr = float(self.training_config.get("server_learning_rate", 1.0))
        beta1 = float(self.training_config.get("server_beta1", 0.9))
        beta2 = float(self.training_config.get("server_beta2", 0.99))
        epsilon = float(self.training_config.get("server_epsilon", 1e-3))

        for param_name in all_weights[0].keys():
            reference_tensor = reference_weights[param_name].detach().cpu()
            if not torch.is_floating_point(reference_tensor):
                aggregated_weights[param_name] = reference_tensor.clone()
                continue
            if self._is_local_normalization_key(param_name):
                aggregated_weights[param_name] = reference_tensor.clone()
                continue

            delta_tensors = []
            for worker_id, weights in enumerate(all_weights):
                local_tensor = weights[param_name].detach().cpu().float()
                delta_tensor = local_tensor - reference_tensor.float()
                if self.quantizer is not None:
                    dequantized_delta, metadata = self.quantizer.quantize_delta(
                        param_name,
                        delta_tensor,
                        worker_id=worker_id,
                        round_index=self.current_round,
                        total_rounds=self.total_rounds,
                    )
                else:
                    dequantized_delta = delta_tensor.clone()
                    original_bytes = float(delta_tensor.numel() * delta_tensor.element_size())
                    metadata = {
                        "compression_ratio": 1.0,
                        "original_bytes": original_bytes,
                        "transmitted_bytes": original_bytes,
                    }
                delta_tensors.append(dequantized_delta)
                total_original_bytes += float(metadata.get("original_bytes", 0.0))
                total_transmitted_bytes += float(metadata.get("transmitted_bytes", 0.0))

            aggregation_mode = str(self.training_config.get("training_aggregation", "sample_weighted")).lower()
            if aggregation_mode == "efficiency_weighted":
                aggregated_delta = self.aggregator.efficiency_weighted_aggregation(
                    delta_tensors,
                    aggregation_weights,
                ).cpu()
            else:
                aggregated_delta = self._weighted_average_tensors(delta_tensors, aggregation_weights).cpu()

            if use_fedadam:
                first_moment = self.server_first_moment.get(param_name)
                second_moment = self.server_second_moment.get(param_name)
                if first_moment is None or first_moment.shape != aggregated_delta.shape:
                    first_moment = torch.zeros_like(aggregated_delta)
                if second_moment is None or second_moment.shape != aggregated_delta.shape:
                    second_moment = torch.zeros_like(aggregated_delta)

                first_moment = beta1 * first_moment + (1.0 - beta1) * aggregated_delta
                second_moment = beta2 * second_moment + (1.0 - beta2) * aggregated_delta.square()
                self.server_first_moment[param_name] = first_moment.detach().cpu()
                self.server_second_moment[param_name] = second_moment.detach().cpu()

                bias_correction1 = 1.0 - (beta1 ** self.server_step)
                bias_correction2 = 1.0 - (beta2 ** self.server_step)
                corrected_first = first_moment / max(bias_correction1, 1e-8)
                corrected_second = second_moment / max(bias_correction2, 1e-8)
                server_update = server_lr * corrected_first / (corrected_second.sqrt() + epsilon)
                aggregated_weights[param_name] = (reference_tensor.float() + server_update).to(
                    dtype=reference_tensor.dtype
                )
            else:
                aggregated_weights[param_name] = (reference_tensor.float() + aggregated_delta).to(
                    dtype=reference_tensor.dtype
                )

        self.last_original_bytes = total_original_bytes
        self.last_transmitted_bytes = total_transmitted_bytes
        self.last_compression_ratio = total_original_bytes / max(total_transmitted_bytes, 1.0)
        return aggregated_weights

    def broadcast_weights(self, weights: Dict[str, torch.Tensor], preserve_local_norm: Optional[bool] = None):
        """Broadcast weights to every worker."""
        if preserve_local_norm is None:
            preserve_local_norm = self._uses_local_normalization() and bool(self.local_normalization_keys)
        self.reference_weights = self._clone_state_dict(weights)
        preserve_names = sorted(self.local_normalization_keys) if preserve_local_norm else None
        calls = [self._worker_call(worker, "set_weights", weights, preserve_names) for worker in self.workers]
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
        return {
            "compression_ratio": float(self.last_compression_ratio),
            "original_bytes": float(self.last_original_bytes),
            "transmitted_bytes": float(self.last_transmitted_bytes),
        }

    def get_quantization_state(self) -> Dict[str, Any]:
        """Return serializable state for quantized FedAvg training."""
        return {
            "current_round": int(self.current_round),
            "total_rounds": int(self.total_rounds),
            "round_sample_counts": [float(count) for count in self.round_sample_counts],
            "local_normalization_keys": sorted(self.local_normalization_keys),
            "reference_weights": self._clone_state_dict(self.reference_weights or {}),
            "server_step": int(self.server_step),
            "server_first_moment": self._clone_state_dict(self.server_first_moment),
            "server_second_moment": self._clone_state_dict(self.server_second_moment),
            "quantizer_state": self.quantizer.get_state() if self.quantizer is not None else {},
        }

    def load_quantization_state(self, state: Optional[Dict[str, Any]]):
        """Restore quantization state from a checkpoint."""
        state = state or {}
        self.current_round = int(state.get("current_round", self.current_round))
        self.total_rounds = int(state.get("total_rounds", self.total_rounds))
        self.round_sample_counts = [float(count) for count in state.get("round_sample_counts", self.round_sample_counts)]
        self.local_normalization_keys = set(state.get("local_normalization_keys", list(self.local_normalization_keys)))
        reference_weights = state.get("reference_weights", {})
        self.reference_weights = self._clone_state_dict(reference_weights) if reference_weights else self.reference_weights
        self.server_step = int(state.get("server_step", self.server_step))
        self.server_first_moment = self._clone_state_dict(state.get("server_first_moment", {}))
        self.server_second_moment = self._clone_state_dict(state.get("server_second_moment", {}))
        if self.quantizer is not None:
            self.quantizer.load_state(state.get("quantizer_state", {}))

    def load_worker_checkpoints(self, checkpoints):
        """Restore workers from per-worker checkpoints."""
        if not checkpoints:
            return
        calls = []
        for worker, checkpoint in zip(self.workers, checkpoints):
            calls.append(self._worker_call(worker, "load_checkpoint", checkpoint))
        self._resolve(calls)
        if self.training_config["ensemble_mode"] == "fedavg" and checkpoints:
            first_state = checkpoints[0].get("model_state_dict", {})
            if first_state:
                self.reference_weights = self._clone_state_dict(first_state)

    def train_ensemble(self, data_loader, num_epochs: int = 10):
        """Train the ensemble using distributed or local workers."""
        self.setup_workers_training(total_epochs=num_epochs)
        epoch_history = []
        ensemble_mode = str(self.training_config.get("ensemble_mode", "independent")).lower()

        for epoch in range(num_epochs):
            epoch_loss_sum = 0.0
            epoch_accuracy_sum = 0.0
            epoch_samples = 0
            worker_round_samples = [0.0] * self.num_workers

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

                for worker_index, result in enumerate(batch_results):
                    if not result:
                        continue
                    num_samples = int(result["num_samples"])
                    worker_round_samples[worker_index] += num_samples
                    epoch_loss_sum += float(result["loss"]) * num_samples
                    epoch_accuracy_sum += float(result["accuracy"]) * num_samples
                    epoch_samples += num_samples

            if ensemble_mode == "fedavg":
                self.current_round = epoch + 1
                self.round_sample_counts = worker_round_samples
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
            "quantization_state": self.ensemble_manager.get_quantization_state(),
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
        self.ensemble_manager.load_quantization_state(model_state.get("quantization_state"))

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
