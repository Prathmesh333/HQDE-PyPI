"""Training preset helpers for common image-classification workloads."""

from __future__ import annotations

from typing import Any, Dict, Optional


def make_cifar_training_config(
    *,
    ensemble_mode: str = "independent",
    batch_assignment: Optional[str] = None,
    learning_rate: float = 0.1,
    optimizer: str = "sgd",
    weight_decay: float = 5e-4,
    min_learning_rate: float = 1e-6,
    warmup_epochs: int = 5,
    warmup_start_factor: float = 0.2,
    label_smoothing: float = 0.1,
    gradient_clip_norm: float = 1.0,
    use_amp: bool = True,
    compile_model: bool = False,
    compile_mode: str = "default",
    training_aggregation: Optional[str] = None,
    prediction_aggregation: Optional[str] = None,
    server_optimizer: str = "fedadam",
    server_learning_rate: float = 1.0,
    server_beta1: float = 0.9,
    server_beta2: float = 0.99,
    server_epsilon: float = 1e-3,
    federated_normalization: str = "local_bn",
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Return a notebook-style CIFAR image-classification recipe.

    The defaults mirror the stronger benchmark setup used in the v4 notebooks:
    small-image ResNet-18, SGD with Nesterov momentum, warmup + cosine decay,
    label smoothing, AMP, and mean prediction aggregation for ensembles.
    """

    ensemble_mode = str(ensemble_mode).lower()
    resolved_batch_assignment = batch_assignment
    if resolved_batch_assignment is None:
        resolved_batch_assignment = "split" if ensemble_mode == "fedavg" else "replicate"

    resolved_training_aggregation = training_aggregation
    if resolved_training_aggregation is None:
        resolved_training_aggregation = "sample_weighted" if ensemble_mode == "fedavg" else "mean"

    resolved_prediction_aggregation = prediction_aggregation
    if resolved_prediction_aggregation is None:
        resolved_prediction_aggregation = "mean"

    config: Dict[str, Any] = {
        "ensemble_mode": ensemble_mode,
        "batch_assignment": resolved_batch_assignment,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "min_learning_rate": min_learning_rate,
        "warmup_epochs": warmup_epochs,
        "warmup_start_factor": warmup_start_factor,
        "label_smoothing": label_smoothing,
        "gradient_clip_norm": gradient_clip_norm,
        "use_amp": use_amp,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "training_aggregation": resolved_training_aggregation,
        "prediction_aggregation": resolved_prediction_aggregation,
        "server_optimizer": server_optimizer,
        "server_learning_rate": server_learning_rate,
        "server_beta1": server_beta1,
        "server_beta2": server_beta2,
        "server_epsilon": server_epsilon,
        "federated_normalization": federated_normalization,
    }
    config.update(overrides)
    return config


__all__ = ["make_cifar_training_config"]
