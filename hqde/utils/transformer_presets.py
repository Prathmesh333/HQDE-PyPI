"""
Training preset configurations for transformer-based models.

This module provides preset configurations optimized for NLP tasks
including CBT text classification, sentiment analysis, and other
sequence classification problems.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def make_transformer_training_config(
    *,
    ensemble_mode: str = "independent",
    batch_assignment: Optional[str] = None,
    learning_rate: float = 5e-4,
    optimizer: str = "adamw",
    weight_decay: float = 1e-2,
    min_learning_rate: float = 1e-6,
    warmup_epochs: int = 3,
    warmup_start_factor: float = 0.1,
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
    server_beta2: float = 0.999,
    server_epsilon: float = 1e-8,
    federated_normalization: str = "shared",  # Transformers don't use BatchNorm
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Return a transformer-optimized training configuration.

    This configuration is tuned for transformer-based text classification
    with AdamW optimizer, warmup, and cosine decay.

    Args:
        ensemble_mode: "independent" for diverse ensembles or "fedavg" for weight averaging
        batch_assignment: "replicate" or "split" (auto-selected if None)
        learning_rate: Initial learning rate (5e-4 is typical for transformers)
        optimizer: Optimizer type ("adamw" recommended for transformers)
        weight_decay: L2 regularization (1e-2 typical for AdamW)
        min_learning_rate: Minimum LR for cosine scheduler
        warmup_epochs: Number of warmup epochs
        warmup_start_factor: Starting factor for warmup
        label_smoothing: Label smoothing factor
        gradient_clip_norm: Gradient clipping threshold
        use_amp: Use automatic mixed precision
        compile_model: Use torch.compile (PyTorch 2.0+)
        compile_mode: Compilation mode
        training_aggregation: Weight aggregation during training
        prediction_aggregation: Prediction aggregation method
        server_optimizer: Server-side optimizer for federated learning
        server_learning_rate: Server learning rate
        server_beta1: Server optimizer beta1
        server_beta2: Server optimizer beta2
        server_epsilon: Server optimizer epsilon
        federated_normalization: Normalization strategy (use "shared" for transformers)
        **overrides: Additional config overrides

    Returns:
        Training configuration dictionary
    """
    ensemble_mode = str(ensemble_mode).lower()
    
    # Auto-select batch assignment
    resolved_batch_assignment = batch_assignment
    if resolved_batch_assignment is None:
        resolved_batch_assignment = "split" if ensemble_mode == "fedavg" else "replicate"

    # Auto-select training aggregation
    resolved_training_aggregation = training_aggregation
    if resolved_training_aggregation is None:
        resolved_training_aggregation = "sample_weighted" if ensemble_mode == "fedavg" else "mean"

    # Auto-select prediction aggregation
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


def make_cbt_training_config(
    *,
    ensemble_mode: str = "independent",
    learning_rate: float = 3e-4,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.05,
    dropout_rate: float = 0.15,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Return a CBT-specific training configuration.

    Optimized for Cognitive Behavioral Therapy text classification with:
    - Lower learning rate for stability
    - More warmup for better convergence
    - Light label smoothing
    - Higher dropout for ensemble diversity

    Args:
        ensemble_mode: Training mode
        learning_rate: Initial learning rate (lower for CBT)
        warmup_epochs: Warmup epochs (more for stability)
        label_smoothing: Label smoothing (light for CBT)
        dropout_rate: Dropout rate (higher for diversity)
        **overrides: Additional overrides

    Returns:
        CBT-optimized training configuration
    """
    config = make_transformer_training_config(
        ensemble_mode=ensemble_mode,
        learning_rate=learning_rate,
        warmup_epochs=warmup_epochs,
        label_smoothing=label_smoothing,
        optimizer="adamw",
        weight_decay=1e-2,
        gradient_clip_norm=1.0,
        use_amp=True,
    )
    
    # Add dropout rate for model initialization
    config["dropout_rate"] = dropout_rate
    
    config.update(overrides)
    return config


def make_lightweight_transformer_config(
    *,
    ensemble_mode: str = "independent",
    learning_rate: float = 1e-3,
    warmup_epochs: int = 2,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Return a configuration for lightweight transformers.

    Optimized for faster training with smaller models:
    - Higher learning rate
    - Less warmup
    - Suitable for quick experiments

    Args:
        ensemble_mode: Training mode
        learning_rate: Initial learning rate (higher for small models)
        warmup_epochs: Warmup epochs (less for small models)
        **overrides: Additional overrides

    Returns:
        Lightweight transformer configuration
    """
    config = make_transformer_training_config(
        ensemble_mode=ensemble_mode,
        learning_rate=learning_rate,
        warmup_epochs=warmup_epochs,
        weight_decay=1e-2,
        label_smoothing=0.1,
        gradient_clip_norm=1.0,
    )
    
    config.update(overrides)
    return config


def make_large_transformer_config(
    *,
    ensemble_mode: str = "fedavg",
    learning_rate: float = 1e-4,
    warmup_epochs: int = 10,
    gradient_clip_norm: float = 0.5,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Return a configuration for large transformer models.

    Optimized for large models with careful training:
    - Lower learning rate
    - More warmup
    - Stricter gradient clipping
    - FedAvg mode for memory efficiency

    Args:
        ensemble_mode: Training mode (fedavg recommended)
        learning_rate: Initial learning rate (lower for large models)
        warmup_epochs: Warmup epochs (more for stability)
        gradient_clip_norm: Gradient clipping (stricter for large models)
        **overrides: Additional overrides

    Returns:
        Large transformer configuration
    """
    config = make_transformer_training_config(
        ensemble_mode=ensemble_mode,
        learning_rate=learning_rate,
        warmup_epochs=warmup_epochs,
        gradient_clip_norm=gradient_clip_norm,
        weight_decay=1e-1,  # Higher weight decay for large models
        label_smoothing=0.1,
        use_amp=True,
        compile_model=True,  # Compile for efficiency
    )
    
    config.update(overrides)
    return config


__all__ = [
    "make_transformer_training_config",
    "make_cbt_training_config",
    "make_lightweight_transformer_config",
    "make_large_transformer_config",
]
