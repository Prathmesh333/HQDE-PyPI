"""
Quantum noise generation module for HQDE framework.

This module provides various quantum-inspired noise generation techniques
for ensemble learning, including quantum differential privacy and
exploration-enhancing noise injection.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math


class QuantumNoiseGenerator:
    """Generates quantum-inspired noise for ensemble learning enhancement."""

    def __init__(self,
                 noise_scale: float = 0.01,
                 quantum_coherence_time: float = 1.0,
                 decoherence_rate: float = 0.1):
        """
        Initialize quantum noise generator.

        Args:
            noise_scale: Base scale for quantum noise
            quantum_coherence_time: Simulated quantum coherence time
            decoherence_rate: Rate of quantum decoherence
        """
        self.noise_scale = noise_scale
        self.coherence_time = quantum_coherence_time
        self.decoherence_rate = decoherence_rate
        self.time_step = 0

    def generate_quantum_dp_noise(self,
                                tensor_shape: torch.Size,
                                epsilon: float = 1.0,
                                delta: float = 1e-5) -> torch.Tensor:
        """
        Generate quantum differential privacy noise.

        Args:
            tensor_shape: Shape of the tensor to add noise to
            epsilon: Privacy parameter epsilon
            delta: Privacy parameter delta

        Returns:
            Quantum differential privacy noise tensor
        """
        # Quantum-enhanced Gaussian mechanism
        sensitivity = 1.0  # Assumed L2 sensitivity
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

        # Generate base Gaussian noise
        base_noise = torch.randn(tensor_shape) * sigma

        # Add quantum coherent oscillations
        coherent_freq = 2 * math.pi / self.coherence_time
        time_factor = math.exp(-self.decoherence_rate * self.time_step)

        # Quantum phase factor
        quantum_phase = torch.exp(1j * coherent_freq * self.time_step * torch.randn(tensor_shape))

        # Combine classical and quantum components
        quantum_noise = base_noise * time_factor * quantum_phase.real

        return quantum_noise

    def generate_exploration_noise(self,
                                 current_weights: torch.Tensor,
                                 exploration_strength: float = 0.1) -> torch.Tensor:
        """
        Generate exploration noise based on current weight distribution.

        Args:
            current_weights: Current weight tensor
            exploration_strength: Strength of exploration noise

        Returns:
            Exploration noise tensor
        """
        # Adaptive noise based on weight variance
        weight_std = torch.std(current_weights)
        adaptive_scale = self.noise_scale * exploration_strength * weight_std

        # Generate correlated quantum noise
        base_noise = torch.randn_like(current_weights)

        # Add quantum correlations through convolution
        if len(current_weights.shape) >= 2:
            # Create quantum correlation kernel
            kernel_size = min(3, min(current_weights.shape[-2:]))
            if kernel_size > 1:
                correlation_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)

                # Apply correlation if tensor has appropriate dimensions
                if len(current_weights.shape) == 4:  # Conv weights
                    correlated_noise = torch.nn.functional.conv2d(
                        base_noise.unsqueeze(0).unsqueeze(0),
                        correlation_kernel,
                        padding=kernel_size//2
                    ).squeeze()
                else:
                    correlated_noise = base_noise
            else:
                correlated_noise = base_noise
        else:
            correlated_noise = base_noise

        exploration_noise = correlated_noise * adaptive_scale

        return exploration_noise

    def generate_entanglement_noise(self,
                                   ensemble_weights: list[torch.Tensor],
                                   entanglement_strength: float = 0.1) -> list[torch.Tensor]:
        """
        Generate entangled noise for ensemble members.

        Args:
            ensemble_weights: List of weight tensors from ensemble members
            entanglement_strength: Strength of entanglement correlations

        Returns:
            List of entangled noise tensors
        """
        num_ensemble = len(ensemble_weights)
        if num_ensemble < 2:
            return [self.generate_exploration_noise(w) for w in ensemble_weights]

        # Generate shared quantum state
        shared_shape = ensemble_weights[0].shape
        shared_quantum_state = torch.randn(shared_shape)

        entangled_noises = []
        for i, weights in enumerate(ensemble_weights):
            # Individual quantum state
            individual_state = torch.randn_like(weights)

            # Entanglement coupling
            entanglement_phase = 2 * math.pi * i / num_ensemble
            coupling_factor = math.cos(entanglement_phase) * entanglement_strength

            # Combine shared and individual components
            entangled_noise = (
                (1 - entanglement_strength) * individual_state +
                coupling_factor * shared_quantum_state
            ) * self.noise_scale

            entangled_noises.append(entangled_noise)

        return entangled_noises

    def generate_quantum_regularization_noise(self,
                                            weights: torch.Tensor,
                                            regularization_strength: float = 0.01) -> torch.Tensor:
        """
        Generate quantum-inspired regularization noise.

        Args:
            weights: Weight tensor to regularize
            regularization_strength: Strength of regularization

        Returns:
            Quantum regularization noise
        """
        # Quantum harmonic oscillator noise
        harmonic_noise = torch.randn_like(weights)

        # Apply quantum energy levels (discrete frequency components)
        for n in range(1, 5):  # First few energy levels
            frequency = math.sqrt(n) * 2 * math.pi
            energy_component = torch.sin(frequency * self.time_step) * math.exp(-n * 0.1)
            harmonic_noise += energy_component * torch.randn_like(weights)

        regularization_noise = harmonic_noise * regularization_strength * self.noise_scale

        return regularization_noise

    def generate_adaptive_quantum_noise(self,
                                      weights: torch.Tensor,
                                      gradient: Optional[torch.Tensor] = None,
                                      loss_value: Optional[float] = None,
                                      **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate adaptive quantum noise based on training dynamics.

        Args:
            weights: Current weight tensor
            gradient: Current gradient tensor (optional)
            loss_value: Current loss value (optional)
            **kwargs: Additional parameters

        Returns:
            Tuple of (quantum_noise, noise_metadata)
        """
        # Base quantum noise
        quantum_noise = torch.randn_like(weights) * self.noise_scale

        # Adapt based on gradient information
        if gradient is not None:
            gradient_magnitude = torch.norm(gradient).item()

            # Increase noise when gradients are small (exploration)
            # Decrease noise when gradients are large (exploitation)
            gradient_factor = 1.0 / (1.0 + gradient_magnitude)
            quantum_noise *= gradient_factor

        # Adapt based on loss value
        if loss_value is not None:
            # Increase noise for high loss (need more exploration)
            loss_factor = 1.0 + math.exp(-loss_value)
            quantum_noise *= loss_factor

        # Add quantum decoherence effect
        decoherence_factor = math.exp(-self.decoherence_rate * self.time_step)
        quantum_noise *= decoherence_factor

        # Update time step for next call
        self.time_step += 1

        # Prepare metadata
        metadata = {
            'noise_scale_used': self.noise_scale,
            'decoherence_factor': decoherence_factor,
            'time_step': self.time_step,
            'adaptive_factors': {
                'gradient_factor': gradient_factor if gradient is not None else 1.0,
                'loss_factor': loss_factor if loss_value is not None else 1.0
            }
        }

        return quantum_noise, metadata

    def apply_quantum_noise_schedule(self,
                                   weights: torch.Tensor,
                                   schedule_type: str = "exponential",
                                   schedule_params: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Apply quantum noise with a specific schedule.

        Args:
            weights: Weight tensor to add noise to
            schedule_type: Type of noise schedule ("exponential", "cosine", "linear")
            schedule_params: Parameters for the schedule

        Returns:
            Noise tensor according to schedule
        """
        if schedule_params is None:
            schedule_params = {}

        # Calculate schedule factor
        if schedule_type == "exponential":
            decay_rate = schedule_params.get("decay_rate", 0.1)
            schedule_factor = math.exp(-decay_rate * self.time_step)
        elif schedule_type == "cosine":
            period = schedule_params.get("period", 100)
            schedule_factor = 0.5 * (1 + math.cos(2 * math.pi * self.time_step / period))
        elif schedule_type == "linear":
            max_steps = schedule_params.get("max_steps", 1000)
            schedule_factor = max(0, 1 - self.time_step / max_steps)
        else:
            schedule_factor = 1.0

        # Generate scheduled quantum noise
        base_noise = torch.randn_like(weights)
        scheduled_noise = base_noise * self.noise_scale * schedule_factor

        return scheduled_noise

    def reset_time_step(self):
        """Reset the internal time step counter."""
        self.time_step = 0

    def get_noise_statistics(self) -> Dict[str, float]:
        """Get current noise generator statistics."""
        return {
            'current_time_step': self.time_step,
            'noise_scale': self.noise_scale,
            'coherence_time': self.coherence_time,
            'decoherence_rate': self.decoherence_rate,
            'current_decoherence_factor': math.exp(-self.decoherence_rate * self.time_step)
        }