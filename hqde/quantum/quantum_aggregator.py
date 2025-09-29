"""
Quantum-inspired ensemble aggregation module.

This module implements quantum-inspired algorithms for aggregating ensemble
predictions and weights, including entanglement simulation and quantum
superposition for model combination.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import math


class EntangledEnsembleManager:
    """Manages quantum entanglement-inspired ensemble correlations."""

    def __init__(self, num_ensembles: int, entanglement_strength: float = 0.1):
        self.num_ensembles = num_ensembles
        self.entanglement_strength = entanglement_strength
        self.entanglement_matrix = self._initialize_entanglement()

    def _initialize_entanglement(self) -> torch.Tensor:
        """Initialize entanglement matrix for ensemble correlations."""
        # Create symmetric entanglement matrix
        matrix = torch.randn(self.num_ensembles, self.num_ensembles)
        matrix = (matrix + matrix.T) / 2  # Make symmetric

        # Apply entanglement strength
        matrix = matrix * self.entanglement_strength

        # Ensure diagonal is 1 (self-entanglement)
        matrix.fill_diagonal_(1.0)

        return matrix

    def compute_entanglement_weights(self, ensemble_states: List[torch.Tensor]) -> torch.Tensor:
        """Compute entanglement-based weights for ensemble aggregation."""
        if len(ensemble_states) != self.num_ensembles:
            raise ValueError(f"Expected {self.num_ensembles} ensemble states, got {len(ensemble_states)}")

        # Compute state similarities for entanglement
        similarities = torch.zeros(self.num_ensembles, self.num_ensembles)

        for i, state_i in enumerate(ensemble_states):
            for j, state_j in enumerate(ensemble_states):
                if i <= j:  # Only compute upper triangle
                    # Cosine similarity between states
                    similarity = torch.cosine_similarity(
                        state_i.flatten(), state_j.flatten(), dim=0
                    )
                    similarities[i, j] = similarities[j, i] = similarity

        # Apply entanglement matrix
        entangled_weights = torch.softmax(
            torch.diagonal(similarities @ self.entanglement_matrix), dim=0
        )

        return entangled_weights

    def apply_entanglement(self, ensemble_predictions: List[torch.Tensor],
                          entanglement_weights: torch.Tensor) -> torch.Tensor:
        """Apply entanglement correlations to ensemble predictions."""
        # Weight predictions by entanglement
        weighted_predictions = []
        for pred, weight in zip(ensemble_predictions, entanglement_weights):
            weighted_predictions.append(pred * weight)

        # Quantum superposition-like combination
        superposition = torch.stack(weighted_predictions, dim=0).sum(dim=0)

        return superposition

    def quantum_measurement(self, superposition_state: torch.Tensor) -> torch.Tensor:
        """Simulate quantum measurement collapse to final prediction."""
        # Add quantum measurement noise
        measurement_noise = torch.randn_like(superposition_state) * 0.01
        measured_state = superposition_state + measurement_noise

        return measured_state


class QuantumEnsembleAggregator:
    """Main quantum-inspired ensemble aggregation system."""

    def __init__(self,
                 num_ensembles: int,
                 entanglement_strength: float = 0.1,
                 quantum_noise_scale: float = 0.01,
                 use_quantum_annealing: bool = False):
        """
        Initialize quantum ensemble aggregator.

        Args:
            num_ensembles: Number of ensemble members
            entanglement_strength: Strength of quantum entanglement simulation
            quantum_noise_scale: Scale of quantum noise injection
            use_quantum_annealing: Whether to use quantum annealing for optimization
        """
        self.num_ensembles = num_ensembles
        self.quantum_noise_scale = quantum_noise_scale
        self.use_quantum_annealing = use_quantum_annealing

        # Initialize quantum components
        self.entanglement_manager = EntangledEnsembleManager(
            num_ensembles, entanglement_strength
        )

    def quantum_superposition_aggregation(self,
                                        ensemble_predictions: List[torch.Tensor],
                                        confidence_scores: Optional[List[float]] = None) -> torch.Tensor:
        """
        Aggregate ensemble predictions using quantum superposition.

        Args:
            ensemble_predictions: List of prediction tensors from ensemble members
            confidence_scores: Optional confidence scores for each ensemble member

        Returns:
            Aggregated prediction tensor
        """
        if len(ensemble_predictions) != self.num_ensembles:
            raise ValueError(f"Expected {self.num_ensembles} predictions, got {len(ensemble_predictions)}")

        # Use confidence scores or equal weights
        if confidence_scores is None:
            confidence_scores = [1.0] * self.num_ensembles

        # Normalize confidence scores to create quantum amplitudes
        confidence_tensor = torch.tensor(confidence_scores, dtype=torch.float32)
        amplitudes = torch.sqrt(torch.softmax(confidence_tensor, dim=0))

        # Create quantum superposition
        superposition = torch.zeros_like(ensemble_predictions[0])
        for pred, amplitude in zip(ensemble_predictions, amplitudes):
            superposition += amplitude * pred

        # Add quantum noise for exploration
        quantum_noise = torch.randn_like(superposition) * self.quantum_noise_scale
        superposition_with_noise = superposition + quantum_noise

        return superposition_with_noise

    def entanglement_based_aggregation(self,
                                     ensemble_predictions: List[torch.Tensor],
                                     ensemble_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate predictions using quantum entanglement simulation.

        Args:
            ensemble_predictions: Prediction tensors from ensemble members
            ensemble_states: Internal state tensors from ensemble members

        Returns:
            Entanglement-weighted aggregated predictions
        """
        # Compute entanglement weights
        entanglement_weights = self.entanglement_manager.compute_entanglement_weights(ensemble_states)

        # Apply entanglement to predictions
        entangled_superposition = self.entanglement_manager.apply_entanglement(
            ensemble_predictions, entanglement_weights
        )

        # Quantum measurement
        final_prediction = self.entanglement_manager.quantum_measurement(entangled_superposition)

        return final_prediction

    def quantum_voting_aggregation(self,
                                 ensemble_predictions: List[torch.Tensor],
                                 voting_weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Quantum-inspired voting aggregation with coherent superposition.

        Args:
            ensemble_predictions: Prediction tensors from ensemble members
            voting_weights: Optional voting weights for each ensemble member

        Returns:
            Quantum-voted aggregated predictions
        """
        if voting_weights is None:
            voting_weights = [1.0] * len(ensemble_predictions)

        # Convert to quantum phase representation
        quantum_phases = []
        for pred, weight in zip(ensemble_predictions, voting_weights):
            # Convert predictions to phase representation
            phase = torch.atan2(torch.sin(pred * math.pi), torch.cos(pred * math.pi))
            weighted_phase = phase * weight
            quantum_phases.append(weighted_phase)

        # Combine phases using quantum interference
        combined_phase = torch.stack(quantum_phases, dim=0).mean(dim=0)

        # Convert back to prediction space
        final_prediction = torch.sin(combined_phase) + 1j * torch.cos(combined_phase)

        # Take real part as final prediction
        return final_prediction.real

    def adaptive_quantum_aggregation(self,
                                   ensemble_predictions: List[torch.Tensor],
                                   ensemble_uncertainties: List[torch.Tensor],
                                   aggregation_mode: str = "auto") -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Adaptive quantum aggregation that chooses the best method based on uncertainty.

        Args:
            ensemble_predictions: Prediction tensors from ensemble members
            ensemble_uncertainties: Uncertainty estimates from ensemble members
            aggregation_mode: Aggregation mode ("auto", "superposition", "entanglement", "voting")

        Returns:
            Tuple of (aggregated_predictions, aggregation_metrics)
        """
        # Compute average uncertainty
        avg_uncertainty = torch.stack(ensemble_uncertainties).mean().item()

        # Choose aggregation method based on uncertainty and mode
        if aggregation_mode == "auto":
            if avg_uncertainty > 0.5:
                chosen_method = "entanglement"  # High uncertainty - use entanglement
            elif avg_uncertainty > 0.2:
                chosen_method = "superposition"  # Medium uncertainty - use superposition
            else:
                chosen_method = "voting"  # Low uncertainty - use voting
        else:
            chosen_method = aggregation_mode

        # Apply chosen aggregation method
        if chosen_method == "superposition":
            confidence_scores = [1.0 / (1.0 + u.mean().item()) for u in ensemble_uncertainties]
            aggregated = self.quantum_superposition_aggregation(ensemble_predictions, confidence_scores)
        elif chosen_method == "entanglement":
            # Use uncertainties as state representations
            aggregated = self.entanglement_based_aggregation(ensemble_predictions, ensemble_uncertainties)
        elif chosen_method == "voting":
            voting_weights = [1.0 / (1.0 + u.mean().item()) for u in ensemble_uncertainties]
            aggregated = self.quantum_voting_aggregation(ensemble_predictions, voting_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {chosen_method}")

        # Compute aggregation metrics
        metrics = {
            "method_used": chosen_method,
            "average_uncertainty": avg_uncertainty,
            "ensemble_diversity": self._compute_ensemble_diversity(ensemble_predictions),
            "quantum_coherence": self._compute_quantum_coherence(ensemble_predictions)
        }

        return aggregated, metrics

    def _compute_ensemble_diversity(self, ensemble_predictions: List[torch.Tensor]) -> float:
        """Compute diversity measure for ensemble predictions."""
        if len(ensemble_predictions) < 2:
            return 0.0

        # Pairwise correlation diversity
        diversities = []
        for i in range(len(ensemble_predictions)):
            for j in range(i + 1, len(ensemble_predictions)):
                pred_i = ensemble_predictions[i].flatten()
                pred_j = ensemble_predictions[j].flatten()

                correlation = torch.corrcoef(torch.stack([pred_i, pred_j]))[0, 1]
                diversity = 1.0 - abs(correlation.item())
                diversities.append(diversity)

        return np.mean(diversities)

    def _compute_quantum_coherence(self, ensemble_predictions: List[torch.Tensor]) -> float:
        """Compute quantum coherence measure for ensemble."""
        if len(ensemble_predictions) < 2:
            return 1.0

        # Compute coherence as phase alignment
        phases = []
        for pred in ensemble_predictions:
            phase = torch.angle(torch.complex(pred, torch.zeros_like(pred)))
            phases.append(phase.flatten())

        stacked_phases = torch.stack(phases)
        phase_variance = torch.var(stacked_phases, dim=0).mean()

        # Coherence is inversely related to phase variance
        coherence = torch.exp(-phase_variance).item()

        return coherence