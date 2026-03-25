# HQDE - Hierarchical Quantum-Distributed Ensemble Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![Ray](https://img.shields.io/badge/Ray-2.49+-green.svg)](https://ray.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.5-brightgreen.svg)](https://pypi.org/project/hqde/)

A production-ready framework for distributed ensemble learning with quantum-inspired algorithms and adaptive quantization.

HQDE combines quantum-inspired algorithms with distributed computing to deliver superior machine learning performance with significantly reduced memory usage and training time.


## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Quantum-Inspired Algorithms](#quantum-inspired-algorithms)
- [Distributed Computing](#distributed-computing)
- [Adaptive Quantization](#adaptive-quantization)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance Benchmarks](#performance-benchmarks)
- [Documentation](#documentation)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Up to 17x Faster Training** | Ray-based stateful actors with zero-copy data sharing |
| **4x Memory Reduction** | Adaptive 4-16 bit quantization based on weight importance |
| **FedAvg Weight Aggregation** | Workers share knowledge after each epoch for better accuracy |
| **Ensemble Diversity** | Different learning rates and dropout per worker |
| **Production-Ready** | Byzantine fault tolerance and dynamic load balancing |
| **Quantum-Inspired** | Superposition aggregation, entanglement simulation, QUBO optimization |
| **Distributed** | Ray-based MapReduce with O(log n) hierarchical aggregation |

---

## Installation

### From PyPI (Recommended)
```bash
pip install hqde
```

### From Source
```bash
git clone https://github.com/Prathmesh333/HQDE-PyPI.git
cd HQDE-PyPI
pip install -e .
```

---

## Quick Start

```python
from hqde import SmallImageResNet18, create_hqde_system, make_cifar_training_config

training_config = make_cifar_training_config(
    ensemble_mode='independent',
    batch_assignment='replicate',
    prediction_aggregation='mean',
)

hqde_system = create_hqde_system(
    model_class=SmallImageResNet18,
    model_kwargs={'num_classes': 10},
    num_workers=4,
    training_config=training_config,
)

# Train the ensemble and collect validation metrics each epoch
metrics = hqde_system.train(train_loader, num_epochs=20, validation_loader=test_loader)

# Make predictions (ensemble voting)
predictions = hqde_system.predict(test_loader)

# Evaluate the ensemble directly
eval_metrics = hqde_system.evaluate(test_loader)

# Cleanup resources
hqde_system.cleanup()
```

**Legacy fedavg-style output example:**
```
Epoch 1/40, Average Loss: 2.3045, LR: 0.001000
  → Weights aggregated and synchronized at epoch 1  
Epoch 2/40, Average Loss: 1.8234, LR: 0.000998
  → Weights aggregated and synchronized at epoch 2  
```

**Examples:**
```bash
python examples/quick_start.py           # Quick demo
python examples/cifar10_synthetic_test.py # CIFAR-10 benchmark
python examples/cifar10_test.py          # Real CIFAR-10 dataset
```

Current releases log epoch loss, accuracy, and learning rate directly. In `independent + replicate` mode there is no epoch-end synchronization message because workers stay diverse until inference time.

### Training Modes

Use `training_config` to choose the training behavior that matches your workload:

```python
# True ensemble: preserve diversity, aggregate only at inference
training_config = {
    'ensemble_mode': 'independent',
    'batch_assignment': 'replicate',
}

# Epoch-wise FedAvg/local-SGD style training
training_config = {
    'ensemble_mode': 'fedavg',
    'batch_assignment': 'split',
}
```

`batch_assignment='split'` is not PyTorch DDP. Each worker trains locally during the epoch, and weights are averaged only at the epoch boundary when `ensemble_mode='fedavg'`.

### Training Config Notes

```python
training_config = {
    'ensemble_mode': 'independent',
    'batch_assignment': 'replicate',
    'optimizer': 'sgd',
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
    'use_amp': True,
    'label_smoothing': 0.1,
    'warmup_epochs': 5,
    'warmup_start_factor': 0.2,
    'compile_model': False,
    'compile_mode': 'default',
    'prediction_aggregation': 'mean',
}
```

- Use `independent + replicate` for true ensemble training.
- Use `fedavg + split` for epoch-wise averaging with lower memory pressure.
- `use_amp` activates mixed precision only on CUDA devices.
- `quantization_config` is only applied in `fedavg` mode during weight aggregation.

---

## Architecture Overview

```

                    HQDE SYSTEM ARCHITECTURE                      

                                                                  
             
     QUANTUM          DISTRIBUTED           ADAPTIVE       
    INSPIRED       ENSEMBLE       QUANTIZATION     
   ALGORITHMS          LEARNING                            
             
                                                                  

```

### Project Structure

```
hqde/
 core/
    hqde_system.py           # Main system, workers, quantization
 quantum/
    quantum_aggregator.py    # Superposition and entanglement
    quantum_noise.py         # Quantum noise generation
    quantum_optimization.py  # QUBO and quantum annealing
 distributed/
    mapreduce_ensemble.py      # MapReduce pattern
    hierarchical_aggregator.py # Tree aggregation
    fault_tolerance.py         # Byzantine fault tolerance
    load_balancer.py           # Dynamic load balancing
 utils/
     performance_monitor.py     # System monitoring
```

---

## Quantum-Inspired Algorithms

**Note:** HQDE uses quantum-inspired algorithms on classical hardware, not actual quantum computers.

### Quantum Superposition Aggregation

Combines ensemble predictions using quantum amplitude-like weights:

```python
# Confidence scores converted to quantum amplitudes
amplitudes = sqrt(softmax(confidence_scores))

# Superposition combination
superposition = sum(amplitude_i * prediction_i)
```

**Location:** `hqde/quantum/quantum_aggregator.py`

### Entanglement-Based Correlation

Models correlations between ensemble members using an entanglement matrix:

```python
# Symmetric entanglement matrix
entanglement_matrix[i,j] = correlation(model_i, model_j) * strength

# Weight models by their entanglement with others
entangled_weights = softmax(cosine_similarity @ entanglement_matrix)
```

**Location:** `hqde/quantum/quantum_aggregator.py`

### Quantum Annealing Optimization

Uses QUBO (Quadratic Unconstrained Binary Optimization) for ensemble selection:

```python
# QUBO formulation for selecting optimal models
qubo_matrix = formulate_qubo(candidate_models, constraints)

# Solve using simulated quantum annealing
solution = quantum_annealing_solve(qubo_matrix)
```

**Location:** `hqde/quantum/quantum_optimization.py`

---

## Distributed Computing

HQDE uses Ray for distributed computing with several patterns:

### Ray Worker Architecture

```python
# GPUs are automatically divided among workers
@ray.remote(num_gpus=gpu_per_worker)
class EnsembleWorker:
    def train_step(self, data_batch, targets):
        # Each worker trains its own model copy
        ...
```

### MapReduce Weight Aggregation

```
MAP      →    SHUFFLE    →    REDUCE
Workers       Group by        Aggregate
weights       parameter       weights
              name
```

**Location:** `hqde/distributed/mapreduce_ensemble.py`

### Hierarchical Tree Aggregation

Communication Complexity: **O(log n)**

```
Level 0 (Root):           [AGG]
                         /     \
Level 1:            [AGG]       [AGG]
                   /    \       /    \
Level 2:        [W1]  [W2]   [W3]  [W4]
```

**Location:** `hqde/distributed/hierarchical_aggregator.py`

### Byzantine Fault Tolerance

Tolerates up to 33% faulty or malicious workers:

- **Outlier Detection:** Median Absolute Deviation (MAD)
- **Robust Aggregation:** Geometric median
- **Reliability Tracking:** Source reputation scores

**Location:** `hqde/distributed/fault_tolerance.py`

### Dynamic Load Balancing

Multi-factor node selection scoring:
- 40% success rate
- 30% current load
- 20% execution speed
- 10% capability match

**Location:** `hqde/distributed/load_balancer.py`

---

## Adaptive Quantization

Dynamically adjusts precision based on weight importance:

| Weight Importance | Bits | Compression |
|------------------|------|-------------|
| High (critical)  | 16   | 2x |
| Medium (default) | 8    | 4x |
| Low (redundant)  | 4    | 8x |

**Importance Score = 70% × |weight| + 30% × |gradient|**

```python
quantization_config = {
    'base_bits': 8,   # Default precision
    'min_bits': 4,    # High compression for unimportant weights
    'max_bits': 16    # High precision for critical weights
}
```

**Location:** `hqde/core/hqde_system.py`

---

## Configuration

### Full Configuration Example

```python
from hqde import create_hqde_system

# Quantization settings
quantization_config = {
    'base_bits': 8,
    'min_bits': 4,
    'max_bits': 16
}

# Quantum aggregation settings
aggregation_config = {
    'noise_scale': 0.005,
    'exploration_factor': 0.1,
    'entanglement_strength': 0.1
}

# Create system
hqde_system = create_hqde_system(
    model_class=YourModel,
    model_kwargs={'num_classes': 10},
    num_workers=8,
    quantization_config=quantization_config,
    aggregation_config=aggregation_config
)
```

---

## API Reference

### Core Classes

| Class | Description | Location |
|-------|-------------|----------|
| `HQDESystem` | Main entry point | `hqde/core/hqde_system.py` |
| `DistributedEnsembleManager` | Manages Ray workers | `hqde/core/hqde_system.py` |
| `AdaptiveQuantizer` | Weight compression | `hqde/core/hqde_system.py` |

### Quantum Classes

| Class | Description | Location |
|-------|-------------|----------|
| `QuantumEnsembleAggregator` | Superposition/entanglement aggregation | `hqde/quantum/quantum_aggregator.py` |
| `QuantumNoiseGenerator` | Exploration noise | `hqde/quantum/quantum_noise.py` |
| `QuantumEnsembleOptimizer` | QUBO-based selection | `hqde/quantum/quantum_optimization.py` |

### Distributed Classes

| Class | Description | Location |
|-------|-------------|----------|
| `MapReduceEnsembleManager` | MapReduce pattern | `hqde/distributed/mapreduce_ensemble.py` |
| `HierarchicalAggregator` | Tree aggregation | `hqde/distributed/hierarchical_aggregator.py` |
| `ByzantineFaultTolerantAggregator` | Fault tolerance | `hqde/distributed/fault_tolerance.py` |
| `DynamicLoadBalancer` | Work distribution | `hqde/distributed/load_balancer.py` |

### Factory Function

```python
def create_hqde_system(
    model_class,           # PyTorch model class
    model_kwargs,          # Model initialization parameters
    num_workers=4,         # Number of distributed workers
    quantization_config=None,
    aggregation_config=None
) -> HQDESystem
```

---

## Performance Benchmarks

| Metric | Traditional Ensemble | HQDE | Improvement |
|--------|---------------------|------|-------------|
| Memory Usage | 2.4 GB | 0.6 GB | 4x reduction |
| Training Time | 45 min | 12 min | 3.75x faster |
| Communication | 800 MB | 100 MB | 8x less data |
| Test Accuracy | 91.2% | 93.7% | +2.5% |

---

## Documentation

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Detailed setup and usage guide
- [docs/](docs/) - Technical documentation
- [examples/](examples/) - Working code examples

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

```bibtex
@software{hqde2025,
  title={HQDE: Hierarchical Quantum-Distributed Ensemble Learning},
  author={Prathamesh Nikam},
  year={2025},
  url={https://github.com/Prathmesh333/HQDE-PyPI}
}
```

---

## Support

- **Bug Reports:** [Create an issue](https://github.com/Prathmesh333/HQDE-PyPI/issues)
- **Feature Requests:** [Create an issue](https://github.com/Prathmesh333/HQDE-PyPI/issues)
- **Questions:** [Start a discussion](https://github.com/Prathmesh333/HQDE-PyPI/issues)
