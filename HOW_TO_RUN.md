# HQDE Framework - Installation and Usage Guide

This guide provides instructions for installing and running the HQDE (Hierarchical Quantum-Distributed Ensemble Learning) framework.

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **GPU**: CUDA-compatible GPU (optional but recommended)
- **Memory**: At least 4GB RAM
- **Storage**: 2GB free space

### Dependencies
The framework automatically installs:
- PyTorch (≥2.8.0)
- Ray (≥2.49.2)
- NumPy (≥2.0.2)
- scikit-learn (≥1.6.1)
- psutil (≥7.1.0)

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

### Using PYTHONPATH (Development)
```powershell
# Windows PowerShell
cd "path\to\HQDE-PyPI"
$env:PYTHONPATH = "."
python examples/cifar10_synthetic_test.py
```

```bash
# Linux/Mac
cd "/path/to/HQDE-PyPI"
export PYTHONPATH=.
python examples/cifar10_synthetic_test.py
```

---

## Quick Start

```python
from hqde import create_hqde_system
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# Create HQDE system
hqde_system = create_hqde_system(
    model_class=MyModel,
    model_kwargs={'num_classes': 10},
    num_workers=4,
    quantization_config={'base_bits': 8, 'min_bits': 4, 'max_bits': 16},
    aggregation_config={'noise_scale': 0.005, 'exploration_factor': 0.1}
)

# Train
metrics = hqde_system.train(train_loader, num_epochs=10)

# Predict
predictions = hqde_system.predict(test_loader)

# Cleanup
hqde_system.cleanup()
```

---

## Examples

| Example | Description | Command |
|---------|-------------|---------|
| Quick Start | Basic demonstration | `python examples/quick_start.py` |
| Synthetic CIFAR-10 | Comprehensive benchmark | `python examples/cifar10_synthetic_test.py` |
| Real CIFAR-10 | Full dataset test | `python examples/cifar10_test.py` |

---

## Configuration

### Quantization
```python
quantization_config = {
    'base_bits': 8,    # Default precision
    'min_bits': 4,     # High compression
    'max_bits': 16     # High precision
}
```

### Quantum Aggregation
```python
aggregation_config = {
    'noise_scale': 0.005,
    'exploration_factor': 0.1,
    'entanglement_strength': 0.1
}
```

### Distributed Workers
```python
num_workers = 4  # Adjust based on available CPU cores/GPUs
```

---

## Performance Benchmarks

| Metric | Traditional Ensemble | HQDE | Improvement |
|--------|---------------------|------|-------------|
| Memory Usage | 2.4 GB | 0.6 GB | 4x reduction |
| Training Time | 45 min | 12 min | 3.75x faster |
| Communication | 800 MB | 100 MB | 8x less data |

---

## Troubleshooting

### Module Not Found
```bash
pip install hqde
# or
export PYTHONPATH=.  # from project root
```

### Ray Issues
```python
import ray
ray.shutdown()
ray.init(ignore_reinit_error=True)
```

### CUDA Out of Memory
- Reduce `batch_size` to 32
- Reduce `num_workers` to 2
- Use CPU: set `device = "cpu"`

---

## Documentation

For detailed technical documentation, see:
- [README.md](README.md) - Overview and API reference
- [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) - Complete technical guide
- [docs/QUANTUM_ALGORITHMS_DEEP_DIVE.md](docs/QUANTUM_ALGORITHMS_DEEP_DIVE.md) - Quantum-inspired algorithms
- [docs/DISTRIBUTED_COMPUTING_DEEP_DIVE.md](docs/DISTRIBUTED_COMPUTING_DEEP_DIVE.md) - Distributed architecture
- [docs/META_LEARNING_AND_QA.md](docs/META_LEARNING_AND_QA.md) - Meta-learning and FAQ

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Prathmesh333/HQDE-PyPI/issues)
- **Repository**: https://github.com/Prathmesh333/HQDE-PyPI