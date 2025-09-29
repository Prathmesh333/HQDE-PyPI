# HQDE Framework - How to Run Guide

This guide provides step-by-step instructions for running the HQDE (Hierarchical Quantum-Distributed Ensemble Learning) framework.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **GPU**: CUDA-compatible GPU (optional but recommended)
- **Memory**: At least 4GB RAM
- **Storage**: 2GB free space

### Required Dependencies
The framework automatically installs these dependencies:
- PyTorch (≥2.8.0)
- Ray (≥2.49.2)
- NumPy (≥2.0.2)
- scikit-learn (≥1.6.1)
- psutil (≥7.1.0)
- Other supporting packages (see `pyproject.toml`)

## 🚀 Quick Start

### Option 1: Using PYTHONPATH (Recommended - No Installation)

```powershell
# Navigate to project directory
cd "path\to\hqde-framework"

# Set Python path (Windows PowerShell)
$env:PYTHONPATH = "."

# Run the synthetic CIFAR-10 test
python examples/cifar10_synthetic_test.py
```

**For Linux/Mac:**
```bash
cd "/path/to/hqde-framework"
export PYTHONPATH=.
python examples/cifar10_synthetic_test.py
```

### Option 2: Install as Package

```powershell
# Navigate to project directory
cd "path\to\hqde-framework"

# Install in development mode
pip install -e .

# Run tests from anywhere
python examples/cifar10_synthetic_test.py
```

## 🧪 Available Tests and Examples

### 1. Quick Start Demo
**Purpose**: Basic HQDE functionality demonstration
```powershell
$env:PYTHONPATH = "."
python examples/quick_start.py
```
**Duration**: ~30 seconds
**Output**: Basic ensemble training with dummy data

### 2. Synthetic CIFAR-10 Test (⭐ Recommended)
**Purpose**: Comprehensive test with realistic image classification data
```powershell
$env:PYTHONPATH = "."
python examples/cifar10_synthetic_test.py
```
**Features**:
- 5000 training samples, 1000 test samples
- CNN model with 3 conv + 2 FC layers
- 4 distributed workers
- Quantum aggregation demonstrations
- Complete performance analysis

**Duration**: ~30 seconds
**Expected Accuracy**: ~86%

### 3. Real CIFAR-10 Test
**Purpose**: Test with actual CIFAR-10 dataset
```powershell
$env:PYTHONPATH = "."
python examples/cifar10_test.py
```
**Note**: Downloads 170MB dataset on first run

## 📁 Generated Output Files

After running tests, you'll find these files:

### Quick Start Demo
- `hqde_model_demo.pth` - Trained model
- `hqde_performance_demo.json` - Performance metrics

### CIFAR-10 Tests
- `hqde_synthetic_cifar10_model.pth` - Trained ensemble model (4.6MB)
- `hqde_synthetic_cifar10_performance.json` - Detailed performance data
- `hqde_synthetic_cifar10_results.json` - Comprehensive test results

## 🔧 Configuration Options

### HQDE System Parameters

You can customize the framework by modifying these parameters in the test scripts:

```python
# Model configuration
model_kwargs = {
    'num_classes': 10,          # Number of output classes
    'hidden_size': 128          # Hidden layer size
}

# Quantization settings
quantization_config = {
    'base_bits': 8,             # Base quantization bits
    'min_bits': 4,              # Minimum bits (aggressive compression)
    'max_bits': 16              # Maximum bits (high precision)
}

# Quantum aggregation settings
aggregation_config = {
    'noise_scale': 0.005,       # Quantum noise scale
    'exploration_factor': 0.1   # Exploration strength
}

# Distributed settings
num_workers = 4                 # Number of ensemble workers
```

### Training Parameters

```python
# Training configuration
train_samples = 5000           # Training dataset size
test_samples = 1000            # Test dataset size
batch_size = 64                # Batch size
num_epochs = 5                 # Training epochs
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Module Not Found Error
```
ModuleNotFoundError: No module named 'hqde'
```
**Solution**: Make sure you're in the correct directory and set PYTHONPATH
```powershell
cd "path\to\hqde-framework"
$env:PYTHONPATH = "."
```

#### 2. Ray Initialization Issues
```
Ray cluster initialization failed
```
**Solution**: Ray will automatically start a local cluster. If issues persist:
```python
import ray
ray.shutdown()  # Clean up any existing Ray processes
ray.init(ignore_reinit_error=True)
```

#### 3. CUDA Out of Memory
```
CUDA out of memory
```
**Solutions**:
- Reduce batch size: `batch_size = 32`
- Reduce number of workers: `num_workers = 2`
- Use CPU: `device = "cpu"`

#### 4. Installation Issues
```
setuptools error: Multiple top-level packages discovered
```
**Solution**: The `pyproject.toml` has been fixed. Try:
```powershell
pip install -e .
```
Or use the PYTHONPATH method instead.

#### 5. Performance Issues
**For better performance**:
- Use GPU: Install CUDA and PyTorch with CUDA support
- Increase workers: `num_workers = 8` (adjust based on your CPU cores)
- Use SSD storage for faster I/O

## 📊 Expected Results

### Performance Benchmarks

| Test | Accuracy | Training Time | Memory Usage |
|------|----------|---------------|--------------|
| Quick Start | ~90% | 5 seconds | <50MB |
| Synthetic CIFAR-10 | ~86% | 18 seconds | <100MB |
| Real CIFAR-10 | ~85-90% | 60+ seconds | <200MB |

### Key Metrics to Look For

1. **Training Convergence**: Loss should decrease from ~2.5 to ~0.7
2. **Test Accuracy**: Should achieve 85%+ on CIFAR-10
3. **Memory Efficiency**: <100MB GPU memory usage
4. **Quantum Diversity**: Ensemble diversity >95%
5. **Processing Speed**: <30 seconds for 5K samples

## 🎯 Understanding the Output

### Training Progress
```
Epoch 1/5, Average Loss: 2.3557, Samples: 5000
Epoch 2/5, Average Loss: 1.6817, Samples: 5000
...
Epoch 5/5, Average Loss: 0.7240, Samples: 5000
```

### Test Results
```
=== Evaluation Performance ===
Test Accuracy: 0.8610 (86.10%)
Test Loss: 0.4054
Accuracy Grade: Good
```

### Per-Class Performance
```
=== Per-Class Accuracy ===
  airplane: 0.930 (93.0%)
  automobile: 0.860 (86.0%)
  bird: 0.920 (92.0%)
  ...
```

### Quantum Features
```
=== Quantum Features Performance ===
Average Ensemble Diversity: 0.968
Quantum Noise Effectiveness: 0.0010
```

## 🔬 Advanced Usage

### Custom Models
To use your own model with HQDE:

```python
import torch.nn as nn
from hqde import create_hqde_system

class CustomModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# Create HQDE system with custom model
hqde_system = create_hqde_system(
    model_class=CustomModel,
    model_kwargs={'input_size': 784, 'num_classes': 10},
    num_workers=4
)
```

### Custom Data Loaders
```python
class CustomDataLoader:
    def __init__(self, data, labels, batch_size=32):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __iter__(self):
        # Your data loading logic
        pass

# Use with HQDE
hqde_system.train(data_loader=CustomDataLoader(data, labels))
```

## 📚 Next Steps

1. **Experiment with Parameters**: Try different quantization settings
2. **Scale Up**: Test with larger datasets and more workers
3. **Custom Applications**: Adapt for your specific ML tasks
4. **Performance Tuning**: Optimize for your hardware configuration
5. **Research Applications**: Explore quantum-inspired algorithms

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the console output
2. **Verify dependencies**: Ensure all required packages are installed
3. **Check system resources**: Monitor CPU, memory, and GPU usage
4. **Review configuration**: Verify your parameter settings
5. **Test with smaller datasets**: Start with the quick start demo

## 📞 Support

For additional support or questions about the HQDE framework, please refer to:
- Project documentation in `README.md`
- Example code in `examples/` directory
- Performance results in generated JSON files

---

**🎉 You're ready to explore the HQDE framework! Start with the synthetic CIFAR-10 test for the best demonstration of all features.**