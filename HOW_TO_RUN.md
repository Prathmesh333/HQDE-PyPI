# HQDE Installation and Usage Guide

This guide describes how to install and run the current HQDE package from this repository. It avoids fixed benchmark claims; use your own executed runs for thesis results.

## Requirements

- Python 3.9 or newer.
- PyTorch.
- Ray is optional. If Ray is unavailable, HQDE uses local workers.
- CUDA is optional but recommended for non-trivial experiments.

## Installation

From PyPI:

```bash
pip install hqde
```

From this source tree:

```bash
git clone https://github.com/Prathmesh333/HQDE-PyPI.git
cd HQDE-PyPI
pip install -e .
```

Development install:

```bash
pip install -e ".[dev]"
```

On Windows PowerShell, when running scripts directly from the repo:

```powershell
cd "D:\MTech 2nd Year\hqde\HQDE-PyPI"
$env:PYTHONPATH = "."
python examples/quick_start.py
```

## Quick Start

```python
from hqde import SmallImageResNet18, create_hqde_system, make_cifar_training_config

training_config = make_cifar_training_config(
    ensemble_mode="independent",
    batch_assignment="replicate",
    prediction_aggregation="mean",
)

system = create_hqde_system(
    model_class=SmallImageResNet18,
    model_kwargs={"num_classes": 10},
    num_workers=2,
    training_config=training_config,
)

metrics = system.train(train_loader, num_epochs=5, validation_loader=test_loader)
eval_metrics = system.evaluate(test_loader)
predictions = system.predict(test_loader)
system.cleanup()
```

## Core Data Format

`HQDESystem` currently expects training and evaluation dataloaders to yield:

```python
(data, targets)
```

or an equivalent tuple/list where `batch[0]` is the input tensor and `batch[1]` is the target tensor.

Dict batches such as `{"input_ids": ..., "attention_mask": ..., "labels": ...}` are not yet supported by the core training loop. The DeBERTa Kaggle notebook uses custom worker code for that style of transformer batch.

## Training Modes

### Independent

```python
training_config = {
    "ensemble_mode": "independent",
    "batch_assignment": "replicate",
}
```

Each worker trains on the full batch. No weight averaging happens at epoch end. Prediction aggregates worker logits.

### FedAvg

```python
training_config = {
    "ensemble_mode": "fedavg",
    "batch_assignment": "split",
    "training_aggregation": "sample_weighted",
}
```

Each batch is split across workers. After each epoch, HQDE aggregates deltas and broadcasts the server model back to workers.

## Optional Quantization

Quantization applies only during `fedavg` aggregation.

```python
quantization_config = {
    "base_bits": 12,
    "min_bits": 8,
    "max_bits": 16,
    "block_size": 1024,
    "warmup_rounds": 5,
}

system = create_hqde_system(
    model_class=SmallImageResNet18,
    model_kwargs={"num_classes": 10},
    num_workers=4,
    training_config={
        "ensemble_mode": "fedavg",
        "batch_assignment": "split",
    },
    quantization_config=quantization_config,
)
```

Compression depends on tensor sizes and skipped parameters. Measure it with:

```python
system.get_performance_metrics()
system.ensemble_manager.get_quantization_metrics()
```

## Examples

| Example | Command | Notes |
|---------|---------|-------|
| Basic package smoke | `python examples/quick_start.py` | Small runnable example. |
| Import smoke | `python test_imports.py` | Checks package imports. |
| Transformer smoke | `python test_transformer_integration.py` | Currently shows the dict-batch HQDE integration gap. |
| Kaggle notebook validation | `python validate_notebook.py` | Checks notebook JSON structure. |

If pytest is installed:

```bash
python -m pytest -q
```

## Kaggle DeBERTa Notebook

Use [examples/cbt_deberta_hqde_kaggle.ipynb](examples/cbt_deberta_hqde_kaggle.ipynb) for the CBT DeBERTa demonstration.

The notebook is self-contained and creates a toy dataset. It has dynamic device handling:

- 2x T4 on Kaggle: multiple DeBERTa workers.
- 1 GPU: fewer workers automatically.
- CPU or smoke mode: set `HQDE_QUICK_TEST=1` before execution for a short run.

Do not treat the toy dataset metrics as benchmark evidence.

## Troubleshooting

### `ModuleNotFoundError: hqde`

Run from the repository root with `PYTHONPATH=.` or install the package:

```bash
pip install -e .
```

### Ray startup issues

Ray is optional. If Ray is installed but in a bad state:

```python
import ray
ray.shutdown()
```

Then restart the process.

### CUDA out of memory

- Reduce batch size.
- Reduce number of workers.
- Use `ensemble_mode="fedavg"` with `batch_assignment="split"`.
- Use a smaller model or shorter sequence length.

### Transformer dict-batch failure

If you see:

```text
ValueError: Training batches must contain at least (data, targets)
```

you are passing a dict dataloader into core `HQDESystem`. Use tuple batches, use the standalone notebook worker, or update `HQDESystem` to support dict batches.

## Documentation

- [README.md](README.md)
- [docs/TRANSFORMER_EXTENSION.md](docs/TRANSFORMER_EXTENSION.md)
- [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)
- [docs/QUANTUM_ALGORITHMS_DEEP_DIVE.md](docs/QUANTUM_ALGORITHMS_DEEP_DIVE.md)
- [docs/DISTRIBUTED_COMPUTING_DEEP_DIVE.md](docs/DISTRIBUTED_COMPUTING_DEEP_DIVE.md)
