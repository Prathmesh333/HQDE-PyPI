# HQDE - Hierarchical Quantum-Distributed Ensemble Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Ray](https://img.shields.io/badge/Ray-optional-green.svg)](https://ray.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.12-brightgreen.svg)](https://pypi.org/project/hqde/)

HQDE is a PyTorch research framework for scalable ensemble learning. It provides a common training API for multiple model replicas, optional Ray-backed workers, epoch-level FedAvg-style synchronization, adaptive delta quantization, and quantum-inspired aggregation utilities.

The project is intended for experimentation and thesis research. Reported accuracy, runtime, and memory numbers should come from your own executed benchmark logs or notebooks; this README does not claim fixed benchmark results.

## What Works Today

| Area | Current status |
|------|----------------|
| Vision/CNN models | Supported through `HQDESystem` with standard `(data, target)` PyTorch dataloaders. |
| Ensemble training | `independent` mode for diverse workers, `fedavg` mode for epoch-level weight averaging. |
| Prediction aggregation | Mean or efficiency-weighted logit aggregation across workers. |
| Quantized communication | Available during `fedavg` aggregation through `AdaptiveQuantizer`. |
| Ray execution | Used when Ray is installed and available; otherwise HQDE falls back to local workers. |
| Transformer modules | Built-in transformer model classes and text utilities exist. Core `HQDESystem` currently expects tensor or tuple batches, not dict batches. |
| DeBERTa CBT notebook | Standalone Kaggle notebook with custom worker code that handles `input_ids`, `attention_mask`, and `labels`. |

## Installation

From PyPI:

```bash
pip install hqde
```

From source:

```bash
git clone https://github.com/Prathmesh333/HQDE-PyPI.git
cd HQDE-PyPI
pip install -e .
```

For development tests, install the dev extras or install pytest separately:

```bash
pip install -e ".[dev]"
```

## Quick Start: Vision Ensemble

```python
from hqde import SmallImageResNet18, create_hqde_system, make_cifar_training_config

training_config = make_cifar_training_config(
    ensemble_mode="independent",
    batch_assignment="replicate",
    prediction_aggregation="mean",
    use_amp=True,
)

hqde_system = create_hqde_system(
    model_class=SmallImageResNet18,
    model_kwargs={"num_classes": 10},
    num_workers=4,
    training_config=training_config,
)

metrics = hqde_system.train(
    train_loader,
    num_epochs=20,
    validation_loader=test_loader,
)

eval_metrics = hqde_system.evaluate(test_loader)
predictions = hqde_system.predict(test_loader)
hqde_system.cleanup()
```

The dataloader must yield either `(data, targets)` or a compatible list/tuple. Current core training does not consume dict batches.

## Training Modes

### Independent Ensemble

```python
training_config = {
    "ensemble_mode": "independent",
    "batch_assignment": "replicate",
    "prediction_aggregation": "mean",
}
```

Each worker receives the same batch and trains its own model copy. Workers remain diverse during training, and predictions are aggregated at inference time.

### FedAvg-Style Epoch Aggregation

```python
training_config = {
    "ensemble_mode": "fedavg",
    "batch_assignment": "split",
    "training_aggregation": "sample_weighted",
    "server_optimizer": "fedadam",
    "federated_normalization": "local_bn",
}
```

Each batch is split across workers. At the end of each epoch, HQDE aggregates model deltas and broadcasts the server state back to workers. This is local-SGD/FedAvg-style training, not PyTorch DDP.

## Quantization

Quantization is applied to model deltas during `fedavg` aggregation when a `quantization_config` is supplied.

```python
quantization_config = {
    "base_bits": 12,
    "min_bits": 8,
    "max_bits": 16,
    "block_size": 1024,
    "warmup_rounds": 5,
    "skip_bias": True,
    "skip_norm": True,
    "error_feedback": True,
}
```

Small tensors, bias tensors, normalization tensors, and non-floating tensors are skipped by default. Compression ratios depend on model size, selected bit widths, and how many tensors are skipped.

## Transformer Status

HQDE ships these transformer classes:

| Model | File | Purpose |
|-------|------|---------|
| `LightweightTransformerClassifier` | `hqde/models/transformers.py` | Small text-classification experiments. |
| `TransformerTextClassifier` | `hqde/models/transformers.py` | General encoder-based text classification. |
| `CBTTransformerClassifier` | `hqde/models/transformers.py` | CBT-themed classifier with optional domain adapter. |

Important current limitation:

- `TextDataLoader` returns dict batches with `input_ids`, `attention_mask`, and `labels`.
- `HQDESystem.train()` currently expects tuple/list batches and calls models as `model(data)`.
- Therefore the documented dict-batch text utility path is not yet fully plug-and-play with core `HQDESystem`.

Working options today:

1. Use the transformer classes directly outside `HQDESystem`.
2. Use tuple-style dataloaders such as `(input_ids, labels)` for simple built-in transformer experiments where `attention_mask` can be omitted.
3. Use the DeBERTa Kaggle notebook for masked HuggingFace-style transformer training; it has custom worker code for dict batches.
4. Add dict-batch support to `HQDESystem` before presenting transformer support as fully plug-and-play.

See [docs/TRANSFORMER_EXTENSION.md](docs/TRANSFORMER_EXTENSION.md) for details.

## Kaggle DeBERTa CBT Notebook

The notebook [examples/cbt_deberta_hqde_kaggle.ipynb](examples/cbt_deberta_hqde_kaggle.ipynb) is a standalone demonstration for CBT cognitive-distortion classification using DeBERTa workers.

It now supports:

- 2x T4 Kaggle GPU execution.
- Single-GPU and CPU smoke-test execution.
- Dynamic device selection instead of hard-coded `cuda:0` and `cuda:1`.
- Safe AMP usage only on CUDA.
- `HQDE_QUICK_TEST=1` for short smoke runs.
- Hugging Face dataset loading via `danthareja/cognitive-distortion` by default.
- Exact-text overlap checks across train, validation, and test splits.

The notebook keeps a synthetic fallback for offline smoke tests via `HQDE_DATASET_SOURCE=synthetic`. Treat synthetic fallback metrics as runtime validation only, not clinical evidence or a benchmark.

## Project Layout

```text
hqde/
  core/
    hqde_system.py              # HQDESystem, workers, FedAvg, quantization
  models/
    vision.py                   # SmallImageResNet18
    transformers.py             # Built-in transformer classifiers
  quantum/
    quantum_aggregator.py       # Quantum-inspired aggregation utilities
    quantum_noise.py
    quantum_optimization.py
  distributed/
    mapreduce_ensemble.py
    hierarchical_aggregator.py
    fault_tolerance.py
    load_balancer.py
  utils/
    data_utils.py
    text_data_utils.py
    training_presets.py
    transformer_presets.py
    performance_monitor.py
```

## Common Commands

```bash
python examples/quick_start.py
python test_imports.py
python test_transformer_integration.py
python validate_notebook.py
```

If pytest is installed:

```bash
python -m pytest -q
```

## Documentation

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Installation and usage guide.
- [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) - Technical architecture notes.
- [docs/TRANSFORMER_EXTENSION.md](docs/TRANSFORMER_EXTENSION.md) - Transformer support status and usage.
- [README_KAGGLE_NOTEBOOKS.md](README_KAGGLE_NOTEBOOKS.md) - Kaggle notebook overview.
- [QUICK_START_KAGGLE.md](QUICK_START_KAGGLE.md) - Kaggle notebook run instructions.

## Results Policy

Do not cite placeholder accuracy, runtime, memory, or compression numbers as thesis results. For thesis reporting:

1. Run the exact script or notebook.
2. Save the command, hardware, seed, package versions, and output files.
3. Report mean and variance across repeated runs where possible.
4. Distinguish synthetic toy data from real datasets.

## Citation

```bibtex
@software{hqde2026,
  title={HQDE: Hierarchical Quantum-Distributed Ensemble Learning},
  author={Prathamesh Nikam},
  year={2026},
  url={https://github.com/Prathmesh333/HQDE-PyPI}
}
```

## License

MIT License. See [LICENSE](LICENSE).
