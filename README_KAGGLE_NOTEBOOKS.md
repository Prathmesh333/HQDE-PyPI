# CBT DeBERTa HQDE Kaggle Notebooks

This repository contains Kaggle-oriented notebooks for demonstrating an ensemble of DeBERTa classifiers on CBT cognitive-distortion text classification.

These notebooks are examples, not validated clinical systems. The synthetic notebook uses a generated toy dataset. Any accuracy or F1 score must be taken from your own executed run and reported with hardware, seed, package versions, and dataset details.

## Available Notebooks

| Notebook | Dataset | Status |
|----------|---------|--------|
| `examples/cbt_deberta_hqde_kaggle.ipynb` | Self-contained synthetic toy data | Updated for dynamic GPU/CPU handling and smoke mode. |
| `examples/cbt_deberta_hqde_kaggle_REAL_DATA.ipynb` | External JSON dataset | Requires the dataset to be uploaded and added in Kaggle. |

## Synthetic Notebook

File:

```text
examples/cbt_deberta_hqde_kaggle.ipynb
```

What it does:

- Generates 100 balanced CBT-style toy conversations inside the notebook.
- Builds a train/validation/test split.
- Fine-tunes DeBERTa worker models with different dropout and learning-rate settings.
- Averages worker logits for ensemble prediction.
- Writes a confusion matrix to `./cbt_output/confusion_matrix.png`.

Runtime behavior:

- Uses up to 2 detected CUDA GPUs.
- Uses fewer workers automatically when only 1 GPU is available.
- Runs in CPU-compatible smoke mode when no GPU is available.
- Enables AMP only on CUDA.
- Supports quick smoke execution with `HQDE_QUICK_TEST=1`.

## Real-Data Notebook

File:

```text
examples/cbt_deberta_hqde_kaggle_REAL_DATA.ipynb
```

Expected dataset file:

```text
cbt_therapy_conversations_full.json
```

Use this notebook only after adding the dataset to the Kaggle notebook environment. The repository does not guarantee fixed results for this file; run it and record the actual metrics.

## Kaggle Setup

1. Open Kaggle Notebooks.
2. Upload the notebook.
3. In settings, choose a GPU accelerator. The synthetic notebook can run with CPU for smoke checks but full DeBERTa training should use GPU.
4. Run the install cell.
5. Run the notebook cells in order.

For a short smoke run, set this before running the configuration cell:

```python
import os
os.environ["HQDE_QUICK_TEST"] = "1"
```

## Outputs

The final cell prints:

- Ensemble test accuracy from the current run.
- Weighted F1 score from the current run.
- Classification report.
- Per-class accuracy.
- Individual worker accuracy.
- Confusion matrix image path.

No fixed expected accuracy is documented here because the result depends on runtime, seeds, hardware, package versions, and whether the dataset is synthetic or real.

## Notes for Thesis Reporting

When using notebook results in a thesis:

1. State whether the dataset is synthetic or real.
2. Include the exact notebook filename and commit hash.
3. Include hardware, GPU count, runtime, seed, and package versions.
4. Avoid presenting the synthetic toy dataset as clinical validation.
5. Repeat runs if you need stable comparisons.

## Related Docs

- [QUICK_START_KAGGLE.md](QUICK_START_KAGGLE.md)
- [REAL_DATASET_INSTRUCTIONS.md](REAL_DATASET_INSTRUCTIONS.md)
- [examples/KAGGLE_NOTEBOOK_INSTRUCTIONS.md](examples/KAGGLE_NOTEBOOK_INSTRUCTIONS.md)
- [docs/TRANSFORMER_EXTENSION.md](docs/TRANSFORMER_EXTENSION.md)
