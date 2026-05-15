# CBT Text Classification Examples

This document describes the CBT text-classification examples in this repository.

The examples are research demonstrations. They are not clinical tools, and they do not establish medical validity. Metrics must be generated from actual runs and documented with hardware, data, seed, and package versions.

## Cognitive-Distortion Labels

The examples use 10 common CBT cognitive-distortion categories:

| ID | Label |
|----|-------|
| 0 | All-or-Nothing Thinking |
| 1 | Overgeneralization |
| 2 | Mental Filter |
| 3 | Disqualifying the Positive |
| 4 | Jumping to Conclusions |
| 5 | Magnification/Catastrophizing |
| 6 | Emotional Reasoning |
| 7 | Should Statements |
| 8 | Labeling |
| 9 | Personalization |

## Available Examples

| File | Description |
|------|-------------|
| `examples/cbt_transformer_example.py` | Uses built-in HQDE transformer utilities. Current core dict-batch limitation may need code changes before this path is fully reliable. |
| `examples/cbt_deberta_hqde_kaggle.ipynb` | Standalone Kaggle DeBERTa ensemble notebook with custom dict-batch worker code. |
| `examples/cbt_deberta_hqde_kaggle.py` | Script form of the DeBERTa workflow. Review before running because it may not include the latest notebook safety patches. |
| `examples/cbt_deberta_hqde_kaggle_REAL_DATA.ipynb` | Real-data notebook variant requiring an external JSON dataset. |

## Recommended Current Path

For masked HuggingFace-style transformer training, use:

```text
examples/cbt_deberta_hqde_kaggle.ipynb
```

That notebook passes `input_ids` and `attention_mask` directly into DeBERTa workers and is not blocked by the core `HQDESystem` tuple-batch assumption.

## Core HQDE Limitation

The package-level text datasets return dict batches:

```python
{
    "input_ids": tensor,
    "attention_mask": tensor,
    "labels": tensor,
}
```

Core `HQDESystem.train()` currently expects tuple/list batches:

```python
(data, targets)
```

Until dict-batch support is implemented in `hqde/core/hqde_system.py`, do not document `TextDataLoader` plus `HQDESystem` as a fully working plug-and-play path.

## Kaggle Notebook Features

The Kaggle notebook:

- Loads `danthareja/cognitive-distortion` from Hugging Face by default.
- Uses the dataset's label metadata to configure the classifier dynamically.
- Deduplicates exact `(text, label)` pairs before splitting.
- Checks exact text overlap across train/validation/test sets.
- Keeps a generated synthetic dataset only as an offline smoke-test fallback.
- Loads `microsoft/deberta-v3-base`.
- Creates multiple workers with different dropout and learning-rate settings.
- Averages worker logits for ensemble prediction.
- Saves a confusion matrix.
- Detects available GPU count dynamically.
- Disables AMP on CPU.
- Supports `HQDE_QUICK_TEST=1` for smoke checks.

## Running the Kaggle Notebook

1. Upload `examples/cbt_deberta_hqde_kaggle.ipynb` to Kaggle.
2. Enable internet and a GPU accelerator if available.
3. Run all cells in order.

For smoke mode:

```python
import os
os.environ["HQDE_QUICK_TEST"] = "1"
```

For offline synthetic fallback:

```python
import os
os.environ["HQDE_DATASET_SOURCE"] = "synthetic"
```

## Reporting Results

When reporting results:

- State which notebook/script was used.
- State the dataset source, for example `danthareja/cognitive-distortion`.
- Include seed, hardware, GPU count, package versions, and runtime.
- Include the printed exact-overlap checks.
- Save the classification report and confusion matrix.
- Do not cite toy synthetic data as clinical evidence.

## Multi-Dataset Thesis Table

For a comparison table across datasets, use:

```bash
python examples/cbt_multi_dataset_comparison.py --quick-test --dry-run
python examples/cbt_multi_dataset_comparison.py --label-mode canonical10 --epochs 5 --max-train-samples 1000 --max-eval-samples 300
```

For Kaggle 2xT4, use `examples/cbt_multi_dataset_hqde_kaggle_2xT4.ipynb`.
It configures 4 HQDE ensemble workers, uses canonical 10-label mapping, and writes the same output tables.

The script writes:

- `benchmark_outputs/cbt_multi_dataset/cbt_multi_dataset_comparison.csv`
- `benchmark_outputs/cbt_multi_dataset/cbt_multi_dataset_comparison.json`
- `benchmark_outputs/cbt_multi_dataset/cbt_multi_dataset_comparison.md`

The default rows are:

- `danthareja/cognitive-distortion`
- `halilbabacan/cognitive_distortions_gpt4`
- `elliott-leow/cognitive_distortion_validation`

For the main thesis table, use `--label-mode canonical10` so compatible dataset labels are mapped into the same 10 CBT categories. The output table reports `missing_classes` when a dataset does not contain one of the canonical categories.

## Next Code Improvement

The next framework improvement should be core dict-batch support for:

- `train`
- `evaluate`
- `predict`
- local workers
- Ray workers

That would make `TextDataLoader` and transformer models genuinely plug-and-play through `HQDESystem`.
