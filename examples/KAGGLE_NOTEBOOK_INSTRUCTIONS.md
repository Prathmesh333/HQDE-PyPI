# Kaggle Notebook Instructions

Use these instructions for:

```text
examples/cbt_deberta_hqde_kaggle.ipynb
```

The notebook is a demonstration of an ensemble DeBERTa workflow for CBT cognitive-distortion classification. It uses generated toy data unless you switch to the real-data notebook and provide a dataset.

## Run on Kaggle

1. Open Kaggle Notebooks.
2. Upload `examples/cbt_deberta_hqde_kaggle.ipynb`.
3. Enable a GPU accelerator in settings if available.
4. Run the install cell.
5. Run cells in order.

The notebook dynamically detects GPU count. It is no longer hard-coded to require exactly two GPUs.

## Smoke Mode

For a short functionality check:

```python
import os
os.environ["HQDE_QUICK_TEST"] = "1"
```

Set this before the configuration cell. Smoke mode uses fewer workers, shorter sequence length, smaller batch size, and one epoch.

## What the Notebook Does

1. Imports dependencies and prints runtime hardware.
2. Builds a configuration from detected hardware.
3. Loads `danthareja/cognitive-distortion` from Hugging Face by default.
4. Tokenizes text with a HuggingFace tokenizer.
5. Defines a DeBERTa classifier.
6. Defines ensemble workers that pass `input_ids` and `attention_mask`.
7. Trains each worker.
8. Averages worker logits for ensemble evaluation.
9. Prints metrics from the current run.
10. Saves a confusion matrix.

## Outputs

The final cell reports:

- Accuracy from the current run.
- Weighted F1 from the current run.
- Classification report.
- Per-class accuracy.
- Individual worker performance.
- Confusion matrix path.
- Exact text overlap counts for train/validation/test splits.

This document intentionally does not provide fixed expected accuracy. Results depend on hardware, random seed, package versions, and dataset quality.

## Hardware Notes

- Full DeBERTa training should use GPU.
- CPU mode is mainly for smoke tests.
- If you get CUDA out-of-memory, reduce `batch_size`, `max_length`, or `num_workers`.
- Kaggle internet must be enabled for the default Hugging Face dataset path.
- Use `HQDE_DATASET_SOURCE=synthetic` only for offline runtime checks.

## Thesis Reporting Notes

For thesis use, record:

- Notebook filename.
- Git commit hash.
- Dataset source, for example `danthareja/cognitive-distortion`.
- Train/validation/test split.
- Exact-overlap check output.
- Random seed.
- Hardware and GPU count.
- Package versions.
- Full output metrics.

Do not present synthetic toy-data metrics as a production or clinical result.

## Multi-Dataset Comparison

For a final thesis or paper table, run:

```bash
python examples/cbt_multi_dataset_comparison.py --backend ray --ray-gpus-per-worker 0.25 --label-mode canonical10 --epochs 5 --max-train-samples 1000 --max-eval-samples 300
```

From the Kaggle UI, use `examples/cbt_multi_dataset_hqde_kaggle_2xT4.ipynb`.
It is preconfigured for 2xT4, 4 HQDE ensemble Ray actors, 4 vCPUs, canonical 10-label CBT mapping, pre-tokenization, `num_gpus=0.25` per worker, and coordinator-side ensemble aggregation.

Use `--quick-test --dry-run` first to verify dataset availability and exact
overlap checks without training:

```bash
python examples/cbt_multi_dataset_comparison.py --quick-test --dry-run
```

The script exports CSV, JSON, and Markdown tables under
`benchmark_outputs/cbt_multi_dataset/`.
