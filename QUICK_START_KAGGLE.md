# Quick Start: Run the CBT DeBERTa HQDE Notebook on Kaggle

Use this guide for:

```text
examples/cbt_deberta_hqde_kaggle.ipynb
```

This notebook is self-contained and uses generated toy CBT-style text. It is useful for testing the DeBERTa ensemble workflow, not for claiming clinical performance.

## Setup

1. Go to `https://www.kaggle.com/code`.
2. Create a new notebook or upload `examples/cbt_deberta_hqde_kaggle.ipynb`.
3. In notebook settings, choose a GPU accelerator when available.
4. Run all cells in order.

The notebook automatically detects available hardware:

| Hardware | Behavior |
|----------|----------|
| 2 CUDA GPUs | Uses up to 4 workers, distributed across GPUs. |
| 1 CUDA GPU | Uses fewer workers on the single GPU. |
| CPU | Runs only for smoke/debug purposes; full DeBERTa training will be slow. |

## Smoke Mode

For a short validation run, set:

```python
import os
os.environ["HQDE_QUICK_TEST"] = "1"
```

Then run from the imports/configuration cells onward. Smoke mode reduces workers, sequence length, batch size, and epochs.

## What the Notebook Produces

The final cell prints metrics from the current run:

- Test accuracy.
- Weighted F1 score.
- Classification report.
- Per-class accuracy.
- Individual worker accuracy.
- Confusion matrix saved to `./cbt_output/confusion_matrix.png`.

This guide intentionally does not list target accuracy numbers. Results depend on hardware, package versions, random seed, and the toy data split.

## Cognitive-Distortion Classes

The toy task uses 10 labels:

1. All-or-Nothing Thinking
2. Overgeneralization
3. Mental Filter
4. Disqualifying the Positive
5. Jumping to Conclusions
6. Magnification/Catastrophizing
7. Emotional Reasoning
8. Should Statements
9. Labeling
10. Personalization

## Configuration Fields

The main settings are in the `Config` class:

```python
model_name = "microsoft/deberta-v3-base"
num_classes = 10
max_length = 256
num_workers = ...
batch_size = ...
num_epochs = 15
learning_rate = 2e-5
weight_decay = 0.01
warmup_ratio = 0.1
use_amp = ...
```

The patched notebook computes `num_workers`, `num_gpus`, `pin_memory`, and AMP usage from the current runtime.

## Troubleshooting

### CUDA out of memory

Reduce:

```python
batch_size = 4
max_length = 128
num_workers = 1
```

### CPU run is too slow

Use `HQDE_QUICK_TEST=1` or run on Kaggle GPU. Full DeBERTa training on CPU is not practical.

### HuggingFace download fails

Check internet access in the notebook and rerun the install/import cells. Kaggle sometimes needs the notebook internet toggle enabled.

### Results look unstable

The synthetic dataset is very small. For thesis-quality reporting, run repeated seeds and use a properly documented dataset.

## Reporting Checklist

Before citing results:

- [ ] Notebook filename and git commit recorded.
- [ ] Dataset source recorded.
- [ ] Hardware and GPU count recorded.
- [ ] Seed and package versions recorded.
- [ ] Full output saved.
- [ ] Synthetic toy results not presented as clinical validation.
