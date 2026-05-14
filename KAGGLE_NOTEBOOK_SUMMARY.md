# Kaggle Notebook Summary

This repository includes a self-contained Kaggle notebook:

```text
examples/cbt_deberta_hqde_kaggle.ipynb
```

The notebook demonstrates a DeBERTa ensemble workflow for CBT cognitive-distortion classification on generated toy data.

## Current Notebook Status

- Valid Jupyter notebook JSON.
- 11 cells: 1 markdown cell and 10 code cells.
- Dynamic device handling for 2 GPUs, 1 GPU, or CPU smoke mode.
- AMP enabled only when CUDA is available.
- DataLoader worker count selected from the runtime environment.
- `HQDE_QUICK_TEST=1` supported for short smoke validation.

## What It Produces

The final notebook cell prints metrics from the current run:

- Test accuracy.
- Weighted F1 score.
- Classification report.
- Per-class accuracy.
- Individual worker metrics.
- Confusion matrix saved under `cbt_output/`.

No fixed expected accuracy is documented here. Run the notebook and record the actual output.

## Important Caveat

The default dataset is synthetic and small. It is suitable for checking the training pipeline, not for clinical claims or final thesis conclusions.

## Recommended Reporting Practice

When using notebook output:

1. Save the executed notebook.
2. Record the git commit.
3. Record package versions and hardware.
4. Record whether `HQDE_QUICK_TEST` was enabled.
5. Include the generated classification report and confusion matrix.
