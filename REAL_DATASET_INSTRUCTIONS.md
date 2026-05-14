# Real Dataset Notebook Instructions

Use this guide for:

```text
examples/cbt_deberta_hqde_kaggle_REAL_DATA.ipynb
```

This notebook variant expects an external JSON dataset. It should be used only after the dataset is uploaded to Kaggle and attached to the notebook.

## Required Dataset

Expected file name:

```text
cbt_therapy_conversations_full.json
```

Before running:

1. Upload the JSON file as a Kaggle dataset.
2. Add that dataset to the notebook through Kaggle's "Add Data" panel.
3. Check the dataset path inside the notebook before running the data-loading cell.

## Running

1. Upload the notebook to Kaggle.
2. Add the dataset.
3. Enable a GPU accelerator when available.
4. Run the cells in order.

## Outputs

The final cells should report metrics from the current run and save visualizations. This document intentionally does not provide target accuracy numbers because those must come from the actual run.

## Reporting

For thesis reporting, include:

- Dataset source and labeling process.
- Number of samples per class.
- Train/validation/test split.
- Hardware and runtime.
- Random seed.
- Package versions.
- Full classification report.

Do not compare real-data and synthetic-data notebooks using undocumented or placeholder metrics.
