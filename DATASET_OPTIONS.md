# Dataset Options

HQDE CBT examples can be run with either generated toy data or a separately provided dataset. The choice changes what conclusions can be drawn.

## Option 1: Synthetic Toy Data

Used by:

```text
examples/cbt_deberta_hqde_kaggle.ipynb
```

Properties:

- Generated inside the notebook.
- Small and balanced.
- Useful for checking code, hardware, tokenization, training, and evaluation.
- Not suitable for clinical or production claims.

## Option 2: External JSON Dataset

Used by:

```text
examples/cbt_deberta_hqde_kaggle_REAL_DATA.ipynb
```

Properties:

- Requires a dataset file to be uploaded and attached.
- Better suited for research reporting if the data source and labels are documented.
- Results depend on dataset quality, split, class balance, and annotation reliability.

## Comparison

| Criterion | Synthetic toy data | External dataset |
|-----------|--------------------|------------------|
| Setup | No extra file | Requires upload |
| Reproducibility | Easy to run | Depends on dataset access |
| Research value | Pipeline validation | Potential experiment data |
| Clinical validity | No | Only if dataset and labels are clinically validated |
| Fixed expected accuracy | Not provided | Not provided |

## Recommendation

Use synthetic data for smoke testing. Use a documented external dataset for thesis results.
