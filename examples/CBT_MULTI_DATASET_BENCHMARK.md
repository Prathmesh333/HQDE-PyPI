# CBT Multi-Dataset Benchmark

This benchmark is intended for thesis and paper tables. It runs the same
DeBERTa-based HQDE-style ensemble protocol across multiple cognitive-distortion
datasets and writes reproducible comparison artifacts.

## Supported Datasets

| Key | Source | Type | Use in paper |
| --- | --- | --- | --- |
| `danthareja` | [`danthareja/cognitive-distortion`](https://huggingface.co/datasets/danthareja/cognitive-distortion) | Human-annotated / Kaggle-derived | Primary CBT cognitive-distortion benchmark |
| `halil-gpt4` | [`halilbabacan/cognitive_distortions_gpt4`](https://huggingface.co/datasets/halilbabacan/cognitive_distortions_gpt4) | LLM-generated | Synthetic/generative robustness comparison |
| `elliott-validation` | [`elliott-leow/cognitive_distortion_validation`](https://huggingface.co/datasets/elliott-leow/cognitive_distortion_validation) | Validation/probe dataset | Secondary validation/probe comparison |

Do not merge these into one dataset for headline metrics. Their label spaces,
collection methods, and intended uses are different. Report them as separate
rows in a comparison table.

## Quick Dataset Check

```bash
python examples/cbt_multi_dataset_comparison.py --quick-test --dry-run
```

This loads each dataset, creates train/validation/test splits, checks exact
text overlap, and writes table artifacts without downloading DeBERTa weights.

## Full Benchmark

```bash
python examples/cbt_multi_dataset_comparison.py \
  --label-mode canonical10 \
  --epochs 5 \
  --max-train-samples 1000 \
  --max-eval-samples 300 \
  --output-dir benchmark_outputs/cbt_multi_dataset
```

For a longer thesis run, increase `--epochs` and remove the sample caps if the
Kaggle runtime allows it.

## Label Modes

Use `--label-mode canonical10` for the main thesis table if you want the same
10 CBT cognitive-distortion categories across datasets:

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

This mode maps compatible labels, for example `Mind Reading` and
`Fortune-telling` to `Jumping to Conclusions`, and drops unmapped labels such
as `No Distortion`. The output table includes `missing_classes` and
`dropped_rows`; if a dataset does not contain one of the 10 categories, the
script reports that explicitly.

Use `--label-mode native` only when you want to evaluate each dataset in its
own original label space. Native mode is useful for robustness checks, but it
is not an apples-to-apples 10-class comparison.

## Kaggle 2xT4 Notebook

Use `examples/cbt_multi_dataset_hqde_kaggle_2xT4.ipynb` when running from the
Kaggle notebook UI. It is configured for:

- 2 Tesla T4 GPUs.
- 4 HQDE ensemble workers.
- 4 vCPUs.
- Single-process DataLoaders for notebook stability.
- Canonical 10-label CBT mapping by default.

Run the dry-run cell first, then the full benchmark cell.

## Kaggle Notes

Enable internet before running the benchmark so the Hugging Face datasets and
model weights can be downloaded. The script defaults to single-process
DataLoaders because Kaggle/Python 3.12 notebook workers can crash with
`DataLoader worker exited unexpectedly`.

Useful environment variables:

```python
import os
os.environ["HQDE_NUM_EPOCHS"] = "5"
os.environ["HQDE_MAX_TRAIN_SAMPLES"] = "1000"
os.environ["HQDE_MAX_EVAL_SAMPLES"] = "300"
os.environ["HQDE_ENSEMBLE_WORKERS"] = "4"
```

## Output Files

The benchmark writes:

- `cbt_multi_dataset_comparison.csv`
- `cbt_multi_dataset_comparison.json`
- `cbt_multi_dataset_comparison.md`
- one `classification_report.json` per completed dataset

The Markdown file is formatted as a paper-ready table. Include the Git commit,
hardware, PyTorch version, HQDE version, random seed, dataset source, and exact
overlap counts when reporting results.
