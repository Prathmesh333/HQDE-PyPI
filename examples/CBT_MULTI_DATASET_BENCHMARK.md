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
  --epochs 5 \
  --max-train-samples 1000 \
  --max-eval-samples 300 \
  --output-dir benchmark_outputs/cbt_multi_dataset
```

For a longer thesis run, increase `--epochs` and remove the sample caps if the
Kaggle runtime allows it.

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
