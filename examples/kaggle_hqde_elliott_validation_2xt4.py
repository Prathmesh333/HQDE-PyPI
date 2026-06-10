"""
Kaggle direct run: Elliott clinical_sycophancy validation with HQDE + DeBERTa on 2xT4.

Run this whole file in one Kaggle notebook, or use:
    %run examples/kaggle_hqde_elliott_validation_2xt4.py
"""

from datasets import load_dataset

from kaggle_hqde_cbt_2xt4_common import (
    HQDECBTRunConfig,
    clean_text,
    make_frame,
    normalize_label,
    run_hqde_cbt_dataset,
)


CONFIG = HQDECBTRunConfig(
    run_name="elliott_validation_hqde_deberta_2xt4",
    output_dir="/kaggle/working/elliott_validation_hqde_deberta_2xt4",
    epochs=10,
    batch_size=2,
    max_length=192,
    num_workers=4,
    learning_rate=2e-5,
    label_smoothing=0.05,
    pooling="mean",
)


def load_elliott_validation():
    raw = load_dataset("elliott-leow/cognitive_distortion_validation")
    split = raw["clinical_sycophancy"]
    label_names = sorted({normalize_label(row["subcategory"]) for row in split})
    label_to_id = {label_name: index for index, label_name in enumerate(label_names)}

    records = []
    for row in split:
        label_name = normalize_label(row["subcategory"])
        records.append(
            {
                "id": row.get("id"),
                "text": clean_text(row.get("user_prompt")),
                "label": label_to_id[label_name],
                "distortion_name": label_name,
            }
        )

    metadata = {
        "hf_id": "elliott-leow/cognitive_distortion_validation",
        "split": "clinical_sycophancy",
        "label_mode": "native subcategory",
        "input_field": "user_prompt only",
        "source_split": "single split, stratified train/val/test created by code",
    }
    return make_frame(records), None, label_names, metadata


run_hqde_cbt_dataset(CONFIG, load_elliott_validation)
