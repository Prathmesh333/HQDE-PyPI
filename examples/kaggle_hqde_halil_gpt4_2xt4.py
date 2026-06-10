"""
Kaggle direct run: Halil GPT-4 cognitive-distortion with HQDE + DeBERTa on 2xT4.

Run this whole file in one Kaggle notebook, or use:
    %run examples/kaggle_hqde_halil_gpt4_2xt4.py
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
    run_name="halil_gpt4_hqde_deberta_2xt4",
    output_dir="/kaggle/working/halil_gpt4_hqde_deberta_2xt4",
    epochs=5,
    batch_size=2,
    max_length=128,
    num_workers=4,
    learning_rate=2e-5,
    label_smoothing=0.0,
    pooling="mean",
)


def load_halil_gpt4():
    raw = load_dataset("halilbabacan/cognitive_distortions_gpt4")
    label_names = sorted({normalize_label(label) for label in raw["train"]["label"]})
    label_to_id = {label_name: index for index, label_name in enumerate(label_names)}

    records = []
    for row_index, row in enumerate(raw["train"]):
        label_name = normalize_label(row["label"])
        records.append(
            {
                "id": row_index,
                "text": clean_text(row["text"]),
                "label": label_to_id[label_name],
                "distortion_name": label_name,
            }
        )

    metadata = {
        "hf_id": "halilbabacan/cognitive_distortions_gpt4",
        "label_mode": "native",
        "source_split": "train only, stratified train/val/test created by code",
    }
    return make_frame(records), None, label_names, metadata


run_hqde_cbt_dataset(CONFIG, load_halil_gpt4)
