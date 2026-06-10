"""
Kaggle direct run: Danthareja cognitive-distortion with HQDE + DeBERTa on 2xT4.

Run this whole file in one Kaggle notebook, or use:
    %run examples/kaggle_hqde_danthareja_2xt4.py
"""

from datasets import load_dataset

from kaggle_hqde_cbt_2xt4_common import (
    HQDECBTRunConfig,
    clean_text,
    make_frame,
    run_hqde_cbt_dataset,
)


CONFIG = HQDECBTRunConfig(
    run_name="danthareja_hqde_deberta_2xt4",
    output_dir="/kaggle/working/danthareja_hqde_deberta_2xt4",
    epochs=8,
    batch_size=2,
    max_length=256,
    num_workers=4,
    learning_rate=2e-5,
    label_smoothing=0.0,
    pooling="mean",
)


def load_danthareja():
    raw = load_dataset("danthareja/cognitive-distortion")
    label_names = list(raw["train"].features["dominant_distortion"].names)

    def convert(split_name: str):
        records = []
        for row in raw[split_name]:
            label = int(row["dominant_distortion"])
            patient_question = clean_text(row.get("patient_question")) or clean_text(row.get(" patient_question"))
            distorted_part = clean_text(row.get("distorted_part"))
            if distorted_part and patient_question:
                text = f"Distorted thought span: {distorted_part}\nPatient question: {patient_question}"
            else:
                text = distorted_part or patient_question
            records.append(
                {
                    "id": row.get("id"),
                    "text": text,
                    "label": label,
                    "distortion_name": label_names[label],
                }
            )
        return make_frame(records)

    train_df = convert("train")
    test_df = convert("test")
    test_df = test_df[~test_df["text"].isin(set(train_df["text"]))].reset_index(drop=True)
    metadata = {
        "hf_id": "danthareja/cognitive-distortion",
        "label_mode": "native",
        "text_mode": "question_plus_distorted",
        "source_split": "published train/test",
    }
    return train_df, test_df, label_names, metadata


run_hqde_cbt_dataset(CONFIG, load_danthareja)
