"""
Standalone Kaggle HQDE + DeBERTa code for one CBT cognitive-distortion dataset.

Notebook-safe: no argparse, no CLI parsing, and no hidden Jupyter arguments.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HQDE_GITHUB_SPEC = "git+https://github.com/Prathmesh333/HQDE-PyPI.git@main"
HQDE_REQUIRED_VERSION = "0.1.13"


def ensure_kaggle_packages() -> None:
    """Install missing Kaggle dependencies without upgrading Kaggle's torch build."""
    try:
        installed_hqde = version("hqde")
    except PackageNotFoundError:
        installed_hqde = None
    if installed_hqde != HQDE_REQUIRED_VERSION:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", HQDE_GITHUB_SPEC]
        )

    packages = []
    required = {
        "ray": "ray>=2.9.0",
        "datasets": "datasets>=2.14.0",
        "transformers": "transformers>=4.45.0",
        "accelerate": "accelerate>=0.20.0",
        "sentencepiece": "sentencepiece>=0.1.99",
        "sklearn": "scikit-learn>=1.3.0",
        "pandas": "pandas>=2.0.0",
        "tqdm": "tqdm>=4.65.0",
    }
    for module_name, package_name in required.items():
        if importlib.util.find_spec(module_name) is None:
            packages.append(package_name)
    if packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


ensure_kaggle_packages()

import hqde
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import logging as transformers_logging

from hqde import create_hqde_system

transformers_logging.set_verbosity_error()


SEED = 42


@dataclass
class HQDECBTRunConfig:
    run_name: str
    output_dir: str
    model_name: str = "microsoft/deberta-v3-base"
    epochs: int = 8
    batch_size: int = 2
    max_length: int = 192
    num_workers: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    label_smoothing: float = 0.0
    val_size: float = 0.1
    test_size: float = 0.2
    max_train_samples: int = 0
    max_eval_samples: int = 0
    seed: int = SEED
    pooling: str = "mean"
    gradient_checkpointing: bool = True
    prediction_aggregation: str = "mean"
    save_model: bool = False


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(value) -> str:
    return "" if value is None else str(value).strip()


def normalize_label(value: str) -> str:
    value = clean_text(value).replace("_", " ").replace("-", " ")
    return " ".join(value.split()).title()


def make_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Dataset loader produced no rows")
    frame = frame[frame["text"].astype(str).str.len() > 0].copy()
    return frame.drop_duplicates(subset=["text"]).reset_index(drop=True)


def safe_stratify(frame: pd.DataFrame):
    counts = frame["label"].value_counts()
    return frame["label"] if len(counts) > 1 and int(counts.min()) >= 2 else None


def stratified_cap(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if not max_rows or len(frame) <= max_rows:
        return frame.reset_index(drop=True)

    labels = sorted(frame["label"].unique())
    per_class = max(max_rows // max(len(labels), 1), 1)
    selected = []
    used = set()

    for label in labels:
        group = frame[frame["label"] == label]
        sample = group.sample(n=min(len(group), per_class), random_state=seed + int(label))
        selected.append(sample)
        used.update(sample.index.tolist())

    capped = pd.concat(selected, axis=0)
    remaining = max_rows - len(capped)
    if remaining > 0:
        rest = frame.drop(index=list(used), errors="ignore")
        if not rest.empty:
            capped = pd.concat([capped, rest.sample(n=min(remaining, len(rest)), random_state=seed)], axis=0)

    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_with_optional_test(
    train_source: pd.DataFrame,
    test_source: Optional[pd.DataFrame],
    config: HQDECBTRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_source = train_source.drop_duplicates(subset=["text"]).reset_index(drop=True)
    train_source = stratified_cap(train_source, config.max_train_samples, config.seed)

    if test_source is not None and not test_source.empty:
        test_df = stratified_cap(test_source, config.max_eval_samples, config.seed + 2)
        train_df, val_df = train_test_split(
            train_source,
            test_size=config.val_size,
            random_state=config.seed,
            stratify=safe_stratify(train_source),
        )
    else:
        train_df, temp_df = train_test_split(
            train_source,
            test_size=config.val_size + config.test_size,
            random_state=config.seed,
            stratify=safe_stratify(train_source),
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=config.test_size / (config.test_size + config.val_size),
            random_state=config.seed,
            stratify=safe_stratify(temp_df),
        )

    val_df = stratified_cap(val_df, config.max_eval_samples, config.seed + 1)
    test_df = stratified_cap(test_df, config.max_eval_samples, config.seed + 2)

    train_texts = set(train_df["text"])
    val_df = val_df[~val_df["text"].isin(train_texts)].reset_index(drop=True)
    test_df = test_df[~test_df["text"].isin(train_texts | set(val_df["text"]))].reset_index(drop=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class DeBERTaHQDEClassifier(nn.Module):
    """DeBERTa model class compatible with hqde.create_hqde_system."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout_rate: float = 0.15,
        pooling: str = "mean",
        gradient_checkpointing: bool = True,
        pad_token_id: int = 0,
    ):
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.pooling = pooling
        self.pad_token_id = int(pad_token_id)
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = dropout_rate
        try:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config, dtype=torch.float32)
        except TypeError:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config, torch_dtype=torch.float32)

        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            if hasattr(self.config, "use_cache"):
                self.config.use_cache = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.float()

    def forward(self, input_ids=None, attention_mask: Optional[torch.Tensor] = None, **_: object) -> torch.Tensor:
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask")
            input_ids = input_ids["input_ids"]
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


class HQDETextDataset(TorchDataset):
    """PyTorch dataset that returns tensor inputs accepted by HQDE workers."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        encoded = tokenizer(
            [str(text) for text in frame["text"].tolist()],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.labels = torch.tensor([int(label) for label in frame["label"].tolist()], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, index: int):
        return self.input_ids[index], self.labels[index]


def tokenize_frame(frame: pd.DataFrame, tokenizer, max_length: int) -> HQDETextDataset:
    return HQDETextDataset(frame, tokenizer, max_length)


def print_hardware() -> None:
    print("HQDE package:", getattr(hqde, "__version__", "unknown"))
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {torch.cuda.get_device_name(index)} ({props.total_memory / 1e9:.2f} GB)")


def init_ray_for_2xt4() -> None:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
        num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        num_cpus=os.cpu_count() or 4,
    )
    print("Ray resources:", ray.cluster_resources())


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_hqde_cbt_dataset(
    config: HQDECBTRunConfig,
    loader: Callable[[], tuple[pd.DataFrame, Optional[pd.DataFrame], list[str], dict]],
) -> dict:
    set_seed(config.seed)
    print("=" * 88)
    print(f"HQDE DeBERTa 2xT4 run: {config.run_name}")
    print("=" * 88)
    print_hardware()

    train_source, test_source, label_names, metadata = loader()
    train_df, val_df, test_df = split_with_optional_test(train_source, test_source, config)
    print("Dataset metadata:", metadata)
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Classes: {len(label_names)}")
    print("Labels:", label_names)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = {
        "run_name": config.run_name,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "num_classes": len(label_names),
        "label_names": label_names,
        "metadata": metadata,
        "hqde_version": getattr(hqde, "__version__", "unknown"),
    }
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_dataset = tokenize_frame(train_df, tokenizer, config.max_length)
    val_dataset = tokenize_frame(val_df, tokenizer, config.max_length)
    test_dataset = tokenize_frame(test_df, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0)

    init_ray_for_2xt4()
    training_config = {
        "learning_rate": config.learning_rate,
        "optimizer": "adamw",
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "label_smoothing": config.label_smoothing,
        "gradient_clip_norm": 1.0,
        "use_amp": torch.cuda.is_available(),
        "ensemble_mode": "independent",
        "batch_assignment": "replicate",
        "prediction_aggregation": config.prediction_aggregation,
        "compile_model": False,
    }
    model_kwargs = {
        "model_name": config.model_name,
        "num_classes": len(label_names),
        "pooling": config.pooling,
        "gradient_checkpointing": config.gradient_checkpointing,
        "pad_token_id": int(tokenizer.pad_token_id or 0),
    }

    system = create_hqde_system(
        DeBERTaHQDEClassifier,
        model_kwargs,
        num_workers=config.num_workers,
        training_config=training_config,
    )

    started_at = time.perf_counter()
    try:
        hqde_train_metrics = system.train(train_loader, num_epochs=config.epochs, validation_loader=val_loader)
        val_logits = system.predict(val_loader)
        test_logits = system.predict(test_loader)
        val_preds = torch.argmax(val_logits, dim=1).numpy()
        test_preds = torch.argmax(test_logits, dim=1).numpy()
    finally:
        clear_memory()

    train_time = time.perf_counter() - started_at
    val_labels = val_df["label"].to_numpy()
    test_labels = test_df["label"].to_numpy()
    metrics = {
        "run_name": config.run_name,
        "hqde_version": getattr(hqde, "__version__", "unknown"),
        "model_name": config.model_name,
        "num_workers": config.num_workers,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "pooling": config.pooling,
        "gradient_checkpointing": config.gradient_checkpointing,
        "prediction_aggregation": config.prediction_aggregation,
        "val_accuracy": accuracy_score(val_labels, val_preds) * 100,
        "val_weighted_f1": f1_score(val_labels, val_preds, average="weighted", zero_division=0) * 100,
        "val_macro_f1": f1_score(val_labels, val_preds, average="macro", zero_division=0) * 100,
        "test_accuracy": accuracy_score(test_labels, test_preds) * 100,
        "test_weighted_f1": f1_score(test_labels, test_preds, average="weighted", zero_division=0) * 100,
        "test_macro_f1": f1_score(test_labels, test_preds, average="macro", zero_division=0) * 100,
        "training_time_sec": round(train_time, 2),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(len(label_names))),
        target_names=label_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "hqde_training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(hqde_train_metrics, handle, indent=2)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    if config.save_model:
        system.save_model(str(output_dir / "hqde_ensemble.pt"))
    system.cleanup()
    clear_memory()
    print(json.dumps(metrics, indent=2))
    print("Saved outputs to:", output_dir)
    return metrics


# ============================================================================
# Dataset-specific Kaggle run
# ============================================================================

"""
Kaggle direct run: Elliott clinical_sycophancy validation with HQDE + DeBERTa on 2xT4.

Run this whole file in one Kaggle notebook, or use:
    %run examples/kaggle_hqde_elliott_validation_2xt4.py
"""

from datasets import load_dataset


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
