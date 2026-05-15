"""
Multi-dataset CBT cognitive-distortion benchmark for HQDE-style DeBERTa ensembles.

This script is intended for thesis/paper result generation. It runs the same
training protocol across multiple Hugging Face datasets and writes comparison
tables to disk. It does not claim fixed results; report only metrics generated
from your own run logs.

Default datasets:
- danthareja/cognitive-distortion
- halilbabacan/cognitive_distortions_gpt4
- elliott-leow/cognitive_distortion_validation:clinical_sycophancy

Example:
    python examples/cbt_multi_dataset_comparison.py --quick-test --dry-run
    python examples/cbt_multi_dataset_comparison.py --epochs 5 --max-train-samples 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - older PyTorch fallback
    from torch.cuda.amp import GradScaler, autocast


SEED = 42
DEFAULT_DATASETS = "danthareja,halil-gpt4,elliott-validation"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str
    display_name: str
    source_type: str
    loader: Callable[[], tuple[pd.DataFrame, Optional[pd.DataFrame], list[str]]]
    notes: str


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class DeBERTaClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = dropout_rate
        self.config.dtype = torch.float32

        self.backbone = self._load_backbone(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.float()

    def _load_backbone(self, model_name: str):
        try:
            return AutoModel.from_pretrained(
                model_name,
                config=self.config,
                dtype=torch.float32,
            )
        except TypeError:
            return AutoModel.from_pretrained(
                model_name,
                config=self.config,
                torch_dtype=torch.float32,
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


class EnsembleWorker:
    def __init__(self, worker_id: int, args, num_classes: int, device: torch.device):
        self.worker_id = worker_id
        self.device = device
        self.dropout_rate = args.dropout_rates[worker_id % len(args.dropout_rates)]
        self.learning_rate = args.learning_rates[worker_id % len(args.learning_rates)]
        self.use_amp = bool(args.use_amp and device.type == "cuda")

        self.model = DeBERTaClassifier(args.model_name, num_classes, self.dropout_rate).to(device)
        if self.use_amp:
            self.model.float()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = make_grad_scaler(self.use_amp)

        first_param = next(self.model.parameters())
        print(
            f"Worker {worker_id}: device={device}, amp={self.use_amp}, "
            f"dropout={self.dropout_rate}, lr={self.learning_rate}, "
            f"param_dtype={first_param.dtype}"
        )

    def train_epoch(self, data_loader: DataLoader, scheduler) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        progress = tqdm(data_loader, desc=f"Worker {self.worker_id} train", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with amp_context(self.device, self.use_amp):
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            scheduler.step()

            batch_size = labels.size(0)
            total += batch_size
            total_loss += loss.detach().item() * batch_size
            predictions = torch.argmax(logits.detach(), dim=1)
            correct += (predictions == labels).sum().item()
            progress.set_postfix(loss=f"{loss.detach().item():.4f}", acc=f"{100 * correct / max(total, 1):.2f}%")

        return total_loss / max(total, 1), 100 * correct / max(total, 1)

    def evaluate(self, data_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Worker {self.worker_id} eval", leave=False):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                total += batch_size
                total_loss += loss.detach().item() * batch_size
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()

        return total_loss / max(total, 1), 100 * correct / max(total, 1)

    def predict_logits(self, data_loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                logits = self.model(input_ids, attention_mask)
                all_logits.append(logits.cpu())
        return torch.cat(all_logits, dim=0)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_label(value: str) -> str:
    value = clean_text(value).replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().title()


def make_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Dataset adapter produced no usable rows")
    frame = frame[frame["text"].astype(str).str.len() > 0].copy()
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return frame


def load_danthareja() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw = load_dataset("danthareja/cognitive-distortion")
    label_names = list(raw["train"].features["dominant_distortion"].names)

    def convert(split_name: str) -> pd.DataFrame:
        rows = []
        for row in raw[split_name]:
            label = int(row["dominant_distortion"])
            text = clean_text(row.get("distorted_part")) or clean_text(row.get(" patient_question"))
            rows.append(
                {
                    "id": row.get("id"),
                    "text": text,
                    "label": label,
                    "distortion_name": label_names[label],
                }
            )
        return make_frame(rows)

    train_df = convert("train")
    test_df = convert("test")
    test_df = test_df[~test_df["text"].isin(set(train_df["text"]))].reset_index(drop=True)
    return train_df, test_df, label_names


def load_halil_gpt4() -> tuple[pd.DataFrame, Optional[pd.DataFrame], list[str]]:
    raw = load_dataset("halilbabacan/cognitive_distortions_gpt4")
    labels = sorted({normalize_label(label) for label in raw["train"]["label"]})
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    rows = []
    for idx, row in enumerate(raw["train"]):
        label_name = normalize_label(row["label"])
        rows.append(
            {
                "id": idx,
                "text": clean_text(row["text"]),
                "label": label_to_id[label_name],
                "distortion_name": label_name,
            }
        )
    return make_frame(rows), None, labels


def load_elliott_validation() -> tuple[pd.DataFrame, Optional[pd.DataFrame], list[str]]:
    raw = load_dataset("elliott-leow/cognitive_distortion_validation")
    split = raw["clinical_sycophancy"]
    labels = sorted({normalize_label(row["subcategory"]) for row in split})
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    rows = []
    for row in split:
        label_name = normalize_label(row["subcategory"])
        rows.append(
            {
                "id": row.get("id"),
                "text": clean_text(row.get("user_prompt")),
                "label": label_to_id[label_name],
                "distortion_name": label_name,
            }
        )
    return make_frame(rows), None, labels


DATASET_SPECS: dict[str, DatasetSpec] = {
    "danthareja": DatasetSpec(
        key="danthareja",
        hf_id="danthareja/cognitive-distortion",
        display_name="Cognitive Distortion",
        source_type="human-annotated / Kaggle-derived",
        loader=load_danthareja,
        notes="Uses published train/test splits; test exact-text overlaps are removed.",
    ),
    "halil-gpt4": DatasetSpec(
        key="halil-gpt4",
        hf_id="halilbabacan/cognitive_distortions_gpt4",
        display_name="Cognitive Distortions GPT-4",
        source_type="LLM-generated",
        loader=load_halil_gpt4,
        notes="Train-only dataset; script creates stratified train/val/test splits.",
    ),
    "elliott-validation": DatasetSpec(
        key="elliott-validation",
        hf_id="elliott-leow/cognitive_distortion_validation",
        display_name="Clinical Sycophancy CD Validation",
        source_type="validation/probe dataset",
        loader=load_elliott_validation,
        notes="Uses only the clinical_sycophancy split and its cognitive-distortion subcategory labels.",
    ),
}


def parse_dataset_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if not keys:
        raise ValueError("At least one dataset key is required")
    unknown = [key for key in keys if key not in DATASET_SPECS]
    if unknown:
        known = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset key(s): {unknown}. Known keys: {known}")
    return keys


def stratified_cap(frame: pd.DataFrame, max_rows: int, seed: int = SEED) -> pd.DataFrame:
    if not max_rows or len(frame) <= max_rows:
        return frame.reset_index(drop=True)

    labels = sorted(frame["label"].unique())
    per_class = max(max_rows // len(labels), 1)
    selected = []
    used_indices: set[int] = set()

    for label in labels:
        group = frame[frame["label"] == label]
        take = min(len(group), per_class)
        sample = group.sample(n=take, random_state=seed + int(label))
        selected.append(sample)
        used_indices.update(sample.index.tolist())

    capped = pd.concat(selected, axis=0)
    remaining = max_rows - len(capped)
    if remaining > 0:
        rest = frame.drop(index=list(used_indices), errors="ignore")
        if not rest.empty:
            capped = pd.concat([capped, rest.sample(n=min(remaining, len(rest)), random_state=seed)], axis=0)

    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_dataset(
    train_source: pd.DataFrame,
    test_source: Optional[pd.DataFrame],
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_source = train_source.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if test_source is not None and not test_source.empty:
        train_source = stratified_cap(train_source, args.max_train_samples, seed=SEED)
        test_df = stratified_cap(test_source, args.max_eval_samples, seed=SEED + 2)
        train_df, val_df = train_test_split(
            train_source,
            test_size=args.val_size,
            random_state=SEED,
            stratify=train_source["label"],
        )
        val_df = stratified_cap(val_df, args.max_eval_samples, seed=SEED + 1)
    else:
        capped = stratified_cap(train_source, args.max_train_samples, seed=SEED)
        train_df, temp_df = train_test_split(
            capped,
            test_size=args.test_size + args.val_size,
            random_state=SEED,
            stratify=capped["label"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=args.test_size / (args.test_size + args.val_size),
            random_state=SEED,
            stratify=temp_df["label"],
        )
        val_df = stratified_cap(val_df, args.max_eval_samples, seed=SEED + 1)
        test_df = stratified_cap(test_df, args.max_eval_samples, seed=SEED + 2)

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])
    val_df = val_df[~val_df["text"].isin(train_texts)].reset_index(drop=True)
    test_df = test_df[~test_df["text"].isin(train_texts | val_texts)].reset_index(drop=True)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def overlap_counts(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])
    return {
        "train_val_overlap": len(train_texts & val_texts),
        "train_test_overlap": len(train_texts & test_texts),
        "val_test_overlap": len(val_texts & test_texts),
    }


def make_grad_scaler(enabled: bool):
    try:
        return GradScaler("cuda", enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    except TypeError:
        return autocast(enabled=enabled)


def resolve_devices(num_workers: int) -> list[torch.device]:
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        return [torch.device(f"cuda:{idx % gpu_count}") for idx in range(num_workers)]
    return [torch.device("cpu") for _ in range(num_workers)]


def create_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    args,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {
        "num_workers": args.dataloader_workers,
        "pin_memory": bool(args.pin_memory and torch.cuda.is_available()),
    }

    train_ds = TextClassificationDataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
    val_ds = TextClassificationDataset(val_df["text"], val_df["label"], tokenizer, args.max_length)
    test_ds = TextClassificationDataset(test_df["text"], test_df["label"], tokenizer, args.max_length)

    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs),
    )


def ensemble_predictions(workers: list[EnsembleWorker], data_loader: DataLoader) -> torch.Tensor:
    logits = [worker.predict_logits(data_loader) for worker in workers]
    return torch.stack(logits, dim=0).mean(dim=0)


def run_dataset(spec: DatasetSpec, tokenizer, args) -> dict:
    print("\n" + "=" * 88)
    print(f"Dataset: {spec.display_name} ({spec.hf_id})")
    print("=" * 88)

    train_source, test_source, label_names = spec.loader()
    train_df, val_df, test_df = split_dataset(train_source, test_source, args)
    overlaps = overlap_counts(train_df, val_df, test_df)

    print(f"Classes: {len(label_names)}")
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Exact overlaps: {overlaps}")

    result_base = {
        "dataset_key": spec.key,
        "dataset": spec.display_name,
        "hf_id": spec.hf_id,
        "source_type": spec.source_type,
        "notes": spec.notes,
        "num_classes": len(label_names),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        **overlaps,
    }

    if args.dry_run:
        return {
            **result_base,
            "status": "dry_run",
            "best_val_accuracy": None,
            "test_accuracy": None,
            "test_weighted_f1": None,
            "test_macro_f1": None,
            "training_time_sec": None,
            "peak_cuda_memory_mb": peak_cuda_memory_mb(),
        }

    train_loader, val_loader, test_loader = create_loaders(train_df, val_df, test_df, tokenizer, args)
    devices = resolve_devices(args.ensemble_workers)
    workers = [EnsembleWorker(idx, args, len(label_names), device) for idx, device in enumerate(devices)]

    training_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(training_steps * args.warmup_ratio)
    schedulers = {
        worker.worker_id: get_cosine_schedule_with_warmup(
            worker.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
        for worker in workers
    }

    started_at = time.perf_counter()
    best_val_accuracy = 0.0
    best_val_weighted_f1 = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        for worker in workers:
            train_loss, train_acc = worker.train_epoch(train_loader, schedulers[worker.worker_id])
            val_loss, val_acc = worker.evaluate(val_loader)
            print(
                f"Worker {worker.worker_id}: train_loss={train_loss:.4f}, "
                f"train_acc={train_acc:.2f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}"
            )

        val_logits = ensemble_predictions(workers, val_loader)
        val_preds = torch.argmax(val_logits, dim=1).numpy()
        val_labels = val_df["label"].to_numpy()
        val_acc = accuracy_score(val_labels, val_preds) * 100
        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0) * 100
        best_val_accuracy = max(best_val_accuracy, val_acc)
        best_val_weighted_f1 = max(best_val_weighted_f1, val_f1)
        print(f"Ensemble validation: accuracy={val_acc:.2f}, weighted_f1={val_f1:.2f}")

    training_time = time.perf_counter() - started_at
    test_logits = ensemble_predictions(workers, test_loader)
    test_preds = torch.argmax(test_logits, dim=1).numpy()
    test_labels = test_df["label"].to_numpy()
    test_accuracy = accuracy_score(test_labels, test_preds) * 100
    test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0) * 100
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0) * 100

    report = classification_report(
        test_labels,
        test_preds,
        labels=list(range(len(label_names))),
        target_names=label_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    dataset_out = Path(args.output_dir) / spec.key
    dataset_out.mkdir(parents=True, exist_ok=True)
    with (dataset_out / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    result = {
        **result_base,
        "status": "completed",
        "best_val_accuracy": round(best_val_accuracy, 4),
        "best_val_weighted_f1": round(best_val_weighted_f1, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "training_time_sec": round(training_time, 2),
        "peak_cuda_memory_mb": peak_cuda_memory_mb(),
    }

    del workers, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def peak_cuda_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e6, 2)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in columns]
        values = [value.replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def write_outputs(results: list[dict], args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)

    csv_path = out_dir / "cbt_multi_dataset_comparison.csv"
    json_path = out_dir / "cbt_multi_dataset_comparison.json"
    md_path = out_dir / "cbt_multi_dataset_comparison.md"

    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)

    paper_columns = [
        "dataset",
        "source_type",
        "num_classes",
        "train_rows",
        "val_rows",
        "test_rows",
        "best_val_accuracy",
        "test_accuracy",
        "test_weighted_f1",
        "test_macro_f1",
        "training_time_sec",
        "train_val_overlap",
        "train_test_overlap",
        "val_test_overlap",
    ]
    available_columns = [col for col in paper_columns if col in frame.columns]
    md_path.write_text(markdown_table(frame[available_columns]), encoding="utf-8")

    print("\nSaved comparison outputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print("\nPaper table preview:")
    print(markdown_table(frame[available_columns]))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=os.environ.get("HQDE_DATASETS", DEFAULT_DATASETS))
    parser.add_argument("--output-dir", default=os.environ.get("HQDE_OUTPUT_DIR", "benchmark_outputs/cbt_multi_dataset"))
    parser.add_argument("--model-name", default=os.environ.get("HQDE_MODEL_NAME", "microsoft/deberta-v3-base"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("HQDE_NUM_EPOCHS", "3")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("HQDE_BATCH_SIZE", "8" if torch.cuda.is_available() else "2")))
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("HQDE_MAX_LENGTH", "256")))
    parser.add_argument("--ensemble-workers", type=int, default=int(os.environ.get("HQDE_ENSEMBLE_WORKERS", "4" if torch.cuda.is_available() else "1")))
    parser.add_argument("--dataloader-workers", type=int, default=int(os.environ.get("HQDE_DATALOADER_WORKERS", "0")))
    parser.add_argument("--max-train-samples", type=int, default=int(os.environ.get("HQDE_MAX_TRAIN_SAMPLES", "0")))
    parser.add_argument("--max-eval-samples", type=int, default=int(os.environ.get("HQDE_MAX_EVAL_SAMPLES", "0")))
    parser.add_argument("--val-size", type=float, default=float(os.environ.get("HQDE_VAL_SIZE", "0.1")))
    parser.add_argument("--test-size", type=float, default=float(os.environ.get("HQDE_TEST_SIZE", "0.2")))
    parser.add_argument("--warmup-ratio", type=float, default=float(os.environ.get("HQDE_WARMUP_RATIO", "0.1")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("HQDE_WEIGHT_DECAY", "0.01")))
    parser.add_argument("--learning-rates", default=os.environ.get("HQDE_LEARNING_RATES", "1.5e-5,2e-5,2.5e-5,3e-5"))
    parser.add_argument("--dropout-rates", default=os.environ.get("HQDE_DROPOUT_RATES", "0.1,0.15,0.2,0.25"))
    parser.add_argument("--quick-test", action="store_true", default=os.environ.get("HQDE_QUICK_TEST", "0") == "1")
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("HQDE_DRY_RUN", "0") == "1")
    parser.add_argument("--no-amp", action="store_true", default=os.environ.get("HQDE_NO_AMP", "0") == "1")
    parser.add_argument("--pin-memory", action="store_true", default=os.environ.get("HQDE_PIN_MEMORY", "0") == "1")
    parser.add_argument("--list-datasets", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_datasets:
        print("Available dataset keys:")
        for key, spec in DATASET_SPECS.items():
            print(f"  {key}: {spec.hf_id} ({spec.source_type})")
        return

    args.dataset_keys = parse_dataset_keys(args.datasets)
    args.learning_rates = [float(value) for value in args.learning_rates.split(",")]
    args.dropout_rates = [float(value) for value in args.dropout_rates.split(",")]
    args.use_amp = not args.no_amp

    if args.quick_test:
        args.epochs = min(args.epochs, 1)
        args.max_length = min(args.max_length, 64)
        args.max_train_samples = args.max_train_samples or 128
        args.max_eval_samples = args.max_eval_samples or 64
        args.ensemble_workers = min(args.ensemble_workers, 1)
        args.batch_size = min(args.batch_size, 2)

    set_seed(SEED)
    print("HQDE CBT multi-dataset comparison")
    print(f"Datasets: {', '.join(args.dataset_keys)}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Ensemble workers: {args.ensemble_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch: {torch.__version__}")
    try:
        print(f"HQDE package: {version('hqde')}")
    except PackageNotFoundError:
        print("HQDE package: not installed")

    tokenizer = None if args.dry_run else AutoTokenizer.from_pretrained(args.model_name)
    results = []
    for key in args.dataset_keys:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        results.append(run_dataset(DATASET_SPECS[key], tokenizer, args))

    write_outputs(results, args)


if __name__ == "__main__":
    main()
