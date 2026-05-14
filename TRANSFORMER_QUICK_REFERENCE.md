# HQDE Transformer Quick Reference

This reference reflects the current code state.

## Status Summary

| Capability | Status |
|------------|--------|
| Import transformer classes | Works. |
| Direct transformer forward with `attention_mask` | Works. |
| Tokenizer and dict dataset utilities | Work independently. |
| Core `HQDESystem` with tuple `(input_ids, labels)` loader | Works as a limited workaround. |
| Core `HQDESystem` with `TextDataLoader` dict batches | Not implemented yet. |
| DeBERTa Kaggle notebook dict-batch path | Works through custom notebook worker code. |

## Imports

```python
from hqde import (
    create_hqde_system,
    TransformerTextClassifier,
    LightweightTransformerClassifier,
    CBTTransformerClassifier,
    SimpleTokenizer,
    TextClassificationDataset,
    CBTDataset,
    TextDataLoader,
    make_transformer_training_config,
    make_cbt_training_config,
)
```

## Direct Model Use

```python
import torch
from hqde import LightweightTransformerClassifier

model = LightweightTransformerClassifier(
    vocab_size=1000,
    num_classes=2,
    d_model=64,
    nhead=2,
    num_encoder_layers=1,
    max_seq_length=64,
)

input_ids = torch.randint(0, 1000, (8, 64))
attention_mask = torch.ones(8, 64, dtype=torch.long)
logits = model(input_ids, attention_mask)
```

## Tokenizer and Dataset Utilities

```python
from hqde import SimpleTokenizer, TextClassificationDataset, TextDataLoader

texts = ["sample one", "sample two"]
labels = [0, 1]

tokenizer = SimpleTokenizer(vocab_size=1000, max_seq_length=32)
tokenizer.build_vocab(texts, min_freq=1)

dataset = TextClassificationDataset(texts, labels, tokenizer)
loader = TextDataLoader.create_text_loader(dataset, batch_size=2, num_workers=0)

batch = next(iter(loader))
print(batch.keys())  # input_ids, attention_mask, labels
```

Do not pass this dict loader directly into `HQDESystem.train()` until core dict-batch support is added.

## Limited HQDE Workaround

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from hqde import create_hqde_system, LightweightTransformerClassifier
from hqde import make_transformer_training_config

input_ids = torch.randint(0, 1000, (32, 64))
labels = torch.randint(0, 2, (32,))
loader = DataLoader(TensorDataset(input_ids, labels), batch_size=8)

system = create_hqde_system(
    model_class=LightweightTransformerClassifier,
    model_kwargs={
        "vocab_size": 1000,
        "num_classes": 2,
        "d_model": 64,
        "nhead": 2,
        "num_encoder_layers": 1,
        "max_seq_length": 64,
    },
    num_workers=2,
    training_config=make_transformer_training_config(
        ensemble_mode="independent",
        use_amp=False,
    ),
)

system.train(loader, num_epochs=1)
predictions = system.predict(loader)
system.cleanup()
```

This omits `attention_mask`; use it only for smoke tests or data without padding concerns.

## Kaggle DeBERTa Notebook

For HuggingFace-style masked transformer training, use:

```text
examples/cbt_deberta_hqde_kaggle.ipynb
```

The notebook has custom workers that handle:

```python
batch["input_ids"]
batch["attention_mask"]
batch["labels"]
```

It supports `HQDE_QUICK_TEST=1` for a short smoke run.

## Training Presets

```python
make_transformer_training_config()
make_cbt_training_config()
make_lightweight_transformer_config()
make_large_transformer_config()
```

These presets return dictionaries for `HQDESystem`. They do not by themselves solve the dict-batch limitation.

## Next Implementation Task

Add a core batch adapter that normalizes dataloader output into:

```python
model_inputs, targets = normalize_batch(batch)
```

Then train/evaluate/predict workers can call either `model(data)` or `model(**model_inputs)` depending on the input type.
