# Transformer Support Status

HQDE includes transformer model classes and text utilities, but full dict-batch transformer support is not yet wired into the core `HQDESystem` training loop.

This page documents what is implemented now, what works, and what still needs to be completed before transformer support can be described as plug-and-play.

## Implemented Components

### Model Classes

Defined in `hqde/models/transformers.py`:

| Class | Purpose |
|-------|---------|
| `LightweightTransformerClassifier` | Small encoder classifier for fast experiments. |
| `TransformerTextClassifier` | Standard transformer encoder classifier. |
| `CBTTransformerClassifier` | CBT-themed classifier with optional domain adapter and emotion head. |

These classes accept:

```python
logits = model(input_ids, attention_mask=None)
```

Direct forward passes with `attention_mask` work.

### Text Utilities

Defined in `hqde/utils/text_data_utils.py`:

| Utility | Purpose |
|---------|---------|
| `SimpleTokenizer` | Word-level tokenizer for simple experiments. |
| `TextClassificationDataset` | Dataset returning tokenized dict samples. |
| `CBTDataset` | CBT-specific dict dataset. |
| `TextDataLoader` | DataLoader factory with a dict collate function. |

`TextDataLoader` returns batches like:

```python
{
    "input_ids": tensor,
    "attention_mask": tensor,
    "labels": tensor,
}
```

### Training Presets

Defined in `hqde/utils/transformer_presets.py`:

- `make_transformer_training_config`
- `make_cbt_training_config`
- `make_lightweight_transformer_config`
- `make_large_transformer_config`

These return HQDE training dictionaries with transformer-friendly defaults such as AdamW, warmup, label smoothing, gradient clipping, and `federated_normalization="shared"`.

## Current Core Limitation

`HQDESystem.train()` currently expects dataloaders to yield tuple/list batches:

```python
data, targets = batch[0], batch[1]
```

Worker training then calls:

```python
outputs = self.model(data_batch)
```

That works for CNNs and simple single-tensor models. It does not pass `attention_mask` into transformer models and rejects dict batches returned by `TextDataLoader`.

Observed smoke-test result:

```text
PASS: direct transformer forward with attention_mask
PASS: HQDE transformer with tuple-loader workaround
FAIL: HQDE transformer documented TextDataLoader path
ValueError: Training batches must contain at least (data, targets)
```

## Working Usage Today

### Direct Transformer Forward

```python
import torch
from hqde.models.transformers import LightweightTransformerClassifier

model = LightweightTransformerClassifier(
    vocab_size=1000,
    num_classes=2,
    d_model=64,
    nhead=2,
    num_encoder_layers=1,
)

input_ids = torch.randint(0, 1000, (4, 32))
attention_mask = torch.ones(4, 32, dtype=torch.long)
logits = model(input_ids, attention_mask)
```

### HQDE Tuple-Loader Workaround

This path omits `attention_mask`, so it is suitable only for simple smoke tests where padding effects are acceptable.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from hqde import create_hqde_system
from hqde.models.transformers import LightweightTransformerClassifier
from hqde.utils.transformer_presets import make_transformer_training_config

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

metrics = system.train(loader, num_epochs=1)
predictions = system.predict(loader)
system.cleanup()
```

### HuggingFace DeBERTa Notebook

Use `examples/cbt_deberta_hqde_kaggle.ipynb` for a masked transformer workflow. That notebook does not rely on `HQDESystem`; it defines a custom worker that explicitly passes:

```python
logits = model(input_ids, attention_mask)
```

## What Is Needed for Plug-and-Play Transformer Support

The core framework should add a batch adapter that supports:

- `(data, targets)` tuple batches for CNNs and existing models.
- Dict batches with `input_ids`, `attention_mask`, and `labels`.
- Optional model kwargs such as `token_type_ids`.
- Train, evaluate, and predict paths.
- Ray serialization compatibility.

The worker call should support both:

```python
outputs = model(data)
```

and:

```python
outputs = model(**model_inputs)
```

Once that is implemented and tested, the examples using `TextDataLoader` can be restored as true plug-and-play examples.

## Testing Checklist

Run:

```bash
python test_transformer_integration.py
```

At the time of this documentation update, the package-level transformer pieces pass, while the core dict-batch integration test fails as described above. That failure should be treated as the next implementation task, not hidden in the docs.

## Notes for Thesis Writing

Use precise language:

- Correct: "HQDE currently supports CNN-style tuple-batch models through the core training loop."
- Correct: "Transformer model classes and utilities are included."
- Correct: "The DeBERTa notebook demonstrates a custom transformer ensemble path."
- Not correct yet: "Transformers are fully plug-and-play through `HQDESystem` with `TextDataLoader`."
