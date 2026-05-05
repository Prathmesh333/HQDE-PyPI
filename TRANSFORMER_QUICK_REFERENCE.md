# HQDE Transformer Extension - Quick Reference

## Installation

```bash
pip install hqde
```

---

## Import Everything You Need

```python
from hqde import (
    # Core
    create_hqde_system,
    
    # Transformer Models
    TransformerTextClassifier,
    LightweightTransformerClassifier,
    CBTTransformerClassifier,
    
    # Text Utilities
    SimpleTokenizer,
    TextClassificationDataset,
    CBTDataset,
    TextDataLoader,
    
    # Training Presets
    make_transformer_training_config,
    make_cbt_training_config,
    make_lightweight_transformer_config,
    make_large_transformer_config
)
```

---

## 5-Minute Quick Start

```python
# 1. Prepare data
texts = ["I always fail", "This always happens", "I feel stupid"]
labels = [0, 1, 2]

# 2. Build tokenizer
tokenizer = SimpleTokenizer(vocab_size=5000, max_seq_length=128)
tokenizer.build_vocab(texts, min_freq=1)

# 3. Create dataset & loader
dataset = CBTDataset(texts, labels, tokenizer)
loader = TextDataLoader.create_text_loader(dataset, batch_size=32)

# 4. Configure model
model_kwargs = {
    'vocab_size': len(tokenizer.word2idx),
    'num_classes': 10,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 4
}

# 5. Create HQDE system
hqde_system = create_hqde_system(
    model_class=CBTTransformerClassifier,
    model_kwargs=model_kwargs,
    num_workers=4,
    training_config=make_cbt_training_config()
)

# 6. Train
hqde_system.train(loader, num_epochs=20)

# 7. Predict
predictions = hqde_system.predict(test_loader)

# 8. Cleanup
hqde_system.cleanup()
```

---

## Model Selection

| Model | When to Use | Config |
|-------|-------------|--------|
| `LightweightTransformerClassifier` | Small datasets, quick experiments | `d_model=64, nhead=2, layers=2` |
| `TransformerTextClassifier` | General text classification | `d_model=256, nhead=8, layers=6` |
| `CBTTransformerClassifier` | Mental health, domain-specific | `d_model=256, nhead=8, layers=4` |

---

## Training Config Selection

| Config | When to Use | Learning Rate |
|--------|-------------|---------------|
| `make_lightweight_transformer_config()` | Small models, fast training | 1e-3 |
| `make_transformer_training_config()` | General purpose | 5e-4 |
| `make_cbt_training_config()` | CBT/domain-specific | 3e-4 |
| `make_large_transformer_config()` | Large models, careful training | 1e-4 |

---

## Common Configurations

### Small & Fast
```python
model_kwargs = {
    'vocab_size': 3000,
    'num_classes': 2,
    'd_model': 64,
    'nhead': 2,
    'num_encoder_layers': 2,
    'max_seq_length': 64
}
config = make_lightweight_transformer_config()
num_workers = 2
num_epochs = 10
```

### Standard
```python
model_kwargs = {
    'vocab_size': 10000,
    'num_classes': 10,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 4,
    'max_seq_length': 128
}
config = make_transformer_training_config()
num_workers = 4
num_epochs = 20
```

### Large & Accurate
```python
model_kwargs = {
    'vocab_size': 30000,
    'num_classes': 20,
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'max_seq_length': 512
}
config = make_large_transformer_config()
num_workers = 8
num_epochs = 30
```

---

## Tokenizer Cheat Sheet

```python
# Create tokenizer
tokenizer = SimpleTokenizer(
    vocab_size=10000,
    max_seq_length=128,
    lowercase=True,
    remove_punctuation=False
)

# Build vocabulary
tokenizer.build_vocab(texts, min_freq=2)

# Encode single text
encoded = tokenizer.encode(
    "Hello world",
    add_special_tokens=True,
    padding=True,
    truncation=True,
    return_attention_mask=True
)
# Returns: {'input_ids': tensor, 'attention_mask': tensor}

# Encode batch
batch_encoded = tokenizer.encode_batch(texts)

# Decode
text = tokenizer.decode(token_ids)

# Vocab size
vocab_size = len(tokenizer.word2idx)
```

---

## Dataset Cheat Sheet

```python
# Text Classification Dataset
dataset = TextClassificationDataset(
    texts=texts,
    labels=labels,
    tokenizer=tokenizer,
    max_seq_length=128
)

# CBT Dataset (with optional emotions)
dataset = CBTDataset(
    texts=texts,
    labels=labels,
    tokenizer=tokenizer,
    max_seq_length=128,
    include_emotions=True,
    emotion_labels=emotion_labels
)

# Create DataLoader
loader = TextDataLoader.create_text_loader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

---

## Training Modes

### Independent (Recommended for Transformers)
```python
training_config = make_transformer_training_config(
    ensemble_mode='independent',
    batch_assignment='replicate'
)
```
- Maximum diversity
- Each worker trains independently
- Best generalization

### FedAvg (Memory Efficient)
```python
training_config = make_transformer_training_config(
    ensemble_mode='fedavg',
    batch_assignment='split'
)
```
- Quantized communication
- Epoch-wise weight averaging
- Good for large models

---

## Hyperparameter Guidelines

### Learning Rate
- **1e-3**: Lightweight models
- **5e-4**: Standard transformers
- **3e-4**: CBT/domain-specific
- **1e-4**: Large models

### Warmup Epochs
- **2-3**: Small models
- **5**: Standard models
- **10**: Large models

### Dropout
- **0.1**: Single model
- **0.15**: Ensemble (diversity)
- **0.2-0.3**: High regularization

### Batch Size
- **16**: Large models, limited memory
- **32**: Standard (recommended)
- **64**: Small models, fast training

### Sequence Length
- **64**: Short texts (tweets)
- **128**: Medium texts (paragraphs)
- **256**: Long texts (articles)
- **512**: Very long texts (documents)

### Workers
- **2**: Memory-constrained
- **4**: Optimal for most cases
- **8**: High-end systems

---

## Common Tasks

### Save Model
```python
hqde_system.save_model('model.pth')

import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
```

### Load Model
```python
hqde_system = create_hqde_system(...)
hqde_system.load_model('model.pth')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
```

### Evaluate
```python
eval_metrics = hqde_system.evaluate(test_loader)
print(f"Accuracy: {eval_metrics['accuracy']:.2%}")
print(f"Loss: {eval_metrics['loss']:.4f}")
```

### Get Performance Metrics
```python
metrics = hqde_system.get_performance_metrics()
for key, value in metrics.items():
    print(f"{key}: {value}")
```

### Inference on New Text
```python
def predict_text(text):
    encoded = tokenizer.encode(text)
    mini_dataset = CBTDataset([text], [0], tokenizer)
    mini_loader = TextDataLoader.create_text_loader(
        mini_dataset, batch_size=1, shuffle=False
    )
    pred = hqde_system.predict(mini_loader)
    class_id = pred.argmax(dim=1).item()
    confidence = torch.softmax(pred, dim=1).max().item()
    return class_id, confidence

class_id, conf = predict_text("I always fail")
print(f"Class: {class_id}, Confidence: {conf:.2%}")
```

---

## Troubleshooting

### Low Accuracy
- ✅ Increase epochs (20-30)
- ✅ Increase model size
- ✅ Add more data
- ✅ Tune learning rate

### Overfitting
- ✅ Increase dropout (0.2-0.3)
- ✅ Add label smoothing (0.1)
- ✅ Reduce model size
- ✅ Add more data

### Slow Training
- ✅ Reduce sequence length
- ✅ Use lightweight model
- ✅ Reduce batch size
- ✅ Use fewer workers

### Out of Memory
- ✅ Reduce batch size (16)
- ✅ Reduce sequence length (64)
- ✅ Use fewer workers (2)
- ✅ Use lightweight model

---

## Examples

### Run CBT Example
```bash
python examples/cbt_transformer_example.py
```

### Run Integration Tests
```bash
python test_transformer_integration.py
```

---

## Documentation

- **Full Guide**: `docs/TRANSFORMER_EXTENSION.md`
- **CBT Guide**: `examples/CBT_CLASSIFICATION_README.md`
- **Summary**: `TRANSFORMER_IMPLEMENTATION_SUMMARY.md`
- **This File**: `TRANSFORMER_QUICK_REFERENCE.md`

---

## Support

- **GitHub**: https://github.com/Prathmesh333/HQDE-PyPI
- **Issues**: https://github.com/Prathmesh333/HQDE-PyPI/issues

---

## Quick Commands

```bash
# Install
pip install hqde

# Run example
python examples/cbt_transformer_example.py

# Test
python test_transformer_integration.py

# Check version
python -c "import hqde; print(hqde.__version__)"
```

---

**That's it! You're ready to use HQDE with transformers! 🚀**
