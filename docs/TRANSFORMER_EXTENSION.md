# HQDE Transformer Extension

## Overview

The HQDE framework now supports **Transformer-based models** for NLP tasks, extending beyond CNNs to enable text classification, sentiment analysis, and specialized applications like **Cognitive Behavioral Therapy (CBT) text classification**.

This extension maintains all the core HQDE benefits:
- **Distributed ensemble learning** with Ray
- **Quantum-inspired aggregation** for better predictions
- **Adaptive quantization** for communication efficiency
- **Byzantine fault tolerance** for robust training
- **Hierarchical aggregation** for scalability

---

## Key Features

### 1. **Transformer Models**

Three transformer architectures are provided:

| Model | Description | Use Case |
|-------|-------------|----------|
| `TransformerTextClassifier` | Standard transformer encoder | General text classification |
| `LightweightTransformerClassifier` | Reduced parameters | Fast training, small datasets |
| `CBTTransformerClassifier` | Domain-adapted for CBT | Mental health text analysis |

### 2. **Text Data Utilities**

- **SimpleTokenizer**: Word-level tokenization with vocabulary building
- **TextClassificationDataset**: PyTorch dataset for text classification
- **CBTDataset**: Specialized dataset for CBT tasks
- **TextDataLoader**: Optimized data loading for transformers

### 3. **Training Presets**

Pre-configured training settings optimized for transformers:

- `make_transformer_training_config()`: General transformer training
- `make_cbt_training_config()`: CBT-specific optimization
- `make_lightweight_transformer_config()`: Fast training
- `make_large_transformer_config()`: Large model training

---

## Architecture Details

### TransformerTextClassifier

```python
TransformerTextClassifier(
    vocab_size=30000,          # Vocabulary size
    num_classes=2,             # Number of output classes
    d_model=256,               # Model dimension
    nhead=8,                   # Number of attention heads
    num_encoder_layers=6,      # Transformer layers
    dim_feedforward=1024,      # FFN dimension
    dropout_rate=0.1,          # Dropout rate
    max_seq_length=512,        # Max sequence length
    activation="gelu",         # Activation function
    use_pooling="cls"          # Pooling: "cls", "mean", or "max"
)
```

**Key Components:**
- Token embedding layer
- Sinusoidal positional encoding
- Multi-head self-attention (Pre-LN for stability)
- Feed-forward networks
- Classification head with dropout

### CBTTransformerClassifier

Specialized for Cognitive Behavioral Therapy text classification:

```python
CBTTransformerClassifier(
    vocab_size=30000,
    num_classes=10,            # e.g., 10 cognitive distortions
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    dim_feedforward=1024,
    dropout_rate=0.15,         # Higher for ensemble diversity
    max_seq_length=512,
    use_domain_adaptation=True # Domain-specific layers
)
```

**Special Features:**
- Domain adaptation layer for CBT context
- Auxiliary emotion classification head
- Optimized for mental health text patterns
- Higher dropout for ensemble diversity

---

## Quick Start

### Basic Text Classification

```python
from hqde import (
    create_hqde_system,
    TransformerTextClassifier,
    SimpleTokenizer,
    TextClassificationDataset,
    TextDataLoader,
    make_transformer_training_config
)

# 1. Prepare data
texts = ["This is great!", "This is terrible!", ...]
labels = [1, 0, ...]  # Positive/Negative

# 2. Build tokenizer
tokenizer = SimpleTokenizer(vocab_size=10000, max_seq_length=128)
tokenizer.build_vocab(texts, min_freq=2)

# 3. Create dataset
dataset = TextClassificationDataset(texts, labels, tokenizer)
train_loader = TextDataLoader.create_text_loader(dataset, batch_size=32)

# 4. Configure HQDE
model_kwargs = {
    'vocab_size': len(tokenizer.word2idx),
    'num_classes': 2,
    'd_model': 128,
    'nhead': 4,
    'num_encoder_layers': 3
}

training_config = make_transformer_training_config(
    ensemble_mode='independent',
    learning_rate=5e-4
)

# 5. Create HQDE system
hqde_system = create_hqde_system(
    model_class=TransformerTextClassifier,
    model_kwargs=model_kwargs,
    num_workers=4,
    training_config=training_config
)

# 6. Train
metrics = hqde_system.train(train_loader, num_epochs=10)

# 7. Predict
predictions = hqde_system.predict(test_loader)

# 8. Cleanup
hqde_system.cleanup()
```

---

## CBT Text Classification Example

### Cognitive Distortion Detection

```python
from hqde import (
    create_hqde_system,
    CBTTransformerClassifier,
    SimpleTokenizer,
    CBTDataset,
    TextDataLoader,
    make_cbt_training_config
)

# CBT cognitive distortions (10 classes)
distortions = [
    "all_or_nothing",           # 0: Black-and-white thinking
    "overgeneralization",       # 1: Broad conclusions from single events
    "mental_filter",            # 2: Focus only on negatives
    "disqualifying_positive",   # 3: Reject positive experiences
    "jumping_to_conclusions",   # 4: Mind reading, fortune telling
    "magnification",            # 5: Catastrophizing or minimizing
    "emotional_reasoning",      # 6: Feelings as facts
    "should_statements",        # 7: Rigid rules
    "labeling",                 # 8: Global labels
    "personalization"           # 9: Taking responsibility for external events
]

# Example texts
texts = [
    "I always fail at everything I try",           # All-or-nothing
    "This always happens to me",                   # Overgeneralization
    "That one mistake ruined everything",          # Mental filter
    "I feel stupid, so I must be stupid",          # Emotional reasoning
    "I should be perfect at this",                 # Should statements
]
labels = [0, 1, 2, 6, 7]

# Build tokenizer
tokenizer = SimpleTokenizer(vocab_size=5000, max_seq_length=128)
tokenizer.build_vocab(texts, min_freq=1)

# Create dataset
train_dataset = CBTDataset(texts, labels, tokenizer)
train_loader = TextDataLoader.create_text_loader(train_dataset, batch_size=32)

# Model configuration
model_kwargs = {
    'vocab_size': len(tokenizer.word2idx),
    'num_classes': 10,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 4,
    'dropout_rate': 0.15,
    'use_domain_adaptation': True
}

# CBT-optimized training config
training_config = make_cbt_training_config(
    ensemble_mode='independent',
    learning_rate=3e-4,
    warmup_epochs=5,
    label_smoothing=0.05
)

# Create HQDE system
hqde_system = create_hqde_system(
    model_class=CBTTransformerClassifier,
    model_kwargs=model_kwargs,
    num_workers=4,
    training_config=training_config
)

# Train
hqde_system.train(train_loader, num_epochs=20)

# Predict on new text
new_text = "I'm a complete failure"
encoded = tokenizer.encode(new_text)
# ... (create mini loader and predict)

hqde_system.cleanup()
```

---

## Training Configurations

### 1. Standard Transformer Config

```python
config = make_transformer_training_config(
    ensemble_mode='independent',
    learning_rate=5e-4,          # Typical for transformers
    optimizer='adamw',           # AdamW recommended
    weight_decay=1e-2,           # L2 regularization
    warmup_epochs=3,             # Warmup for stability
    label_smoothing=0.1,         # Regularization
    gradient_clip_norm=1.0,      # Prevent exploding gradients
    use_amp=True                 # Mixed precision
)
```

### 2. CBT-Specific Config

```python
config = make_cbt_training_config(
    ensemble_mode='independent',
    learning_rate=3e-4,          # Lower for stability
    warmup_epochs=5,             # More warmup
    label_smoothing=0.05,        # Light smoothing
    dropout_rate=0.15            # Higher for diversity
)
```

### 3. Lightweight Config

```python
config = make_lightweight_transformer_config(
    ensemble_mode='independent',
    learning_rate=1e-3,          # Higher for small models
    warmup_epochs=2              # Less warmup needed
)
```

### 4. Large Model Config

```python
config = make_large_transformer_config(
    ensemble_mode='fedavg',      # Memory efficient
    learning_rate=1e-4,          # Lower for large models
    warmup_epochs=10,            # More warmup
    gradient_clip_norm=0.5       # Stricter clipping
)
```

---

## Model Comparison

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| Lightweight | ~500K | Fast | Good | Quick experiments, small datasets |
| Standard | ~5M | Medium | Better | General text classification |
| CBT | ~5M | Medium | Best (CBT) | Mental health, domain-specific |

---

## Ensemble Modes for Transformers

### Independent Mode (Recommended)

```python
training_config = make_transformer_training_config(
    ensemble_mode='independent',
    batch_assignment='replicate'
)
```

**Benefits:**
- Maximum ensemble diversity
- Each worker trains independently
- Better generalization
- Ideal for transformers

### FedAvg Mode

```python
training_config = make_transformer_training_config(
    ensemble_mode='fedavg',
    batch_assignment='split'
)
```

**Benefits:**
- Memory efficient (quantized communication)
- Epoch-wise weight averaging
- Good for large models
- Reduced communication overhead

---

## Advanced Features

### 1. Multi-Task Learning (CBT)

```python
# CBT model with emotion prediction
model = CBTTransformerClassifier(
    num_classes=10,              # Cognitive distortions
    use_domain_adaptation=True
)

# Forward pass with emotions
logits, emotion_logits = model(
    input_ids,
    attention_mask,
    return_emotions=True
)
```

### 2. Custom Tokenization

For production, use HuggingFace tokenizers:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Encode
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)
```

### 3. Pooling Strategies

```python
model = TransformerTextClassifier(
    use_pooling='cls'    # CLS token (default)
    # use_pooling='mean'  # Mean pooling
    # use_pooling='max'   # Max pooling
)
```

---

## Performance Tips

### 1. **Batch Size**
- Start with 32 for standard models
- Reduce to 16 for large models
- Increase to 64 for lightweight models

### 2. **Sequence Length**
- 128: Short texts (tweets, sentences)
- 256: Medium texts (paragraphs)
- 512: Long texts (documents)

### 3. **Learning Rate**
- 5e-4: Standard transformers
- 3e-4: CBT/domain-specific
- 1e-3: Lightweight models
- 1e-4: Large models

### 4. **Warmup**
- 2-3 epochs: Small models
- 5 epochs: Standard models
- 10 epochs: Large models

### 5. **Workers**
- 4 workers: Optimal for most cases
- 2 workers: Memory-constrained
- 8 workers: High-end systems

---

## Real-World Applications

### 1. **Mental Health**
- Cognitive distortion detection
- Therapy session analysis
- Patient sentiment tracking
- Intervention recommendation

### 2. **Customer Support**
- Ticket classification
- Sentiment analysis
- Urgency detection
- Intent recognition

### 3. **Content Moderation**
- Toxic comment detection
- Hate speech identification
- Misinformation flagging

### 4. **Medical**
- Clinical note classification
- Symptom extraction
- Diagnosis prediction
- Treatment recommendation

---

## Example: Complete CBT Pipeline

See `examples/cbt_transformer_example.py` for a complete working example that demonstrates:

1. Data preparation and preprocessing
2. Tokenizer building
3. Dataset creation
4. HQDE system configuration
5. Ensemble training
6. Evaluation and prediction
7. Inference on new examples

Run the example:

```bash
python examples/cbt_transformer_example.py
```

---

## Comparison: CNNs vs Transformers in HQDE

| Aspect | CNNs | Transformers |
|--------|------|--------------|
| **Input** | Images (32x32, 224x224) | Text sequences (tokens) |
| **Architecture** | Conv layers + pooling | Self-attention + FFN |
| **Position** | Implicit (spatial) | Explicit (positional encoding) |
| **Parallelization** | High | Very high |
| **Memory** | Lower | Higher |
| **Training Speed** | Faster | Slower |
| **Interpretability** | Moderate | High (attention weights) |
| **Use Cases** | Vision tasks | NLP tasks |

**Both benefit equally from HQDE's:**
- Distributed ensemble learning
- Quantum-inspired aggregation
- Adaptive quantization
- Byzantine fault tolerance

---

## Future Enhancements

Planned features for transformer support:

1. **Pre-trained Models**: Integration with HuggingFace models
2. **Multi-Modal**: Vision-language transformers
3. **Efficient Attention**: Linear attention, sparse attention
4. **Knowledge Distillation**: Teacher-student ensembles
5. **Continual Learning**: Incremental training on new data

---

## Citation

If you use HQDE with transformers for CBT or NLP research:

```bibtex
@software{hqde_transformers2025,
  title={HQDE: Hierarchical Quantum-Distributed Ensemble Learning with Transformers},
  author={HQDE Team},
  year={2025},
  url={https://github.com/Prathmesh333/HQDE-PyPI}
}
```

---

## Support

For questions or issues with transformer models:
- **GitHub Issues**: [Create an issue](https://github.com/Prathmesh333/HQDE-PyPI/issues)
- **Example Code**: `examples/cbt_transformer_example.py`
- **Documentation**: This file and `README.md`
