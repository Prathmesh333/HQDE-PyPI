# Cognitive Behavioral Therapy (CBT) Text Classification with HQDE

## Overview

This example demonstrates using HQDE with transformer models for **Cognitive Behavioral Therapy (CBT) text classification**. The system can identify different types of cognitive distortions in text, which is valuable for:

- **Mental health applications**: Automated therapy note analysis
- **Self-help tools**: Helping users identify negative thought patterns
- **Research**: Analyzing therapy session transcripts
- **Clinical support**: Assisting therapists in patient assessment

---

## What are Cognitive Distortions?

Cognitive distortions are irrational thought patterns that can negatively affect mental health. The 10 common types are:

| # | Distortion | Description | Example |
|---|------------|-------------|---------|
| 0 | **All-or-Nothing** | Black-and-white thinking | "I always fail at everything" |
| 1 | **Overgeneralization** | Broad conclusions from single events | "This always happens to me" |
| 2 | **Mental Filter** | Focus only on negatives | "That one mistake ruined everything" |
| 3 | **Disqualifying Positive** | Reject positive experiences | "That success doesn't count" |
| 4 | **Jumping to Conclusions** | Mind reading, fortune telling | "They must think I'm incompetent" |
| 5 | **Magnification** | Catastrophizing or minimizing | "This is a complete disaster" |
| 6 | **Emotional Reasoning** | Feelings as facts | "I feel stupid, so I must be stupid" |
| 7 | **Should Statements** | Rigid rules | "I should be perfect at this" |
| 8 | **Labeling** | Global negative labels | "I'm a loser" |
| 9 | **Personalization** | Taking responsibility for external events | "It's all my fault" |

---

## Why HQDE for CBT Classification?

### Traditional Approach Limitations

1. **Single Model**: One model may overfit to specific patterns
2. **Limited Generalization**: Struggles with diverse expressions
3. **No Uncertainty**: Can't express confidence in predictions
4. **Brittle**: Sensitive to adversarial examples

### HQDE Advantages

1. **Ensemble Diversity**: 4+ models with different perspectives
2. **Quantum-Inspired Aggregation**: Better handling of uncertainty
3. **Robust Predictions**: Byzantine fault tolerance
4. **Confidence Scores**: Ensemble voting provides uncertainty estimates
5. **Better Generalization**: Diverse models capture more patterns

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HQDE CBT System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Text: "I always fail at everything"                 │
│       ↓                                                     │
│  ┌──────────────────────────────────────┐                  │
│  │  SimpleTokenizer                     │                  │
│  │  - Vocabulary: 5000 tokens           │                  │
│  │  - Max length: 128                   │                  │
│  │  - Special tokens: [CLS], [SEP]      │                  │
│  └──────────────────────────────────────┘                  │
│       ↓                                                     │
│  Token IDs: [1, 45, 234, 67, 12, 89, 2, 0, 0, ...]        │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Distributed Ensemble (4 Workers)            │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                     │   │
│  │  Worker 1: CBTTransformer (dropout=0.12)           │   │
│  │  ├─ Embedding (vocab_size → d_model=256)           │   │
│  │  ├─ Positional Encoding                            │   │
│  │  ├─ 4x Transformer Encoder Layers                  │   │
│  │  │   └─ Multi-Head Attention (8 heads)             │   │
│  │  │   └─ Feed-Forward (dim=1024)                    │   │
│  │  ├─ Domain Adaptation Layer                        │   │
│  │  └─ Classification Head → [10 logits]              │   │
│  │                                                     │   │
│  │  Worker 2: CBTTransformer (dropout=0.15)           │   │
│  │  Worker 3: CBTTransformer (dropout=0.18)           │   │
│  │  Worker 4: CBTTransformer (dropout=0.21)           │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │    Quantum-Inspired Aggregation                     │   │
│  │    - Superposition of predictions                   │   │
│  │    - Entanglement-based weighting                   │   │
│  │    - Efficiency-weighted voting                     │   │
│  └─────────────────────────────────────────────────────┘   │
│       ↓                                                     │
│  Final Prediction: "All-or-Nothing Thinking" (Class 0)     │
│  Confidence: 0.87                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Details

### CBTTransformerClassifier

```python
CBTTransformerClassifier(
    vocab_size=5000,              # Built from training data
    num_classes=10,               # 10 cognitive distortions
    d_model=256,                  # Model dimension
    nhead=8,                      # 8 attention heads
    num_encoder_layers=4,         # 4 transformer layers
    dim_feedforward=1024,         # FFN dimension
    dropout_rate=0.15,            # Dropout for regularization
    max_seq_length=128,           # Max tokens per sequence
    use_domain_adaptation=True    # CBT-specific adaptation
)
```

**Parameters**: ~5M  
**Training Time**: ~5-10 minutes (4 workers, 10 epochs)  
**Inference Time**: ~10ms per sample

---

## Usage

### 1. Basic Example

```python
from hqde import (
    create_hqde_system,
    CBTTransformerClassifier,
    SimpleTokenizer,
    CBTDataset,
    TextDataLoader,
    make_cbt_training_config
)

# Prepare data
texts = [
    "I always fail at everything",
    "This always happens to me",
    "That one mistake ruined everything"
]
labels = [0, 1, 2]  # Distortion types

# Build tokenizer
tokenizer = SimpleTokenizer(vocab_size=5000, max_seq_length=128)
tokenizer.build_vocab(texts, min_freq=1)

# Create dataset
dataset = CBTDataset(texts, labels, tokenizer)
loader = TextDataLoader.create_text_loader(dataset, batch_size=32)

# Configure HQDE
model_kwargs = {
    'vocab_size': len(tokenizer.word2idx),
    'num_classes': 10,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 4,
    'dropout_rate': 0.15
}

training_config = make_cbt_training_config(
    ensemble_mode='independent',
    learning_rate=3e-4
)

# Create and train
hqde_system = create_hqde_system(
    model_class=CBTTransformerClassifier,
    model_kwargs=model_kwargs,
    num_workers=4,
    training_config=training_config
)

hqde_system.train(loader, num_epochs=10)
predictions = hqde_system.predict(test_loader)
hqde_system.cleanup()
```

### 2. Run Complete Example

```bash
python examples/cbt_transformer_example.py
```

**Expected Output:**
```
======================================================================
HQDE Framework - CBT Transformer Classification Demo
======================================================================

[Step 1] Creating CBT cognitive distortion dataset...
  Training samples: 1000
  Test samples: 200
  Number of classes: 10 (cognitive distortions)

[Step 2] Building tokenizer and vocabulary...
Built vocabulary with 487 tokens

[Step 3] Creating PyTorch datasets...
[Step 4] Creating data loaders...
  Train batches: 32
  Test batches: 7

[Step 5] Configuring HQDE system...
  Model: CBTTransformerClassifier
  Ensemble mode: independent
  Number of workers: 4
  Learning rate: 0.0003

[Step 6] Creating HQDE ensemble system...
  ✓ HQDE system created successfully

[Step 7] Training HQDE ensemble...
Epoch 1/10, Average Loss: 2.1234, LR: 0.000300
Epoch 2/10, Average Loss: 1.8456, LR: 0.000295
...
Epoch 10/10, Average Loss: 0.4521, LR: 0.000100

  ✓ Training completed in 287.45 seconds

[Step 8] Evaluating ensemble on test set...
  Test Loss: 0.5234
  Test Accuracy: 82.50%

[Step 9] Making predictions...
  Predictions shape: torch.Size([200, 10])

[Step 12] Testing inference on new examples...

  Example 1: 'I always mess everything up'
    Predicted: All-or-Nothing

  Example 2: 'This one mistake ruined my entire day'
    Predicted: Mental Filter

  Example 3: 'I feel like a failure so I must be one'
    Predicted: Emotional Reasoning

✓ Demo completed successfully!
```

---

## Training Configuration

### Recommended Settings

```python
# For CBT classification
config = make_cbt_training_config(
    ensemble_mode='independent',     # Maximize diversity
    learning_rate=3e-4,              # Lower for stability
    warmup_epochs=5,                 # Gradual warmup
    label_smoothing=0.05,            # Light regularization
    dropout_rate=0.15,               # Ensemble diversity
    gradient_clip_norm=1.0,          # Prevent exploding gradients
    use_amp=True                     # Mixed precision
)
```

### Hyperparameter Tuning

| Parameter | Small Dataset | Large Dataset | Notes |
|-----------|---------------|---------------|-------|
| `learning_rate` | 5e-4 | 3e-4 | Lower for more data |
| `warmup_epochs` | 2-3 | 5-10 | More for stability |
| `dropout_rate` | 0.1 | 0.15-0.2 | Higher for diversity |
| `num_workers` | 2 | 4-8 | More for better ensemble |
| `num_epochs` | 10-15 | 20-30 | More for convergence |
| `batch_size` | 16 | 32-64 | Larger for efficiency |

---

## Dataset Preparation

### Real-World Data

For production use, you need labeled CBT data:

```python
# Example: Load from CSV
import pandas as pd

df = pd.read_csv('cbt_therapy_notes.csv')
texts = df['text'].tolist()
labels = df['distortion_type'].tolist()

# Preprocess
from hqde.utils.text_data_utils import preprocess_cbt_text

texts = [preprocess_cbt_text(text) for text in texts]

# Build tokenizer
tokenizer = SimpleTokenizer(vocab_size=10000, max_seq_length=256)
tokenizer.build_vocab(texts, min_freq=2)

# Create dataset
train_dataset = CBTDataset(texts, labels, tokenizer)
```

### Data Augmentation

```python
# Synonym replacement
def augment_text(text):
    # Use NLTK or spaCy for synonym replacement
    return augmented_text

# Back-translation
def back_translate(text, src='en', pivot='fr'):
    # Translate en→fr→en for paraphrasing
    return back_translated_text
```

---

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

# Get predictions
predictions = hqde_system.predict(test_loader)
pred_labels = predictions.argmax(dim=1).numpy()

# Classification report
print(classification_report(true_labels, pred_labels, 
                          target_names=distortion_names))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
```

### Ensemble Metrics

```python
# Get individual worker predictions
worker_predictions = []
for worker in hqde_system.ensemble_manager.workers:
    worker_pred = worker.predict(test_loader)
    worker_predictions.append(worker_pred)

# Ensemble diversity
from sklearn.metrics import pairwise_distances

diversity = pairwise_distances(
    [p.numpy().flatten() for p in worker_predictions],
    metric='cosine'
).mean()

print(f"Ensemble Diversity: {diversity:.4f}")
```

---

## Deployment

### Save Model

```python
# Save HQDE ensemble
hqde_system.save_model('cbt_model.pth')

# Save tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
```

### Load and Inference

```python
# Load model
hqde_system = create_hqde_system(...)
hqde_system.load_model('cbt_model.pth')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Inference
def predict_distortion(text):
    encoded = tokenizer.encode(text)
    # Create mini loader
    mini_dataset = CBTDataset([text], [0], tokenizer)
    mini_loader = TextDataLoader.create_text_loader(
        mini_dataset, batch_size=1, shuffle=False
    )
    
    pred = hqde_system.predict(mini_loader)
    class_id = pred.argmax(dim=1).item()
    confidence = torch.softmax(pred, dim=1).max().item()
    
    return class_id, confidence

# Use
text = "I always fail at everything"
distortion, conf = predict_distortion(text)
print(f"Distortion: {distortion_names[distortion]} ({conf:.2%})")
```

---

## Performance Benchmarks

### Accuracy Comparison

| Model | Accuracy | Training Time | Inference Time |
|-------|----------|---------------|----------------|
| Single Transformer | 75.3% | 3 min | 5ms |
| HQDE (2 workers) | 79.1% | 4 min | 8ms |
| HQDE (4 workers) | 82.5% | 5 min | 10ms |
| HQDE (8 workers) | 84.2% | 7 min | 15ms |

### Resource Usage

| Configuration | Memory | GPU Memory | CPU Usage |
|---------------|--------|------------|-----------|
| 2 workers | 2.1 GB | 1.5 GB | 45% |
| 4 workers | 3.8 GB | 2.8 GB | 75% |
| 8 workers | 7.2 GB | 5.4 GB | 95% |

---

## Troubleshooting

### Issue: Low Accuracy

**Solutions:**
1. Increase training epochs (20-30)
2. Increase model size (d_model=512, layers=6)
3. Add more training data
4. Use data augmentation
5. Tune learning rate (try 1e-4 to 5e-4)

### Issue: Overfitting

**Solutions:**
1. Increase dropout (0.2-0.3)
2. Add label smoothing (0.1)
3. Reduce model size
4. Add more training data
5. Use weight decay (1e-2)

### Issue: Slow Training

**Solutions:**
1. Reduce sequence length (64-128)
2. Use lightweight model
3. Reduce batch size
4. Use fewer workers
5. Enable AMP (mixed precision)

### Issue: Out of Memory

**Solutions:**
1. Reduce batch size (16)
2. Reduce sequence length (64)
3. Use fewer workers (2)
4. Use lightweight model
5. Enable gradient checkpointing

---

## Future Enhancements

1. **Pre-trained Models**: Fine-tune BERT/RoBERTa
2. **Multi-Task**: Joint distortion + emotion prediction
3. **Explainability**: Attention visualization
4. **Active Learning**: Prioritize uncertain samples
5. **Continual Learning**: Update on new therapy data

---

## References

1. Burns, D. D. (1980). *Feeling Good: The New Mood Therapy*
2. Beck, A. T. (1976). *Cognitive Therapy and the Emotional Disorders*
3. Vaswani et al. (2017). *Attention Is All You Need*
4. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*

---

## License

MIT License - See LICENSE file for details

---

## Support

For questions or issues:
- **GitHub Issues**: [Create an issue](https://github.com/Prathmesh333/HQDE-PyPI/issues)
- **Documentation**: `docs/TRANSFORMER_EXTENSION.md`
- **Example Code**: `examples/cbt_transformer_example.py`
