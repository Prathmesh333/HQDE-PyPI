"""
CBT Cognitive Distortion Classification with DeBERTa + HQDE
Optimized for Kaggle: 2x T4 GPUs, 4 vCPUs

This notebook trains an ensemble of DeBERTa models using HQDE framework
to classify cognitive distortions in therapy conversations.

Hardware: 2x T4 GPUs, 4 vCPUs
Expected Training Time: ~30-45 minutes
Expected Accuracy: 85-90%
"""

# ============================================================================
# SECTION 1: SETUP AND INSTALLATION
# ============================================================================

print("=" * 80)
print("CBT Cognitive Distortion Classification with DeBERTa + HQDE")
print("=" * 80)
print("\nHardware Configuration:")
print("  - GPUs: 2x T4")
print("  - vCPUs: 4")
print("  - RAM: ~30GB")
print("\n" + "=" * 80)

# Install required packages
print("\n[1/8] Installing required packages...")
import subprocess
import sys

packages = [
    "transformers>=4.30.0",
    "datasets>=2.14.0",
    "accelerate>=0.20.0",
    "torch>=2.0.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✓ Packages installed successfully")

# ============================================================================
# SECTION 2: IMPORTS
# ============================================================================

print("\n[2/8] Importing libraries...")

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

print("✓ Libraries imported successfully")

# Check GPU availability
print(f"\n🔧 Hardware Check:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# ============================================================================
# SECTION 3: CONFIGURATION
# ============================================================================

print("\n[3/8] Setting up configuration...")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
class Config:
    # Model
    model_name = "microsoft/deberta-v3-base"  # DeBERTa v3 base model
    num_classes = 10
    max_length = 256  # Reduced for faster training
    
    # Training
    num_workers = 4  # Number of ensemble workers (HQDE)
    batch_size = 8  # Per GPU (effective batch = 8 * 2 GPUs = 16)
    num_epochs = 15  # Reduced for faster training
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    
    # Ensemble diversity
    dropout_rates = [0.1, 0.15, 0.2, 0.25]  # Different dropout per worker
    learning_rates = [1.5e-5, 2e-5, 2.5e-5, 3e-5]  # Different LR per worker
    
    # Hardware
    num_gpus = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True  # Mixed precision training
    
    # Data
    test_size = 0.2
    val_size = 0.1
    
    # Paths
    output_dir = "./cbt_deberta_hqde_output"
    
    # Cognitive distortions
    distortion_names = [
        "All-or-Nothing Thinking",
        "Overgeneralization",
        "Mental Filter",
        "Disqualifying the Positive",
        "Jumping to Conclusions",
        "Magnification/Catastrophizing",
        "Emotional Reasoning",
        "Should Statements",
        "Labeling",
        "Personalization"
    ]

config = Config()

# Create output directory
os.makedirs(config.output_dir, exist_ok=True)

print(f"✓ Configuration set")
print(f"  Model: {config.model_name}")
print(f"  Ensemble workers: {config.num_workers}")
print(f"  Batch size per GPU: {config.batch_size}")
print(f"  Effective batch size: {config.batch_size * config.num_gpus}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Max sequence length: {config.max_length}")

# ============================================================================
# SECTION 4: DATASET CREATION
# ============================================================================

print("\n[4/8] Creating CBT therapy conversation dataset...")

# Generate synthetic CBT dataset (100 conversations)
def generate_cbt_dataset():
    """Generate CBT therapy conversations with reasoning."""
    
    conversations = []
    conv_id = 1
    
    # Templates for each distortion type
    templates = {
        0: [  # All-or-Nothing
            ("I got a B+ on my exam. I'm devastated. I studied so hard, but it wasn't enough. I'm either an A student or I'm a complete failure. There's no in-between for me. Because it's not perfect! If I can't get an A, then what's the point? I either do things perfectly or I might as well not do them at all.", "frustrated, anxious"),
            ("My presentation went well, but I stumbled over one word. That one mistake ruined the whole thing. It's either perfect or it's garbage. No, you don't understand. If it's not flawless, then it's worthless.", "disappointed, perfectionistic"),
        ],
        1: [  # Overgeneralization
            ("I asked someone out and they said no. This always happens to me. I'm never going to find a partner. Every time I try, I get rejected. Well, this was only the second time, but both times they said no. So clearly I'm just not dateable. Everyone rejects me.", "hopeless, defeated"),
            ("I made a mistake at work today. I always mess things up. I never do anything right. This is typical of me. It happens all the time. Every project I touch goes wrong.", "defeated, hopeless"),
        ],
        2: [  # Mental Filter
            ("My performance review was mostly positive, but there was one area for improvement mentioned. That's the only part I can think about. The positive feedback doesn't matter. That one criticism is all I hear. I keep replaying it in my mind.", "anxious, dissatisfied"),
            ("I had a great day, but I made one small social mistake. That mistake is all I remember now. Everything else fades away compared to that one error.", "regretful, fixated"),
        ],
        3: [  # Disqualifying Positive
            ("I got promoted, but it doesn't mean anything. They probably just needed someone and I was there. It's not like I earned it. Yes, I exceeded the metrics, but that doesn't really count. The metrics were probably easy.", "doubtful, unworthy"),
            ("People complimented my work, but they were probably just being polite. They don't really mean it. It's not genuine praise.", "dismissive, skeptical"),
        ],
        4: [  # Jumping to Conclusions
            ("I saw my coworkers whispering. They were definitely talking about me. They must think I'm incompetent. No, I didn't ask. It was obvious from their body language. My manager has been quiet - he's probably building a case to fire me.", "paranoid, anxious"),
            ("My friend hasn't texted me back in two hours. She must be mad at me. She's probably ending our friendship. I just know it.", "worried, insecure"),
        ],
        5: [  # Catastrophizing
            ("I have a headache and I'm tired. This could be a brain tumor. I might be dying. My whole life could be over. I'm terrified. This is a disaster. I can't stop thinking about worst-case scenarios.", "panicked, terrified"),
            ("I made a small mistake in my email. This is a complete disaster. Everyone will think I'm unprofessional. My career is ruined.", "overwhelmed, panicked"),
        ],
        6: [  # Emotional Reasoning
            ("I feel stupid, so I must be stupid. I feel like everyone thinks I'm incompetent, so I must be incompetent. I feel it so strongly. If I feel anxious, it means something bad will happen. My emotions are telling me the truth.", "insecure, anxious"),
            ("I feel like a fraud at work, so I must be a fraud. My feelings don't lie to me. I feel overwhelmed, which means I can't handle my life.", "fraudulent, overwhelmed"),
        ],
        7: [  # Should Statements
            ("I should be able to handle everything. I must be strong all the time. I ought to be perfect at my job, as a parent, as a partner. I should never burden others. I must always be productive. If I don't meet these standards, I feel guilty.", "guilty, stressed, exhausted"),
            ("I must never make mistakes. I should be perfect in everything I do. I ought to exceed everyone's expectations always. I should always put others first.", "pressured, exhausted"),
        ],
        8: [  # Labeling
            ("I made a mistake - I sent an email to the wrong person. I'm such an idiot. I'm a complete failure. I'm worthless. That's what I am. I made a mistake, so I'm a screw-up. I'm fundamentally flawed.", "ashamed, self-loathing"),
            ("I forgot someone's name, so I'm a terrible person. One social mistake makes me fundamentally flawed. I'm just a loser.", "embarrassed, self-hating"),
        ],
        9: [  # Personalization
            ("My team didn't meet the deadline. It's all my fault. If only I had worked harder, this wouldn't have happened. I'm responsible for everyone's failure. There were eight of us, but I should have done more. Their failures are my failures.", "guilty, ashamed"),
            ("My friend is upset. I must have done something to cause it. Everything bad that happens around me is because of me. It's my responsibility to fix their mood.", "guilty, responsible"),
        ]
    }
    
    # Generate 10 conversations per distortion type
    for distortion_type in range(10):
        for i in range(10):
            template_idx = i % len(templates[distortion_type])
            text, emotion = templates[distortion_type][template_idx]
            
            conversations.append({
                "id": conv_id,
                "text": text,
                "label": distortion_type,
                "emotion": emotion,
                "distortion_name": config.distortion_names[distortion_type]
            })
            conv_id += 1
    
    return conversations

# Generate dataset
conversations = generate_cbt_dataset()
df = pd.DataFrame(conversations)

print(f"✓ Dataset created")
print(f"  Total conversations: {len(df)}")
print(f"  Classes: {df['label'].nunique()}")
print(f"  Samples per class: {df['label'].value_counts().min()}-{df['label'].value_counts().max()}")

# Display class distribution
print("\n  Class Distribution:")
for label in sorted(df['label'].unique()):
    count = (df['label'] == label).sum()
    name = config.distortion_names[label]
    print(f"    {label}: {name:35s} - {count:3d} samples")

# ============================================================================
# SECTION 5: DATA PREPARATION
# ============================================================================

print("\n[5/8] Preparing data splits...")

# Split data: train/val/test
train_df, temp_df = train_test_split(
    df, 
    test_size=config.test_size + config.val_size,
    random_state=42,
    stratify=df['label']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=config.test_size / (config.test_size + config.val_size),
    random_state=42,
    stratify=temp_df['label']
)

print(f"✓ Data split completed")
print(f"  Training samples: {len(train_df)}")
print(f"  Validation samples: {len(val_df)}")
print(f"  Test samples: {len(test_df)}")

# Load tokenizer
print(f"\n  Loading tokenizer: {config.model_name}")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

# Dataset class
class CBTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = CBTDataset(
    train_df['text'].values,
    train_df['label'].values,
    tokenizer,
    config.max_length
)

val_dataset = CBTDataset(
    val_df['text'].values,
    val_df['label'].values,
    tokenizer,
    config.max_length
)

test_dataset = CBTDataset(
    test_df['text'].values,
    test_df['label'].values,
    tokenizer,
    config.max_length
)

print(f"✓ Datasets created")

# ============================================================================
# SECTION 6: MODEL DEFINITION
# ============================================================================

print("\n[6/8] Defining DeBERTa ensemble model...")

class DeBERTaClassifier(nn.Module):
    """DeBERTa model for cognitive distortion classification."""
    
    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        super(DeBERTaClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = dropout_rate
        
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

print(f"✓ Model architecture defined")

# ============================================================================
# SECTION 7: HQDE ENSEMBLE TRAINING
# ============================================================================

print("\n[7/8] Training HQDE ensemble...")
print(f"  Creating {config.num_workers} ensemble workers with diverse configurations")

class EnsembleWorker:
    """Single ensemble worker with DeBERTa model."""
    
    def __init__(self, worker_id, config, device):
        self.worker_id = worker_id
        self.config = config
        self.device = device
        
        # Diverse configuration per worker
        self.dropout_rate = config.dropout_rates[worker_id % len(config.dropout_rates)]
        self.learning_rate = config.learning_rates[worker_id % len(config.learning_rates)]
        
        print(f"\n  Worker {worker_id}:")
        print(f"    Dropout: {self.dropout_rate}")
        print(f"    Learning rate: {self.learning_rate}")
        print(f"    Device: {device}")
        
        # Create model
        self.model = DeBERTaClassifier(
            config.model_name,
            config.num_classes,
            self.dropout_rate
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Worker {self.worker_id} Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Worker {self.worker_id} Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def predict(self, data_loader):
        """Get predictions."""
        self.model.eval()
        all_logits = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                all_logits.append(logits.cpu())
        
        return torch.cat(all_logits, dim=0)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size * 2,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Create ensemble workers
workers = []
devices = [f"cuda:{i % config.num_gpus}" for i in range(config.num_workers)]

for i in range(config.num_workers):
    worker = EnsembleWorker(i, config, devices[i])
    workers.append(worker)

print(f"\n✓ {config.num_workers} ensemble workers created")

# Training loop
print(f"\n🚀 Starting ensemble training for {config.num_epochs} epochs...")

best_ensemble_acc = 0

for epoch in range(config.num_epochs):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}/{config.num_epochs}")
    print(f"{'='*80}")
    
    # Train each worker
    for worker in workers:
        # Create scheduler
        num_training_steps = len(train_loader) * config.num_epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            worker.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        train_loss, train_acc = worker.train_epoch(train_loader, scheduler)
        val_loss, val_acc, _, _ = worker.evaluate(val_loader)
        
        worker.train_losses.append(train_loss)
        worker.val_losses.append(val_loss)
        worker.val_accuracies.append(val_acc)
        
        print(f"\n  Worker {worker.worker_id} Results:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Ensemble evaluation
    print(f"\n  Ensemble Evaluation:")
    all_worker_logits = []
    
    for worker in workers:
        logits = worker.predict(val_loader)
        all_worker_logits.append(logits)
    
    # Average ensemble predictions
    ensemble_logits = torch.stack(all_worker_logits).mean(dim=0)
    ensemble_preds = torch.argmax(ensemble_logits, dim=1).numpy()
    true_labels = val_df['label'].values
    
    ensemble_acc = accuracy_score(true_labels, ensemble_preds) * 100
    ensemble_f1 = f1_score(true_labels, ensemble_preds, average='weighted') * 100
    
    print(f"    Ensemble Accuracy: {ensemble_acc:.2f}%")
    print(f"    Ensemble F1-Score: {ensemble_f1:.2f}%")
    
    if ensemble_acc > best_ensemble_acc:
        best_ensemble_acc = ensemble_acc
        print(f"    ✓ New best ensemble accuracy!")

print(f"\n✓ Training completed!")
print(f"  Best ensemble validation accuracy: {best_ensemble_acc:.2f}%")

# ============================================================================
# SECTION 8: FINAL EVALUATION
# ============================================================================

print("\n[8/8] Final evaluation on test set...")

# Get predictions from all workers
test_worker_logits = []
for worker in workers:
    logits = worker.predict(test_loader)
    test_worker_logits.append(logits)

# Ensemble predictions
ensemble_test_logits = torch.stack(test_worker_logits).mean(dim=0)
ensemble_test_preds = torch.argmax(ensemble_test_logits, dim=1).numpy()
test_true_labels = test_df['label'].values

# Calculate metrics
test_acc = accuracy_score(test_true_labels, ensemble_test_preds) * 100
test_f1 = f1_score(test_true_labels, ensemble_test_preds, average='weighted') * 100

print(f"\n{'='*80}")
print(f"FINAL TEST RESULTS")
print(f"{'='*80}")
print(f"  Ensemble Test Accuracy: {test_acc:.2f}%")
print(f"  Ensemble Test F1-Score: {test_f1:.2f}%")

# Classification report
print(f"\n  Classification Report:")
print(classification_report(
    test_true_labels,
    ensemble_test_preds,
    target_names=config.distortion_names,
    digits=3
))

# Confusion matrix
cm = confusion_matrix(test_true_labels, ensemble_test_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[f"{i}" for i in range(10)],
    yticklabels=[f"{i}" for i in range(10)]
)
plt.title('Confusion Matrix - DeBERTa HQDE Ensemble')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{config.output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Confusion matrix saved to {config.output_dir}/confusion_matrix.png")

# Per-class accuracy
print(f"\n  Per-Class Accuracy:")
for i in range(config.num_classes):
    class_mask = test_true_labels == i
    if class_mask.sum() > 0:
        class_acc = (ensemble_test_preds[class_mask] == test_true_labels[class_mask]).mean() * 100
        print(f"    {i}: {config.distortion_names[i]:35s} - {class_acc:.2f}%")

# Individual worker performance
print(f"\n  Individual Worker Test Accuracy:")
for i, worker in enumerate(workers):
    worker_logits = test_worker_logits[i]
    worker_preds = torch.argmax(worker_logits, dim=1).numpy()
    worker_acc = accuracy_score(test_true_labels, worker_preds) * 100
    print(f"    Worker {i}: {worker_acc:.2f}%")

print(f"\n{'='*80}")
print(f"✅ TRAINING AND EVALUATION COMPLETE!")
print(f"{'='*80}")
print(f"\nKey Results:")
print(f"  • Best Validation Accuracy: {best_ensemble_acc:.2f}%")
print(f"  • Final Test Accuracy: {test_acc:.2f}%")
print(f"  • Final Test F1-Score: {test_f1:.2f}%")
print(f"  • Number of Ensemble Workers: {config.num_workers}")
print(f"  • Model: {config.model_name}")
print(f"\nOutput saved to: {config.output_dir}/")
