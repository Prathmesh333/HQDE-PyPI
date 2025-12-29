"""
Simple test to verify HQDE is doing real training, not mock data.
"""

import torch
import torch.nn as nn
import numpy as np
from hqde import create_hqde_system

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Create synthetic dataset
torch.manual_seed(42)
np.random.seed(42)

# Generate simple linearly separable data
train_data = []
train_labels = []
for i in range(500):
    # Class 0: negative values
    if i < 250:
        x = torch.randn(10) - 2
        train_data.append(x)
        train_labels.append(0)
    # Class 1: positive values
    else:
        x = torch.randn(10) + 2
        train_data.append(x)
        train_labels.append(1)

train_data = torch.stack(train_data)
train_labels = torch.tensor(train_labels)

# Create simple data loader
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("=" * 60)
print("Testing HQDE with Simple Linearly Separable Data")
print("=" * 60)
print(f"Training samples: {len(train_data)}")
print(f"Features: {train_data.shape[1]}")
print(f"Classes: 2")
print()

# Create HQDE system
print("Creating HQDE system with 2 workers...")
hqde_system = create_hqde_system(
    model_class=SimpleModel,
    model_kwargs={},
    num_workers=2
)

# Train
print("\nTraining for 10 epochs...")
print("-" * 60)
training_metrics = hqde_system.train(train_loader, num_epochs=10)
print("-" * 60)

# Test predictions
print("\nTesting predictions on training data...")

# First, let's check the model weights directly
import ray
print(f"Number of workers: {len(hqde_system.ensemble_manager.workers)}")

# Get weights from first worker to see if they've changed
worker_weights = hqde_system.ensemble_manager.workers[0].get_weights.remote()
weights_dict = ray.get(worker_weights)
print(f"Model weights - fc layer mean: {weights_dict['fc.weight'].mean().item():.4f}")
print(f"Model weights - fc layer std: {weights_dict['fc.weight'].std().item():.4f}")

predictions = hqde_system.predict(train_loader)

if predictions.numel() > 0:
    # Calculate accuracy
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == train_labels).float().mean().item()

    print(f"Training Accuracy: {accuracy:.2%}")
    print()

    # Verify it's better than random (50% for binary classification)
    if accuracy > 0.6:
        print("SUCCESS! Model is learning (accuracy > 60%)")
        print(f"   The model achieved {accuracy:.1%} accuracy on a simple task.")
        print("   This confirms REAL training is happening, not mock data!")
    elif accuracy > 0.51:
        print("PARTIAL SUCCESS! Model shows some learning")
        print(f"   Accuracy: {accuracy:.1%} (slightly better than random)")
    else:
        print("ISSUE! Model not learning properly")
        print(f"   Accuracy: {accuracy:.1%} (no better than random)")
else:
    print("ERROR! No predictions received")

# Cleanup
hqde_system.cleanup()
print("\nTest completed!")
