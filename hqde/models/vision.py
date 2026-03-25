"""Vision backbones aligned with the benchmark notebooks."""

from __future__ import annotations

import torch.nn as nn
import torchvision


class SmallImageResNet18(nn.Module):
    """ResNet-18 adapted for 32x32 image classification workloads."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.0):
        super().__init__()
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        if dropout_rate > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(model.fc.in_features, num_classes),
            )
        self.model = model

    def forward(self, x):
        return self.model(x)
