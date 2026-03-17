"""
Data utilities for HQDE framework.

This module provides practical transform and dataloader helpers for image
classification workloads, including large-dataset settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass
class DataLoaderConfig:
    """Configuration for building efficient PyTorch dataloaders."""

    batch_size: int = 64
    num_workers: Optional[int] = None
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: Optional[bool] = None
    distributed: bool = False


class DataLoader:
    """Factory wrapper for building optimized PyTorch dataloaders."""

    @staticmethod
    def recommended_num_workers(max_workers: int = 8) -> int:
        """Return a conservative worker count for image pipelines."""
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count // 2, max_workers))

    @staticmethod
    def seed_worker(worker_id: int):
        """Seed dataloader workers for reproducible augmentation."""
        del worker_id
        seed = torch.initial_seed() % (2**32)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def create(
        dataset,
        batch_size: int = 64,
        is_training: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        distributed: bool = False,
        drop_last: Optional[bool] = None,
        shuffle: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> TorchDataLoader:
        """
        Create a dataloader configured for image classification workloads.

        When `distributed=True`, a `DistributedSampler` is created and `shuffle`
        is disabled on the dataloader itself.
        """

        num_workers = (
            DataLoader.recommended_num_workers()
            if num_workers is None
            else max(int(num_workers), 0)
        )
        drop_last = is_training if drop_last is None else drop_last
        shuffle = (is_training and not distributed) if shuffle is None else shuffle

        sampler = None
        if distributed:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError(
                    "Distributed dataloader requested but torch.distributed is not initialized."
                )
            sampler = DistributedSampler(dataset, shuffle=is_training, drop_last=drop_last)
            shuffle = False

        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "worker_init_fn": DataLoader.seed_worker,
            "generator": generator,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor

        return TorchDataLoader(**loader_kwargs)

    @staticmethod
    def create_loader_pair(
        train_dataset,
        val_dataset,
        config: Optional[DataLoaderConfig] = None,
    ) -> Tuple[TorchDataLoader, TorchDataLoader]:
        """Create train/validation loaders from one shared config."""
        config = config or DataLoaderConfig()
        train_loader = DataLoader.create(
            train_dataset,
            batch_size=config.batch_size,
            is_training=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            distributed=config.distributed,
            drop_last=config.drop_last,
        )
        val_loader = DataLoader.create(
            val_dataset,
            batch_size=config.batch_size,
            is_training=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            distributed=config.distributed,
            drop_last=False,
        )
        return train_loader, val_loader


class DataPreprocessor:
    """Build dataset transforms for image classification."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    @staticmethod
    def build_classification_transforms(
        image_size: Union[int, Tuple[int, int]] = 224,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        is_training: bool = True,
        auto_augment: Optional[str] = None,
        random_erasing: float = 0.0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        """Build a standard image-classification transform pipeline."""
        if is_training:
            ops = [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
            ]
            augment = (auto_augment or "").lower()
            if augment == "randaugment":
                ops.append(transforms.RandAugment())
            elif augment == "autoaugment":
                ops.append(transforms.AutoAugment())
            elif augment == "trivialaugmentwide":
                ops.append(transforms.TrivialAugmentWide())

            ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            if random_erasing > 0:
                ops.append(transforms.RandomErasing(p=random_erasing))
            return transforms.Compose(ops)

        if isinstance(image_size, int):
            resize_size = int(image_size * 256 / 224)
        else:
            resize_size = tuple(int(dim * 256 / 224) for dim in image_size)

        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def cifar10_transforms(is_training: bool = True):
        """Convenience transform builder for CIFAR-10 style datasets."""
        if is_training:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(DataPreprocessor.CIFAR10_MEAN, DataPreprocessor.CIFAR10_STD),
                ]
            )

        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(DataPreprocessor.CIFAR10_MEAN, DataPreprocessor.CIFAR10_STD),
            ]
        )
