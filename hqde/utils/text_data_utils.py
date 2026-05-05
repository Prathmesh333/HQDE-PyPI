"""
Text data utilities for transformer-based models.

This module provides text preprocessing, tokenization, and data loading
utilities for NLP tasks including CBT text classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
import re
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np


@dataclass
class TextDataConfig:
    """Configuration for text data processing."""
    
    max_seq_length: int = 512
    vocab_size: int = 30000
    batch_size: int = 32
    num_workers: int = 4
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    unk_token_id: int = 3
    mask_token_id: int = 4


class SimpleTokenizer:
    """
    Simple word-level tokenizer for text classification.
    
    For production use, consider using HuggingFace tokenizers or
    torchtext tokenizers for better performance.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        max_seq_length: int = 512,
        lowercase: bool = True,
        remove_punctuation: bool = False
    ):
        """Initialize tokenizer."""
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        
        self.special_tokens = [
            self.pad_token,
            self.cls_token,
            self.sep_token,
            self.unk_token,
            self.mask_token
        ]
        
        # Vocabulary
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._build_special_tokens()

    def _build_special_tokens(self):
        """Build special token mappings."""
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        Build vocabulary from texts.

        Args:
            texts: List of text strings
            min_freq: Minimum frequency for a word to be included
        """
        # Count word frequencies
        word_freq: Dict[str, int] = {}
        
        for text in texts:
            tokens = self._tokenize_text(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build vocabulary (reserve space for special tokens)
        current_idx = len(self.special_tokens)
        for word, freq in sorted_words:
            if freq >= min_freq and current_idx < self.vocab_size:
                if word not in self.word2idx:
                    self.word2idx[word] = current_idx
                    self.idx2word[current_idx] = word
                    current_idx += 1
        
        print(f"Built vocabulary with {len(self.word2idx)} tokens")

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize a single text."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Simple whitespace tokenization
        tokens = text.split()
        
        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Add [CLS] and [SEP] tokens
            padding: Pad to max_seq_length
            truncation: Truncate to max_seq_length
            return_attention_mask: Return attention mask

        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'
        """
        tokens = self._tokenize_text(text)
        
        # Convert to IDs
        token_ids = [
            self.word2idx.get(token, self.word2idx[self.unk_token])
            for token in tokens
        ]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.word2idx[self.cls_token]] + token_ids + [self.word2idx[self.sep_token]]
        
        # Truncation
        if truncation and len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Padding
        if padding:
            padding_length = self.max_seq_length - len(token_ids)
            token_ids = token_ids + [self.word2idx[self.pad_token]] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long)
        }
        
        if return_attention_mask:
            result['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        
        return result

    def encode_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of texts."""
        encoded_list = [self.encode(text, **kwargs) for text in texts]
        
        # Stack tensors
        result = {
            'input_ids': torch.stack([enc['input_ids'] for enc in encoded_list]),
        }
        
        if 'attention_mask' in encoded_list[0]:
            result['attention_mask'] = torch.stack([enc['attention_mask'] for enc in encoded_list])
        
        return result

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = [
            self.idx2word.get(idx, self.unk_token)
            for idx in token_ids
            if idx != self.word2idx[self.pad_token]
        ]
        
        # Remove special tokens
        tokens = [
            token for token in tokens
            if token not in self.special_tokens
        ]
        
        return ' '.join(tokens)


class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: SimpleTokenizer,
        max_seq_length: int = 512
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text strings
            labels: List of integer labels
            tokenizer: Tokenizer instance
            max_seq_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        # Add label
        encoded['labels'] = torch.tensor(label, dtype=torch.long)
        
        return encoded


class CBTDataset(Dataset):
    """
    Dataset for Cognitive Behavioral Therapy text classification.
    
    Supports classification of:
    - Cognitive distortions
    - Therapy session notes
    - Patient sentiment
    - Therapeutic interventions
    """

    # Common CBT cognitive distortions
    COGNITIVE_DISTORTIONS = [
        "all_or_nothing",
        "overgeneralization",
        "mental_filter",
        "disqualifying_positive",
        "jumping_to_conclusions",
        "magnification",
        "emotional_reasoning",
        "should_statements",
        "labeling",
        "personalization"
    ]

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: SimpleTokenizer,
        max_seq_length: int = 512,
        include_emotions: bool = False,
        emotion_labels: Optional[List[int]] = None
    ):
        """
        Initialize CBT dataset.

        Args:
            texts: List of therapy-related texts
            labels: List of cognitive distortion labels
            tokenizer: Tokenizer instance
            max_seq_length: Maximum sequence length
            include_emotions: Whether to include emotion labels
            emotion_labels: Optional emotion labels (if include_emotions=True)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.include_emotions = include_emotions
        self.emotion_labels = emotion_labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        # Add main label
        encoded['labels'] = torch.tensor(label, dtype=torch.long)
        
        # Add emotion label if available
        if self.include_emotions and self.emotion_labels is not None:
            encoded['emotion_labels'] = torch.tensor(
                self.emotion_labels[idx],
                dtype=torch.long
            )
        
        return encoded


class TextDataLoader:
    """Factory for creating text data loaders."""

    @staticmethod
    def create_text_loader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> TorchDataLoader:
        """
        Create a data loader for text classification.

        Args:
            dataset: Text dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer

        Returns:
            PyTorch DataLoader
        """
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=TextDataLoader.collate_fn
        )

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        # Stack all tensors
        collated = {}
        
        for key in batch[0].keys():
            collated[key] = torch.stack([item[key] for item in batch])
        
        return collated


def create_cbt_sample_data(
    num_samples: int = 1000,
    num_classes: int = 10,
    vocab_size: int = 5000,
    max_seq_length: int = 128
) -> Tuple[List[str], List[int]]:
    """
    Create sample CBT data for testing.

    Args:
        num_samples: Number of samples to generate
        num_classes: Number of cognitive distortion classes
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (texts, labels)
    """
    # Sample CBT-related phrases
    cbt_phrases = [
        "I always fail at everything",
        "Nobody likes me",
        "This is a disaster",
        "I should be perfect",
        "Everyone thinks I'm stupid",
        "I can't do anything right",
        "Things will never get better",
        "I'm a complete failure",
        "This proves I'm worthless",
        "I must be the worst person"
    ]
    
    texts = []
    labels = []
    
    for _ in range(num_samples):
        # Generate random text from phrases
        num_phrases = np.random.randint(1, 5)
        text = ' '.join(np.random.choice(cbt_phrases, num_phrases))
        
        # Random label
        label = np.random.randint(0, num_classes)
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def preprocess_cbt_text(text: str) -> str:
    """
    Preprocess CBT-related text.

    Args:
        text: Raw text

    Returns:
        Preprocessed text
    """
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep only alphanumeric and basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?\'-]', '', text)
    
    return text
