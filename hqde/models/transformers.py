"""
Transformer-based models for text classification tasks.

This module provides transformer architectures optimized for NLP tasks
including CBT (Cognitive Behavioral Therapy) text classification, sentiment
analysis, and other sequence classification problems.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerTextClassifier(nn.Module):
    """
    Transformer-based text classifier for CBT and general NLP tasks.
    
    This model uses a standard transformer encoder architecture with
    positional encoding and a classification head.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        num_classes: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout_rate: float = 0.1,
        max_seq_length: int = 512,
        activation: str = "gelu",
        use_pooling: str = "cls",  # "cls", "mean", or "max"
    ):
        """
        Initialize Transformer Text Classifier.

        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout_rate: Dropout probability
            max_seq_length: Maximum sequence length
            activation: Activation function ("relu" or "gelu")
            use_pooling: Pooling strategy for sequence representation
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_pooling = use_pooling
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
                           (1 for real tokens, 0 for padding)

        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Transpose for positional encoding: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, d_model]
        
        # Create padding mask for transformer (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Pooling strategy
        if self.use_pooling == "cls":
            # Use first token (CLS token)
            pooled = encoded[:, 0, :]
        elif self.use_pooling == "mean":
            # Mean pooling over sequence
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size())
                sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = encoded.mean(dim=1)
        elif self.use_pooling == "max":
            # Max pooling over sequence
            pooled = encoded.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.use_pooling}")
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class LightweightTransformerClassifier(nn.Module):
    """
    Lightweight transformer for faster training and inference.
    
    Optimized for CBT text classification with reduced parameters.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        num_classes: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout_rate: float = 0.1,
        max_seq_length: int = 256,
    ):
        """Initialize lightweight transformer classifier."""
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout_rate)
        
        # Lightweight transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Simple classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Padding mask
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # Encode
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # CLS token pooling
        pooled = encoded[:, 0, :]
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class CBTTransformerClassifier(nn.Module):
    """
    Specialized transformer for Cognitive Behavioral Therapy text classification.
    
    Designed for tasks like:
    - Identifying cognitive distortions
    - Classifying therapy session notes
    - Sentiment analysis in mental health contexts
    - Detecting therapeutic interventions
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        num_classes: int = 10,  # e.g., 10 types of cognitive distortions
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout_rate: float = 0.15,
        max_seq_length: int = 512,
        use_domain_adaptation: bool = True,
    ):
        """
        Initialize CBT-specific transformer.

        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of CBT categories (e.g., cognitive distortions)
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout_rate: Dropout rate (higher for ensemble diversity)
            max_seq_length: Maximum sequence length
            use_domain_adaptation: Add domain-specific layers
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_domain_adaptation = use_domain_adaptation
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Domain adaptation layer (optional)
        if use_domain_adaptation:
            self.domain_adapter = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, d_model)
            )
        
        # Multi-head classification (for multi-label scenarios)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Auxiliary emotion classifier (optional, for CBT context)
        self.emotion_head = nn.Linear(d_model, 7)  # 7 basic emotions
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_emotions: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_emotions: Whether to return emotion predictions

        Returns:
            Logits [batch_size, num_classes] or tuple of (logits, emotions)
        """
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Padding mask
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # Encode
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # CLS token representation
        cls_output = encoded[:, 0, :]
        
        # Domain adaptation
        if self.use_domain_adaptation:
            cls_output = self.domain_adapter(cls_output) + cls_output  # Residual
        
        # Main classification
        logits = self.classifier(cls_output)
        
        # Optional emotion prediction
        if return_emotions:
            emotion_logits = self.emotion_head(cls_output)
            return logits, emotion_logits
        
        return logits


# Convenience aliases
SmallTransformerClassifier = LightweightTransformerClassifier
StandardTransformerClassifier = TransformerTextClassifier
