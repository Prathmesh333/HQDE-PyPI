"""Reusable model backbones shipped with HQDE."""

from .vision import SmallImageResNet18

# Import transformer models
try:
    from .transformers import (
        TransformerTextClassifier,
        LightweightTransformerClassifier,
        CBTTransformerClassifier,
        SmallTransformerClassifier,
        StandardTransformerClassifier
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TransformerTextClassifier = None
    LightweightTransformerClassifier = None
    CBTTransformerClassifier = None
    SmallTransformerClassifier = None
    StandardTransformerClassifier = None
    TRANSFORMERS_AVAILABLE = False

__all__ = [
    "SmallImageResNet18",
    "TransformerTextClassifier",
    "LightweightTransformerClassifier",
    "CBTTransformerClassifier",
    "SmallTransformerClassifier",
    "StandardTransformerClassifier",
]
