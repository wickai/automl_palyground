"""Model registry"""
from .classification import CLASSIFICATION_MODELS, ALGO_DISPLAY_NAMES as CLF_ALGO_NAMES
from .regression import REGRESSION_MODELS

__all__ = [
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS", 
    "CLF_ALGO_NAMES"
]
