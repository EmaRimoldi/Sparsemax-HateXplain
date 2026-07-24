"""Sparsemax-supervised attention experiments on HateXplain."""

from .config import ExperimentConfig, load_config
from .sparsemax import sparsemax, sparsemax_loss

__all__ = ["ExperimentConfig", "load_config", "sparsemax", "sparsemax_loss"]
__version__ = "0.1.0"
