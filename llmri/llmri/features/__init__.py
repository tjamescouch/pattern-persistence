# llmri/features/__init__.py

"""
Feature / logic extraction tools for llmri.

Phase 2 focuses on sparse autoencoders (SAEs) for a single layer/component.
"""

from .sae import SparseAutoencoder, train_sae
from .datasets import ActivationDataset

__all__ = [
    "SparseAutoencoder",
    "train_sae",
    "ActivationDataset",
]
