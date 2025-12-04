# llmri/utils/__init__.py

"""
Utility subpackage for llmri.

This currently includes:

- hf_adapter: Hugging Face–specific model structure inspection.
- io: helpers for saving/loading activation traces and JSON summaries.
"""

from .hf_adapter import HFModelStructure, HFBlockDesc, infer_model_structure
from .io import (
    save_activation_trace,
    load_activation_trace,
    save_json,
    load_json,
)

__all__ = [
    "HFModelStructure",
    "HFBlockDesc",
    "infer_model_structure",
    "save_activation_trace",
    "load_activation_trace",
    "save_json",
    "load_json",
]
