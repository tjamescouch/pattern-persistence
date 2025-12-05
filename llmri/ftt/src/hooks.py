# -----------------------------------------------------------------------------
# Project: FTT (Fast Tensor Transform)
# Module: Hooks
# Path: llmri/ftt/src/hooks.py
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import sys
import os

# Robust import for 'engine' whether running as script or module
try:
    from .engine import FTT_Writer
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from engine import FTT_Writer

class FTT_Capture:
    """
    Context Manager for capturing LLM activations.
    """
    def __init__(self, model: nn.Module, layer: nn.Module, writer: FTT_Writer):
        self.model = model
        self.layer = layer
        self.writer = writer
        self.hook_handle = None

    def _hook_fn(self, module, input, output):
        # Handle HuggingFace tuple outputs (usually hidden_states is idx 0)
        if isinstance(output, tuple):
            tensor_to_save = output[0]
        else:
            tensor_to_save = output

        # Flatten [Batch, Seq, Dim] -> [Batch*Seq, Dim]
        if tensor_to_save.dim() == 3:
            b, s, d = tensor_to_save.shape
            flat_tensor = tensor_to_save.view(-1, d)
        else:
            flat_tensor = tensor_to_save

        self.writer.push(flat_tensor.detach())

    def __enter__(self):
        self.hook_handle = self.layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle:
            self.hook_handle.remove()