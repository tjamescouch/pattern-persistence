# llmri/stats.py

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .core import ModelInfo
from .trace import ActivationTrace


def summarize_weights(model: nn.Module, info: ModelInfo) -> Dict[str, Any]:
    """
    Very basic Phase 1 summary: per-layer Frobenius norms of selected components.
    """
    res: Dict[str, Any] = {"layers": {}}

    with torch.no_grad():
        for layer in info.layer_infos:
            layer_summary: Dict[str, Any] = {}
            for comp in layer.components:
                # Resolve param
                module = model
                parts = comp.name.split(".")
                for p in parts[:-1]:
                    module = getattr(module, p)
                weight = getattr(module, parts[-1])
                norm = weight.norm().item()
                layer_summary[comp.component] = {"weight_norm": norm}
            res["layers"][str(layer.index)] = layer_summary

    return res


def summarize_activations(trace: ActivationTrace) -> Dict[str, Any]:
    """
    Basic summary per activation slice: mean and std of activations.
    """
    res: Dict[str, Any] = {"slices": {}}
    for key, sl in trace.slices.items():
        t = sl.tensor
        res["slices"][key] = {
            "layer_index": sl.layer_index,
            "component": sl.component,
            "mean": t.mean().item(),
            "std": t.std().item(),
        }
    return res