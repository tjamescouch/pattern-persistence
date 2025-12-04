# llmri/trace.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    # Only needed for type checking; avoids circular import at runtime.
    from .core import ModelInfo


@dataclass
class ActivationSlice:
    layer_index: int
    component: str   # e.g. "mlp", "attn"
    tensor: torch.Tensor  # [batch, seq, dim] or [batch, heads, seq, dim_head]
    token_ids: torch.Tensor  # [batch, seq]


@dataclass
class ActivationTrace:
    model_id: str
    tokenizer_id: str
    meta: Dict[str, Any]
    slices: Dict[str, ActivationSlice]  # key: f"{layer_index}:{component}"


class ActivationTracer:
    """
    Attach forward hooks to selected submodules of a model and capture activations
    as we run text through it.

    Phase 1: only supports llama-like HF models (model.model.layers list).
    Components understood: "mlp", "attn".
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        info: "ModelInfo",
        layers: Sequence[int],
        components: Sequence[str],
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.layers = list(layers)
        self.components = list(components)
        self.device = device

        self._hooks: List[Any] = []
        self._buffer: Dict[str, List[torch.Tensor]] = {}

    # --------------------------------------------------------------------- #
    # Hook management
    # --------------------------------------------------------------------- #

    def __enter__(self) -> "ActivationTracer":
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._remove_hooks()

    def _register_hooks(self) -> None:
        """
        Register hooks on the modules corresponding to requested (layer, component) pairs.
        For Phase 1, we hook:
          - block.mlp        -> component "mlp"
          - block.self_attn  -> component "attn"
        """
        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            raise RuntimeError(
                "ActivationTracer currently supports llama-like models only in Phase 1 "
                "(expected model.model.layers)."
            )

        blocks = list(self.model.model.layers)

        for layer_idx in self.layers:
            if layer_idx < 0 or layer_idx >= len(blocks):
                raise IndexError(f"Requested layer {layer_idx}, but model has {len(blocks)} layers.")

            block = blocks[layer_idx]

            if "mlp" in self.components and hasattr(block, "mlp"):
                h = block.mlp.register_forward_hook(
                    self._make_hook(layer_idx, "mlp")
                )
                self._hooks.append(h)

            if "attn" in self.components and hasattr(block, "self_attn"):
                h = block.self_attn.register_forward_hook(
                    self._make_hook(layer_idx, "attn")
                )
                self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, layer_idx: int, component: str):
        key = f"{layer_idx}:{component}"

        def hook(module, inputs, output):
            # output typically [batch, seq, dim] or [batch, heads, seq, dim_head]
            if key not in self._buffer:
                self._buffer[key] = []
            self._buffer[key].append(output.detach().to("cpu"))

        return hook

    # --------------------------------------------------------------------- #
    # Public run methods
    # --------------------------------------------------------------------- #

    def run_on_text(self, text: str) -> ActivationTrace:
        return self.run_on_batch([text])

    def run_on_batch(self, texts: Sequence[str]) -> ActivationTrace:
        self._buffer.clear()

        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad(), self:
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        token_ids = input_ids.to("cpu")

        slices: Dict[str, ActivationSlice] = {}
        for key, chunk_list in self._buffer.items():
            tensor = torch.cat(chunk_list, dim=0)  # [batch, seq, dim] etc.
            layer_str, component = key.split(":")
            layer_idx = int(layer_str)

            slices[key] = ActivationSlice(
                layer_index=layer_idx,
                component=component,
                tensor=tensor,
                token_ids=token_ids,
            )

        meta = {
            "num_texts": len(texts),
        }

        trace = ActivationTrace(
            model_id=self.info.model_id,
            tokenizer_id=self.info.model_id,  # HF: same as model; can decouple later
            meta=meta,
            slices=slices,
        )
        return trace

    def run_on_dataset(self, dataset, text_field: str = "text", max_samples: int | None = None) -> ActivationTrace:
        """
        Simple dataset runner for Phase 1: collects all samples into one batch.
        """
        texts: List[str] = []
        for i, item in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
            texts.append(item[text_field])

        return self.run_on_batch(texts)

    def run_and_save(self, texts: Sequence[str], out_path: str) -> ActivationTrace:
        from .utils.io import save_activation_trace  # local import to avoid cycles

        trace = self.run_on_batch(texts)
        save_activation_trace(trace, out_path)
        return trace