# llmri/features/datasets.py

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from ..utils.io import load_activation_trace


class ActivationDataset(Dataset):
    """
    Dataset of per-token activations for a specific (layer, component).

    Each item is:
      (activation: Tensor[d_model], token_id: int)

    For Phase 2 we eagerly load & flatten all traces into memory for simplicity.
    """

    def __init__(
        self,
        trace_paths: Sequence[str | Path],
        layer: int,
        component: str,
        max_tokens: int | None = None,
    ) -> None:
        self.layer = layer
        self.component = component
        self._activations: torch.Tensor
        self._token_ids: torch.Tensor

        acts_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []

        key = f"{layer}:{component}"

        for path in trace_paths:
            trace = load_activation_trace(path)
            if key not in trace.slices:
                raise KeyError(
                    f"Trace {path} does not contain slice '{key}'. "
                    f"Available slices: {list(trace.slices.keys())}"
                )
            sl = trace.slices[key]
            # sl.tensor: [batch, seq, dim]
            # sl.token_ids: [batch, seq]
            t = sl.tensor
            tokens = sl.token_ids

            if t.dim() != 3:
                raise ValueError(f"Expected activations [B, S, D], got {tuple(t.shape)} for {path}")

            b, s, d = t.shape
            acts_flat = t.reshape(b * s, d)
            tokens_flat = tokens.reshape(b * s)

            acts_list.append(acts_flat)
            token_list.append(tokens_flat)

        if not acts_list:
            raise ValueError("No activations loaded for dataset.")

        activations = torch.cat(acts_list, dim=0)  # [N, d_model]
        token_ids = torch.cat(token_list, dim=0)   # [N]

        if max_tokens is not None and activations.size(0) > max_tokens:
            # Randomly subsample to max_tokens
            idx = torch.randperm(activations.size(0))[:max_tokens]
            activations = activations[idx]
            token_ids = token_ids[idx]

        self._activations = activations
        self._token_ids = token_ids

    @property
    def d_model(self) -> int:
        return int(self._activations.size(1))

    def __len__(self) -> int:
        return int(self._activations.size(0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self._activations[idx]   # [d_model]
        token_id = int(self._token_ids[idx].item())
        return x, token_id
