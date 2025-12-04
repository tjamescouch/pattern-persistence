# llmri/features/sae.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import ActivationDataset


class SparseAutoencoder(nn.Module):
    """
    Basic sparse autoencoder:

      z = ReLU(W_enc x)
      x_hat = W_dec z

    No bias by default, L1 penalty on z.
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_model, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, d_model]
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return z, x_hat

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.encoder(x))
        return z


@dataclass
class SAETrainConfig:
    d_model: int
    hidden_dim: int
    batch_size: int = 256
    lr: float = 1e-3
    l1: float = 1e-3
    epochs: int = 5
    device: str = "cuda"


def train_sae(
    dataset: ActivationDataset,
    cfg: SAETrainConfig,
) -> SparseAutoencoder:
    """
    Train a SparseAutoencoder on the given ActivationDataset.
    """
    device = torch.device(cfg.device)
    model = SparseAutoencoder(d_model=cfg.d_model, hidden_dim=cfg.hidden_dim).to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for x, _token_id in loader:
            x = x.to(device)  # [B, d_model]

            z, x_hat = model(x)

            recon_loss = torch.nn.functional.mse_loss(x_hat, x)
            l1_pen = cfg.l1 * torch.mean(torch.abs(z))

            loss = recon_loss + l1_pen

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item()) * x.size(0)
            count += x.size(0)

        avg_loss = total_loss / max(count, 1)
        print(f"[SAE] epoch {epoch}/{cfg.epochs}  loss={avg_loss:.6f}")

    return model


def summarize_sae_features(
    model: SparseAutoencoder,
    dataset: ActivationDataset,
    tokenizer,
    top_k: int = 20,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Compute simple per-feature stats and top tokens.

    For each feature j:
      - mean_activation: mean over all codes z[:, j]
      - sparsity: fraction of z[:, j] > 0
      - top_tokens: tokens with highest activations for that feature
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)

    # We'll accumulate statistics in a streaming manner.
    n_features = model.encoder.out_features

    sum_act = torch.zeros(n_features, dtype=torch.float64)
    count_nonzero = torch.zeros(n_features, dtype=torch.float64)
    total_count = 0

    # For top tokens, we keep a list of (activation, token_id) per feature,
    # then sort and take top_k at the end. Simple but fine for Phase 2.
    top_lists: List[List[tuple[float, int]]] = [[] for _ in range(n_features)]

    with torch.no_grad():
        for x, token_ids in loader:
            x = x.to(device_obj)  # [B, d_model]
            z = model.encode(x)   # [B, n_features]
            z_cpu = z.cpu()
            token_ids_cpu = token_ids.clone()

            total_count += z_cpu.size(0)

            # Update global stats
            sum_act += z_cpu.sum(dim=0).double()
            count_nonzero += (z_cpu > 0).sum(dim=0).double()

            # For each feature, pick top activations in this batch
            # and record (activation, token_id).
            # To keep it cheap, we only take the top_k per batch.
            B = z_cpu.size(0)
            for j in range(n_features):
                col = z_cpu[:, j]  # [B]
                if torch.all(col == 0):
                    continue
                k = min(top_k, B)
                vals, idx = torch.topk(col, k)
                for v, i in zip(vals.tolist(), idx.tolist()):
                    top_lists[j].append((v, int(token_ids_cpu[i].item())))

    mean_act = (sum_act / max(total_count, 1)).tolist()
    sparsity = (count_nonzero / max(total_count, 1)).tolist()

    # Build feature dicts
    features_info: List[Dict[str, Any]] = []
    for j in range(n_features):
        lst = top_lists[j]
        if lst:
            # sort by activation descending
            lst.sort(key=lambda t: t[0], reverse=True)
            # keep top_k globally
            lst = lst[:top_k]
        # map tokens
        top_tokens_info: List[Dict[str, Any]] = []
        for v, tok_id in lst:
            try:
                tok_str = tokenizer.decode([tok_id]).strip()
            except Exception:
                tok_str = ""
            top_tokens_info.append(
                {
                    "token_id": tok_id,
                    "token": tok_str,
                    "activation": v,
                }
            )

        features_info.append(
            {
                "id": j,
                "mean_activation": mean_act[j],
                "sparsity": sparsity[j],
                "top_tokens": top_tokens_info,
            }
        )

    summary: Dict[str, Any] = {
        "n_features": n_features,
        "features": features_info,
    }
    return summary
