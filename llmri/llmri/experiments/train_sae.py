# llmri/experiments/train_sae.py

"""
Train a simple sparse autoencoder (SAE) on stored activations from a single layer.

Example:

  python -m llmri.experiments.train_sae \
    --activations activations/mistral7b_l10_mlp_shakespeare_4k.pt \
    --layer-index 10 \
    --component mlp \
    --n-features 1024 \
    --epochs 5 \
    --device mps \
    --batch-size 4096 \
    --out features/mistral7b_l10_mlp_sae.pt \
    --features-json features/mistral7b_l10_mlp_sae_features.json

Notes:
  * We expect the activations file to be an ActivationTrace saved by
    `llmri trace activations` (the same format that `llmri stats activations` uses).
  * If the trace contains exactly one slice, we use it.
  * If it contains multiple slices, you can optionally specify --layer-index and --component
    to select the slice (e.g. layer 10, component "mlp").
  * The SAE is a basic linear encoder + ReLU + linear decoder with L2 recon + L1 sparsity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from llmri.utils.io import load_activation_trace


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def resolve_device(device_str: str) -> torch.device:
    s = device_str.lower()
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if s == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    raise RuntimeError(f"Unknown device '{device_str}'")


# ---------------------------------------------------------------------------
# SAE model
# ---------------------------------------------------------------------------


class SAENet(nn.Module):
    """
    Simple sparse autoencoder:
      x ∈ R^d → encoder → ReLU(z) → decoder → x̂ ∈ R^d
    """

    def __init__(self, d_model: int, n_features: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, d_model]
        Returns:
          x_rec: [batch, d_model]
          z:     [batch, n_features]
        """
        z = self.encoder(x)
        z = F.relu(z)
        x_rec = self.decoder(z)
        return x_rec, z


# ---------------------------------------------------------------------------
# Activation loading
# ---------------------------------------------------------------------------


def _select_slice_object(
    trace,
    layer_index: Optional[int],
    component: Optional[str],
) -> Any:
    """
    Internal helper: pick an ActivationSlice object from trace.slices.
    Supports both dict-like and sequence-like traces.

    For dict-like traces, if layer_index and component are provided, we first
    try the key "layer:component" (e.g. "10:mlp"), matching ActivationDataset. :contentReference[oaicite:1]{index=1}
    """
    slices = getattr(trace, "slices", None)
    if slices is None:
        raise RuntimeError(f"ActivationTrace has no 'slices' attribute (type={type(trace)})")

    # Dict-like: typical for our traces.
    if isinstance(slices, dict):
        keys = list(slices.keys())
        if layer_index is None and component is None:
            if len(keys) != 1:
                raise RuntimeError(
                    f"Trace has {len(keys)} slices but no layer/component was specified. "
                    f"Available keys: {keys}"
                )
            return slices[keys[0]]

        # If both layer and component are specified, try the "L:comp" key first.
        if layer_index is not None and component is not None:
            key = f"{layer_index}:{component}"
            if key in slices:
                return slices[key]

        # Fallback: try to match via attributes on the slice objects, if present.
        selected = None
        for sl in slices.values():
            li = getattr(sl, "layer_index", None)
            comp = getattr(sl, "component", None)
            if layer_index is not None and li != layer_index:
                continue
            if component is not None and comp != component:
                continue
            selected = sl
            break

        if selected is None:
            raise RuntimeError(
                f"Could not find slice with layer_index={layer_index} component={component!r}. "
                f"Available keys: {keys}"
            )
        return selected

    # Sequence-like: older traces.
    seq = list(slices)
    if layer_index is None and component is None:
        if len(seq) != 1:
            raise RuntimeError(
                f"Trace has {len(seq)} slices but no layer/component was specified"
            )
        return seq[0]

    selected = None
    for sl in seq:
        li = getattr(sl, "layer_index", None)
        comp = getattr(sl, "component", None)
        if layer_index is not None and li != layer_index:
            continue
        if component is not None and comp != component:
            continue
        selected = sl
        break

    if selected is None:
        raise RuntimeError(
            f"Could not find slice with layer_index={layer_index} component={component!r} "
            f"in sequence-like slices"
        )
    return selected


def pick_slice_from_trace(
    path: Path,
    layer_index: Optional[int] = None,
    component: Optional[str] = None,
) -> torch.Tensor:
    """
    Load an ActivationTrace from `path` and return a tensor of activations [N, d_model]
    for a single (layer, component) slice.

    Handles both:
      * per-token activations: slice.tensor of shape [B, S, D]
      * already-flattened activations: slice.acts / slice.activations / slice.tensor of shape [N, D]
    """
    trace = load_activation_trace(path)
    sl = _select_slice_object(trace, layer_index, component)

    # Extract activations tensor (field name may be 'acts', 'activations', or 'tensor').
    if hasattr(sl, "acts"):
        acts = sl.acts
    elif hasattr(sl, "activations"):
        acts = sl.activations
    elif hasattr(sl, "tensor"):
        acts = sl.tensor
    else:
        raise RuntimeError(
            f"Selected slice has no 'acts', 'activations', or 'tensor' field (type={type(sl)})"
        )

    # Flatten if necessary.
    if acts.ndim == 3:
        b, s, d = acts.shape
        acts = acts.reshape(b * s, d)
    elif acts.ndim != 2:
        raise RuntimeError(
            f"Expected activations of shape [N, d_model] or [B, S, d_model], "
            f"got {tuple(acts.shape)}"
        )

    return acts.cpu().to(torch.float32)  # [N, d_model] on CPU


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    activations: Path
    layer_index: Optional[int]
    component: Optional[str]
    n_features: int
    epochs: int
    batch_size: int
    lr: float
    l1_coeff: float
    max_samples: Optional[int]
    device: torch.device
    out: Path
    features_json: Optional[Path]


def train_sae(cfg: TrainConfig, acts: Optional[torch.Tensor] = None) -> SAENet:
    """
    Train the SAE on the given activations.

    If `acts` is None we reload from disk; normally `main()` preloads and passes it in
    so we only hit the disk once.
    """
    if acts is None:
        print(f"[load] activations from {cfg.activations}", flush=True)
        acts = pick_slice_from_trace(cfg.activations, cfg.layer_index, cfg.component)
    else:
        print("[train] using activations passed from caller", flush=True)

    # Optionally subsample (explicit parameter; default None = use all)
    if cfg.max_samples is not None and acts.shape[0] > cfg.max_samples:
        acts = acts[: cfg.max_samples]
        print(f"[data] truncated to first {cfg.max_samples} samples", flush=True)

    acts = acts.to(torch.float32)

    N, d_model = acts.shape
    print(f"[data] N={N}, d_model={d_model}", flush=True)

    dataset = TensorDataset(acts)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    device = cfg.device
    print(
        f"[train] device={device}, n_features={cfg.n_features}, epochs={cfg.epochs}",
        flush=True,
    )

    model = SAENet(d_model=d_model, n_features=cfg.n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        running_loss = 0.0
        running_recon = 0.0
        running_l1 = 0.0
        total_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_rec, z = model(batch_x)

            recon_loss = F.mse_loss(x_rec, batch_x)
            l1 = z.abs().mean()
            loss = recon_loss + cfg.l1_coeff * l1

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_recon += float(recon_loss.item())
            running_l1 += float(l1.item())
            total_batches += 1

        avg_loss = running_loss / max(1, total_batches)
        avg_recon = running_recon / max(1, total_batches)
        avg_l1 = running_l1 / max(1, total_batches)

        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.6f} "
            f"recon={avg_recon:.6f} l1={avg_l1:.6f}",
            flush=True,
        )

    print("[train] finished training", flush=True)
    return model


def save_sae_checkpoint(model: SAENet, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "encoder.weight": model.encoder.weight.detach().cpu(),
        "encoder.bias": model.encoder.bias.detach().cpu(),
        "decoder.weight": model.decoder.weight.detach().cpu(),
        "decoder.bias": model.decoder.bias.detach().cpu(),
    }
    torch.save(state, out_path)
    print(f"[save] SAE checkpoint written to {out_path}", flush=True)


def compute_feature_stats(
    model: SAENet,
    acts: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Compute mean activation and sparsity per feature over `acts`.
    """
    model.eval()
    acts = acts.to(torch.float32)

    dataset = TensorDataset(acts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    n_features = model.encoder.out_features
    sum_z = torch.zeros(n_features, dtype=torch.float64)
    count = 0
    zero_count = torch.zeros(n_features, dtype=torch.int64)

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            _, z = model(batch_x)  # [B, n_features]
            z_cpu = z.cpu()

            sum_z += z_cpu.sum(dim=0).to(torch.float64)
            zero_count += (z_cpu == 0).sum(dim=0)
            count += z_cpu.shape[0]

    mean_activation = (sum_z / max(1, count)).tolist()
    sparsity = (zero_count.double() / max(1, count)).tolist()

    features = []
    for fid in range(n_features):
        features.append(
            {
                "id": fid,
                "mean_activation": float(mean_activation[fid]),
                "sparsity": float(sparsity[fid]),
                # top_tokens intentionally omitted; can be filled later.
            }
        )

    return {
        "n_features": n_features,
        "features": features,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a sparse autoencoder on layer activations."
    )
    p.add_argument(
        "--activations",
        type=str,
        required=True,
        help="Path to activations .pt file (ActivationTrace).",
    )
    p.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Layer index to select from trace (if multiple slices).",
    )
    p.add_argument(
        "--component",
        type=str,
        default=None,
        help="Component name to select from trace (e.g. 'mlp').",
    )
    p.add_argument(
        "--n-features",
        type=int,
        required=True,
        help="Number of SAE latent features.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for training.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (Adam).",
    )
    p.add_argument(
        "--l1-coeff",
        type=float,
        default=1e-3,
        help="L1 coefficient for sparsity penalty.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples from activations.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on: cpu | cuda | mps.",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for SAE checkpoint (.pt).",
    )
    p.add_argument(
        "--features-json",
        type=str,
        default=None,
        help="Optional JSON file for basic feature stats (mean, sparsity).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    cfg = TrainConfig(
        activations=Path(args.activations),
        layer_index=args.layer_index,
        component=args.component,
        n_features=args.n_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_coeff=args.l1_coeff,
        max_samples=args.max_samples,
        device=device,
        out=Path(args.out),
        features_json=(Path(args.features_json) if args.features_json else None),
    )

    # Preload activations once (for both training and stats).
    acts = pick_slice_from_trace(cfg.activations, cfg.layer_index, cfg.component)
    print(f"[data] activations shape: {tuple(acts.shape)}", flush=True)

    # Maybe truncate if user requested
    if cfg.max_samples is not None and acts.shape[0] > cfg.max_samples:
        acts = acts[: cfg.max_samples]
        print(f"[data] truncated to first {cfg.max_samples} samples", flush=True)

    N, d_model = acts.shape
    print(f"[data] N={N}, d_model={d_model}", flush=True)

    # Train SAE
    model = train_sae(cfg, acts=acts)

    # Save checkpoint
    save_sae_checkpoint(model, cfg.out)

    # Optionally write basic feature stats JSON
    if cfg.features_json is not None:
        print(
            f"[stats] computing feature stats for JSON export -> {cfg.features_json}",
            flush=True,
        )
        stats = compute_feature_stats(model, acts, batch_size=cfg.batch_size, device=device)
        cfg.features_json.parent.mkdir(parents=True, exist_ok=True)
        with cfg.features_json.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)
        print(f"[stats] wrote feature stats JSON to {cfg.features_json}", flush=True)


if __name__ == "__main__":
    main()