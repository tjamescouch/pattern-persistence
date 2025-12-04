# llmri/interventions.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .features.sae import SparseAutoencoder


# ----------------------------------------------------------------------
# Config / hook
# ----------------------------------------------------------------------


@dataclass
class SAEFeatureEditConfig:
    """
    Configuration for a single SAE feature intervention.

    scale semantics:
      0.0  -> ablate feature (remove its SAE contribution)
      1.0  -> baseline / no-op
      >1.0 -> boost feature
      <0.0 -> invert feature
    """
    layer_index: int
    feature_id: int
    scale: float = 0.0
    device: str = "cuda"


class SAEFeatureIntervention:
    """
    Forward hook that edits a single SAE feature at a specific layer.

    We assume a standard HF layout:
      - LLaMA-style: model.model.layers[layer_index].mlp
      - GPT-style:   model.transformer.h[layer_index].mlp

    The SAE is used to define a *direction* in activation space:
      h_sae       = D E h
      h_sae_edit  = D (S_f(scale) E h)   # scale one latent coord
      delta       = h_sae_edit - h_sae
      h'          = h + delta

    So:
      - scale = 1  => h' = h  (delta = 0)
      - scale = 0  => remove this feature's SAE contribution
      - others     => push along that SAE-defined direction
    """

    def __init__(
        self,
        model: PreTrainedModel,
        sae: SparseAutoencoder,
        cfg: SAEFeatureEditConfig,
    ) -> None:
        self.model = model
        self.sae = sae
        self.cfg = cfg
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

        self.device = torch.device(cfg.device)
        self.sae.to(self.device)
        self.sae.eval()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def install(self) -> None:
        module = self._get_target_module()
        hook = self._make_hook()
        self._handle = module.register_forward_hook(hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_target_module(self) -> nn.Module:
        m = self.model
        i = self.cfg.layer_index

        # LLaMA-style: model.model.layers[i].mlp
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            block = m.model.layers[i]
            if hasattr(block, "mlp"):
                return block.mlp
            raise RuntimeError(f"Layer {i} has no 'mlp' attribute (LLaMA-style).")

        # GPT-style: model.transformer.h[i].mlp
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            block = m.transformer.h[i]
            if hasattr(block, "mlp"):
                return block.mlp
            raise RuntimeError(f"Layer {i} has no 'mlp' attribute (GPT-style).")

        raise RuntimeError(
            "Unsupported model architecture for SAEFeatureIntervention: "
            "expected LLaMA-style (model.model.layers) or GPT-style (transformer.h)."
        )

    def _make_hook(self):
        sae = self.sae
        feature_id = int(self.cfg.feature_id)
        scale = float(self.cfg.scale)
        device = self.device

        # If scale is effectively 1, we can skip all SAE work â€“ delta = 0.
        if abs(scale - 1.0) < 1e-6:
            def identity_hook(_module, _inputs, output):
                return output
            return identity_hook

        def edit_tensor(h: torch.Tensor) -> torch.Tensor:
            # h: [..., d_model]
            if h.dim() < 2:
                return h

            *prefix, d = h.shape
            flat = h.reshape(-1, d).to(device)

            with torch.no_grad():
                # Encode once
                z = sae.encode(flat)  # [N, F]

                # Out-of-range feature id => true no-op
                if feature_id < 0 or feature_id >= z.size(1):
                    return h

                # Baseline SAE reconstruction
                h_sae = sae.decoder(z)  # [N, d]

                # Edited SAE reconstruction (scale one latent coordinate)
                z_edit = z.clone()
                z_edit[:, feature_id] = z_edit[:, feature_id] * scale
                h_sae_edit = sae.decoder(z_edit)

                # Inject only the delta: keep the original h as the base
                delta = h_sae_edit - h_sae
                edited_flat = flat + delta

            edited = edited_flat.reshape(*prefix, d)
            return edited

        def hook(_module, _inputs, output):
            # Typical case: tensor output [B, S, D]
            if isinstance(output, torch.Tensor):
                return edit_tensor(output)

            # Some blocks return (hidden_states, ...) tuples
            if isinstance(output, (tuple, list)) and len(output) > 0:
                first = output[0]
                if isinstance(first, torch.Tensor):
                    edited_first = edit_tensor(first)
                    if isinstance(output, tuple):
                        return (edited_first,) + output[1:]
                    else:
                        return [edited_first, *output[1:]]
            return output

        return hook


# ----------------------------------------------------------------------
# Generation + debugging
# ----------------------------------------------------------------------


def _generate_with_scores(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run greedy generation and return (sequences, first_step_logits).
    """
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = out.sequences          # [B, T_total]
    scores = out.scores                # list len=max_new_tokens, each [B, vocab]
    first_logits = scores[0][0]        # [vocab]
    return sequences, first_logits


def _topk_prob_debug(
    base_logits: torch.Tensor,
    edit_logits: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Build a small table of top-k probs for the first generated token,
    using the union of baseline and edited top-k indices.
    """
    base_probs = F.softmax(base_logits, dim=0)
    edit_probs = F.softmax(edit_logits, dim=0)

    k = min(k, base_probs.shape[0])

    base_topk = torch.topk(base_probs, k)
    edit_topk = torch.topk(edit_probs, k)
    idx = torch.unique(torch.cat([base_topk.indices, edit_topk.indices]))

    entries: List[Dict[str, Any]] = []
    for i in idx:
        i_int = int(i)
        p0 = float(base_probs[i_int])
        p1 = float(edit_probs[i_int])
        try:
            tok = tokenizer.decode([i_int])
        except Exception:
            tok = f"<id:{i_int}>"
        entries.append(
            {
                "token_id": i_int,
                "token": tok,
                "p_baseline": p0,
                "p_edited": p1,
                "delta": p1 - p0,
            }
        )

    entries.sort(key=lambda e: e["p_baseline"], reverse=True)
    return entries


def run_sae_feature_intervention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sae: SparseAutoencoder,
    layer_index: int,
    feature_id: int,
    scale: float,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 40,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Run the model on a prompt with and without a SAE feature intervention.

    Returns:
      (baseline_text, edited_text, debug_metrics)
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    sae.to(dev)
    sae.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(dev)

    # Baseline
    with torch.no_grad():
        base_seqs, base_first_logits = _generate_with_scores(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
    baseline_text = tokenizer.decode(base_seqs[0], skip_special_tokens=True)

    # Edited
    cfg = SAEFeatureEditConfig(
        layer_index=layer_index,
        feature_id=feature_id,
        scale=scale,
        device=device,
    )
    intervention = SAEFeatureIntervention(model, sae, cfg)
    intervention.install()
    try:
        with torch.no_grad():
            edit_seqs, edit_first_logits = _generate_with_scores(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
            )
    finally:
        intervention.remove()

    edited_text = tokenizer.decode(edit_seqs[0], skip_special_tokens=True)

    # Metrics on first generated token
    diff = edit_first_logits - base_first_logits
    l2 = diff.norm().item()
    cos = F.cosine_similarity(base_first_logits, edit_first_logits, dim=0).item()

    base_top = int(base_first_logits.argmax())
    edit_top = int(edit_first_logits.argmax())

    debug: Dict[str, Any] = {
        "applied_feature_id": int(feature_id),
        "applied_scale": float(scale),
        "first_logit_l2_diff": float(l2),
        "first_logit_cosine": float(cos),
        "first_top_token_baseline": base_top,
        "first_top_token_edited": edit_top,
        "first_top_token_changed": bool(base_top != edit_top),
    }

    debug["topk_probs"] = _topk_prob_debug(
        base_first_logits,
        edit_first_logits,
        tokenizer=tokenizer,
        k=10,
    )

    return baseline_text, edited_text, debug