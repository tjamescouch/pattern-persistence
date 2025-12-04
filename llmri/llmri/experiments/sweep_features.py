# llmri/experiments/sweep_features.py

"""
Sweep SAE feature interventions on a language model layer and record their effects.

Usage (example):

  python -m llmri.experiments.sweep_features \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --device mps \
    --sae-checkpoint features/llama3_l10_mlp_sae.pt \
    --layer-index 10 \
    --feature-ids 0 1 2 3 4 5 6 7 \
    --scales 0.0 2.0 4.0 6.0 \
    --prompts-file data/prompts_chat.txt \
    --chat-template llama3 \
    --max-new-tokens 64 \
    --topk 10 \
    --out results/llama3_feature_sweep.json

This script:

  1. Loads a base LLM and an SAE for a specific layer.
  2. Precomputes baseline logits + completions per prompt.
  3. For each (feature, scale, prompt) triple:
       - runs an edited forward pass with a hook that
         encodes→scales feature→decodes at the chosen layer,
       - computes metrics vs the baseline logits,
       - generates an edited completion,
       - saves everything into a JSON file consumable by the viewer.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

@dataclass
class SAE:
    """
    Simple linear SAE wrapper: ReLU encoder, linear decoder.

    We assume:
      - W_enc: [n_features, d_model]
      - W_dec: [d_model, n_features]
      - b_enc: [n_features] or None
      - b_dec: [d_model] or None
    """

    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: Optional[torch.Tensor] = None
    b_dec: Optional[torch.Tensor] = None

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[1]

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, d_model]
        returns z: [N, n_features]
        """
        z = x @ self.W_enc.T
        if self.b_enc is not None:
            z = z + self.b_enc
        z = F.relu(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [N, n_features]
        returns x_rec: [N, d_model]
        """
        x = z @ self.W_dec.T
        if self.b_dec is not None:
            x = x + self.b_dec
        return x


def load_sae_checkpoint(path: Path, d_model_expected: Optional[int] = None, device: torch.device = torch.device("cpu")) -> SAE:
    """
    Load SAE weights from a checkpoint.

    We try to be robust to common key/shape conventions:
      - encoder.weight: [n_features, d_model] or [d_model, n_features]
      - decoder.weight: [d_model, n_features] or [n_features, d_model]
      - optional encoder.bias, decoder.bias
    """
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"SAE checkpoint at {path} is not a state dict")

    def find_key(candidates: Sequence[str]) -> Optional[str]:
        for k in candidates:
            if k in ckpt:
                return k
        return None

    enc_k = find_key(["encoder.weight", "W_enc", "sae.encoder.weight", "enc.weight"])
    dec_k = find_key(["decoder.weight", "W_dec", "sae.decoder.weight", "dec.weight"])
    if enc_k is None or dec_k is None:
        raise RuntimeError(
            f"Could not find encoder/decoder weights in SAE checkpoint {path}. "
            f"Available keys: {list(ckpt.keys())[:20]}"
        )

    W_enc_raw = ckpt[enc_k].float()  # [*, *]
    W_dec_raw = ckpt[dec_k].float()

    # Infer d_model and n_features from shapes.
    def orient_enc(W: torch.Tensor, d_model: Optional[int]) -> torch.Tensor:
        if d_model is not None:
            if W.shape[1] == d_model:
                return W  # [n_features, d_model]
            if W.shape[0] == d_model:
                return W.T  # [n_features, d_model]
            raise RuntimeError(f"encoder.weight shape {tuple(W.shape)} incompatible with d_model={d_model}")
        # If d_model not given, pick orientation where second dim > first (heuristic)
        if W.shape[1] >= W.shape[0]:
            return W
        return W.T

    def orient_dec(W: torch.Tensor, d_model: int, n_features: int) -> torch.Tensor:
        # Want [d_model, n_features]
        if W.shape == (d_model, n_features):
            return W
        if W.shape == (n_features, d_model):
            return W.T
        raise RuntimeError(
            f"decoder.weight shape {tuple(W.shape)} incompatible with (d_model={d_model}, n_features={n_features})"
        )

    W_enc = orient_enc(W_enc_raw, d_model_expected)
    d_model = W_enc.shape[1]
    n_features = W_enc.shape[0]
    W_dec = orient_dec(W_dec_raw, d_model, n_features)

    # Biases (optional)
    b_enc_k = find_key(["encoder.bias", "b_enc", "sae.encoder.bias", "enc.bias"])
    b_dec_k = find_key(["decoder.bias", "b_dec", "sae.decoder.bias", "dec.bias"])
    b_enc = ckpt[b_enc_k].float() if b_enc_k is not None else None
    b_dec = ckpt[b_dec_k].float() if b_dec_k is not None else None

    sae = SAE(
        W_enc=W_enc.to(device),
        W_dec=W_dec.to(device),
        b_enc=(b_enc.to(device) if b_enc is not None else None),
        b_dec=(b_dec.to(device) if b_dec is not None else None),
    )
    return sae


# ---------------------------------------------------------------------------
# Model + hook utilities
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


def load_model_and_tokenizer(model_name: str, device: torch.device) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    print(f"[load] model={model_name!r} on device={device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type in ("cuda", "mps") else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def get_decoder_layers(model: PreTrainedModel) -> Sequence[torch.nn.Module]:
    # LLaMA-style: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # type: ignore[no-any-return]
    # GPT-style: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # type: ignore[no-any-return]
    raise RuntimeError("Don't know how to find decoder layers on this model")


def get_mlp_module(model: PreTrainedModel, layer_index: int) -> torch.nn.Module:
    layers = get_decoder_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(f"layer_index {layer_index} out of range (0..{len(layers)-1})")
    layer = layers[layer_index]
    if hasattr(layer, "mlp"):
        return layer.mlp  # type: ignore[no-any-return]
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward  # type: ignore[no-any-return]
    raise RuntimeError(f"Layer {layer_index} has no 'mlp' or 'feed_forward' attribute")


@contextmanager
def feature_intervention_hook(
    model: PreTrainedModel,
    sae: SAE,
    layer_index: int,
    feature_id: int,
    scale: float,
) -> Iterable[None]:
    """
    Register a forward hook on the given layer's MLP that:
      - encodes h -> z (SAE),
      - rescales z[:, feature_id] by 'scale',
      - decodes back to hidden space.

    Used as:

      with feature_intervention_hook(...):
          logits = forward_for_logits(...)
          text = generate_text(...)
    """
    if feature_id < 0 or feature_id >= sae.n_features:
        raise ValueError(f"feature_id {feature_id} not in [0, {sae.n_features})")

    mlp = get_mlp_module(model, layer_index)

    def hook(_module, _inputs, output):
        # output: [batch, seq, d_model]
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Expected tensor output from MLP; got something else")
        h = output
        bsz, seqlen, d = h.shape
        h_flat = h.reshape(-1, d)  # [N, d_model]

        # SAE encode/decode
        z = sae.encode(h_flat)  # [N, n_features]
        z[:, feature_id] = z[:, feature_id] * scale
        h_new = sae.decode(z)   # [N, d_model]

        return h_new.reshape(bsz, seqlen, d)

    handle = mlp.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Tokenization and generation helpers
# ---------------------------------------------------------------------------

def encode_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
    chat_template: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Encode a prompt, optionally using a chat template.

    If chat_template == "llama3", we assume a LLaMA-3-style chat model
    and call tokenizer.apply_chat_template([...], tokenize=True, ...).

    Otherwise we just call tokenizer(prompt, ...).
    """
    if chat_template == "llama3":
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
        )
    # Move tensors to device
    return {k: v.to(device) for k, v in enc.items()}


def forward_logits(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Run a forward pass and return the logits at the final position of the input.

    Returns: logits: [vocab_size]
    """
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits  # [1, seq, vocab]
        last_logits = logits[:, -1, :].squeeze(0)  # [vocab]
        return last_logits


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Generate a completion and return the decoded text (completion only).
    """
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # gen_out: [1, seq_total]
    full = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    # For simplicity, just return the tail part after the prompt tokens
    # (this is approximate when using chat templates, but good enough for inspection).
    completion_ids = gen_out[0][input_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion.strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    l2_diff: float
    cosine_similarity: float
    top_token_changed: bool
    top_token_id_baseline: int
    top_token_id_edited: int
    topk_deltas: List[Dict[str, Any]]


def compute_run_metrics(
    baseline_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    topk: int,
) -> RunMetrics:
    """
    Compare baseline vs edited next-token logits.

    Returns:
      - L2(logits)
      - cosine similarity(logits)
      - whether argmax token id changed
      - list of per-token probability deltas for the union of top-k tokens.
    """
    # Move to CPU float
    b = baseline_logits.detach().float().cpu()
    e = edited_logits.detach().float().cpu()

    # L2 / cosine on logits
    l2 = float(torch.norm(b - e, p=2).item())
    cos = float(F.cosine_similarity(b.unsqueeze(0), e.unsqueeze(0)).item())

    # Probabilities
    p_b = F.softmax(b, dim=-1)
    p_e = F.softmax(e, dim=-1)

    # Top-1 ids
    t_b = int(torch.argmax(p_b).item())
    t_e = int(torch.argmax(p_e).item())
    changed = t_b != t_e

    # Union-of-topk
    k = max(1, topk)
    topk_b = torch.topk(p_b, k=k)
    topk_e = torch.topk(p_e, k=k)
    ids_union = set(topk_b.indices.tolist()) | set(topk_e.indices.tolist())

    deltas: List[Dict[str, Any]] = []
    for tid in sorted(ids_union, key=lambda i: float(p_b[i]), reverse=True):
        pb = float(p_b[tid].item())
        pe = float(p_e[tid].item())
        delta = pe - pb
        token_str = tokenizer.decode([tid])
        deltas.append(
            {
                "token_id": tid,
                "token": token_str,
                "p_base": pb,
                "p_edit": pe,
                "delta": delta,
            }
        )

    return RunMetrics(
        l2_diff=l2,
        cosine_similarity=cos,
        top_token_changed=changed,
        top_token_id_baseline=t_b,
        top_token_id_edited=t_e,
        topk_deltas=deltas,
    )


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep SAE feature interventions over prompts.")
    p.add_argument("--model", type=str, required=True, help="HF model name or path.")
    p.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    p.add_argument("--sae-checkpoint", type=str, required=True, help="Path to SAE checkpoint (.pt).")
    p.add_argument("--layer-index", type=int, required=True, help="Layer index whose MLP is hooked.")
    p.add_argument("--feature-ids", type=int, nargs="+", required=True, help="List of SAE feature ids to sweep.")
    p.add_argument("--scales", type=float, nargs="+", required=True, help="List of feature scales (e.g. 0 2 4 6).")
    p.add_argument("--prompts-file", type=str, required=True, help="File with one prompt per line.")
    p.add_argument("--chat-template", type=str, default="none",
                   help="Chat template mode: 'none' or 'llama3'.")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p.")
    p.add_argument("--topk", type=int, default=10, help="Top-k tokens to include in probability deltas.")
    p.add_argument("--out", type=str, required=True, help="Output JSON file.")
    return p.parse_args()


def load_prompts(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
    if not lines:
        raise RuntimeError(f"No non-empty prompts in {path}")
    return lines


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    torch.set_grad_enabled(False)

    # Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Load SAE
    sae_path = Path(args.sae_checkpoint)
    print(f"[load] SAE from {sae_path}", flush=True)
    sae = load_sae_checkpoint(sae_path, d_model_expected=model.config.hidden_size, device=device)
    print(f"[sae] d_model={sae.d_model}, n_features={sae.n_features}", flush=True)

    # Prompts
    prompts = load_prompts(Path(args.prompts_file))
    print(f"[prompts] Loaded {len(prompts)} prompts from {args.prompts_file}", flush=True)

    # Pre-encode prompts and compute baseline logits + completions
    encoded_inputs: List[Dict[str, torch.Tensor]] = []
    baseline_logits: List[torch.Tensor] = []
    baseline_texts: List[str] = []

    print("[baseline] Computing baseline logits + completions per prompt...", flush=True)
    for i, prompt in enumerate(prompts):
        enc = encode_prompt(tokenizer, prompt, device, chat_template=args.chat_template if args.chat_template != "none" else None)
        encoded_inputs.append(enc)

        # Baseline logits
        logits = forward_logits(model, enc)
        baseline_logits.append(logits)

        # Baseline text
        completion = generate_text(
            model,
            tokenizer,
            enc,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        baseline_texts.append(completion)

        print(f"  [baseline] prompt[{i}]='{prompt[:60]}...' done.", flush=True)

    # Sweep
    feature_ids: List[int] = args.feature_ids
    scales: List[float] = args.scales

    runs: List[Dict[str, Any]] = []

    total = len(prompts) * len(feature_ids) * len(scales)
    print(f"[sweep] Starting sweep over {len(prompts)} prompts, {len(feature_ids)} features, {len(scales)} scales "
          f"(total {total} runs)", flush=True)

    run_idx = 0
    for p_idx, prompt in enumerate(prompts):
        for feat in feature_ids:
            for scale in scales:
                run_idx += 1
                print(f"[sweep] #{run_idx}/{total} prompt={p_idx} feat={feat} scale={scale}", flush=True)

                enc = encoded_inputs[p_idx]

                # Edited logits + text under hook
                with feature_intervention_hook(model, sae, args.layer_index, feat, scale):
                    edited_logits = forward_logits(model, enc)
                    edited_text = generate_text(
                        model,
                        tokenizer,
                        enc,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                metrics = compute_run_metrics(
                    baseline_logits[p_idx],
                    edited_logits,
                    tokenizer,
                    topk=args.topk,
                )

                run_record: Dict[str, Any] = {
                    "prompt": prompt,
                    "feature_id": int(feat),
                    "scale": float(scale),
                    "baseline_text": baseline_texts[p_idx],
                    "edited_text": edited_text,
                    "metrics": {
                        "l2_diff": metrics.l2_diff,
                        "cosine_similarity": metrics.cosine_similarity,
                        "top_token_changed": metrics.top_token_changed,
                        "top_token_id_baseline": metrics.top_token_id_baseline,
                        "top_token_id_edited": metrics.top_token_id_edited,
                        "topk_deltas": metrics.topk_deltas,
                    },
                }
                runs.append(run_record)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model,
        "device": str(device),
        "layer_index": args.layer_index,
        "sae_checkpoint": str(sae_path),
        "feature_ids": [int(f) for f in feature_ids],
        "scales": [float(s) for s in scales],
        "prompts": prompts,
        "runs": runs,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)

    print(f"[done] Wrote sweep results to {out_path}", flush=True)


if __name__ == "__main__":
    main()