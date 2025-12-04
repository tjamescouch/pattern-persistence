# llmri/cli/sweep.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from llmri.core import load_model
from llmri.features.sae import SparseAutoencoder
from llmri.interventions import run_sae_feature_intervention


def _parse_feature_ids(s: str) -> List[int]:
    # allow "1,2,3" or "1 2 3"
    if "," in s:
        parts = s.split(",")
    else:
        parts = s.split()
    return [int(p) for p in parts if p.strip()]


def _parse_scales(s: str) -> List[float]:
    if "," in s:
        parts = s.split(",")
    else:
        parts = s.split()
    return [float(p) for p in parts if p.strip()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep SAE feature interventions over prompts/features/scales."
    )

    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model id or local path (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu, cuda, mps, etc.).",
    )
    p.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index where the SAE is attached.",
    )
    p.add_argument(
        "--sae-checkpoint",
        type=str,
        required=True,
        help="Path to SAE checkpoint (.pt) trained for this layer/component.",
    )
    p.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Text file with one prompt per line.",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap on number of prompts to use.",
    )
    p.add_argument(
        "--feature-ids",
        type=_parse_feature_ids,
        required=True,
        help='Feature ids to sweep, e.g. "149 393 712" or "149,393,712".',
    )
    p.add_argument(
        "--scales",
        type=_parse_scales,
        required=True,
        help='Scales to sweep, e.g. "0.0 2.0 -2.0" or "0.0,2.0,-2.0".',
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Max new tokens to generate for each sample (default: 40).",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL file path.",
    )
    return p


def load_sae(ckpt_path: Path, device: str) -> SparseAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"]
    enc_weight = state_dict["encoder.weight"]
    hidden_dim, d_model = enc_weight.shape

    sae = SparseAutoencoder(d_model=d_model, hidden_dim=hidden_dim)
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()
    return sae


def load_prompts(path: Path, max_prompts: int | None = None) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
            if max_prompts is not None and len(lines) >= max_prompts:
                break
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    device = args.device
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] Loading model {args.model} on device={device}")
    lm = load_model(args.model, device=device)
    model = lm.model
    tokenizer = lm.tokenizer

    print(f"[sweep] Loading SAE from {args.sae_checkpoint}")
    sae = load_sae(Path(args.sae_checkpoint), device=device)

    prompts = load_prompts(Path(args.prompts_file), max_prompts=args.max_prompts)
    print(f"[sweep] Using {len(prompts)} prompts from {args.prompts_file}")
    print(f"[sweep] Feature ids: {args.feature_ids}")
    print(f"[sweep] Scales: {args.scales}")
    print(f"[sweep] Writing JSONL to {out_path}")

    total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for pi, prompt in enumerate(prompts):
            for fid in args.feature_ids:
                for scale in args.scales:
                    baseline, edited, dbg = run_sae_feature_intervention(
                        model=model,
                        tokenizer=tokenizer,
                        sae=sae,
                        layer_index=args.layer,
                        feature_id=fid,
                        scale=scale,
                        prompt=prompt,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                    )

                    rec = {
                        "prompt_index": pi,
                        "prompt": prompt,
                        "feature_id": fid,
                        "scale": scale,
                        "layer_index": args.layer,
                        "model_id": args.model,
                        "baseline": baseline,
                        "edited": edited,
                        "debug": dbg,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    total += 1

                    if total % 10 == 0:
                        print(f"[sweep] Wrote {total} samples...", flush=True)

    print(f"[sweep] Done. Total samples: {total}")


if __name__ == "__main__":
    main()