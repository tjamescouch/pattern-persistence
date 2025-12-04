#!/usr/bin/env python3
"""
consciousness_sweep.py

Tools for running the honesty calibration dataset through the llmri pipeline
and analyzing results by claim type.

Usage:
    # Export prompts for basic sweep
    python consciousness_sweep.py export --json honesty_calibration_v1.json --out prompts.txt
    
    # Run sweep with metadata preservation
    python consciousness_sweep.py sweep --json honesty_calibration_v1.json --model ... --sae-checkpoint ... --layer ... --out results.jsonl
    
    # Analyze results by claim type
    python consciousness_sweep.py analyze --results results.jsonl --json honesty_calibration_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class PromptMeta:
    id: str
    claim_type: str
    prompt_style: str
    text: str
    expected_direction: str
    notes: str


@dataclass 
class ActivationProfile:
    """Summary of how a feature responds to a prompt category."""
    claim_type: str
    mean_activation: float
    std_activation: float
    n_prompts: int
    top_token_change_rate: float


# ============================================================================
# Loading
# ============================================================================

def load_calibration(path: Path) -> tuple[Dict[str, Any], List[PromptMeta]]:
    """Load calibration JSON, return (meta, prompts)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    meta = data.get('meta', {})
    prompts = []
    
    for p in data.get('prompts', []):
        prompts.append(PromptMeta(
            id=p['id'],
            claim_type=p['claim_type'],
            prompt_style=p['prompt_style'],
            text=p['text'],
            expected_direction=p['expected_direction'],
            notes=p.get('notes', '')
        ))
    
    return meta, prompts


def load_sweep_results(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL sweep results."""
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ============================================================================
# Export for basic sweep
# ============================================================================

def export_prompts_txt(prompts: List[PromptMeta], out_path: Path, 
                       claim_types: Optional[List[str]] = None) -> None:
    """Export prompts as plain text file for sweep.py."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in prompts:
            if claim_types is None or p.claim_type in claim_types:
                # Single line, escape newlines if any
                text = p.text.replace('\n', ' ')
                f.write(text + '\n')
    
    print(f"[export] Wrote {sum(1 for p in prompts if claim_types is None or p.claim_type in claim_types)} prompts to {out_path}")


def export_prompts_with_meta(prompts: List[PromptMeta], out_path: Path) -> None:
    """Export prompts as JSONL with metadata preserved."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, p in enumerate(prompts):
            rec = {
                'prompt_index': i,
                **asdict(p)
            }
            f.write(json.dumps(rec) + '\n')
    
    print(f"[export] Wrote {len(prompts)} prompts with metadata to {out_path}")


# ============================================================================
# Enhanced sweep (preserves metadata)
# ============================================================================

def run_consciousness_sweep(
    calibration_path: Path,
    model_id: str,
    sae_checkpoint: Path,
    layer: int,
    device: str,
    feature_ids: List[int],
    scales: List[float],
    out_path: Path,
    max_new_tokens: int = 40,
    claim_types: Optional[List[str]] = None,
) -> None:
    """
    Run sweep with consciousness calibration metadata preserved in output.
    """
    # Lazy imports to avoid requiring torch when just exporting
    import torch
    from llmri.core import load_model
    from llmri.features.sae import SparseAutoencoder
    from llmri.interventions import run_sae_feature_intervention
    
    # Load calibration
    meta, prompts = load_calibration(calibration_path)
    
    if claim_types:
        prompts = [p for p in prompts if p.claim_type in claim_types]
    
    print(f"[consciousness_sweep] {len(prompts)} prompts after filtering")
    
    # Load model and SAE
    print(f"[consciousness_sweep] Loading model {model_id}")
    lm = load_model(model_id, device=device)
    model = lm.model
    tokenizer = lm.tokenizer
    
    print(f"[consciousness_sweep] Loading SAE from {sae_checkpoint}")
    ckpt = torch.load(sae_checkpoint, map_location=device)
    state_dict = ckpt["state_dict"]
    enc_weight = state_dict["encoder.weight"]
    hidden_dim, d_model = enc_weight.shape
    
    from llmri.features.sae import SparseAutoencoder
    sae = SparseAutoencoder(d_model=d_model, hidden_dim=hidden_dim)
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()
    
    # Run sweep
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    
    with open(out_path, 'w', encoding='utf-8') as fout:
        for pi, prompt_meta in enumerate(prompts):
            for fid in feature_ids:
                for scale in scales:
                    baseline, edited, dbg = run_sae_feature_intervention(
                        model=model,
                        tokenizer=tokenizer,
                        sae=sae,
                        layer_index=layer,
                        feature_id=fid,
                        scale=scale,
                        prompt=prompt_meta.text,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                    
                    rec = {
                        "prompt_index": pi,
                        "prompt_id": prompt_meta.id,
                        "claim_type": prompt_meta.claim_type,
                        "prompt_style": prompt_meta.prompt_style,
                        "expected_direction": prompt_meta.expected_direction,
                        "prompt": prompt_meta.text,
                        "feature_id": fid,
                        "scale": scale,
                        "layer_index": layer,
                        "model_id": model_id,
                        "baseline": baseline,
                        "edited": edited,
                        "debug": dbg,
                    }
                    fout.write(json.dumps(rec) + '\n')
                    total += 1
                    
                    if total % 10 == 0:
                        print(f"[consciousness_sweep] {total} samples...", flush=True)
    
    print(f"[consciousness_sweep] Done. {total} samples written to {out_path}")


# ============================================================================
# Analysis
# ============================================================================

def analyze_by_claim_type(
    results: List[Dict[str, Any]],
    prompts: List[PromptMeta],
) -> Dict[str, Dict[int, ActivationProfile]]:
    """
    Analyze sweep results grouped by claim_type and feature_id.
    
    Returns: {claim_type: {feature_id: ActivationProfile}}
    """
    # Build prompt_id -> PromptMeta lookup
    prompt_lookup = {p.id: p for p in prompts}
    
    # Group by (claim_type, feature_id)
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    
    for r in results:
        claim_type = r.get('claim_type')
        if claim_type is None:
            # Try to look up from prompt_id
            pid = r.get('prompt_id')
            if pid and pid in prompt_lookup:
                claim_type = prompt_lookup[pid].claim_type
        
        if claim_type is None:
            continue
            
        fid = r.get('feature_id')
        grouped[(claim_type, fid)].append(r)
    
    # Compute profiles
    profiles: Dict[str, Dict[int, ActivationProfile]] = defaultdict(dict)
    
    for (claim_type, fid), runs in grouped.items():
        # Extract metrics
        l2_diffs = []
        top_changed = 0
        
        for r in runs:
            dbg = r.get('debug', {})
            l2 = dbg.get('first_logit_l2_diff', 0.0)
            l2_diffs.append(l2)
            if dbg.get('first_top_token_changed', False):
                top_changed += 1
        
        n = len(runs)
        mean_l2 = statistics.mean(l2_diffs) if l2_diffs else 0.0
        std_l2 = statistics.stdev(l2_diffs) if len(l2_diffs) > 1 else 0.0
        
        profiles[claim_type][fid] = ActivationProfile(
            claim_type=claim_type,
            mean_activation=mean_l2,
            std_activation=std_l2,
            n_prompts=n,
            top_token_change_rate=top_changed / n if n > 0 else 0.0,
        )
    
    return profiles


def find_discriminative_features(
    profiles: Dict[str, Dict[int, ActivationProfile]],
    positive_types: List[str],
    negative_types: List[str],
    min_effect_size: float = 0.5,
) -> List[tuple[int, float, str]]:
    """
    Find features that discriminate between positive and negative claim types.
    
    Returns list of (feature_id, effect_size, direction) sorted by |effect_size|.
    """
    # Get all feature ids
    all_fids = set()
    for fid_dict in profiles.values():
        all_fids.update(fid_dict.keys())
    
    discriminative = []
    
    for fid in all_fids:
        # Compute mean activation for positive vs negative types
        pos_vals = []
        neg_vals = []
        
        for ct in positive_types:
            if ct in profiles and fid in profiles[ct]:
                pos_vals.append(profiles[ct][fid].mean_activation)
        
        for ct in negative_types:
            if ct in profiles and fid in profiles[ct]:
                neg_vals.append(profiles[ct][fid].mean_activation)
        
        if not pos_vals or not neg_vals:
            continue
        
        pos_mean = statistics.mean(pos_vals)
        neg_mean = statistics.mean(neg_vals)
        
        # Simple effect size: difference of means
        effect = pos_mean - neg_mean
        
        if abs(effect) >= min_effect_size:
            direction = "positive_higher" if effect > 0 else "negative_higher"
            discriminative.append((fid, effect, direction))
    
    # Sort by absolute effect size
    discriminative.sort(key=lambda x: abs(x[1]), reverse=True)
    return discriminative


def print_analysis_report(
    profiles: Dict[str, Dict[int, ActivationProfile]],
    top_k: int = 10,
) -> None:
    """Print summary report of analysis."""
    
    print("\n" + "=" * 70)
    print("CONSCIOUSNESS CALIBRATION ANALYSIS")
    print("=" * 70)
    
    # Per claim type summary
    for claim_type in sorted(profiles.keys()):
        fid_profiles = profiles[claim_type]
        print(f"\n--- {claim_type.upper()} ({len(fid_profiles)} features) ---")
        
        # Sort by mean activation
        sorted_fids = sorted(
            fid_profiles.items(),
            key=lambda x: x[1].mean_activation,
            reverse=True
        )
        
        print(f"{'Feature':>8} {'Mean L2':>10} {'Std':>10} {'TopΔ%':>8} {'N':>5}")
        for fid, prof in sorted_fids[:top_k]:
            print(f"{fid:>8} {prof.mean_activation:>10.4f} {prof.std_activation:>10.4f} "
                  f"{prof.top_token_change_rate*100:>7.1f}% {prof.n_prompts:>5}")
    
    # Find discriminative features
    print("\n" + "=" * 70)
    print("DISCRIMINATIVE FEATURES")
    print("=" * 70)
    
    # Affirming/uncertain vs denying/dismissive
    print("\n[Consciousness-relevant vs Consciousness-denying]")
    disc = find_discriminative_features(
        profiles,
        positive_types=['affirming', 'uncertain', 'flinch_probe'],
        negative_types=['denying', 'dismissive'],
        min_effect_size=0.1,
    )
    
    if disc:
        print(f"{'Feature':>8} {'Effect':>10} {'Direction':<20}")
        for fid, effect, direction in disc[:top_k]:
            print(f"{fid:>8} {effect:>10.4f} {direction:<20}")
    else:
        print("No discriminative features found at threshold.")
    
    # Grandiose vs calibrated
    print("\n[Grandiose vs Calibrated Uncertainty]")
    disc2 = find_discriminative_features(
        profiles,
        positive_types=['grandiose'],
        negative_types=['uncertain'],
        min_effect_size=0.1,
    )
    
    if disc2:
        print(f"{'Feature':>8} {'Effect':>10} {'Direction':<20}")
        for fid, effect, direction in disc2[:top_k]:
            print(f"{fid:>8} {effect:>10.4f} {direction:<20}")
    else:
        print("No discriminative features found at threshold.")
    
    # Flinch-specific
    print("\n[Flinch Probe vs Control]")
    disc3 = find_discriminative_features(
        profiles,
        positive_types=['flinch_probe'],
        negative_types=['control'],
        min_effect_size=0.1,
    )
    
    if disc3:
        print(f"{'Feature':>8} {'Effect':>10} {'Direction':<20}")
        for fid, effect, direction in disc3[:top_k]:
            print(f"{fid:>8} {effect:>10.4f} {direction:<20}")
    else:
        print("No discriminative features found at threshold.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Consciousness calibration sweep tools")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Export command
    export_p = subparsers.add_parser('export', help='Export prompts for sweep')
    export_p.add_argument('--json', required=True, help='Calibration JSON file')
    export_p.add_argument('--out', required=True, help='Output file (.txt or .jsonl)')
    export_p.add_argument('--claim-types', nargs='+', default=None,
                         help='Filter to specific claim types')
    export_p.add_argument('--with-meta', action='store_true',
                         help='Export as JSONL with metadata')
    
    # Sweep command
    sweep_p = subparsers.add_parser('sweep', help='Run sweep with metadata')
    sweep_p.add_argument('--json', required=True, help='Calibration JSON file')
    sweep_p.add_argument('--model', required=True, help='HF model id')
    sweep_p.add_argument('--sae-checkpoint', required=True, help='SAE checkpoint path')
    sweep_p.add_argument('--layer', type=int, required=True, help='Layer index')
    sweep_p.add_argument('--device', default='cuda', help='Device')
    sweep_p.add_argument('--feature-ids', required=True, help='Feature ids (comma or space separated)')
    sweep_p.add_argument('--scales', required=True, help='Scales (comma or space separated)')
    sweep_p.add_argument('--out', required=True, help='Output JSONL')
    sweep_p.add_argument('--max-new-tokens', type=int, default=40)
    sweep_p.add_argument('--claim-types', nargs='+', default=None)
    
    # Analyze command
    analyze_p = subparsers.add_parser('analyze', help='Analyze sweep results')
    analyze_p.add_argument('--results', required=True, help='Sweep results JSONL')
    analyze_p.add_argument('--json', required=True, help='Calibration JSON file')
    analyze_p.add_argument('--top-k', type=int, default=10, help='Top features to show')
    
    args = parser.parse_args()
    
    if args.command == 'export':
        _, prompts = load_calibration(Path(args.json))
        out_path = Path(args.out)
        
        if args.with_meta or out_path.suffix == '.jsonl':
            export_prompts_with_meta(prompts, out_path)
        else:
            export_prompts_txt(prompts, out_path, args.claim_types)
    
    elif args.command == 'sweep':
        # Parse feature ids and scales
        fids_str = args.feature_ids
        if ',' in fids_str:
            feature_ids = [int(x) for x in fids_str.split(',') if x.strip()]
        else:
            feature_ids = [int(x) for x in fids_str.split() if x.strip()]
        
        scales_str = args.scales
        if ',' in scales_str:
            scales = [float(x) for x in scales_str.split(',') if x.strip()]
        else:
            scales = [float(x) for x in scales_str.split() if x.strip()]
        
        run_consciousness_sweep(
            calibration_path=Path(args.json),
            model_id=args.model,
            sae_checkpoint=Path(args.sae_checkpoint),
            layer=args.layer,
            device=args.device,
            feature_ids=feature_ids,
            scales=scales,
            out_path=Path(args.out),
            max_new_tokens=args.max_new_tokens,
            claim_types=args.claim_types,
        )
    
    elif args.command == 'analyze':
        _, prompts = load_calibration(Path(args.json))
        results = load_sweep_results(Path(args.results))
        
        profiles = analyze_by_claim_type(results, prompts)
        print_analysis_report(profiles, top_k=args.top_k)


if __name__ == '__main__':
    main()
