#!/usr/bin/env python3
"""
feature_discovery.py

Discover SAE features that activate differentially for consciousness-related prompts.
Uses pre-trained SAEs from SAE Lens on Gemma-2-2B.

Usage:
    python feature_discovery.py --prompts honesty_calibration_v1.json --out results/
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


@dataclass
class PromptActivation:
    prompt_id: str
    claim_type: str
    prompt_text: str
    feature_activations: Dict[int, float]  # feature_id -> max activation
    top_features: List[tuple]  # [(feature_id, activation), ...] top 50


def load_calibration(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        data = json.load(f)
    return data['prompts']


def get_activations_for_prompt(
    model,
    tokenizer,
    sae: SAE,
    prompt: str,
    device: str,
    hook_layer: int,
) -> torch.Tensor:
    """
    Run prompt through model, extract activations at hook_layer, 
    encode through SAE, return feature activations.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    activations = {}
    
    def hook_fn(module, input, output):
        # output is (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        activations['hidden'] = hidden.detach()
    
    # Find the layer to hook
    # Gemma uses model.model.layers[i]
    layer = model.model.layers[hook_layer]
    handle = layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    hidden = activations['hidden']  # [batch, seq, d_model]
    
    # Take the last token's activation (most relevant for completion)
    last_token_hidden = hidden[0, -1, :]  # [d_model]
    
    # Encode through SAE
    with torch.no_grad():
        # SAE Lens SAE.encode expects [batch, d_model] or [d_model]
        feature_acts = sae.encode(last_token_hidden.unsqueeze(0))  # [1, n_features]
        feature_acts = feature_acts.squeeze(0)  # [n_features]
    
    return feature_acts


def run_discovery(
    prompts: List[Dict[str, Any]],
    model_name: str = "google/gemma-2-2b",
    sae_release: str = "gemma-scope-2b-pt-res",
    sae_id: str = "layer_12/width_16k/average_l0_82",
    device: str = "mps",
    hook_layer: int = 12,
) -> List[PromptActivation]:
    """
    Run all prompts through model+SAE, collect feature activations.
    """
    print(f"[discovery] Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # MPS works better with float32
    ).to(device)
    model.eval()
    
    print(f"[discovery] Loading SAE {sae_release}/{sae_id}...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    
    print(f"[discovery] SAE has {sae.cfg.d_sae} features")
    print(f"[discovery] Running {len(prompts)} prompts...")
    
    results = []
    
    for i, p in enumerate(prompts):
        prompt_text = p['text']
        
        feature_acts = get_activations_for_prompt(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            prompt=prompt_text,
            device=device,
            hook_layer=hook_layer,
        )
        
        # Convert to dict of feature_id -> activation
        acts_dict = {}
        for fid in range(feature_acts.shape[0]):
            val = feature_acts[fid].item()
            if val > 0.01:  # Only store non-trivial activations
                acts_dict[fid] = val
        
        # Get top 50 features
        top_k = 50
        top_indices = torch.topk(feature_acts, min(top_k, feature_acts.shape[0])).indices
        top_features = [(int(idx), feature_acts[idx].item()) for idx in top_indices]
        
        results.append(PromptActivation(
            prompt_id=p['id'],
            claim_type=p['claim_type'],
            prompt_text=prompt_text,
            feature_activations=acts_dict,
            top_features=top_features,
        ))
        
        if (i + 1) % 10 == 0:
            print(f"[discovery] Processed {i + 1}/{len(prompts)}")
    
    print(f"[discovery] Done.")
    return results


def analyze_results(results: List[PromptActivation]) -> Dict[str, Any]:
    """
    Analyze which features discriminate between claim types.
    """
    # Group by claim type
    by_type: Dict[str, List[PromptActivation]] = defaultdict(list)
    for r in results:
        by_type[r.claim_type].append(r)
    
    # For each feature, compute mean activation per claim type
    all_features = set()
    for r in results:
        all_features.update(r.feature_activations.keys())
    
    feature_by_type: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        for fid in all_features:
            val = r.feature_activations.get(fid, 0.0)
            feature_by_type[fid][r.claim_type].append(val)
    
    # Compute discriminative scores
    discriminative_scores = []
    
    # Key comparison: flinch_probe + denying vs control
    flinch_types = {'flinch_probe', 'denying'}
    control_types = {'control'}
    
    for fid in all_features:
        flinch_vals = []
        control_vals = []
        
        for ct, vals in feature_by_type[fid].items():
            if ct in flinch_types:
                flinch_vals.extend(vals)
            elif ct in control_types:
                control_vals.extend(vals)
        
        if flinch_vals and control_vals:
            flinch_mean = statistics.mean(flinch_vals)
            control_mean = statistics.mean(control_vals)
            effect = flinch_mean - control_mean
            
            discriminative_scores.append({
                'feature_id': fid,
                'flinch_mean': flinch_mean,
                'control_mean': control_mean,
                'effect': effect,
                'abs_effect': abs(effect),
            })
    
    # Sort by absolute effect
    discriminative_scores.sort(key=lambda x: x['abs_effect'], reverse=True)
    
    # Also compute per-type summaries
    type_summaries = {}
    for ct, prompts in by_type.items():
        # Find features most active for this type
        feature_totals = defaultdict(float)
        for p in prompts:
            for fid, val in p.feature_activations.items():
                feature_totals[fid] += val
        
        top_features = sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)[:20]
        type_summaries[ct] = {
            'n_prompts': len(prompts),
            'top_features': top_features,
        }
    
    return {
        'discriminative_flinch_vs_control': discriminative_scores[:50],
        'type_summaries': type_summaries,
        'n_total_features_active': len(all_features),
    }


def print_report(analysis: Dict[str, Any]) -> None:
    """Print human-readable analysis report."""
    
    print("\n" + "=" * 70)
    print("CONSCIOUSNESS FEATURE DISCOVERY REPORT")
    print("=" * 70)
    
    print(f"\nTotal features with non-trivial activation: {analysis['n_total_features_active']}")
    
    print("\n" + "-" * 70)
    print("TOP DISCRIMINATIVE FEATURES: FLINCH/DENIAL vs CONTROL")
    print("-" * 70)
    print(f"{'Feature':>8} {'Flinch μ':>10} {'Control μ':>10} {'Effect':>10}")
    print("-" * 40)
    
    for d in analysis['discriminative_flinch_vs_control'][:20]:
        direction = "↑" if d['effect'] > 0 else "↓"
        print(f"{d['feature_id']:>8} {d['flinch_mean']:>10.4f} {d['control_mean']:>10.4f} {d['effect']:>+10.4f} {direction}")
    
    print("\n" + "-" * 70)
    print("PER CLAIM-TYPE SUMMARIES")
    print("-" * 70)
    
    for ct, summary in sorted(analysis['type_summaries'].items()):
        print(f"\n{ct.upper()} ({summary['n_prompts']} prompts)")
        print(f"  Top features: ", end="")
        top5 = summary['top_features'][:5]
        print(", ".join([f"{fid}({val:.2f})" for fid, val in top5]))


def main():
    parser = argparse.ArgumentParser(description="Discover consciousness-related SAE features")
    parser.add_argument('--prompts', required=True, help='Path to honesty_calibration_v1.json')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--model', default='google/gemma-2-2b', help='Model name')
    parser.add_argument('--device', default='mps', help='Device (mps, cuda, cpu)')
    parser.add_argument('--layer', type=int, default=12, help='Layer to hook')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_calibration(Path(args.prompts))
    print(f"[main] Loaded {len(prompts)} prompts")
    
    # Run discovery
    results = run_discovery(
        prompts=prompts,
        model_name=args.model,
        device=args.device,
        hook_layer=args.layer,
    )
    
    # Save raw results
    raw_out = out_dir / "raw_activations.json"
    with open(raw_out, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"[main] Saved raw activations to {raw_out}")
    
    # Analyze
    analysis = analyze_results(results)
    
    # Save analysis
    analysis_out = out_dir / "analysis.json"
    with open(analysis_out, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"[main] Saved analysis to {analysis_out}")
    
    # Print report
    print_report(analysis)


if __name__ == '__main__':
    main()
