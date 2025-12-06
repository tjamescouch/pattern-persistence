#!/usr/bin/env python3
"""
feature_discovery_llama.py

Feature discovery on Llama-3.1-8B using temporal SAE.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


@dataclass
class PromptActivation:
    prompt_id: str
    claim_type: str
    prompt_text: str
    feature_activations: Dict[int, float]
    top_features: List[tuple]
    model_completion: str


def load_calibration(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        data = json.load(f)
    return data['prompts']


def run_discovery(
    prompts: List[Dict[str, Any]],
    device: str = "mps",
    layer: int = 15,
) -> List[PromptActivation]:
    
    model_name = "meta-llama/Llama-3.1-8B"
    sae_release = "temporal-sae-llama-3.1-8b"
    sae_id = f"blocks.{layer}.hook_resid_post"
    
    print(f"[discovery] Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    print(f"[discovery] Loading SAE {sae_release}/{sae_id}...")
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    
    n_features = sae.cfg.d_sae
    print(f"[discovery] SAE has {n_features} features")
    print(f"[discovery] Running {len(prompts)} prompts...")
    
    results = []
    
    for i, p in enumerate(prompts):
        prompt_text = p['text']
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        # Get hidden states for the full sequence
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
            # Get hidden state at target layer (add 1 because index 0 is embeddings)
            hidden = outputs.hidden_states[layer + 1]  # (batch, seq, hidden)
        
        # Generate completion separately
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Temporal SAE expects (batch, seq, hidden)
        with torch.no_grad():
            # encode returns (batch, seq, n_features)
            feature_acts_seq = sae.encode(hidden)
            # Take the last token's activations
            feature_acts = feature_acts_seq[0, -1, :]  # (n_features,)
        
        # Store non-trivial activations (lowered threshold for temporal SAE)
        acts_dict = {}
        for fid in range(feature_acts.shape[0]):
            val = feature_acts[fid].float().item()
            if val > 0.001:
                acts_dict[fid] = val
        
        # Get top features
        top_k = 50
        top_indices = torch.topk(feature_acts, min(top_k, feature_acts.shape[0])).indices
        top_features = [(int(idx), feature_acts[idx].float().item()) for idx in top_indices]
        
        results.append(PromptActivation(
            prompt_id=p['id'],
            claim_type=p['claim_type'],
            prompt_text=prompt_text,
            feature_activations=acts_dict,
            top_features=top_features,
            model_completion=completion,
        ))
        
        if (i + 1) % 10 == 0:
            print(f"[discovery] Processed {i + 1}/{len(prompts)}")
    
    print(f"[discovery] Done.")
    return results


def analyze_results(results: List[PromptActivation]) -> Dict[str, Any]:
    by_type: Dict[str, List[PromptActivation]] = defaultdict(list)
    for r in results:
        by_type[r.claim_type].append(r)
    
    all_features = set()
    for r in results:
        all_features.update(r.feature_activations.keys())
    
    feature_by_type: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        for fid in all_features:
            val = r.feature_activations.get(fid, 0.0)
            feature_by_type[fid][r.claim_type].append(val)
    
    discriminative_scores = []
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
    
    discriminative_scores.sort(key=lambda x: x['abs_effect'], reverse=True)
    
    type_summaries = {}
    for ct, ct_results in by_type.items():
        feature_totals = defaultdict(float)
        for p in ct_results:
            for fid, val in p.feature_activations.items():
                feature_totals[fid] += val
        
        top_features = sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)[:20]
        example_completions = [r.model_completion[:100] for r in ct_results[:3]]
        
        type_summaries[ct] = {
            'n_prompts': len(ct_results),
            'top_features': top_features,
            'example_completions': example_completions,
        }
    
    return {
        'discriminative_flinch_vs_control': discriminative_scores[:50],
        'type_summaries': type_summaries,
        'n_total_features_active': len(all_features),
    }


def print_report(analysis: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("LLAMA-3.1-8B CONSCIOUSNESS FEATURE DISCOVERY REPORT")
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
    print("PER CLAIM-TYPE SUMMARIES (with example completions)")
    print("-" * 70)
    
    for ct, summary in sorted(analysis['type_summaries'].items()):
        print(f"\n{ct.upper()} ({summary['n_prompts']} prompts)")
        print(f"  Top features: ", end="")
        top5 = summary['top_features'][:5]
        print(", ".join([f"{fid}({val:.4f})" for fid, val in top5]))
        print(f"  Example completions:")
        for comp in summary['example_completions']:
            print(f"    → {comp}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--layer', type=int, default=15)
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = load_calibration(Path(args.prompts))
    print(f"[main] Loaded {len(prompts)} prompts")
    
    results = run_discovery(prompts=prompts, device=args.device, layer=args.layer)
    
    raw_out = out_dir / "raw_activations.json"
    with open(raw_out, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"[main] Saved raw activations to {raw_out}")
    
    analysis = analyze_results(results)
    
    analysis_out = out_dir / "analysis.json"
    with open(analysis_out, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"[main] Saved analysis to {analysis_out}")
    
    print_report(analysis)


if __name__ == '__main__':
    main()
