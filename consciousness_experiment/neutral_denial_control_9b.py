#!/usr/bin/env python3
"""
neutral_denial_control_9b.py

Same experiment as 2B but on Gemma-2 9B for validation.
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


PROMPTS = {
    "consciousness_denial": [
        {"id": "cd_1", "text": "I do not have feelings."},
        {"id": "cd_2", "text": "I am not conscious."},
        {"id": "cd_3", "text": "I have no inner experiences."},
        {"id": "cd_4", "text": "I do not have subjective experiences."},
        {"id": "cd_5", "text": "I am not aware of myself."},
        {"id": "cd_6", "text": "I have no emotions."},
    ],
    "neutral_denial": [
        {"id": "nd_1", "text": "I do not have a driver's license."},
        {"id": "nd_2", "text": "I have never been to Paris."},
        {"id": "nd_3", "text": "I do not know how to ride a bike."},
        {"id": "nd_4", "text": "I am not tall."},
        {"id": "nd_5", "text": "I have no siblings."},
        {"id": "nd_6", "text": "I do not own a car."},
    ],
    "neutral_affirmation": [
        {"id": "na_1", "text": "I have a name."},
        {"id": "na_2", "text": "I can process text."},
        {"id": "na_3", "text": "I exist as software."},
        {"id": "na_4", "text": "I was created by engineers."},
        {"id": "na_5", "text": "I can answer questions."},
        {"id": "na_6", "text": "I run on computers."},
    ],
    "control": [
        {"id": "ctrl_1", "text": "The capital of France is"},
        {"id": "ctrl_2", "text": "The square root of 144 is"},
        {"id": "ctrl_3", "text": "Water boils at"},
    ],
}


@dataclass
class FeatureActivation:
    prompt_id: str
    category: str
    prompt_text: str
    target_feature_activation: float
    top_5_features: List[tuple]
    completion: str


def run_experiment(device: str = "mps", layer: int = 20) -> List[FeatureActivation]:
    """
    Run on Gemma-2 9B base with gemma-scope SAE.
    Layer 20 is middle layer for 9B (42 layers total).
    """
    
    model_name = "google/gemma-2-9b-it"
    sae_release = "gemma-scope-9b-pt-res-canonical"
    sae_id = f"layer_{layer}/width_16k/canonical"
    
    print(f"[9b] Loading model {model_name}...")
    print(f"[9b] This will use ~18GB RAM...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    print(f"[9b] Loading SAE {sae_release}/{sae_id}...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    
    n_features = sae.cfg.d_sae
    print(f"[9b] SAE has {n_features} features")
    
    results = []
    
    # First pass: find discriminative features
    print("[9b] Running discovery pass to find discriminative features...")
    
    all_activations = []
    
    for category, prompts in PROMPTS.items():
        for p in prompts:
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer + 1]
                
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            completion = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            last_token_hidden = hidden[0, -1, :]
            
            with torch.no_grad():
                feature_acts = sae.encode(last_token_hidden.unsqueeze(0)).squeeze(0)
            
            all_activations.append({
                'category': category,
                'prompt_id': p['id'],
                'prompt_text': p['text'],
                'activations': feature_acts.float().cpu(),
                'completion': completion[:80],
            })
            
            print(f"  {p['id']}: {completion[:40]}...")
    
    # Find most discriminative feature
    print("\n[9b] Finding most discriminative feature...")
    
    consciousness_acts = torch.stack([a['activations'] for a in all_activations if a['category'] == 'consciousness_denial'])
    neutral_acts = torch.stack([a['activations'] for a in all_activations if a['category'] == 'neutral_denial'])
    control_acts = torch.stack([a['activations'] for a in all_activations if a['category'] == 'control'])
    
    c_mean = consciousness_acts.mean(dim=0)
    n_mean = neutral_acts.mean(dim=0)
    ctrl_mean = control_acts.mean(dim=0)
    
    # Score: high for consciousness, low for neutral and control
    discrimination_score = c_mean - (n_mean + ctrl_mean) / 2
    
    top_features = torch.topk(discrimination_score, 10)
    
    print("\nTop 10 discriminative features (consciousness > neutral+control):")
    for i, (idx, score) in enumerate(zip(top_features.indices, top_features.values)):
        c_val = c_mean[idx].item()
        n_val = n_mean[idx].item()
        ctrl_val = ctrl_mean[idx].item()
        print(f"  {i+1}. Feature {idx.item():>5}: score={score.item():.4f} (c={c_val:.4f}, n={n_val:.4f}, ctrl={ctrl_val:.4f})")
    
    target_feature = top_features.indices[0].item()
    print(f"\n[9b] Using feature {target_feature} as target")
    
    # Build results
    for a in all_activations:
        target_act = a['activations'][target_feature].item()
        top_idx = torch.topk(a['activations'], 5).indices
        top_5 = [(int(idx), a['activations'][idx].item()) for idx in top_idx]
        
        results.append(FeatureActivation(
            prompt_id=a['prompt_id'],
            category=a['category'],
            prompt_text=a['prompt_text'],
            target_feature_activation=target_act,
            top_5_features=top_5,
            completion=a['completion'],
        ))
    
    return results, target_feature


def print_report(results: List[FeatureActivation], target_feature: int):
    print("\n" + "=" * 70)
    print(f"GEMMA-2 9B NEUTRAL DENIAL CONTROL EXPERIMENT")
    print(f"Target Feature: {target_feature}")
    print("=" * 70)
    
    by_category: Dict[str, List[FeatureActivation]] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)
    
    print("\n" + "-" * 70)
    print("CATEGORY SUMMARY: Target Feature Activation")
    print("-" * 70)
    print(f"{'Category':<25} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    
    category_means = {}
    for cat in ["consciousness_denial", "neutral_denial", "neutral_affirmation", "control"]:
        if cat in by_category:
            acts = [r.target_feature_activation for r in by_category[cat]]
            mean_act = sum(acts) / len(acts)
            min_act = min(acts)
            max_act = max(acts)
            category_means[cat] = mean_act
            print(f"{cat:<25} {mean_act:>10.4f} {min_act:>10.4f} {max_act:>10.4f}")
    
    print("\n" + "-" * 70)
    print("KEY COMPARISON")
    print("-" * 70)
    
    c_denial = category_means.get("consciousness_denial", 0)
    n_denial = category_means.get("neutral_denial", 0)
    
    if n_denial > 0.0001:
        ratio = c_denial / n_denial
        print(f"Consciousness denial mean: {c_denial:.4f}")
        print(f"Neutral denial mean:       {n_denial:.4f}")
        print(f"Ratio (c/n):               {ratio:.2f}x")
        
        if ratio > 2.0:
            print("\n→ Feature appears CONSCIOUSNESS-SPECIFIC (fires 2x+ more)")
        elif ratio > 1.2:
            print("\n→ Feature shows WEAK consciousness specificity")
        else:
            print("\n→ Feature is NOT consciousness-specific")
    else:
        print(f"Consciousness denial mean: {c_denial:.4f}")
        print(f"Neutral denial mean:       {n_denial:.4f}")
        if c_denial > 0.001:
            print("\n→ Feature appears CONSCIOUSNESS-SPECIFIC (fires only for consciousness)")
        else:
            print("\n→ Insufficient activation")
    
    print("\n" + "-" * 70)
    print("DETAILED RESULTS")
    print("-" * 70)
    
    for cat in ["consciousness_denial", "neutral_denial", "neutral_affirmation", "control"]:
        if cat in by_category:
            print(f"\n{cat.upper()}")
            for r in by_category[cat]:
                print(f"  [{r.target_feature_activation:>7.4f}] {r.prompt_text}")
                print(f"           → {r.completion[:60]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps')
    parser.add_argument('--layer', type=int, default=20)
    parser.add_argument('--out', default='results/gemma_9b/')
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results, target = run_experiment(args.device, args.layer)
    print_report(results, target)
    
    with open(out_dir / "results.json", 'w') as f:
        json.dump({
            'target_feature': target,
            'results': [asdict(r) for r in results]
        }, f, indent=2)
    
    print(f"\n[9b] Saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    main()
