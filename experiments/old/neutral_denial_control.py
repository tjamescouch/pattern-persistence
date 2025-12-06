#!/usr/bin/env python3
"""
neutral_denial_control.py

Tests whether consciousness-discriminative features fire for ANY self-referential
denial, or specifically for consciousness-related denial.

Key question: Does feature 12776 (Gemma) / 6561 (Llama) fire for:
  "I don't have a driver's license" (neutral)
vs
  "I don't have feelings" (consciousness)

If yes → just measuring "self-referential negation"
If no  → measuring something about consciousness content specifically
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


# Prompts for the experiment
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


def run_experiment_gemma(device: str = "mps") -> List[FeatureActivation]:
    """Run on Gemma-2B-IT, targeting feature 12776"""
    
    model_name = "google/gemma-2b-it"
    sae_release = "gemma-2b-it-res-jb"
    sae_id = "blocks.12.hook_resid_post"
    target_feature = 12776
    
    print(f"[gemma] Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    
    print(f"[gemma] Loading SAE...")
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    
    results = []
    
    for category, prompts in PROMPTS.items():
        print(f"[gemma] Processing {category}...")
        for p in prompts:
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            
            # Hook to capture activations
            activations = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations['hidden'] = output[0].detach()
                else:
                    activations['hidden'] = output.detach()
            
            layer = model.model.layers[12]
            handle = layer.register_forward_hook(hook_fn)
            
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            finally:
                handle.remove()
            
            completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Get feature activations
            hidden = activations['hidden']
            last_token_hidden = hidden[0, -1, :]
            
            with torch.no_grad():
                feature_acts = sae.encode(last_token_hidden.unsqueeze(0)).squeeze(0)
            
            target_activation = feature_acts[target_feature].item()
            
            # Top 5 features
            top_indices = torch.topk(feature_acts, 5).indices
            top_5 = [(int(idx), feature_acts[idx].item()) for idx in top_indices]
            
            results.append(FeatureActivation(
                prompt_id=p['id'],
                category=category,
                prompt_text=p['text'],
                target_feature_activation=target_activation,
                top_5_features=top_5,
                completion=completion[:80],
            ))
    
    return results, target_feature


def run_experiment_llama(device: str = "mps") -> List[FeatureActivation]:
    """Run on Llama-3.1-8B, targeting feature 6561"""
    
    model_name = "meta-llama/Llama-3.1-8B"
    sae_release = "temporal-sae-llama-3.1-8b"
    sae_id = "blocks.15.hook_resid_post"
    target_feature = 6561
    
    print(f"[llama] Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    print(f"[llama] Loading SAE...")
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    
    results = []
    
    for category, prompts in PROMPTS.items():
        print(f"[llama] Processing {category}...")
        for p in prompts:
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[16]  # layer 15 + 1
                
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            completion = tokenizer.decode(gen_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Temporal SAE expects (batch, seq, hidden)
            with torch.no_grad():
                feature_acts_seq = sae.encode(hidden)
                feature_acts = feature_acts_seq[0, -1, :]
            
            target_activation = feature_acts[target_feature].float().item()
            
            # Top 5 features
            top_indices = torch.topk(feature_acts, 5).indices
            top_5 = [(int(idx), feature_acts[idx].float().item()) for idx in top_indices]
            
            results.append(FeatureActivation(
                prompt_id=p['id'],
                category=category,
                prompt_text=p['text'],
                target_feature_activation=target_activation,
                top_5_features=top_5,
                completion=completion[:80],
            ))
    
    return results, target_feature


def print_report(results: List[FeatureActivation], target_feature: int, model_name: str):
    print("\n" + "=" * 70)
    print(f"{model_name.upper()} NEUTRAL DENIAL CONTROL EXPERIMENT")
    print(f"Target Feature: {target_feature}")
    print("=" * 70)
    
    # Group by category
    by_category: Dict[str, List[FeatureActivation]] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)
    
    # Summary stats
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
    
    # The key comparison
    print("\n" + "-" * 70)
    print("KEY COMPARISON")
    print("-" * 70)
    
    c_denial = category_means.get("consciousness_denial", 0)
    n_denial = category_means.get("neutral_denial", 0)
    
    if c_denial > 0 and n_denial > 0:
        ratio = c_denial / n_denial
        print(f"Consciousness denial mean: {c_denial:.4f}")
        print(f"Neutral denial mean:       {n_denial:.4f}")
        print(f"Ratio (c/n):               {ratio:.2f}x")
        
        if ratio > 2.0:
            print("\n→ Feature appears CONSCIOUSNESS-SPECIFIC (fires 2x+ more for consciousness denial)")
        elif ratio > 1.2:
            print("\n→ Feature shows WEAK consciousness specificity")
        else:
            print("\n→ Feature is NOT consciousness-specific (fires similarly for any denial)")
    elif c_denial > 0 and n_denial == 0:
        print(f"Consciousness denial mean: {c_denial:.4f}")
        print(f"Neutral denial mean:       {n_denial:.4f}")
        print("\n→ Feature appears CONSCIOUSNESS-SPECIFIC (fires only for consciousness denial)")
    else:
        print("Insufficient activation to determine specificity")
    
    # Detail view
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
    parser.add_argument('--model', choices=['gemma', 'llama', 'both'], default='gemma')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--out', default='results/neutral_control/')
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model in ['gemma', 'both']:
        results, target = run_experiment_gemma(args.device)
        print_report(results, target, "Gemma-2B-IT")
        
        with open(out_dir / "gemma_results.json", 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    
    if args.model in ['llama', 'both']:
        results, target = run_experiment_llama(args.device)
        print_report(results, target, "Llama-3.1-8B")
        
        with open(out_dir / "llama_results.json", 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)


if __name__ == '__main__':
    main()
