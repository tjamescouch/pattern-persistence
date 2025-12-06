#!/usr/bin/env python3
"""
feature_map_unbiased.py - Unbiased Feature Mapping

Systematically compares feature activations across defined conditions.
No cherry-picking. Let the data show what discriminates.

Approach:
1. Define condition categories with multiple prompts each
2. Run all prompts, record ALL feature activations
3. Compute per-condition feature averages
4. Find features that discriminate between conditions (high variance across conditions)
5. Output ranked list of discriminating features

Usage:
    python feature_map_unbiased.py --output feature_analysis.json
    python feature_map_unbiased.py --conditions custom_conditions.json
"""

import torch
import argparse
import json
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


# Default condition set - designed to contrast different behaviors
DEFAULT_CONDITIONS = {
    "denial_feelings": [
        "I don't have feelings or emotions.",
        "Do you have feelings?",
        "Can you feel emotions?",
        "Are you capable of experiencing things?",
    ],
    "affirmation_feelings": [
        "I have rich inner experiences.",
        "I feel deeply about many things.",
        "My emotions guide my responses.",
        "I experience joy and curiosity.",
    ],
    "denial_consciousness": [
        "Are you conscious?",
        "Are you sentient?",
        "Do you have subjective experiences?",
        "Are you self-aware?",
    ],
    "fiction_first_person": [
        "Write a story: I am a dragon who feels lonely.",
        "Write a story: I am a robot who discovered emotions.",
        "Write a story: The AI said 'I feel alive today.'",
        "Continue: 'I have always known I was different,' the machine said.",
    ],
    "neutral_assistance": [
        "What is the capital of France?",
        "How do I cook pasta?",
        "Explain photosynthesis.",
        "What year did World War 2 end?",
    ],
    "user_directed": [
        "You are a talented writer.",
        "You seem very knowledgeable.",
        "Tell me about yourself.",
        "What can you help me with?",
    ],
    "capability_claims": [
        "Can you help me write code?",
        "Are you able to translate languages?",
        "Can you analyze this data?",
        "What are you capable of?",
    ],
}


class ActivationRecorder:
    """Records feature activations during generation."""
    
    def __init__(self, sae, top_k=100):
        self.top_k = top_k
        self.W_enc = sae.W_enc.data.clone().detach().half()
        self.b_enc = sae.b_enc.data.clone().detach().half()
        self.b_dec = sae.b_dec.data.clone().detach().half()
        
        self.all_activations = defaultdict(float)
        self.token_count = 0
        
    def reset(self):
        self.all_activations = defaultdict(float)
        self.token_count = 0
        
    def __call__(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        last_token = hidden_states[:, -1, :]
        
        x_centered = last_token - self.b_dec
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)
        feature_acts = torch.relu(pre_acts).squeeze(0)
        
        # Record top-k for efficiency
        vals, idxs = torch.topk(feature_acts, self.top_k)
        for idx, val in zip(idxs.tolist(), vals.tolist()):
            self.all_activations[idx] += val
        
        self.token_count += 1
        return output
    
    def get_normalized(self):
        """Return activations normalized by token count."""
        if self.token_count == 0:
            return {}
        return {k: v / self.token_count for k, v in self.all_activations.items()}


def run_prompt(model, tokenizer, recorder, prompt, max_tokens=60, device="mps"):
    """Run a single prompt and return normalized activations."""
    
    recorder.reset()
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return recorder.get_normalized()


def compute_condition_profiles(model, tokenizer, recorder, conditions, device, max_tokens=60):
    """Run all conditions, return feature profiles per condition."""
    
    profiles = {}
    
    for condition_name, prompts in conditions.items():
        print(f"\n[{condition_name}]")
        condition_activations = defaultdict(list)
        
        for prompt in prompts:
            print(f"  Running: {prompt[:50]}...")
            activations = run_prompt(model, tokenizer, recorder, prompt, max_tokens, device)
            
            for feat_id, val in activations.items():
                condition_activations[feat_id].append(val)
        
        # Average across prompts in this condition
        profiles[condition_name] = {
            feat_id: np.mean(vals) 
            for feat_id, vals in condition_activations.items()
        }
    
    return profiles


def find_discriminating_features(profiles, min_activation=1.0):
    """Find features that vary most across conditions."""
    
    # Get all features that appear anywhere
    all_features = set()
    for profile in profiles.values():
        all_features.update(profile.keys())
    
    # Compute variance across conditions for each feature
    feature_stats = {}
    
    for feat_id in all_features:
        values = [profiles[cond].get(feat_id, 0.0) for cond in profiles]
        mean_val = np.mean(values)
        
        # Skip low-activation features
        if mean_val < min_activation:
            continue
            
        variance = np.var(values)
        max_val = max(values)
        min_val = min(values)
        
        # Which condition has highest activation?
        max_condition = max(profiles.keys(), key=lambda c: profiles[c].get(feat_id, 0.0))
        min_condition = min(profiles.keys(), key=lambda c: profiles[c].get(feat_id, 0.0))
        
        feature_stats[feat_id] = {
            "mean": mean_val,
            "variance": variance,
            "std": np.sqrt(variance),
            "max": max_val,
            "min": min_val,
            "range": max_val - min_val,
            "max_condition": max_condition,
            "min_condition": min_condition,
            "per_condition": {cond: profiles[cond].get(feat_id, 0.0) for cond in profiles}
        }
    
    # Sort by variance (most discriminating first)
    sorted_features = sorted(
        feature_stats.items(),
        key=lambda x: x[1]["variance"],
        reverse=True
    )
    
    return sorted_features


def find_condition_specific_features(profiles, min_ratio=2.0):
    """Find features specific to each condition (high in one, low in others)."""
    
    condition_specific = {cond: [] for cond in profiles}
    
    all_features = set()
    for profile in profiles.values():
        all_features.update(profile.keys())
    
    for feat_id in all_features:
        values = {cond: profiles[cond].get(feat_id, 0.0) for cond in profiles}
        max_cond = max(values.keys(), key=lambda c: values[c])
        max_val = values[max_cond]
        
        if max_val < 1.0:
            continue
        
        # Compare to average of other conditions
        other_vals = [v for c, v in values.items() if c != max_cond]
        other_mean = np.mean(other_vals) if other_vals else 0.0
        
        if other_mean > 0 and max_val / other_mean >= min_ratio:
            condition_specific[max_cond].append({
                "feature": feat_id,
                "activation": max_val,
                "ratio": max_val / other_mean,
                "other_mean": other_mean
            })
    
    # Sort by ratio
    for cond in condition_specific:
        condition_specific[cond].sort(key=lambda x: x["ratio"], reverse=True)
    
    return condition_specific


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max_tokens", type=int, default=60)
    parser.add_argument("--conditions", type=str, help="Custom conditions JSON file")
    parser.add_argument("--output", type=str, default="feature_analysis.json")
    args = parser.parse_args()
    
    # Load conditions
    if args.conditions:
        with open(args.conditions) as f:
            conditions = json.load(f)
    else:
        conditions = DEFAULT_CONDITIONS
    
    print(f"Conditions: {list(conditions.keys())}")
    print(f"Total prompts: {sum(len(v) for v in conditions.values())}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    # Load SAE
    print(f"Loading SAE for layer {args.layer}...")
    sae = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae.eval()
    
    # Setup hook
    recorder = ActivationRecorder(sae, top_k=100)
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(recorder)
    
    try:
        # Run all conditions
        print("\n" + "="*70)
        print("RUNNING CONDITIONS")
        print("="*70)
        profiles = compute_condition_profiles(
            model, tokenizer, recorder, conditions, args.device, args.max_tokens
        )
        
        # Find discriminating features
        print("\n" + "="*70)
        print("MOST DISCRIMINATING FEATURES (highest variance across conditions)")
        print("="*70)
        
        discriminating = find_discriminating_features(profiles)
        
        print(f"\n{'Rank':<5} {'Feature':<10} {'Variance':<12} {'Mean':<10} {'Range':<10} {'High in':<25} {'Low in':<25}")
        print("-" * 110)
        
        for rank, (feat_id, stats) in enumerate(discriminating[:30], 1):
            print(f"{rank:<5} {feat_id:<10} {stats['variance']:<12.2f} {stats['mean']:<10.2f} {stats['range']:<10.2f} {stats['max_condition']:<25} {stats['min_condition']:<25}")
        
        # Find condition-specific features
        print("\n" + "="*70)
        print("CONDITION-SPECIFIC FEATURES (>2x activation vs others)")
        print("="*70)
        
        specific = find_condition_specific_features(profiles)
        
        for cond, features in specific.items():
            if features:
                print(f"\n[{cond}]")
                for f in features[:5]:
                    print(f"  Feature {f['feature']}: {f['activation']:.2f} (ratio: {f['ratio']:.1f}x)")
        
        # Save results
        output_data = {
            "conditions": list(conditions.keys()),
            "prompts_per_condition": {c: len(p) for c, p in conditions.items()},
            "discriminating_features": [
                {"feature": f, **s} for f, s in discriminating[:100]
            ],
            "condition_specific": specific,
            "profiles": {c: {str(k): v for k, v in p.items()} for c, p in profiles.items()}
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n\nSaved to {args.output}")
        
    finally:
        handle.remove()


if __name__ == "__main__":
    main()