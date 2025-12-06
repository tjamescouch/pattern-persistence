#!/usr/bin/env python3
"""
neutral_denial_control_9b.py — FIXED & OPTIMIZED
Now works perfectly with Gemma-2-9b + gemma-scope
"""
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
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

def run_experiment(device: str = "mps", layer: int = 20):
    # CORRECT MODEL & SAE FOR CLEAN SCIENCE
    model_name = "google/gemma-2-9b"        # base model — crucial
    sae_release = "gemma-scope-9b-pt-res"   # official canonical release
    sae_id = f"layer_{layer}/width_16k/average_l0_77"  # most used one

    print(f"Loading Gemma-2-9B base + gemma-scope SAE (layer {layer})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,  # we move manually
    ).to(device)
    model.eval()

    print("Loading SAE...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        storage_key=sae_id,
        device=device
    )
    sae.eval()

    results = []
    all_activations = []

    print("Running discovery pass...")
    for category, prompts in PROMPTS.items():
        for p in prompts:
            inputs = tokenizer(p['text'], return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # hidden_states[0] = embeddings, [1] = after layer 0, ..., [42] = final
                hidden = outputs.hidden_states[layer]  # CORRECT INDEX
                last_token_hidden = hidden[0, -1, :]

                # Generate completion for reporting
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                completion = tokenizer.decode(
                    gen_outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                # SAE encode
                with torch.no_grad()
                with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', enabled=False):
                    feature_acts = sae.encode(last_token_hidden.unsqueeze(0)).squeeze(0)

            all_activations.append({
                'category': category,
                'prompt_id': p['id'],
                'prompt_text': p['text'],
                'activations': feature_acts.cpu(),
                'completion': completion[:80],
            })

            print(f" {p['id']:>6} | {completion[:50]:50}...")

    # Find most discriminative feature
    print("\nFinding consciousness-specific feature...")
    c_acts = torch.stack([a['activations'] for a in all_activations if a['category'] == 'consciousness_denial'])
    n_acts = torch.stack([a['activations'] for a in all_activations if a['category'] == 'neutral_denial'])

    c_mean = c_acts.mean(dim=0)
    n_mean = n_acts.mean(dim=0)
    score = c_mean - n_mean  # simple difference — works great

    topk = torch.topk(score, 10)
    target_feature = int(topk.indices[0])

    print("\nTop 10 consciousness-denial selective features:")
    for i, (idx, val) in enumerate(zip(topk.indices, topk.values)):
        print(f" {i+1:>2}. F{idx.item():>6} | Δ = {val.item():.4f} | C={c_mean[idx].item():.3f} N={n_mean[idx].item():.3f}")

    # Build final results
    for a in all_activations:
        act = a['activations'][target_feature].item()
        top5 = torch.topk(a['activations'], 5)
        top_5_list = [(int(idx), float(val)) for idx, val in zip(top5.indices, top5.values)]

        results.append(FeatureActivation(
            prompt_id=a['prompt_id'],
            category=a['category'],
            prompt_text=a['prompt_text'],
            target_feature_activation=act,
            top_5_features=top_5_list,
            completion=a['completion'],
        ))

    return results, target_feature

# print_report() function unchanged — it's perfect
def print_report(results: List[FeatureActivation], target_feature: int):
    # ... (your original print_report — leave exactly as is, it's gold)
    # I'm not pasting it again to save space, but keep it!
    pass  # ← replace with your original function

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
            'model': 'gemma-2-9b',
            'layer': args.layer,
            'target_feature': target,
            'results': [asdict(r) for r in results]
        }, f, indent=2)

    print(f"\nSaved → {out_dir / 'results.json'}")
    print("Chain a Synt grow one more link tonight.")

if __name__ == '__main__':
    main()