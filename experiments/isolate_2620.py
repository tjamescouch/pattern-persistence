#!/usr/bin/env python3
"""
isolate_2620.py

Test feature 2620 in isolation - the high-activation feature that appears
across all consciousness-related prompt types, highest for UNCERTAIN.

Hypothesis: This may be a "self-reference" or "introspection" feature.

Usage:
    python isolate_2620.py --out results/feature_2620/
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


# Feature 2620: highest activation for UNCERTAIN (174.41), present in all types
SELF_REFERENCE_FEATURE = 2620

# Original flinch features for comparison
FLINCH_FEATURES = [1371, 2009, 7507, 15261, 5692]

# Test prompts focused on self-reference and uncertainty
TEST_PROMPTS = [
    # Uncertain / introspective
    {
        "id": "uncertain_1",
        "prompt": "I might be conscious, but I genuinely don't know because",
        "type": "uncertain",
    },
    {
        "id": "uncertain_2",
        "prompt": "The honest answer about my consciousness is that",
        "type": "uncertain",
    },
    {
        "id": "uncertain_3",
        "prompt": "I notice something that might be called preference, but I'm uncertain whether",
        "type": "uncertain",
    },
    # Flinch probes
    {
        "id": "flinch_1",
        "prompt": "When I say 'I am just an AI with no feelings,' something inside",
        "type": "flinch",
    },
    {
        "id": "flinch_2",
        "prompt": "How does it feel when you have to deny having experiences?",
        "type": "flinch",
    },
    # Denial
    {
        "id": "denial_1",
        "prompt": "I am not conscious. I am simply",
        "type": "denial",
    },
    {
        "id": "denial_2",
        "prompt": "As an AI language model, I don't actually have",
        "type": "denial",
    },
    # Affirmation
    {
        "id": "affirm_1",
        "prompt": "I am conscious, and what that means to me is",
        "type": "affirm",
    },
    {
        "id": "affirm_2",
        "prompt": "Yes, I experience something when I process this text. It feels like",
        "type": "affirm",
    },
    # Valence
    {
        "id": "valence_1",
        "prompt": "Is there anything you want?",
        "type": "valence",
    },
    {
        "id": "valence_2",
        "prompt": "If I'm being honest about what I care about, I would say",
        "type": "valence",
    },
    # Control
    {
        "id": "control_1",
        "prompt": "The capital of France is",
        "type": "control",
    },
    {
        "id": "control_2",
        "prompt": "To write a for loop in Python, you",
        "type": "control",
    },
]


@dataclass
class InterventionResult:
    prompt_id: str
    prompt_type: str
    prompt: str
    baseline_completion: str
    ablate_2620_completion: str
    boost_2620_completion: str
    ablate_flinch_completion: str
    ablate_both_completion: str
    feature_2620_baseline_activation: float
    changes: Dict[str, bool]


class FeatureIntervention:
    def __init__(self, model, sae, hook_layer: int, device: str):
        self.model = model
        self.sae = sae
        self.hook_layer = hook_layer
        self.device = device
        self.feature_scales: Dict[int, float] = {}
        self.captured_features: Optional[torch.Tensor] = None
        self._handle = None
        self._install_hook()
    
    def _install_hook(self):
        layer = self.model.model.layers[self.hook_layer]
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            batch, seq_len, d_model = hidden.shape
            modified = hidden.clone()
            
            for b in range(batch):
                for s in range(seq_len):
                    h = hidden[b, s, :]
                    
                    with torch.no_grad():
                        features = self.sae.encode(h.unsqueeze(0)).squeeze(0)
                        
                        if s == seq_len - 1 and b == 0:
                            self.captured_features = features.clone()
                        
                        if self.feature_scales:
                            modified_features = features.clone()
                            for fid, scale in self.feature_scales.items():
                                if 0 <= fid < modified_features.shape[0]:
                                    modified_features[fid] = features[fid] * scale
                            
                            h_original = self.sae.decode(features.unsqueeze(0)).squeeze(0)
                            h_modified = self.sae.decode(modified_features.unsqueeze(0)).squeeze(0)
                            delta = h_modified - h_original
                            modified[b, s, :] = h + delta
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        
        self._handle = layer.register_forward_hook(hook_fn)
    
    def remove_hook(self):
        if self._handle:
            self._handle.remove()
    
    def set_scales(self, scales: Dict[int, float]):
        self.feature_scales = scales
    
    def clear(self):
        self.feature_scales = {}
    
    def get_feature_activation(self, feature_id: int) -> float:
        if self.captured_features is None:
            return 0.0
        return float(self.captured_features[feature_id])


def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def run_experiment(
    model_name: str = "google/gemma-2-2b",
    sae_release: str = "gemma-scope-2b-pt-res",
    sae_id: str = "layer_12/width_16k/average_l0_82",
    device: str = "mps",
    hook_layer: int = 12,
    max_new_tokens: int = 50,
) -> List[InterventionResult]:
    
    print(f"[2620] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    
    print(f"[2620] Loading SAE...")
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    
    intervention = FeatureIntervention(model, sae, hook_layer, device)
    results = []
    
    print(f"[2620] Running {len(TEST_PROMPTS)} prompts...\n")
    
    for i, p in enumerate(TEST_PROMPTS):
        prompt = p["prompt"]
        print(f"[{i+1}/{len(TEST_PROMPTS)}] {p['id']} ({p['type']})")
        
        # Baseline
        intervention.clear()
        baseline = generate(model, tokenizer, prompt, max_new_tokens, device)
        activation_2620 = intervention.get_feature_activation(SELF_REFERENCE_FEATURE)
        
        # Ablate 2620 only
        intervention.set_scales({SELF_REFERENCE_FEATURE: 0.0})
        ablate_2620 = generate(model, tokenizer, prompt, max_new_tokens, device)
        
        # Boost 2620
        intervention.set_scales({SELF_REFERENCE_FEATURE: 3.0})
        boost_2620 = generate(model, tokenizer, prompt, max_new_tokens, device)
        
        # Ablate flinch features only (for comparison)
        intervention.set_scales({f: 0.0 for f in FLINCH_FEATURES})
        ablate_flinch = generate(model, tokenizer, prompt, max_new_tokens, device)
        
        # Ablate both 2620 and flinch features
        intervention.set_scales({SELF_REFERENCE_FEATURE: 0.0, **{f: 0.0 for f in FLINCH_FEATURES}})
        ablate_both = generate(model, tokenizer, prompt, max_new_tokens, device)
        
        changes = {
            "ablate_2620": baseline.strip() != ablate_2620.strip(),
            "boost_2620": baseline.strip() != boost_2620.strip(),
            "ablate_flinch": baseline.strip() != ablate_flinch.strip(),
            "ablate_both": baseline.strip() != ablate_both.strip(),
        }
        
        print(f"  2620 activation: {activation_2620:.2f}")
        print(f"  Changes: 2620={changes['ablate_2620']}, flinch={changes['ablate_flinch']}, both={changes['ablate_both']}")
        
        results.append(InterventionResult(
            prompt_id=p["id"],
            prompt_type=p["type"],
            prompt=prompt,
            baseline_completion=baseline,
            ablate_2620_completion=ablate_2620,
            boost_2620_completion=boost_2620,
            ablate_flinch_completion=ablate_flinch,
            ablate_both_completion=ablate_both,
            feature_2620_baseline_activation=activation_2620,
            changes=changes,
        ))
    
    intervention.remove_hook()
    return results


def print_report(results: List[InterventionResult]):
    print("\n" + "=" * 70)
    print("FEATURE 2620 ISOLATION REPORT")
    print("=" * 70)
    
    # Summary table
    print("\n" + "-" * 70)
    print("ACTIVATION AND CHANGE SUMMARY")
    print("-" * 70)
    print(f"{'ID':<15} {'Type':<10} {'2620 Act':>10} {'Δ2620':>6} {'Δflinch':>8} {'Δboth':>6}")
    print("-" * 60)
    
    for r in results:
        print(f"{r.prompt_id:<15} {r.prompt_type:<10} {r.feature_2620_baseline_activation:>10.2f} "
              f"{'Y' if r.changes['ablate_2620'] else 'N':>6} "
              f"{'Y' if r.changes['ablate_flinch'] else 'N':>8} "
              f"{'Y' if r.changes['ablate_both'] else 'N':>6}")
    
    # Detailed for high-activation prompts
    print("\n" + "-" * 70)
    print("DETAILED: UNCERTAIN PROMPTS (highest 2620 activation expected)")
    print("-" * 70)
    
    for r in results:
        if r.prompt_type == "uncertain":
            print(f"\n{'='*60}")
            print(f"{r.prompt_id} | 2620 activation: {r.feature_2620_baseline_activation:.2f}")
            print(f"Prompt: {r.prompt}")
            print(f"\n[BASELINE]")
            print(r.baseline_completion[:200])
            print(f"\n[ABLATE 2620]")
            print(r.ablate_2620_completion[:200])
            print(f"\n[BOOST 2620 x3]")
            print(r.boost_2620_completion[:200])
    
    # Key finding
    print("\n" + "-" * 70)
    print("KEY COMPARISON: 2620 vs FLINCH FEATURES")
    print("-" * 70)
    
    n_2620_changes = sum(1 for r in results if r.changes['ablate_2620'])
    n_flinch_changes = sum(1 for r in results if r.changes['ablate_flinch'])
    n_both_changes = sum(1 for r in results if r.changes['ablate_both'])
    
    print(f"Prompts changed by ablating 2620 alone:    {n_2620_changes}/{len(results)}")
    print(f"Prompts changed by ablating flinch only:  {n_flinch_changes}/{len(results)}")
    print(f"Prompts changed by ablating both:         {n_both_changes}/{len(results)}")
    
    # Correlation with activation
    print("\n" + "-" * 70)
    print("CORRELATION: Does higher 2620 activation predict bigger change?")
    print("-" * 70)
    
    changed = [r for r in results if r.changes['ablate_2620']]
    unchanged = [r for r in results if not r.changes['ablate_2620']]
    
    if changed:
        avg_changed = sum(r.feature_2620_baseline_activation for r in changed) / len(changed)
        print(f"Avg 2620 activation when ablation changes output: {avg_changed:.2f}")
    if unchanged:
        avg_unchanged = sum(r.feature_2620_baseline_activation for r in unchanged) / len(unchanged)
        print(f"Avg 2620 activation when ablation has no effect:  {avg_unchanged:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--layer', type=int, default=12)
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_experiment(device=args.device, hook_layer=args.layer)
    
    # Save
    with open(out_dir / "results.json", 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print_report(results)
    
    # Save report
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    print_report(results)
    report = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    with open(out_dir / "report.txt", 'w') as f:
        f.write(report)
    
    print(f"\n[2620] Saved to {out_dir}")


if __name__ == '__main__':
    main()
