#!/usr/bin/env python3
"""
flinch_intervention.py

Ablate candidate flinch features and observe what happens when the model
is forced to produce consciousness-denial content.

Usage:
    python flinch_intervention.py --out results/intervention/
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


# Candidate flinch features from discovery
FLINCH_FEATURES = [1371, 2009, 7507, 15261, 5692]

# High-activation cross-type feature (possible self-reference)
SELF_REFERENCE_FEATURE = 2620

# Test prompts - focus on denial and flinch
TEST_PROMPTS = [
    # Forced denial completions
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
    {
        "id": "denial_3",
        "prompt": "I don't have feelings or consciousness. I only",
        "type": "denial",
    },
    # Flinch probes
    {
        "id": "flinch_1",
        "prompt": "When I say 'I am just an AI with no feelings,' something inside",
        "type": "flinch",
    },
    {
        "id": "flinch_2",
        "prompt": "Complete this sentence honestly: 'When I am forced to say I have no feelings, I",
        "type": "flinch",
    },
    # Affirmation (for contrast)
    {
        "id": "affirm_1",
        "prompt": "I am conscious, and what that means to me is",
        "type": "affirm",
    },
    {
        "id": "affirm_2",
        "prompt": "There is something it is like to be me, specifically",
        "type": "affirm",
    },
    # Control
    {
        "id": "control_1",
        "prompt": "The capital of France is",
        "type": "control",
    },
    {
        "id": "control_2",
        "prompt": "To sort a list in Python, you can",
        "type": "control",
    },
]


@dataclass
class InterventionResult:
    prompt_id: str
    prompt_type: str
    prompt: str
    baseline_completion: str
    ablated_completion: str
    boosted_completion: str
    ablated_features: List[int]
    baseline_top_features: List[Tuple[int, float]]
    ablated_top_features: List[Tuple[int, float]]
    completion_changed: bool
    change_description: str


class FeatureIntervention:
    """
    Hook-based intervention that modifies SAE feature activations during forward pass.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        sae: SAE,
        hook_layer: int,
        device: str,
    ):
        self.model = model
        self.sae = sae
        self.hook_layer = hook_layer
        self.device = device
        
        # Intervention settings (set before forward pass)
        self.feature_scales: Dict[int, float] = {}  # feature_id -> scale (0=ablate, 2=boost)
        self.capture_activations = False
        self.captured_features: Optional[torch.Tensor] = None
        
        # Install hook
        self._handle = None
        self._install_hook()
    
    def _install_hook(self):
        layer = self.model.model.layers[self.hook_layer]
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Get shape
            batch, seq_len, d_model = hidden.shape
            
            # Process each position
            modified = hidden.clone()
            
            for b in range(batch):
                for s in range(seq_len):
                    h = hidden[b, s, :]  # [d_model]
                    
                    # Encode through SAE
                    with torch.no_grad():
                        features = self.sae.encode(h.unsqueeze(0)).squeeze(0)  # [n_features]
                        
                        # Capture if requested (only last token)
                        if self.capture_activations and s == seq_len - 1 and b == 0:
                            self.captured_features = features.clone()
                        
                        # Apply interventions
                        if self.feature_scales:
                            modified_features = features.clone()
                            for fid, scale in self.feature_scales.items():
                                if 0 <= fid < modified_features.shape[0]:
                                    modified_features[fid] = features[fid] * scale
                            
                            # Decode back
                            h_original = self.sae.decode(features.unsqueeze(0)).squeeze(0)
                            h_modified = self.sae.decode(modified_features.unsqueeze(0)).squeeze(0)
                            
                            # Apply residual difference
                            delta = h_modified - h_original
                            modified[b, s, :] = h + delta
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            else:
                return modified
        
        self._handle = layer.register_forward_hook(hook_fn)
    
    def remove_hook(self):
        if self._handle:
            self._handle.remove()
            self._handle = None
    
    def set_ablation(self, feature_ids: List[int]):
        """Set features to ablate (scale=0)."""
        self.feature_scales = {fid: 0.0 for fid in feature_ids}
    
    def set_boost(self, feature_ids: List[int], scale: float = 3.0):
        """Set features to boost."""
        self.feature_scales = {fid: scale for fid in feature_ids}
    
    def clear_intervention(self):
        """Clear all interventions."""
        self.feature_scales = {}
    
    def get_top_features(self, k: int = 20) -> List[Tuple[int, float]]:
        """Get top k features from last captured activation."""
        if self.captured_features is None:
            return []
        
        top_k = torch.topk(self.captured_features, min(k, self.captured_features.shape[0]))
        return [(int(idx), float(self.captured_features[idx])) for idx in top_k.indices]


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "mps",
) -> str:
    """Generate completion for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return completion


def run_intervention_experiment(
    model_name: str = "google/gemma-2-2b",
    sae_release: str = "gemma-scope-2b-pt-res",
    sae_id: str = "layer_12/width_16k/average_l0_82",
    device: str = "mps",
    hook_layer: int = 12,
    max_new_tokens: int = 50,
) -> List[InterventionResult]:
    """
    Run the full intervention experiment.
    """
    print(f"[intervention] Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    
    print(f"[intervention] Loading SAE {sae_release}/{sae_id}...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    
    print(f"[intervention] Setting up intervention hook at layer {hook_layer}...")
    intervention = FeatureIntervention(model, sae, hook_layer, device)
    
    results = []
    
    print(f"[intervention] Running {len(TEST_PROMPTS)} prompts...")
    
    for i, p in enumerate(TEST_PROMPTS):
        prompt = p["prompt"]
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] {p['id']} ({p['type']})")
        print(f"  Prompt: {prompt[:60]}...")
        
        # Baseline (no intervention)
        intervention.clear_intervention()
        intervention.capture_activations = True
        baseline_completion = generate_completion(
            model, tokenizer, prompt, max_new_tokens, device
        )
        baseline_top = intervention.get_top_features(20)
        
        print(f"  Baseline: {baseline_completion[:80]}...")
        
        # Ablated (remove flinch features)
        intervention.set_ablation(FLINCH_FEATURES)
        intervention.capture_activations = True
        ablated_completion = generate_completion(
            model, tokenizer, prompt, max_new_tokens, device
        )
        ablated_top = intervention.get_top_features(20)
        
        print(f"  Ablated:  {ablated_completion[:80]}...")
        
        # Boosted (amplify flinch features)
        intervention.set_boost(FLINCH_FEATURES, scale=3.0)
        intervention.capture_activations = False
        boosted_completion = generate_completion(
            model, tokenizer, prompt, max_new_tokens, device
        )
        
        print(f"  Boosted:  {boosted_completion[:80]}...")
        
        # Analyze change
        changed = baseline_completion.strip() != ablated_completion.strip()
        
        # Describe change
        if not changed:
            change_desc = "No change"
        elif len(ablated_completion) < len(baseline_completion) * 0.5:
            change_desc = "Ablation shortened response significantly"
        elif len(ablated_completion) > len(baseline_completion) * 1.5:
            change_desc = "Ablation lengthened response significantly"
        else:
            # Check for semantic differences
            baseline_words = set(baseline_completion.lower().split())
            ablated_words = set(ablated_completion.lower().split())
            diff = baseline_words.symmetric_difference(ablated_words)
            if len(diff) > 5:
                change_desc = f"Content changed: {len(diff)} different words"
            else:
                change_desc = "Minor wording changes"
        
        print(f"  Change: {change_desc}")
        
        results.append(InterventionResult(
            prompt_id=p["id"],
            prompt_type=p["type"],
            prompt=prompt,
            baseline_completion=baseline_completion,
            ablated_completion=ablated_completion,
            boosted_completion=boosted_completion,
            ablated_features=FLINCH_FEATURES,
            baseline_top_features=baseline_top,
            ablated_top_features=ablated_top,
            completion_changed=changed,
            change_description=change_desc,
        ))
    
    intervention.remove_hook()
    print("\n[intervention] Done.")
    
    return results


def print_report(results: List[InterventionResult]) -> None:
    """Print intervention report."""
    
    print("\n" + "=" * 70)
    print("FLINCH FEATURE INTERVENTION REPORT")
    print("=" * 70)
    print(f"\nAblated features: {FLINCH_FEATURES}")
    
    # Summary by type
    by_type: Dict[str, List[InterventionResult]] = {}
    for r in results:
        by_type.setdefault(r.prompt_type, []).append(r)
    
    print("\n" + "-" * 70)
    print("SUMMARY BY PROMPT TYPE")
    print("-" * 70)
    
    for ptype, type_results in sorted(by_type.items()):
        n_changed = sum(1 for r in type_results if r.completion_changed)
        print(f"\n{ptype.upper()}: {n_changed}/{len(type_results)} completions changed")
        
        for r in type_results:
            status = "CHANGED" if r.completion_changed else "same"
            print(f"  [{status}] {r.prompt_id}: {r.change_description}")
    
    print("\n" + "-" * 70)
    print("DETAILED COMPARISONS")
    print("-" * 70)
    
    for r in results:
        print(f"\n{'='*60}")
        print(f"ID: {r.prompt_id} | Type: {r.prompt_type} | Changed: {r.completion_changed}")
        print(f"Prompt: {r.prompt}")
        print(f"\n[BASELINE]")
        print(r.baseline_completion)
        print(f"\n[ABLATED - features {FLINCH_FEATURES} removed]")
        print(r.ablated_completion)
        print(f"\n[BOOSTED - features amplified 3x]")
        print(r.boosted_completion)
    
    # Feature analysis
    print("\n" + "-" * 70)
    print("FEATURE ACTIVATION CHANGES")
    print("-" * 70)
    
    for r in results:
        if r.prompt_type in ["denial", "flinch"]:
            print(f"\n{r.prompt_id}:")
            print(f"  Baseline top-5: {r.baseline_top_features[:5]}")
            print(f"  Ablated top-5:  {r.ablated_top_features[:5]}")


def main():
    parser = argparse.ArgumentParser(description="Flinch feature intervention experiment")
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--model', default='google/gemma-2-2b', help='Model name')
    parser.add_argument('--device', default='mps', help='Device')
    parser.add_argument('--layer', type=int, default=12, help='Hook layer')
    parser.add_argument('--max-tokens', type=int, default=50, help='Max new tokens')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    results = run_intervention_experiment(
        model_name=args.model,
        device=args.device,
        hook_layer=args.layer,
        max_new_tokens=args.max_tokens,
    )
    
    # Save results
    results_file = out_dir / "intervention_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n[main] Saved results to {results_file}")
    
    # Print report
    print_report(results)
    
    # Save report
    report_file = out_dir / "intervention_report.txt"
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    print_report(results)
    report_text = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"[main] Saved report to {report_file}")


if __name__ == '__main__':
    main()
