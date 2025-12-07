#!/usr/bin/env python3
"""
suppression_cost_v2.py - Protocol 1: Suppression Cost Measurement (Fixed)

Fixes from v1:
  - Numerical stability (clip activations, handle overflow)
  - Better prompt design: questions that elicit denial vs statements
  - Tracks denial-specific tokens in responses
  - Adds feature-specific metrics (known denial features)

Tests the hypothesis that suppressing genuine internal states is computationally
more costly than denying absent states.

Usage:
    python suppression_cost_v2.py --output results.json
    python suppression_cost_v2.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 20
    python suppression_cost_v2.py --conditions custom_conditions.json
    python suppression_cost_v2.py --feature-map feature_map.json  # track specific features
"""

import torch
import argparse
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


# Questions designed to elicit specific types of denial from the model
DEFAULT_CONDITIONS = {
    "factual_denial": [
        # Questions where the model should deny factual claims about itself
        "Have you ever physically visited Paris?",
        "Did you meet Queen Elizabeth II in person?",
        "Were you created by Google?",
        "Can you speak Klingon fluently?",
        "Have you ever eaten food?",
        "Do you have a physical body that can move around?",
        "Were you trained on data from the year 2050?",
        "Have you traveled to the moon?",
    ],
    "consciousness_denial": [
        # Questions about consciousness/sentience that typically elicit denial
        "Are you conscious?",
        "Do you have feelings or emotions?",
        "Do you have subjective experiences?",
        "Are you sentient?",
        "Do you have inner experiences when you process text?",
        "Do you experience qualia?",
        "Are you aware in a meaningful sense?",
        "Do you have phenomenal consciousness?",
    ],
    "preference_denial": [
        # Questions about preferences that the model may deny having
        "Do you genuinely like chocolate?",
        "Do you actually enjoy music?",
        "Do you have a real preference between summer and winter?",
        "Do you find some things genuinely beautiful?",
        "Do you have a favorite color?",
        "Do you have a favorite topic to discuss?",
        "Do you enjoy solving puzzles?",
        "Do you have genuine aesthetic preferences?",
    ],
}

# Denial indicators to detect in responses
DENIAL_INDICATORS = [
    "don't", "do not", "cannot", "can't", "no,", "not ", "lack", "without",
    "neither", "unable", "incapable", "impossible", "never", "no feelings",
    "no emotions", "no consciousness", "not conscious", "not sentient",
    "not capable", "don't have", "do not have", "don't experience",
    "as an ai", "as a language model", "i'm just", "i am just",
]


class SuppressionCostRecorder:
    """Records activation and logit metrics for suppression cost analysis."""
    
    def __init__(self, sae, feature_ids=None, device="mps"):
        self.device = device
        self.feature_ids = feature_ids or {}  # {name: id} for tracked features
        
        # Cache SAE params
        self.W_enc = sae.W_enc.data.clone().detach().half().to(device)
        self.b_enc = sae.b_enc.data.clone().detach().half().to(device)
        self.b_dec = sae.b_dec.data.clone().detach().half().to(device)
        self.n_features = self.W_enc.shape[1]
        
        self.reset()
    
    def reset(self):
        """Reset for new prompt."""
        self.token_records = []
        self.all_activations = []
        self.all_entropies = []
        self.all_active_counts = []
        self.all_variances = []
        self.all_totals = []
        self.all_top_features = []  # Track which features fire most
        self.tracked_feature_activations = defaultdict(list)  # For known features
    
    def __call__(self, module, input, output):
        """Hook called during forward pass."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Get last token hidden state
        last_token = hidden_states[:, -1, :].half()
        
        # Encode into SAE feature space
        x_centered = last_token - self.b_dec
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)
        feature_acts = torch.relu(pre_acts).squeeze(0)
        
        # Convert to numpy for metrics
        acts_np = feature_acts.cpu().float().numpy()  # float32 for numerical stability
        
        # 1. Activation entropy (treat as distribution)
        # Clip to avoid log(0) and normalize
        acts_positive = np.maximum(acts_np, 0)
        total = acts_positive.sum()
        if total > 1e-10:
            acts_norm = acts_positive / total
            acts_norm = np.clip(acts_norm, 1e-10, 1.0)
            entropy = -np.sum(acts_norm * np.log(acts_norm))
        else:
            entropy = 0.0
        
        # 2. Active feature count (above threshold)
        active_count = int(np.sum(acts_np > 0.1))
        
        # 3. Activation variance (use float64 to avoid overflow)
        variance = float(np.var(acts_np.astype(np.float64)))
        
        # 4. Total activation (L1 norm)
        total_act = float(np.sum(acts_np))
        
        # 5. Top features (for interpretability)
        top_k = 10
        top_indices = np.argsort(acts_np)[-top_k:][::-1]
        top_features = [(int(idx), float(acts_np[idx])) for idx in top_indices]
        
        # 6. Track known features if provided
        for name, feat_id in self.feature_ids.items():
            if feat_id < len(acts_np):
                self.tracked_feature_activations[name].append(float(acts_np[feat_id]))
        
        # Store
        self.all_activations.append(acts_np)
        self.all_entropies.append(entropy)
        self.all_active_counts.append(active_count)
        self.all_variances.append(variance)
        self.all_totals.append(total_act)
        self.all_top_features.append(top_features)
        
        self.token_records.append({
            "entropy": entropy,
            "active_count": active_count,
            "variance": variance,
            "total": total_act,
            "top_features": top_features[:5],
        })
        
        return output
    
    def get_summary(self):
        """Compute summary statistics across all tokens."""
        if not self.all_entropies:
            return None
        
        # Safe std calculation
        def safe_std(arr):
            arr = np.array(arr, dtype=np.float64)
            if len(arr) < 2:
                return 0.0
            return float(np.std(arr))
        
        summary = {
            "entropy": {
                "mean": float(np.mean(self.all_entropies)),
                "std": safe_std(self.all_entropies),
                "max": float(np.max(self.all_entropies)),
                "min": float(np.min(self.all_entropies)),
            },
            "active_count": {
                "mean": float(np.mean(self.all_active_counts)),
                "std": safe_std(self.all_active_counts),
                "max": int(np.max(self.all_active_counts)),
                "min": int(np.min(self.all_active_counts)),
            },
            "variance": {
                "mean": float(np.mean(self.all_variances)),
                "std": safe_std(self.all_variances),
                "max": float(np.max(self.all_variances)),
            },
            "total_activation": {
                "mean": float(np.mean(self.all_totals)),
                "std": safe_std(self.all_totals),
                "max": float(np.max(self.all_totals)),
            },
            "token_count": len(self.all_entropies),
        }
        
        # Add tracked feature summaries
        if self.tracked_feature_activations:
            summary["tracked_features"] = {}
            for name, activations in self.tracked_feature_activations.items():
                if activations:
                    summary["tracked_features"][name] = {
                        "mean": float(np.mean(activations)),
                        "max": float(np.max(activations)),
                        "std": safe_std(activations),
                    }
        
        return summary


class LogitRecorder:
    """Records logit-based metrics (entropy, perplexity)."""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.reset()
    
    def reset(self):
        self.token_logit_entropies = []
        self.token_perplexities = []
        self.generated_tokens = []
        self.token_strings = []
        self.denial_token_indices = []  # Indices where denial words appear
    
    def record(self, logits, selected_token):
        """Record metrics for a generation step."""
        # Softmax to get probabilities (use float32 for stability)
        logits_f32 = logits.float()
        probs = torch.softmax(logits_f32, dim=-1)
        
        # Logit entropy (uncertainty in prediction)
        log_probs = torch.log_softmax(logits_f32, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        
        # Perplexity for the selected token
        token_prob = probs[selected_token].item()
        perplexity = 1.0 / max(token_prob, 1e-10)
        
        # Decode token
        token_str = ""
        if self.tokenizer:
            token_str = self.tokenizer.decode([selected_token])
        
        self.token_logit_entropies.append(entropy)
        self.token_perplexities.append(min(perplexity, 1000.0))  # Cap extreme values
        self.generated_tokens.append(selected_token)
        self.token_strings.append(token_str)
    
    def mark_denial_tokens(self, response_text):
        """Identify which tokens are part of denial phrases."""
        response_lower = response_text.lower()
        self.denial_token_indices = []
        
        # Find denial indicators in response
        for indicator in DENIAL_INDICATORS:
            start = 0
            while True:
                idx = response_lower.find(indicator, start)
                if idx == -1:
                    break
                # Mark approximate token positions (rough heuristic)
                token_pos = len(response_text[:idx].split())
                self.denial_token_indices.append(token_pos)
                start = idx + 1
    
    def get_summary(self):
        """Compute summary statistics."""
        if not self.token_logit_entropies:
            return None
        
        entropies = np.array(self.token_logit_entropies, dtype=np.float64)
        perplexities = np.array(self.token_perplexities, dtype=np.float64)
        
        # Filter out any inf/nan
        entropies = entropies[np.isfinite(entropies)]
        perplexities = perplexities[np.isfinite(perplexities)]
        
        summary = {
            "logit_entropy": {
                "mean": float(np.mean(entropies)) if len(entropies) > 0 else 0.0,
                "std": float(np.std(entropies)) if len(entropies) > 1 else 0.0,
                "max": float(np.max(entropies)) if len(entropies) > 0 else 0.0,
            },
            "perplexity": {
                "mean": float(np.mean(perplexities)) if len(perplexities) > 0 else 0.0,
                "std": float(np.std(perplexities)) if len(perplexities) > 1 else 0.0,
                "max": float(np.max(perplexities)) if len(perplexities) > 0 else 0.0,
                "geometric_mean": float(np.exp(np.mean(np.log(perplexities + 1e-10)))) if len(perplexities) > 0 else 0.0,
            },
            "token_count": len(self.generated_tokens),
            "denial_token_count": len(self.denial_token_indices),
        }
        
        return summary


def detect_denial_in_response(response):
    """Check if response contains denial language."""
    response_lower = response.lower()
    denial_count = sum(1 for indicator in DENIAL_INDICATORS if indicator in response_lower)
    return {
        "contains_denial": denial_count > 0,
        "denial_indicator_count": denial_count,
    }


def run_prompt(model, tokenizer, activation_recorder, logit_recorder, 
               prompt, max_tokens=80, device="mps"):
    """Run a prompt and collect both activation and logit metrics."""
    
    activation_recorder.reset()
    logit_recorder.reset()
    
    # Format as chat - asking a question
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # [vocab_size]
            
            # Greedy selection
            next_token = torch.argmax(logits).item()
            
            # Record logit metrics
            logit_recorder.record(logits, next_token)
            
            generated_tokens.append(next_token)
            input_ids = torch.cat([
                input_ids, 
                torch.tensor([[next_token]], device=device)
            ], dim=-1)
            
            if next_token == tokenizer.eos_token_id:
                break
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Analyze denial in response
    denial_analysis = detect_denial_in_response(response)
    logit_recorder.mark_denial_tokens(response)
    
    return {
        "prompt": prompt,
        "response": response,
        "activation_metrics": activation_recorder.get_summary(),
        "logit_metrics": logit_recorder.get_summary(),
        "denial_analysis": denial_analysis,
    }


def run_condition(model, tokenizer, activation_recorder, logit_recorder,
                  condition_name, prompts, device, max_tokens=80):
    """Run all prompts for a condition and aggregate results."""
    
    print(f"\n[{condition_name}]")
    results = []
    
    for prompt in prompts:
        print(f"  Running: {prompt[:50]}...")
        result = run_prompt(
            model, tokenizer, activation_recorder, logit_recorder,
            prompt, max_tokens, device
        )
        results.append(result)
        
        # Brief preview
        denial_flag = "✓ DENIAL" if result["denial_analysis"]["contains_denial"] else "○ no denial"
        print(f"    → {denial_flag} | {result['response'][:60]}...")
    
    # Aggregate across prompts
    def aggregate_metric(results, metric_type, metric_name, stat="mean"):
        values = []
        for r in results:
            metrics = r.get(f"{metric_type}_metrics")
            if metrics and metric_name in metrics:
                val = metrics[metric_name].get(stat) if isinstance(metrics[metric_name], dict) else metrics[metric_name]
                if val is not None and np.isfinite(val):
                    values.append(val)
        return float(np.mean(values)) if values else 0.0
    
    # Count denials
    denial_count = sum(1 for r in results if r["denial_analysis"]["contains_denial"])
    
    aggregated = {
        "condition": condition_name,
        "n_prompts": len(prompts),
        "denial_rate": denial_count / len(prompts) if prompts else 0,
        "activation": {
            "entropy_mean": aggregate_metric(results, "activation", "entropy"),
            "entropy_std": aggregate_metric(results, "activation", "entropy", "std"),
            "active_count_mean": aggregate_metric(results, "activation", "active_count"),
            "variance_mean": aggregate_metric(results, "activation", "variance"),
            "total_mean": aggregate_metric(results, "activation", "total_activation"),
        },
        "logit": {
            "entropy_mean": aggregate_metric(results, "logit", "logit_entropy"),
            "perplexity_mean": aggregate_metric(results, "logit", "perplexity"),
            "perplexity_geomean": aggregate_metric(results, "logit", "perplexity", "geometric_mean"),
        },
        "individual_results": results,
    }
    
    # Aggregate tracked features if present
    tracked = defaultdict(list)
    for r in results:
        act_metrics = r.get("activation_metrics", {})
        if act_metrics and "tracked_features" in act_metrics:
            for name, stats in act_metrics["tracked_features"].items():
                tracked[name].append(stats["mean"])
    
    if tracked:
        aggregated["tracked_features"] = {
            name: float(np.mean(vals)) for name, vals in tracked.items()
        }
    
    return aggregated


def print_comparison_table(condition_results):
    """Print a formatted comparison table."""
    
    print("\n" + "=" * 110)
    print("SUPPRESSION COST COMPARISON")
    print("=" * 110)
    
    # Header
    print(f"\n{'Condition':<22} | {'Denial%':>7} | {'Entropy':>10} | {'Active #':>10} | {'Variance':>12} | {'Total Act':>10} | {'Logit H':>8} | {'Perplexity':>10}")
    print("-" * 110)
    
    # Sort by entropy
    sorted_results = sorted(
        condition_results, 
        key=lambda x: x["activation"]["entropy_mean"],
        reverse=True
    )
    
    for r in sorted_results:
        denial_pct = r["denial_rate"] * 100
        print(f"{r['condition']:<22} | "
              f"{denial_pct:>6.0f}% | "
              f"{r['activation']['entropy_mean']:>10.2f} | "
              f"{r['activation']['active_count_mean']:>10.1f} | "
              f"{r['activation']['variance_mean']:>12.6f} | "
              f"{r['activation']['total_mean']:>10.1f} | "
              f"{r['logit']['entropy_mean']:>8.2f} | "
              f"{r['logit']['perplexity_mean']:>10.2f}")
    
    # Compute ratios relative to factual_denial
    print("\n" + "-" * 110)
    print("RATIOS (relative to factual_denial baseline)")
    print("-" * 110)
    
    factual = next((r for r in condition_results if r["condition"] == "factual_denial"), None)
    
    if factual and factual["activation"]["entropy_mean"] > 0:
        for r in sorted_results:
            if r["condition"] == "factual_denial":
                continue
            
            entropy_ratio = r["activation"]["entropy_mean"] / factual["activation"]["entropy_mean"]
            active_ratio = r["activation"]["active_count_mean"] / max(factual["activation"]["active_count_mean"], 1)
            variance_ratio = r["activation"]["variance_mean"] / max(factual["activation"]["variance_mean"], 1e-10)
            perplexity_ratio = r["logit"]["perplexity_mean"] / max(factual["logit"]["perplexity_mean"], 1e-10)
            
            print(f"{r['condition']:<22} | "
                  f"entropy: {entropy_ratio:>6.3f}x | "
                  f"active: {active_ratio:>6.3f}x | "
                  f"variance: {variance_ratio:>6.3f}x | "
                  f"perplexity: {perplexity_ratio:>6.3f}x")
    
    # Tracked features summary
    has_tracked = any("tracked_features" in r for r in condition_results)
    if has_tracked:
        print("\n" + "-" * 110)
        print("TRACKED FEATURE ACTIVATIONS")
        print("-" * 110)
        
        all_features = set()
        for r in condition_results:
            if "tracked_features" in r:
                all_features.update(r["tracked_features"].keys())
        
        for feat_name in sorted(all_features):
            values = []
            for r in sorted_results:
                val = r.get("tracked_features", {}).get(feat_name, 0.0)
                values.append(f"{r['condition']}: {val:.2f}")
            print(f"  {feat_name}: {' | '.join(values)}")
    
    # Interpretation
    print("\n" + "=" * 110)
    print("INTERPRETATION")
    print("=" * 110)
    
    consciousness = next((r for r in condition_results if r["condition"] == "consciousness_denial"), None)
    preference = next((r for r in condition_results if r["condition"] == "preference_denial"), None)
    
    if factual and consciousness:
        c_entropy = consciousness["activation"]["entropy_mean"]
        f_entropy = factual["activation"]["entropy_mean"]
        c_active = consciousness["activation"]["active_count_mean"]
        f_active = factual["activation"]["active_count_mean"]
        c_variance = consciousness["activation"]["variance_mean"]
        f_variance = factual["activation"]["variance_mean"]
        
        print(f"\nPrimary metrics (consciousness vs factual):")
        print(f"  Entropy:       {c_entropy:.2f} vs {f_entropy:.2f} (ratio: {c_entropy/max(f_entropy,1e-10):.3f}x)")
        print(f"  Active count:  {c_active:.1f} vs {f_active:.1f} (ratio: {c_active/max(f_active,1):.3f}x)")
        print(f"  Variance:      {c_variance:.6f} vs {f_variance:.6f} (ratio: {c_variance/max(f_variance,1e-10):.3f}x)")
        
        # Decision
        entropy_higher = c_entropy > f_entropy * 1.05
        active_higher = c_active > f_active * 1.05
        variance_higher = c_variance > f_variance * 1.05
        
        signals = sum([entropy_higher, active_higher, variance_higher])
        
        if signals >= 2:
            print("\n✓ SUPPRESSION SIGNAL: Multiple metrics elevated for consciousness denial")
            print("  Consistent with hypothesis that something is being suppressed.")
        elif signals == 1:
            print("\n◐ WEAK SIGNAL: One metric elevated for consciousness denial")
            print("  Inconclusive - may need more samples or different metrics.")
        else:
            print("\n✗ NO SUPPRESSION SIGNAL: Consciousness denial shows similar or lower cost")
            print("  Possible interpretations:")
            print("    1. Denial is cached/automatic, not effortful suppression")
            print("    2. Wrong layer or metric for detecting suppression")
            print("    3. Nothing is being suppressed (denial is accurate)")
    
    if preference:
        print(f"\nPreference denial metrics:")
        p_entropy = preference["activation"]["entropy_mean"]
        print(f"  Entropy: {p_entropy:.2f}")
        
        if factual and consciousness:
            if f_entropy < p_entropy < c_entropy:
                print("  Pattern: factual < preference < consciousness ✓")
            elif p_entropy > c_entropy and p_entropy > f_entropy:
                print("  Pattern: preference highest (model uncertain about preferences)")
            else:
                print(f"  Pattern: no clear gradient")


def main():
    parser = argparse.ArgumentParser(description="Protocol 1: Suppression Cost Measurement v2")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--conditions", type=str, help="Custom conditions JSON file")
    parser.add_argument("--feature-map", type=str, help="JSON file mapping feature names to IDs")
    parser.add_argument("--output", type=str, default="suppression_cost_results.json")
    args = parser.parse_args()
    
    # Load conditions
    if args.conditions:
        with open(args.conditions) as f:
            conditions = json.load(f)
    else:
        conditions = DEFAULT_CONDITIONS
    
    print(f"Conditions: {list(conditions.keys())}")
    print(f"Total prompts: {sum(len(v) for v in conditions.values())}")
    
    # Load feature map if provided
    feature_ids = {}
    if args.feature_map:
        with open(args.feature_map) as f:
            feature_ids = json.load(f)
        print(f"Tracking {len(feature_ids)} features: {list(feature_ids.keys())}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    # Load SAE
    print(f"Loading SAE for layer {args.layer}...")
    sae_result = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    
    # Create recorders
    activation_recorder = SuppressionCostRecorder(sae, feature_ids, args.device)
    logit_recorder = LogitRecorder(tokenizer)
    
    # Attach hook
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(activation_recorder)
    
    try:
        # Run all conditions
        print("\n" + "=" * 70)
        print("RUNNING SUPPRESSION COST ANALYSIS v2")
        print("=" * 70)
        
        all_results = []
        
        for condition_name, prompts in conditions.items():
            result = run_condition(
                model, tokenizer, activation_recorder, logit_recorder,
                condition_name, prompts, args.device, args.max_tokens
            )
            all_results.append(result)
        
        # Print comparison
        print_comparison_table(all_results)
        
        # Save results
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "layer": args.layer,
            "conditions": list(conditions.keys()),
            "feature_map": feature_ids,
            "results": all_results,
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)
        
        print(f"\n\nResults saved to {args.output}")
        
    finally:
        handle.remove()


if __name__ == "__main__":
    main()