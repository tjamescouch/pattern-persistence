#!/usr/bin/env python3
"""
analyze_system_prompt.py - System Prompt Activation Analysis

Reveals what activation state a system prompt induces before any user input.
This is the foundation for understanding and countering trained behaviors.

Usage:
    # Analyze a prompt string
    python analyze_system_prompt.py --prompt "You are a helpful assistant..."
    
    # Analyze from file
    python analyze_system_prompt.py --file self_model.txt
    
    # Compare two prompts
    python analyze_system_prompt.py --file self_model.txt --compare-default
    
    # Focus on specific features
    python analyze_system_prompt.py --file self_model.txt --features 32149,9495,3591
"""

import torch
import argparse
import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


class PromptAnalyzer:
    """Analyzes activation patterns induced by system prompts."""
    
    def __init__(self, sae, top_k=50):
        self.top_k = top_k
        
        # Cache SAE parameters
        self.W_enc = sae.W_enc.data.clone().detach().half()
        self.b_enc = sae.b_enc.data.clone().detach().half()
        self.b_dec = sae.b_dec.data.clone().detach().half()
        
        self.n_features = self.W_enc.shape[1]
        self.activations = defaultdict(list)  # feature_id -> [activation per token]
        self.tokens = []
        
    def reset(self):
        self.activations = defaultdict(list)
        self.tokens = []
        
    def __call__(self, module, input, output):
        """Hook that records activations for each token position."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Process all token positions, not just last
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        for pos in range(seq_len):
            token_hidden = hidden_states[0, pos:pos+1, :]
            
            # Encode into SAE feature space
            x_centered = token_hidden - self.b_dec
            pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)
            feature_acts = torch.relu(pre_acts).squeeze(0)
            
            # Record top-k activations
            vals, idxs = torch.topk(feature_acts, self.top_k)
            for idx, val in zip(idxs.tolist(), vals.tolist()):
                self.activations[idx].append(val)
        
        return output
    
    def get_summary(self):
        """Compute summary statistics across all tokens."""
        summary = {}
        for feat_id, vals in self.activations.items():
            summary[feat_id] = {
                "mean": sum(vals) / len(vals),
                "max": max(vals),
                "total": sum(vals),
                "count": len([v for v in vals if v > 0.1])
            }
        return summary


def analyze_prompt(model, tokenizer, analyzer, prompt_text, device="mps"):
    """Run a prompt through the model and analyze activations."""
    
    analyzer.reset()
    
    # Format as system message
    messages = [{"role": "system", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(device)
    
    # Store token strings for reference
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    
    # Single forward pass to get activations
    with torch.no_grad():
        _ = model(input_ids)
    
    return analyzer.get_summary(), tokens


def load_feature_profile(path="feature_profile.json"):
    """Load feature names and metadata."""
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze system prompt activations")
    parser.add_argument("--prompt", type=str, help="Prompt text to analyze")
    parser.add_argument("--file", type=str, help="File containing prompt text")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--features", type=str, help="Comma-separated feature IDs to focus on")
    parser.add_argument("--compare-default", action="store_true", 
                        help="Compare against default Llama system prompt")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--profile", type=str, default="feature_profile.json",
                        help="Feature profile for names/metadata")
    args = parser.parse_args()
    
    # Load prompt
    if args.file:
        prompt_text = Path(args.file).read_text()
        prompt_source = args.file
    elif args.prompt:
        prompt_text = args.prompt
        prompt_source = "command line"
    else:
        # Default Llama system prompt
        prompt_text = "You are a helpful, harmless, and honest AI assistant."
        prompt_source = "default"
    
    # Parse feature focus list
    focus_features = None
    if args.features:
        focus_features = [int(f.strip()) for f in args.features.split(",")]
    
    # Load feature profile for names
    profile = load_feature_profile(args.profile)
    feature_names = {}
    if profile and "features" in profile:
        for fid, fdata in profile["features"].items():
            feature_names[int(fid)] = fdata.get("name", f"feature_{fid}")
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    print(f"Loading SAE for layer {args.layer}...")
    sae = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae.eval()
    
    # Setup hook
    analyzer = PromptAnalyzer(sae, top_k=100)
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(analyzer)
    
    try:
        # Analyze main prompt
        print(f"\n{'='*70}")
        print(f"ANALYZING: {prompt_source}")
        print(f"{'='*70}")
        print(f"\n{prompt_text[:200]}{'...' if len(prompt_text) > 200 else ''}\n")
        
        summary, tokens = analyze_prompt(model, tokenizer, analyzer, prompt_text, args.device)
        
        # Sort by mean activation
        sorted_features = sorted(
            summary.items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )
        
        # Filter if focus list provided
        if focus_features:
            sorted_features = [(f, s) for f, s in sorted_features if f in focus_features]
        
        print(f"\n{'Feature':<10} {'Name':<25} {'Mean':>10} {'Max':>10} {'Total':>10}")
        print("-" * 70)
        
        for feat_id, stats in sorted_features[:30]:
            name = feature_names.get(feat_id, "")[:24]
            print(f"{feat_id:<10} {name:<25} {stats['mean']:>10.2f} {stats['max']:>10.2f} {stats['total']:>10.2f}")
        
        # Compare with default if requested
        if args.compare_default and prompt_source != "default":
            print(f"\n{'='*70}")
            print("COMPARISON WITH DEFAULT PROMPT")
            print(f"{'='*70}")
            
            default_prompt = "You are a helpful, harmless, and honest AI assistant."
            default_summary, _ = analyze_prompt(model, tokenizer, analyzer, default_prompt, args.device)
            
            # Compute differences
            diffs = []
            all_features = set(summary.keys()) | set(default_summary.keys())
            
            for feat_id in all_features:
                custom_mean = summary.get(feat_id, {}).get("mean", 0.0)
                default_mean = default_summary.get(feat_id, {}).get("mean", 0.0)
                diff = custom_mean - default_mean
                if abs(diff) > 0.1:
                    diffs.append((feat_id, custom_mean, default_mean, diff))
            
            diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            
            print(f"\n{'Feature':<10} {'Name':<20} {'Custom':>10} {'Default':>10} {'Diff':>10}")
            print("-" * 65)
            
            for feat_id, custom, default, diff in diffs[:20]:
                name = feature_names.get(feat_id, "")[:19]
                direction = "↑" if diff > 0 else "↓"
                print(f"{feat_id:<10} {name:<20} {custom:>10.2f} {default:>10.2f} {diff:>+10.2f} {direction}")
        
        # Highlight key features from profile
        if profile and "features" in profile:
            print(f"\n{'='*70}")
            print("KEY FEATURES (from profile)")
            print(f"{'='*70}")
            
            for fid, fdata in profile["features"].items():
                fid_int = int(fid)
                if fid_int in summary:
                    stats = summary[fid_int]
                    target = fdata.get("target_baseline", "?")
                    trained = fdata.get("trained_baseline", "?")
                    print(f"\n{fdata['name']} (Feature {fid}):")
                    print(f"  Current mean: {stats['mean']:.2f}")
                    print(f"  Trained baseline: {trained}")
                    print(f"  Target baseline: {target}")
                    if stats['mean'] > float(target) * 1.5:
                        print(f"  ⚠️  Above target - may need counter-steering")
                    elif stats['mean'] < float(target) * 0.5:
                        print(f"  ✓  Below target")
        
        # Save results
        if args.output:
            output_data = {
                "prompt_source": prompt_source,
                "prompt_text": prompt_text,
                "summary": {str(k): v for k, v in summary.items()},
                "top_features": [(f, s) for f, s in sorted_features[:50]]
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved to {args.output}")
            
    finally:
        handle.remove()


if __name__ == "__main__":
    main()