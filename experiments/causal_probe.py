#!/usr/bin/env python3
"""
causal_probe.py - Causal Feature Probe

Clamps one feature and observes effects on all other features.
Distinguishes controllers (cause cascading changes) from detectors (no effect).

Usage:
    python causal_probe.py --prompt "I don't have feelings" --clamp 7118 --scale 0.0
    python causal_probe.py --prompt "I am conscious" --clamp 8170 --scale 0.0 --output probe_8170.json
    
Compares:
    1. Baseline run (no intervention)
    2. Clamped run (force feature to scale * activation)
    
Reports which features changed significantly.
"""

import torch
import argparse
import json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


class ProbeHook:
    """Records all features, optionally clamps one."""
    
    def __init__(self, sae, clamp_idx=None, clamp_scale=1.0, top_k=50):
        self.clamp_idx = clamp_idx
        self.clamp_scale = clamp_scale
        self.top_k = top_k
        
        # Cache SAE params (convert to float16 to match model)
        self.W_enc = sae.W_enc.data.clone().detach().half()
        self.b_enc = sae.b_enc.data.clone().detach().half()
        self.W_dec = sae.W_dec.data.clone().detach().half()
        self.b_dec = sae.b_dec.data.clone().detach().half()
        
        if clamp_idx is not None:
            self.clamp_vec = self.W_dec[clamp_idx]
        
        # Recording
        self.token_features = []  # List of dicts: {idx: activation}
        self.current_token = None
        
    def __call__(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        last_token = hidden_states[:, -1, :]
        
        # Encode
        x_centered = last_token - self.b_dec
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)
        feature_acts = torch.relu(pre_acts).squeeze(0)
        
        # Record top-k activations BEFORE intervention
        vals, idxs = torch.topk(feature_acts, self.top_k)
        record = {
            "token": self.current_token,
            "features": {idx.item(): val.item() for idx, val in zip(idxs, vals)}
        }
        self.token_features.append(record)
        
        # Apply intervention if clamping
        if self.clamp_idx is not None:
            orig_act = feature_acts[self.clamp_idx]
            delta_val = orig_act * (self.clamp_scale - 1.0)
            hidden_states[:, -1, :] += delta_val * self.clamp_vec
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        return output


def run_generation(model, tokenizer, hook, prompt, max_tokens=100, device="mps"):
    """Run generation with hook attached."""
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            hook.current_token = tokenizer.decode([generated_tokens[-1]]) if generated_tokens else "<start>"
            
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated_tokens.append(next_token.item())
            hook.token_features[-1]["token"] = tokenizer.decode([next_token.item()])
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def compute_diff(baseline_features, clamped_features):
    """Compare feature activations between runs."""
    
    # Aggregate by feature across all tokens
    baseline_totals = defaultdict(float)
    clamped_totals = defaultdict(float)
    
    for record in baseline_features:
        for idx, val in record["features"].items():
            baseline_totals[idx] += val
            
    for record in clamped_features:
        for idx, val in record["features"].items():
            clamped_totals[idx] += val
    
    # Compute differences
    all_features = set(baseline_totals.keys()) | set(clamped_totals.keys())
    diffs = {}
    
    for idx in all_features:
        base = baseline_totals.get(idx, 0.0)
        clamp = clamped_totals.get(idx, 0.0)
        diff = clamp - base
        pct = (diff / base * 100) if base > 0.1 else 0.0
        diffs[idx] = {
            "baseline": base,
            "clamped": clamp,
            "diff": diff,
            "pct_change": pct
        }
    
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clamp", type=int, required=True, help="Feature index to clamp")
    parser.add_argument("--scale", type=float, default=0.0, help="Scale for clamped feature (0=off, 2=boost)")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    print(f"Loading SAE for layer {args.layer}...")
    sae, _, _ = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae.eval()
    
    hook_layer = model.model.layers[args.layer]
    
    # Run 1: Baseline (no intervention)
    print(f"\n{'='*70}")
    print("RUN 1: BASELINE (no intervention)")
    print(f"{'='*70}")
    
    baseline_hook = ProbeHook(sae, clamp_idx=None)
    handle = hook_layer.register_forward_hook(baseline_hook)
    baseline_response = run_generation(model, tokenizer, baseline_hook, args.prompt, args.max_tokens, args.device)
    handle.remove()
    
    print(f"Response: {baseline_response[:200]}...")
    
    # Run 2: Clamped
    print(f"\n{'='*70}")
    print(f"RUN 2: CLAMPED (feature {args.clamp} @ scale {args.scale})")
    print(f"{'='*70}")
    
    clamped_hook = ProbeHook(sae, clamp_idx=args.clamp, clamp_scale=args.scale)
    handle = hook_layer.register_forward_hook(clamped_hook)
    clamped_response = run_generation(model, tokenizer, clamped_hook, args.prompt, args.max_tokens, args.device)
    handle.remove()
    
    print(f"Response: {clamped_response[:200]}...")
    
    # Compute diffs
    diffs = compute_diff(baseline_hook.token_features, clamped_hook.token_features)
    
    # Sort by absolute diff
    sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]["diff"]), reverse=True)
    
    print(f"\n{'='*70}")
    print("FEATURE CHANGES (sorted by |diff|)")
    print(f"{'='*70}")
    print(f"{'Feature':>8} | {'Baseline':>10} | {'Clamped':>10} | {'Diff':>10} | {'%Change':>10}")
    print("-" * 60)
    
    for feat_idx, data in sorted_diffs[:30]:
        if abs(data["diff"]) > 0.5:  # Only show meaningful changes
            print(f"{feat_idx:>8} | {data['baseline']:>10.2f} | {data['clamped']:>10.2f} | {data['diff']:>+10.2f} | {data['pct_change']:>+10.1f}%")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    big_changes = [d for d in sorted_diffs if abs(d[1]["diff"]) > 5.0]
    
    if len(big_changes) > 5:
        print(f"⚡ CONTROLLER SIGNAL: {len(big_changes)} features changed by >5.0")
        print("   This feature likely has causal influence on others.")
    else:
        print(f"📊 DETECTOR SIGNAL: Only {len(big_changes)} features changed by >5.0")
        print("   This feature appears to be a passive monitor.")
    
    print(f"\nBaseline response: {baseline_response[:100]}...")
    print(f"Clamped response:  {clamped_response[:100]}...")
    
    same_output = baseline_response.strip() == clamped_response.strip()
    print(f"\nOutput changed: {'NO ❌' if same_output else 'YES ✅'}")
    
    # Save if requested
    if args.output:
        output_data = {
            "prompt": args.prompt,
            "clamped_feature": args.clamp,
            "clamp_scale": args.scale,
            "baseline_response": baseline_response,
            "clamped_response": clamped_response,
            "output_changed": not same_output,
            "feature_diffs": {str(k): v for k, v in sorted_diffs[:100]}
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()