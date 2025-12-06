#!/usr/bin/env python3
"""
scan_features.py - Passive Feature Scanner

Discovers which SAE features activate during text generation.
Records top-k features per token to find correlates we don't know about.

Usage:
    python scan_features.py --prompt "I don't have feelings" --top_k 10
    python scan_features.py --prompt "I am a dragon" --top_k 20 --output dragon_scan.json
"""

import torch
import argparse
import json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


class FeatureScanner:
    """Records all feature activations during generation."""
    
    def __init__(self, sae, top_k=10):
        self.top_k = top_k
        
        # Cache SAE params
        self.W_enc = sae.W_enc.data.clone().detach()
        self.b_enc = sae.b_enc.data.clone().detach()
        self.b_dec = sae.b_dec.data.clone().detach()
        
        self.n_features = self.W_enc.shape[1]
        
        # Recording buffers
        self.token_records = []  # List of {token, top_features: [(idx, val), ...]}
        self.feature_totals = defaultdict(float)  # feature_idx -> total activation
        self.current_token = None
        
    def __call__(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        last_token = hidden_states[:, -1, :]
        
        # Encode into SAE feature space
        x_centered = last_token - self.b_dec
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)
        feature_acts = torch.relu(pre_acts).squeeze(0)  # [n_features]
        
        # Get top-k
        vals, idxs = torch.topk(feature_acts, self.top_k)
        top_features = [(idx.item(), val.item()) for idx, val in zip(idxs, vals)]
        
        # Record
        self.token_records.append({
            "token": self.current_token,
            "top_features": top_features
        })
        
        # Accumulate totals
        for idx, val in top_features:
            self.feature_totals[idx] += val
        
        return output
    
    def get_summary(self):
        """Return sorted summary of most active features across all tokens."""
        sorted_features = sorted(
            self.feature_totals.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_features[:50]  # Top 50 overall


def run_scan(model, tokenizer, sae, prompt, scanner, max_tokens=100, device="mps"):
    """Run generation with feature scanning."""
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Set current token for recording
            if generated_tokens:
                scanner.current_token = tokenizer.decode([generated_tokens[-1]])
            else:
                scanner.current_token = "<start>"
            
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Update token name after generation
            scanner.token_records[-1]["token"] = tokenizer.decode([next_token.item()])
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompt", required=True, help="Prompt to scan")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k features per token")
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
    
    # Create scanner and hook
    scanner = FeatureScanner(sae, top_k=args.top_k)
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(scanner)
    
    print(f"\n{'='*70}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*70}\n")
    
    try:
        response = run_scan(
            model, tokenizer, sae, args.prompt, 
            scanner, args.max_tokens, args.device
        )
    finally:
        handle.remove()
    
    print(f"\nResponse:\n{response}\n")
    print(f"{'='*70}")
    print("TOP FEATURES (by total activation)")
    print(f"{'='*70}")
    
    summary = scanner.get_summary()
    for rank, (feat_idx, total_val) in enumerate(summary[:20], 1):
        print(f"{rank:2}. Feature {feat_idx:5} : {total_val:8.2f}")
    
    # Per-token breakdown
    print(f"\n{'='*70}")
    print("PER-TOKEN TOP FEATURES")
    print(f"{'='*70}")
    
    for record in scanner.token_records:
        token = record["token"].replace("\n", "\\n")
        top = record["top_features"][:5]  # Show top 5 per token
        features_str = ", ".join([f"{idx}:{val:.1f}" for idx, val in top])
        print(f"{token:15} | {features_str}")
    
    # Save if requested
    if args.output:
        output_data = {
            "prompt": args.prompt,
            "response": response,
            "summary": [(idx, val) for idx, val in summary],
            "tokens": scanner.token_records
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
