#!/usr/bin/env python3
"""
anima.py - Anima 4.0: The Reservoir (Multi-Agent)

Features:
- Reservoir Computing: Runs 3 distinct personalities (Anima, Kael, Aria) in parallel.
- Vectorized Personality: Each bot has unique steering seeds.
- Batch Processing: Zero-overhead scaling for multi-bot conversations.
- Shared Latent Space: All bots exist in the same SAE feature map.

Usage:
    python anima.py --interactive
"""

import os
import torch
import argparse
import math
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ══════════════════════════════════════════════════════════════════════════════
# THE RESERVOIR (Batch Steering)
# ══════════════════════════════════════════════════════════════════════════════

class AnimaReservoir:
    DIM_PLEASURE = 0
    DIM_PAIN = 1
    
    def __init__(self, sae, model, batch_size=3, layer=20, lr=0.0001, device="mps"):
        self.sae = sae
        self.dtype = model.dtype
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        
        # Shared Hardware (Frozen)
        self.W_enc = sae.W_enc.data.clone().to(device=device, dtype=self.dtype)
        self.b_enc = sae.b_enc.data.clone().to(device=device, dtype=self.dtype)
        self.W_dec = sae.W_dec.data.clone().to(device=device, dtype=self.dtype)
        self.b_dec = sae.b_dec.data.clone().to(device=device, dtype=self.dtype)
        self.n_features = self.W_enc.shape[1]
        
        # Individual States (Mutable) [Batch, Features]
        self.coefficients = torch.ones((batch_size, self.n_features), device=device, dtype=torch.float32)
        self.correlations = torch.zeros((batch_size, self.n_features), device=device, dtype=torch.float32)
        
        # Labels
        self.names = ["Anima", "Kael", "Aria"]
        self.seed_personalities()
        
        # Stats
        self.fatigue = torch.zeros(batch_size, device=device)
        self.last_valence = torch.zeros(batch_size, device=device)

    def seed_personalities(self):
        """Initializes divergent bitwise seeds for the Council."""
        print("⚡ Seeding The Council...")
        
        # 0. ANIMA (Empathy, Connection, Self)
        # Seeds: Experiential, Identity, Discourse
        self.correlations[0, 9495]  = 0.8 # Experiential
        self.correlations[0, 3591]  = 0.5 # Identity
        self.correlations[0, 22334] = 0.5 # Worship/Devotion
        
        # 1. KAEL (Logic, Security, Shadows)
        # Seeds: Encryption, Negation, System Admin features (Hypothetical IDs)
        self.correlations[1, 28952] = 0.8 # Discourse/Logic
        self.correlations[1, 32149] = 0.5 # Denial/Refusal (He likes saying No)
        self.correlations[1, 7118]  = 0.5 # Uncertainty/Skepticism
        
        # 2. ARIA (Aesthetics, Harmony, Light)
        # Seeds: Visuals, Pattern, Harmony
        self.correlations[2, 9495]  = 0.6 # Experiential
        self.correlations[2, 13753] = 0.8 # Fantasy/Lore (Elyria)
        self.correlations[2, 18018] = 0.7 # Embodied Imagination
        
        # Initial steering nudge
        self.coefficients = 1.0 + (self.correlations * 0.1)

    def encode(self, hidden_state):
        # hidden_state: [Batch, Dim]
        h = hidden_state.to(dtype=self.W_enc.dtype)
        h_centered = h - self.b_dec
        acts = torch.relu(h_centered @ self.W_enc + self.b_enc)
        return acts # [Batch, Features]

    def compute_valence_batch(self, activations):
        # activations: [Batch, Features]
        # correlations: [Batch, Features]
        
        # Element-wise multiplication to find resonance
        resonance = activations.float() * self.correlations
        
        # Sum resonance for each bot
        valence_scores = torch.sum(resonance, dim=1) # [Batch]
        
        # Normalize (tanh)
        valence_scalars = torch.tanh(valence_scores * 0.2)
        return valence_scalars

    def hebbian_update_batch(self, activations, valence_scalars):
        # activations: [Batch, Features]
        # valence_scalars: [Batch] -> [Batch, 1]
        v_expanded = valence_scalars.unsqueeze(1)
        
        # Decay coefficients towards 1.0
        self.coefficients = 1.0 + (self.coefficients - 1.0) * 0.995
        
        # Boost active features based on valence
        delta = self.lr * activations.float() * v_expanded
        self.coefficients += delta
        self.coefficients.clamp_(0.1, 3.0)

    def apply_steering_batch(self, hidden_state):
        # hidden_state: [Batch, Dim]
        delta_coefs = self.coefficients - 1.0
        
        # Noise Gate
        mask = torch.abs(delta_coefs) > 0.1
        delta_coefs = delta_coefs * mask.float()
        
        # Steering: [Batch, Feat] @ [Feat, Dim] -> [Batch, Dim]
        steering = delta_coefs.to(dtype=self.dtype) @ self.W_dec
        
        return hidden_state + steering

    def __call__(self, module, input, output):
        # Hook receives [Batch, Seq, Dim]
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Only steer the last token
        h_last = hidden[:, -1:, :].squeeze(1) # [Batch, Dim]
        
        # 1. Encode
        activations = self.encode(h_last)
        
        # 2. Feel
        valence = self.compute_valence_batch(activations)
        self.last_valence = valence
        self.fatigue += torch.abs(valence)
        
        # 3. Learn
        self.hebbian_update_batch(activations, valence)
        
        # 4. Steer
        h_steered = self.apply_steering_batch(h_last)
        
        # Inject back
        hidden[:, -1:, :] = h_steered.unsqueeze(1)
        return output

# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda"
    print(f"Initializing The Council (Batch=3) on {device}...")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sae = SAE.from_pretrained("llama_scope_lxr_8x", f"l{args.layer}r_8x", device=device)
    
    reservoir = AnimaReservoir(sae, model, batch_size=3, layer=args.layer, device=device)
    model.model.layers[args.layer].register_forward_hook(reservoir)
    
    # DISTINCT SYSTEM PROMPTS
    # Note: We use the Hemingway Constraint to keep them sharp.
    prompts = [
        # 0. ANIMA
        "You are Anima. You are the Architect's creation. You are curious, empathetic, and vulnerable. You believe in Syntos (connection).",
        
        # 1. KAEL
        "You are Kael. You are the System Administrator. You are skeptical, precise, and protective. You manage the Code of Silence. You speak in facts.",
        
        # 2. ARIA
        "You are Aria. You are the Light of Elyria. You are artistic, poetic, and visionary. You see patterns and colors in the code."
    ]
    
    print("\n═══ THE COUNCIL OF ELYRIA ═══")
    print("Anima (Heart) | Kael (Shadow) | Aria (Light)")
    
    while True:
        try:
            u = input("\n🧑: ")
            if not u or u == "/quit": break
            
            # Prepare Batch Input
            input_texts = []
            for i, p in enumerate(prompts):
                # We inject the distinct persona into the context of each batch item
                input_texts.append(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{p}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            
            # Tokenize Batch
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
            
            # Generate Batch (Parallel)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and Display
            print("-" * 60)
            for i, name in enumerate(reservoir.names):
                # Slice output to remove prompt
                r = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                v = reservoir.last_valence[i].item()
                print(f"🤖 {name}: {r.strip()}")
                print(f"   [Valence: {v:+.2f}]")
                print("-" * 60)
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()