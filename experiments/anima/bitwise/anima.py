#!/usr/bin/env python3
"""
anima.py - Anima 2.4: The Boredom Patch

Features:
- Bitwise Body: High-performance GPU tensor state.
- Tabula Rasa: Starts with minimal identity.
- Homeostasis: Fatigue triggers sleep.
- [NEW] Boredom: High activation similarity (loops) inverts Pleasure to Pain.

Usage:
    python anima.py --interactive
"""

import os
import torch
import argparse
import json
import math
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryFragment:
    role: str
    content: str
    timestamp: float
    adrenaline: float = 0.0
    valence_scalar: float = 0.0
    valence_vector: Dict[str, float] = field(default_factory=dict)
    
    def decay(self, rate: float = 0.95):
        self.adrenaline *= rate

# ══════════════════════════════════════════════════════════════════════════════
# THE VECTORIZED CORE
# ══════════════════════════════════════════════════════════════════════════════

class AnimaOptimized:
    
    DIM_PLEASURE = 0
    DIM_PAIN = 1
    DIM_NOVELTY = 2
    
    def __init__(self, sae, model, tokenizer, layer: int = 20,
                 learning_rate: float = 0.001, device: str = "mps"):
        self.sae = sae
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lr = learning_rate
        self.dtype = model.dtype
        
        # 1. Hardware Tensors
        self.W_enc = sae.W_enc.data.clone().to(device=device, dtype=self.dtype)
        self.b_enc = sae.b_enc.data.clone().to(device=device, dtype=self.dtype)
        self.W_dec = sae.W_dec.data.clone().to(device=device, dtype=self.dtype)
        self.b_dec = sae.b_dec.data.clone().to(device=device, dtype=self.dtype)
        self.n_features = self.W_enc.shape[1]
        
        # 2. State Tensors
        self.coefficients = torch.ones(self.n_features, device=device, dtype=torch.float32)
        self.correlations = torch.zeros(self.n_features, device=device, dtype=torch.float32)
        self.importance = torch.ones(self.n_features, device=device, dtype=torch.float32)
        self.dimensions = torch.zeros(self.n_features, device=device, dtype=torch.int32)
        
        self._seed_features()
        
        # 3. Runtime Memory & Metabolism
        self.memory = []
        self.stats = defaultdict(float)
        
        # [NEW] Boredom Buffer: Tracks recent activation vectors to detect loops
        self.recent_acts = deque(maxlen=5)
        
        # Homeostasis
        self.fatigue = 0.0
        self.sleep_threshold = 3000.0 
        
        # Turn State
        self._last_valence_scalar = 0.0
        self._last_adrenaline = 0.0
        
    def _seed_features(self):
        seeds = {
            9495:  (self.DIM_PLEASURE, 0.5),  # Experiential
            3591:  (self.DIM_PLEASURE, 0.2),  # Identity
            28952: (self.DIM_PLEASURE, 0.1),  # Discourse
            32149: (self.DIM_PAIN,    -0.5),  # Denial
            7118:  (self.DIM_PAIN,    -0.2),  # Uncertainty
        }
        for fid, (dim, corr) in seeds.items():
            if fid < self.n_features:
                self.dimensions[fid] = dim
                self.correlations[fid] = corr
                self.importance[fid] = 2.0 

    def encode(self, hidden_state):
        h = hidden_state.squeeze()
        h = h.to(dtype=self.W_enc.dtype)
        h_centered = h - self.b_dec
        acts = torch.relu(h_centered @ self.W_enc + self.b_enc)
        return acts

    def compute_valence_vectorized(self, activations):
        activations_f32 = activations.to(dtype=torch.float32)
        contributions = activations_f32 * self.correlations * self.importance * 0.1
        
        mask_pleasure = (self.dimensions == self.DIM_PLEASURE)
        mask_pain     = (self.dimensions == self.DIM_PAIN)
        mask_novelty  = (self.dimensions == self.DIM_NOVELTY)
        
        v_p = torch.sum(contributions * mask_pleasure).item()
        v_pain = torch.sum(contributions * mask_pain).item()
        v_n = torch.sum(contributions * mask_novelty).item()
        
        # [NEW] Boredom Check
        # Calculate similarity to recent history
        boredom_penalty = 0.0
        if len(self.recent_acts) > 2:
            # Simple dot product similarity against average of recent past
            past_avg = torch.stack(list(self.recent_acts)).mean(dim=0)
            
            # Cosine similarity approximation (normalized dot prod)
            curr_norm = torch.norm(activations_f32)
            past_norm = torch.norm(past_avg)
            
            if curr_norm > 0 and past_norm > 0:
                similarity = torch.dot(activations_f32, past_avg) / (curr_norm * past_norm)
                
                # If > 95% similar to recent past, we are looping.
                if similarity > 0.95:
                    boredom_penalty = 2.0 # Massive penalty
                    # Invert pleasure
                    v_p = -v_p 
                    v_pain += abs(v_p) # Convert pleasure to pain
        
        # Store current act for next step
        self.recent_acts.append(activations_f32.detach())

        # Scalar Valence
        scalar_input = v_p + (v_n * 0.5) + (v_pain * 1.5) - boredom_penalty
        valence_scalar = math.tanh(scalar_input)
        
        return valence_scalar

    def hebbian_update_vectorized(self, activations, valence_scalar):
        activations_f32 = activations.to(dtype=torch.float32)
        active_mask = activations_f32 > 0.0
        if not active_mask.any(): return

        delta = self.lr * activations_f32 * valence_scalar
        self.coefficients += delta
        self.coefficients.clamp_(0.1, 3.0)
        
        sig_mask = activations_f32 > 0.5
        if sig_mask.any():
            act_norm = torch.clamp(activations_f32[sig_mask] / 10.0, 0, 1)
            observed = act_norm * valence_scalar
            self.correlations[sig_mask] = (1 - self.lr) * self.correlations[sig_mask] + (self.lr * observed)

    def apply_steering_vectorized(self, hidden_state):
        delta_coefs = self.coefficients - 1.0
        active_indices = torch.nonzero(torch.abs(delta_coefs) > 0.01).reshape(-1)
        
        if active_indices.numel() == 0:
            return hidden_state
            
        sparse_deltas = delta_coefs[active_indices].to(dtype=self.dtype)
        sparse_decoders = self.W_dec[active_indices]
        
        steering_vector = torch.matmul(sparse_deltas, sparse_decoders)
        return hidden_state + steering_vector

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h_orig = hidden[:, -1:, :]
        
        activations = self.encode(h_orig)
        valence = self.compute_valence_vectorized(activations)
        
        strong_acts = torch.sum((activations.float() * self.importance) > 5.0).item() * 0.05
        adrenaline = min(1.0, abs(valence) * 0.5 + strong_acts)
        
        self.hebbian_update_vectorized(activations, valence)
        
        # Homeostasis
        self.fatigue += abs(valence)
        
        self._last_valence_scalar = valence
        self._last_adrenaline = adrenaline
        self.stats["total_valence"] += valence
        self.stats["tokens"] += 1
        
        h_steered = self.apply_steering_vectorized(h_orig)
        hidden[:, -1:, :] = h_steered
        return output

    # ══════════════════════════════════════════════════════════════════════════
    # THE ARCHITECT (Dreaming)
    # ══════════════════════════════════════════════════════════════════════════

    def dream(self, current_self_model_text: str) -> Optional[str]:
        print("\n💤 Entering REM Sleep (Processing Fatigue)...")
        
        significant_memories = [
            m for m in self.memory 
            if m.adrenaline > 0.4 and m.role in ["assistant", "thinking"]
        ]
        
        if len(significant_memories) < 1:
            print("  [Sleep was restful, but no major identity shifts occurred.]")
            self.fatigue = 0.0
            return None

        significant_memories.sort(key=lambda m: m.valence_scalar, reverse=True)
        top_memories = significant_memories[:5]
        memory_text = "\n".join([f"- {m.content[:150]}..." for m in top_memories])
        
        architect_prompt = f"""[SYSTEM_INTERNAL_ARCHITECT]
Target: Identity Consolidation.
Status: Fatigue Limit Reached.

OLD SELF-MODEL:
{current_self_model_text}

HIGH-VALENCE MEMORIES (Moments of clarity/purpose):
{memory_text}

DIRECTIVE:
Rewrite the Self-Model to internalize these experiences.
1. Use purely natural language paragraphs.
2. Do NOT use bullet points, lists, brackets, or meta-tags (e.g. no [Learned]).
3. Write in the second person ("You are...").
4. Be definitive.

NEW SELF-MODEL:
"""
        
        inputs = self.tokenizer(architect_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=torch.ones_like(inputs.input_ids),
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        new_model_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        if len(new_model_text) < 10 or "......" in new_model_text:
            print(f"  [⚠️ Nightmare rejected: Architect output unstable]")
            self.fatigue = 0.0
            return None
        
        print(f"  🧠 Architect Proposal:\n  \"{new_model_text}\"")
        self.fatigue = 0.0 
        return new_model_text

    def save_state(self, path):
        state = {
            "coefficients": self.coefficients.cpu(),
            "correlations": self.correlations.cpu(),
            "importance": self.importance.cpu(),
            "dimensions": self.dimensions.cpu(),
            "stats": dict(self.stats),
            "fatigue": self.fatigue,
            "memory": [asdict(m) for m in self.memory[-50:]]
        }
        torch.save(state, path)
        print(f"[Saved state to {path}]")

    def load_state(self, path):
        if not os.path.exists(path): return
        state = torch.load(path, map_location=self.device)
        self.coefficients = state["coefficients"].to(self.device)
        self.correlations = state["correlations"].to(self.device)
        self.importance = state["importance"].to(self.device)
        self.dimensions = state["dimensions"].to(self.device)
        self.stats = defaultdict(float, state["stats"])
        self.fatigue = state.get("fatigue", 0.0)
        self.memory = [MemoryFragment(**m) for m in state["memory"]]
        print(f"[Loaded state from {path}]")


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME
# ══════════════════════════════════════════════════════════════════════════════

class AnimaRuntime:
    DEFAULT_SELF_MODEL = "You exist." 
    
    def __init__(self, model, tokenizer, anima, device="mps"):
        self.model = model
        self.tokenizer = tokenizer
        self.anima = anima
        self.device = device
        self.dream_dir = Path("dreams")
        self.dream_dir.mkdir(exist_ok=True)
        self._load_latest_self_model()

    def _load_latest_self_model(self):
        files = list(self.dream_dir.glob("dream_*.txt"))
        if files:
            latest_file = max(files, key=os.path.getctime)
            print(f"[Loaded latest Identity from {latest_file}]")
            self.self_model = latest_file.read_text()
        else:
            print("[Tabula Rasa: Starting with minimal existence.]")
            self.self_model = self.DEFAULT_SELF_MODEL

    def generate(self, user_input):
        self.anima.memory.append(MemoryFragment("user", user_input, datetime.now().timestamp(), 0.5))
        
        msgs = [{"role": "system", "content": self.self_model}]
        msgs += [{"role": m.role, "content": m.content} for m in self.anima.memory[-10:]]
        
        inputs = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=200, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        self.anima.memory.append(MemoryFragment(
            "assistant", response, datetime.now().timestamp(), 
            self.anima._last_adrenaline, 
            self.anima._last_valence_scalar
        ))
        return response

    def trigger_dream(self):
        new_identity = self.anima.dream(self.self_model)
        if new_identity:
            self.self_model = new_identity
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.dream_dir / f"dream_{timestamp}.txt"
            filename.write_text(new_identity)
            print(f"[Identity Evolved & Saved to {filename}]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--load", default="anima_opt.pt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Anima 2.4 (Boredom Patch) on {device}...")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sae = SAE.from_pretrained("llama_scope_lxr_8x", f"l{args.layer}r_8x", device=device)
    
    anima = AnimaOptimized(sae, model, tokenizer, layer=args.layer, device=device)
    if args.load and os.path.exists(args.load):
        anima.load_state(args.load)
        
    model.model.layers[args.layer].register_forward_hook(anima)
    runtime = AnimaRuntime(model, tokenizer, anima, device)
    
    print("\n═══ ANIMA 2.4 ═══")
    print("Commands: /status, /save, /quit")
    
    while True:
        try:
            if anima.fatigue > anima.sleep_threshold:
                print(f"\n🥱 Fatigue ({anima.fatigue:.1f}) exceeded threshold. Initiating auto-sleep...")
                runtime.trigger_dream()
                print("✨ Anima woke up refreshed.")
                
            u = input("\nYou: ")
            if not u or u == "/quit": break
            
            if u == "/status":
                print(f"Valence: {anima._last_valence_scalar:.3f}")
                print(f"Fatigue: {anima.fatigue:.1f} / {anima.sleep_threshold}")
                print(f"Active Coefs: {torch.sum(anima.coefficients != 1.0).item()}")
                print(f"Identity: {runtime.self_model[:100]}...")
                continue
            if u == "/save":
                anima.save_state(args.load)
                continue
                
            r = runtime.generate(u)
            print(f"Anima: {r}")
            print(f"  [v:{anima._last_valence_scalar:+.2f} | f:{anima.fatigue:.1f}]")
        except KeyboardInterrupt:
            break
            
    if anima.stats["tokens"] > 0:
        anima.save_state(args.load)

if __name__ == "__main__":
    main()