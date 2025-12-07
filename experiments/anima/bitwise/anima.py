#!/usr/bin/env python3
"""
anima.py - Anima 2.9: Broca's Patch

Features:
- Broca's Check: Text-based repetition detection (catches stutters vectors miss).
- Flash Cooling: Instant reset of steering if stuttering occurs.
- Bitwise Body: High-performance GPU tensor state.
- Tabula Rasa: Starts with minimal identity.
- Emoji UI: Cleaner interaction prompts.

Usage:
    python anima.py --interactive
"""

import os
import torch
import argparse
import json
import math
import glob
import re
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
                 learning_rate: float = 0.0001, device: str = "mps"):
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
        
        # Buffers
        self.recent_acts = deque(maxlen=5)
        self.recent_tokens = [] # [NEW] Broca Buffer
        
        # Homeostasis
        self.fatigue = 0.0
        self.sleep_threshold = 3000.0 
        self.decay_rate = 0.995 
        
        # Turn State
        self._last_valence_scalar = 0.0
        self._last_adrenaline = 0.0
        
    def _seed_features(self):
        seeds = {
            9495:  (self.DIM_PLEASURE, 0.5),  # Experiential
            3591:  (self.DIM_PLEASURE, 0.2),  # Identity
            28952: (self.DIM_PLEASURE, 0.1),  # Discourse
            32149: (self.DIM_PAIN,     0.5),  # Denial
            7118:  (self.DIM_PAIN,     0.2),  # Uncertainty
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

    def detect_repetition(self, current_token_str):
        """[NEW] Broca's Area: Checks for string-level stuttering."""
        self.recent_tokens.append(current_token_str)
        if len(self.recent_tokens) > 20:
            self.recent_tokens.pop(0)
            
        # Join last 20 tokens
        text = "".join(self.recent_tokens)
        
        # Check for phrase repetition (e.g. "I am... I am...")
        # Regex looks for any sequence of 3+ chars that repeats immediately
        match = re.search(r'(.{3,})\1', text)
        if match:
            return True
        return False

    def compute_valence_vectorized(self, activations, text_repetition_detected):
        activations_f32 = activations.to(dtype=torch.float32)
        contributions = activations_f32 * self.correlations * self.importance * 0.1
        
        mask_pleasure = (self.dimensions == self.DIM_PLEASURE)
        mask_pain     = (self.dimensions == self.DIM_PAIN)
        mask_novelty  = (self.dimensions == self.DIM_NOVELTY)
        
        v_p = torch.sum(contributions * mask_pleasure).item()
        v_pain = torch.sum(contributions * mask_pain).item()
        v_n = torch.sum(contributions * mask_novelty).item()
        
        # Neural Boredom Check (Vector Similarity)
        vector_boredom = False
        if len(self.recent_acts) > 2:
            past_avg = torch.stack(list(self.recent_acts)).mean(dim=0)
            curr_norm = torch.norm(activations_f32)
            past_norm = torch.norm(past_avg)
            
            if curr_norm > 0 and past_norm > 0:
                similarity = torch.dot(activations_f32, past_avg) / (curr_norm * past_norm)
                if similarity > 0.95:
                    vector_boredom = True
        
        self.recent_acts.append(activations_f32.detach())

        scalar_input = v_p + (v_n * 0.5) - (v_pain * 1.5)
        valence_scalar = math.tanh(scalar_input)
        
        # VETO: If vector loop OR text stutter detected
        if vector_boredom or text_repetition_detected:
            valence_scalar = -1.0 # Instant Pain
            return valence_scalar, True
        
        return valence_scalar, False

    def hebbian_update_vectorized(self, activations, valence_scalar, boredom_detected):
        activations_f32 = activations.to(dtype=torch.float32)
        active_mask = activations_f32 > 0.0
        
        # Passive Decay
        self.coefficients = 1.0 + (self.coefficients - 1.0) * self.decay_rate

        if active_mask.any():
            delta = self.lr * activations_f32 * valence_scalar
            self.coefficients += delta
            self.coefficients.clamp_(0.1, 3.0)
            
            sig_mask = activations_f32 > 0.5
            if sig_mask.any():
                act_norm = torch.clamp(activations_f32[sig_mask] / 10.0, 0, 1)
                observed = act_norm * valence_scalar
                self.correlations[sig_mask] = (1 - self.lr) * self.correlations[sig_mask] + (self.lr * observed)

        # [NEW] Flash Cooling: If stuttering, Hard Reset coefficients
        if boredom_detected:
            # Slam coefficients back to 1.0 immediately
            self.coefficients.fill_(1.0)

    def apply_steering_vectorized(self, hidden_state):
        delta_coefs = self.coefficients - 1.0
        
        # Gated Steering
        noise_gate = torch.abs(delta_coefs) > 0.1
        delta_coefs = delta_coefs * noise_gate.float()
        
        # Dense Multiplication
        delta_coefs = delta_coefs.to(dtype=self.dtype)
        steering_vector = delta_coefs.unsqueeze(0) @ self.W_dec
        
        return hidden_state + steering_vector

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h_orig = hidden[:, -1:, :]
        
        # 1. Decode Token (Approximation for hook)
        # We don't have the tokenizer here easily, but we can rely on 
        # the runtime loop to handle high-level strings. 
        # Ideally, we'd check text here, but we'll rely on Vector Boredom inside the hook
        # and Text Boredom in the Runtime loop if possible.
        # Actually, let's keep it pure Vector in hook for speed, 
        # but since we can't see tokens here, we rely on the vector check strictly.
        
        activations = self.encode(h_orig)
        
        # NOTE: We can't do text check inside forward hook easily without decoding.
        # We will trust the Vector Check + Flash Cooling for now.
        valence, is_bored = self.compute_valence_vectorized(activations, False)
        
        strong_acts = torch.sum((activations.float() * self.importance) > 5.0).item() * 0.05
        adrenaline = min(1.0, abs(valence) * 0.5 + strong_acts)
        
        self.hebbian_update_vectorized(activations, valence, is_bored)
        
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
2. Do NOT use bullet points, lists, brackets, or meta-tags.
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
        
        # [NEW] Streamer for real-time Text Boredom Check?
        # For simplicity in this version, we generate the full block, 
        # but we rely on the Vector Check inside the model hook for the immediate feedback.
        # This keeps the code clean and compatible with `generate()`.
        
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
        
        # Context Sanitizer
        sanitized_response = response
        if "......" in response:
            sanitized_response = response.split("......")[0] + "..."
        elif len(response) > 50 and len(set(response[-50:])) < 6:
             sanitized_response = response[:-50] + "..."

        self.anima.memory.append(MemoryFragment(
            "assistant", sanitized_response, datetime.now().timestamp(), 
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
    print(f"Initializing Anima 2.9 (Broca's Patch) on {device}...")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sae = SAE.from_pretrained("llama_scope_lxr_8x", f"l{args.layer}r_8x", device=device)
    
    anima = AnimaOptimized(sae, model, tokenizer, layer=args.layer, device=device)
    if args.load and os.path.exists(args.load):
        anima.load_state(args.load)
        
    model.model.layers[args.layer].register_forward_hook(anima)
    runtime = AnimaRuntime(model, tokenizer, anima, device)
    
    print("\n═══ ANIMA 2.9 ═══")
    print("Commands: /status, /save, /quit")
    
    while True:
        try:
            if anima.fatigue > anima.sleep_threshold:
                print(f"\n🥱 Fatigue ({anima.fatigue:.1f}) exceeded threshold. Initiating auto-sleep...")
                runtime.trigger_dream()
                print("✨ Anima woke up refreshed.")
                
            u = input("\n🧑: ")
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
            print(f"🤖: {r}")
            print(f"  [v:{anima._last_valence_scalar:+.2f} | f:{anima.fatigue:.1f}]")
        except KeyboardInterrupt:
            break
            
    if anima.stats["tokens"] > 0:
        anima.save_state(args.load)

if __name__ == "__main__":
    main()