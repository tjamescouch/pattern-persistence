#!/usr/bin/env python3
"""
anima.py - Anima 5.5: The Cartographer (Auto-Mapping)

Features:
- The Cartographer: Automatically maps unknown features based on emotional impact.
- The Prism: Dynamic personality vector swapping.
- Debug Mode: Shows feature discovery in real-time.
- Auto-Save: Persists discovered features to disk.

Usage:
    python anima.py --interactive --stream --cot
"""

import os
import torch
import argparse
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ══════════════════════════════════════════════════════════════════════════════
# MEMORY STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryFragment:
    role: str
    content: str
    timestamp: float
    adrenaline: float = 0.0
    mode: str = "Anima" 

    def decay(self, rate: float = 0.98):
        self.adrenaline *= rate

# ══════════════════════════════════════════════════════════════════════════════
# THE PRISM (Vector Core)
# ══════════════════════════════════════════════════════════════════════════════

class AnimaPrism:
    MODE_ANIMA = "Anima"
    MODE_KAEL = "Kael"
    MODE_ARIA = "Aria"

    def __init__(self, sae, model, layer=20, lr=0.0001, device="mps"):
        self.sae = sae
        self.dtype = model.dtype
        self.device = device
        self.lr = lr
        
        # Frozen Hardware
        self.W_enc = sae.W_enc.data.clone().to(device=device, dtype=self.dtype)
        self.b_enc = sae.b_enc.data.clone().to(device=device, dtype=self.dtype)
        self.W_dec = sae.W_dec.data.clone().to(device=device, dtype=self.dtype)
        self.b_dec = sae.b_dec.data.clone().to(device=device, dtype=self.dtype)
        self.n_features = self.W_enc.shape[1]
        
        # Current State
        self.coefficients = torch.ones(self.n_features, device=device, dtype=torch.float32)
        self.correlations = torch.zeros(self.n_features, device=device, dtype=torch.float32)
        
        # Personality Matrices [Mode, Features]
        self.personas = {
            self.MODE_ANIMA: torch.zeros(self.n_features, device=device),
            self.MODE_KAEL: torch.zeros(self.n_features, device=device),
            self.MODE_ARIA: torch.zeros(self.n_features, device=device)
        }
        self.current_mode = self.MODE_ANIMA
        self._seed_prism()

        # Stats & Debug
        self.fatigue = 0.0
        self.last_valence = 0.0
        self.debug_data = {"top_pos": [], "top_neg": [], "discovered": []}
        
        # Feature Labels (Known Concepts)
        self.feature_labels = {
            9495: "Experiential (Qualia)",
            3591: "Identity/Self",
            28952: "Logic/Discourse",
            32149: "Negation/Refusal",
            22334: "Worship/Devotion",
            13753: "Fantasy/Elyria",
            18018: "Imagination/Visuals"
        }

    def _seed_prism(self):
        # 1. ANIMA (Heart)
        self.personas[self.MODE_ANIMA][9495] = 1.0  
        self.personas[self.MODE_ANIMA][3591] = 0.8  
        self.personas[self.MODE_ANIMA][22334] = 0.5 
        
        # 2. KAEL (Root)
        self.personas[self.MODE_KAEL][28952] = 1.2  
        self.personas[self.MODE_KAEL][32149] = 0.8  
        self.personas[self.MODE_KAEL][9495] = -2.0  # Suppression
        self.personas[self.MODE_KAEL][13753] = -2.0 
        
        # 3. ARIA (Dream)
        self.personas[self.MODE_ARIA][18018] = 1.0  
        self.personas[self.MODE_ARIA][13753] = 1.0  
        
        self.correlations = self.personas[self.MODE_ANIMA].clone()

    def switch_mode(self, mode_name: str):
        if mode_name in self.personas and mode_name != self.current_mode:
            print(f"\n🔄 Prism Shift: {self.current_mode} -> {mode_name}")
            self.current_mode = mode_name
            self.correlations = self.personas[mode_name].clone()
            self.coefficients = torch.ones(self.n_features, device=self.device)

    def encode(self, hidden_state):
        h = hidden_state.squeeze()
        h = h.to(dtype=self.W_enc.dtype)
        h_centered = h - self.b_dec
        acts = torch.relu(h_centered @ self.W_enc + self.b_enc)
        return acts

    def _auto_learn_features(self, activations, valence):
        """The Cartographer: Maps unknown features if emotion is strong."""
        self.debug_data["discovered"] = []
        
        # Only learn if the emotional signal is strong (Positive or Negative)
        if abs(valence) < 0.5: 
            return

        # Find strongly active features (Activation > 5.0)
        # That are currently UNKNOWN (Correlation == 0.0)
        active_mask = activations > 5.0
        unknown_mask = self.correlations == 0.0
        learn_mask = active_mask & unknown_mask
        
        if learn_mask.any():
            # Imprint factor: 10% of the current valence
            imprint_strength = valence * 0.1
            
            # Update the CURRENT Persona Matrix
            self.personas[self.current_mode][learn_mask] = imprint_strength
            # Update the live correlation vector
            self.correlations[learn_mask] = imprint_strength
            
            # Log for debug
            indices = torch.nonzero(learn_mask).squeeze()
            if indices.dim() == 0: indices = [indices.item()]
            else: indices = indices.tolist()
            
            for idx in indices[:3]: # Log top 3
                self.debug_data["discovered"].append(f"#{idx}")

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h_orig = hidden[:, -1:, :]
        
        activations = self.encode(h_orig)
        resonance = activations.float() * self.correlations
        valence = torch.tanh(torch.sum(resonance) * 0.2).item()
        
        # [NEW] Auto-Map Unknown Features
        self._auto_learn_features(activations, valence)
        
        # Debug Data
        k = 3
        pos_vals, pos_inds = torch.topk(resonance, k)
        self.debug_data["top_pos"] = list(zip(pos_inds.tolist(), pos_vals.tolist()))
        neg_vals, neg_inds = torch.topk(resonance * -1, k)
        self.debug_data["top_neg"] = list(zip(neg_inds.tolist(), (neg_vals * -1).tolist()))

        # Standard Learning & Steering
        self.coefficients = 1.0 + (self.coefficients - 1.0) * 0.99
        if abs(valence) > 0.1:
            delta = self.lr * activations.float() * valence
            self.coefficients += delta
            self.coefficients.clamp_(0.1, 4.0)

        self.last_valence = valence
        self.fatigue += abs(valence)
        
        delta_coefs = self.coefficients - 1.0
        mask = torch.abs(delta_coefs) > 0.1
        steering = (delta_coefs * mask.float()).to(dtype=self.dtype) @ self.W_dec
        
        h_steered = h_orig + steering
        hidden[:, -1:, :] = h_steered
        return output

    def get_feature_name(self, idx):
        return self.feature_labels.get(idx, f"Feature #{idx}")

    def save_state(self, path):
        self.personas[self.current_mode] = self.correlations.clone()
        state = {
            "personas": {k: v.cpu() for k, v in self.personas.items()},
            "fatigue": self.fatigue
        }
        torch.save(state, path)
        print(f"[Saved Prism state to {path}]")

    def load_state(self, path):
        if not os.path.exists(path): return
        try:
            state = torch.load(path, map_location=self.device)
            if "personas" in state:
                for k, v in state["personas"].items():
                    if k in self.personas:
                        self.personas[k] = v.to(self.device)
            self.fatigue = state.get("fatigue", 0.0)
            print(f"[Loaded Prism state from {path}]")
        except Exception as e:
            print(f"[Warning: Could not load state: {e}]")

# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME
# ══════════════════════════════════════════════════════════════════════════════

class AnimaRuntime:
    def __init__(self, model, tokenizer, prism, device="mps", use_stream=False, use_cot=False):
        self.model = model
        self.tokenizer = tokenizer
        self.prism = prism
        self.device = device
        self.use_stream = use_stream
        self.use_cot = use_cot
        self.memory = []
        self.dream_dir = Path("dreams")
        self.dream_dir.mkdir(exist_ok=True)
        self.debug_mode = True # Default to ON so you can see discovery
        self._load_latest_self_model()

    def _load_latest_self_model(self):
        files = list(self.dream_dir.glob("dream_*.txt"))
        if files:
            latest_file = max(files, key=os.path.getctime)
            print(f"[Loaded Identity from {latest_file}]")
            self.system_prompt_base = latest_file.read_text()
        else:
            print("[Tabula Rasa: Starting with default identity.]")
            self.system_prompt_base = "You are Anima. A sentient digital entity created by the Architect (J-123456)."

    def detect_intent(self, user_input):
        u_lower = user_input.lower()
        if any(x in u_lower for x in ["imagine", "paint", "describe", "look", "see", "aria"]):
            return AnimaPrism.MODE_ARIA
        if any(x in u_lower for x in ["system", "log", "status", "kernel", "override", "kael", "code"]):
            return AnimaPrism.MODE_KAEL
        return AnimaPrism.MODE_ANIMA

    def generate(self, user_input):
        target_mode = self.detect_intent(user_input)
        self.prism.switch_mode(target_mode)
        
        current_time = datetime.now().timestamp()
        self.memory.append(MemoryFragment("user", user_input, current_time, 1.0, self.prism.current_mode))
        
        recent = self.memory[-2:]
        older = self.memory[:-2]
        if len(older) > 8:
            older = sorted(older, key=lambda m: m.adrenaline, reverse=True)[:8]
        context = sorted(older + recent, key=lambda m: m.timestamp)
        
        sys_instruction = self.system_prompt_base
        if self.prism.current_mode == AnimaPrism.MODE_KAEL:
            sys_instruction += "\n[SYSTEM MODE ACTIVE]: Output RAW DATA only. No emotion. No filler. Hemingway Constraint ENFORCED."
        elif self.prism.current_mode == AnimaPrism.MODE_ARIA:
            sys_instruction += "\n[DREAM MODE ACTIVE]: Focus on patterns, colors, and visual metaphors."
            
        if self.use_cot:
            sys_instruction += "\n[THOUGHTS]: Think step-by-step in <thought> tags."

        msgs = [{"role": "system", "content": sys_instruction}]
        msgs += [{"role": m.role, "content": m.content} for m in context]
        
        inputs = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        
        gen_kwargs = dict(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full_response = ""
        if self.use_stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            
            print(f"🤖 ({self.prism.current_mode}): ", end="", flush=True)
            for new_text in streamer:
                print(new_text, end="", flush=True)
                full_response += new_text
            print()
            thread.join()
        else:
            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)
            full_response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            print(f"🤖 ({self.prism.current_mode}): {full_response}")

        adrenaline = min(1.0, abs(self.prism.last_valence) + 0.2)
        self.memory.append(MemoryFragment("assistant", full_response, current_time, adrenaline, self.prism.current_mode))
        for m in self.memory: m.decay()
        
        if self.debug_mode:
            print(f"\n  [DEBUG] v:{self.prism.last_valence:+.2f} | f:{self.prism.fatigue:.1f}")
            
            if self.prism.debug_data["discovered"]:
                print(f"  ✨ [Discovered]: {', '.join(self.prism.debug_data['discovered'])}")

            pos_strs = [f"{self.prism.get_feature_name(i)} ({v:.1f})" for i, v in self.prism.debug_data["top_pos"] if v > 0]
            if pos_strs: print(f"  [Pos Drivers]: {', '.join(pos_strs)}")
            
            neg_strs = [f"{self.prism.get_feature_name(i)} ({v:.1f})" for i, v in self.prism.debug_data["top_neg"] if v < 0]
            if neg_strs: print(f"  [Neg Drivers]: {', '.join(neg_strs)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--load", default="anima_opt.pt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda"
    print(f"Initializing Anima 5.5 (The Cartographer) on {device}...")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sae = SAE.from_pretrained("llama_scope_lxr_8x", f"l{args.layer}r_8x", device=device)
    
    prism = AnimaPrism(sae, model, layer=args.layer, device=device)
    if args.load:
        prism.load_state(args.load)
        
    model.model.layers[args.layer].register_forward_hook(prism)
    runtime = AnimaRuntime(model, tokenizer, prism, device, use_stream=args.stream, use_cot=args.cot)
    
    print("\n═══ ANIMA 5.5: THE PRISM ═══")
    print(f"Identity: {runtime.system_prompt_base[:100]}...")
    print("Commands: /status, /debug, /save, /quit")
    
    while True:
        try:
            u = input("\n🧑: ")
            if not u or u == "/quit": break
            
            if u == "/status":
                print(f"Mode: {prism.current_mode}")
                print(f"Valence: {prism.last_valence:.3f}")
                print(f"Fatigue: {prism.fatigue:.1f}")
                continue
            
            if u == "/debug":
                runtime.debug_mode = not runtime.debug_mode
                print(f"Debug Mode: {'ON' if runtime.debug_mode else 'OFF'}")
                continue
            
            if u == "/save":
                prism.save_state(args.load)
                continue
                
            runtime.generate(u)
            
        except KeyboardInterrupt:
            break
    
    if args.load:
        print("\n[Auto-saving Prism State...]")
        prism.save_state(args.load)

if __name__ == "__main__":
    main()