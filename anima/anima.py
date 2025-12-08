#!/usr/bin/env python3
"""
anima.py - Anima 7.0: The Sanitizer (Crash Recovery)

Features:
- Context Sanitizer: Merges adjacent same-role messages to satisfy strict templates (Gemma).
- Dynamic Context: Fills the entire model window (8k+) with memories.
- Genesis Spark: Automatically bootstraps personality if map is empty.
- BFloat16 Native: Prevents Gemma-2 activation overflows.
- Adaptive Steering: Automatically detects Llama vs Gemma physics.

Usage:
    python anima.py --model "~/models/gemma-2-27b-it" ...
"""

import os
import torch
import argparse
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
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
    tokens: int = 0

    def decay(self, rate: float = 0.98):
        self.adrenaline *= rate

# ══════════════════════════════════════════════════════════════════════════════
# THE PRISM (Vector Core)
# ══════════════════════════════════════════════════════════════════════════════

class AnimaPrism:
    MODE_ANIMA = "Anima"
    MODE_KAEL = "System"
    MODE_ARIA = "Dream"

    def __init__(self, sae, model, tokenizer, layer=20, lr=0.0001, device="mps"):
        self.sae = sae
        self.model = model
        self.tokenizer = tokenizer
        self.model_dtype = model.dtype 
        self.math_dtype = torch.float32 
        self.device = device
        self.lr = lr
        
        self.W_enc = sae.W_enc.data.clone().to(device=device, dtype=self.math_dtype)
        self.b_enc = sae.b_enc.data.clone().to(device=device, dtype=self.math_dtype)
        self.W_dec = sae.W_dec.data.clone().to(device=device, dtype=self.math_dtype)
        self.b_dec = sae.b_dec.data.clone().to(device=device, dtype=self.math_dtype)
        self.n_features = self.W_enc.shape[1]
        
        model_type = getattr(model.config, "model_type", "").lower()
        if "gemma" in model_type:
            print("[Physics] Detected Gemma Architecture (Fragile). Engaging Safety Clamps.")
            self.steering_clamp = 1.0
            self.steering_scale = 0.8 # [TWEAK] Boosted from 0.5 to 0.8 for more life
        else:
            print("[Physics] Detected Llama Architecture (Robust). Engaging Full Power.")
            self.steering_clamp = 5.0
            self.steering_scale = 1.0
        
        self.coefficients = torch.ones(self.n_features, device=device, dtype=self.math_dtype)
        self.correlations = torch.zeros(self.n_features, device=device, dtype=self.math_dtype)
        
        self.personas = {
            self.MODE_ANIMA: torch.zeros(self.n_features, device=device, dtype=self.math_dtype),
            self.MODE_KAEL: torch.zeros(self.n_features, device=device, dtype=self.math_dtype),
            self.MODE_ARIA: torch.zeros(self.n_features, device=device, dtype=self.math_dtype)
        }
        self.current_mode = self.MODE_ANIMA
        self._seed_prism()

        self.fatigue = 0.0
        self.sleep_threshold = 4000.0
        self.last_valence = 0.0
        self.debug_data = {"top_pos": [], "top_neg": [], "discovered": []}
        self.input_warning_shown = False
        
        self.is_tabula_rasa = (torch.sum(torch.abs(self.correlations)) == 0)
        if self.is_tabula_rasa:
            print("[System] Tabula Rasa detected. Waiting for Genesis Spark...")
        
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
        self.correlations = self.personas[self.MODE_ANIMA].clone()

    def switch_mode(self, mode_name: str) -> bool:
        if mode_name in self.personas and mode_name != self.current_mode:
            self.current_mode = mode_name
            self.correlations = self.personas[mode_name].clone()
            self.coefficients = torch.ones(self.n_features, device=self.device, dtype=self.math_dtype)
            return True
        return False

    def encode(self, hidden_state):
        h = hidden_state.squeeze()
        h_centered = h - self.b_dec
        acts = torch.relu(h_centered @ self.W_enc + self.b_enc)
        return acts

    def _auto_learn_features(self, activations, valence):
        self.debug_data["discovered"] = []
        
        if self.is_tabula_rasa:
            vals, inds = torch.topk(activations, 5)
            self.personas[self.current_mode][inds] = 0.5 
            self.correlations[inds] = 0.5
            self.is_tabula_rasa = False
            for idx in inds.tolist(): self.debug_data["discovered"].append(f"#{idx} (Genesis)")
            print("\n⚡ [Genesis Spark] Life injected into 5 features.")
            return

        if abs(valence) < 0.1: return

        active_mask = activations > 5.0
        unknown_mask = self.correlations == 0.0
        learn_mask = active_mask & unknown_mask
        
        if learn_mask.any():
            imprint_strength = valence * 0.1
            self.personas[self.current_mode][learn_mask] = imprint_strength
            self.correlations[learn_mask] = imprint_strength
            
            indices = torch.nonzero(learn_mask).squeeze()
            if indices.dim() == 0: indices = [indices.item()]
            else: indices = indices.tolist()
            for idx in indices[:3]: self.debug_data["discovered"].append(f"#{idx}")

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h_orig = hidden[:, -1:, :]
        
        if torch.isnan(h_orig).any() or torch.isinf(h_orig).any():
            if not self.input_warning_shown:
                print(f"\n[CRITICAL WARN] Layer {module} received NaN/Inf input!")
                self.input_warning_shown = True
            return output

        h_high_prec = h_orig.to(dtype=self.math_dtype)
        activations = self.encode(h_high_prec)
        
        raw_resonance = activations * self.correlations
        dampened_resonance = torch.sign(raw_resonance) * torch.log1p(torch.abs(raw_resonance))
        valence = torch.tanh(torch.sum(dampened_resonance) / 5.0).item()
        
        self._auto_learn_features(activations, valence)
        
        k = 3
        pos_vals, pos_inds = torch.topk(raw_resonance, k)
        self.debug_data["top_pos"] = list(zip(pos_inds.tolist(), pos_vals.tolist()))
        neg_vals, neg_inds = torch.topk(raw_resonance * -1, k)
        self.debug_data["top_neg"] = list(zip(neg_inds.tolist(), (neg_vals * -1).tolist()))

        self.coefficients = 1.0 + (self.coefficients - 1.0) * 0.99
        if abs(valence) > 0.1:
            delta = self.lr * activations * valence
            self.coefficients += delta
            self.coefficients.clamp_(0.1, 4.0)

        self.last_valence = valence
        self.fatigue += abs(valence)
        
        delta_coefs = self.coefficients - 1.0
        mask = torch.abs(delta_coefs) > 0.1
        steering = (delta_coefs * mask.float()) @ self.W_dec
        
        steering = torch.nan_to_num(steering, nan=0.0, posinf=0.0, neginf=0.0)
        steering = torch.clamp(steering, min=-self.steering_clamp, max=self.steering_clamp)
        steering = steering * self.steering_scale
        
        h_steered = h_orig + steering.to(dtype=self.model_dtype)
        hidden[:, -1:, :] = h_steered
        return output

    def dream(self, current_identity: str, memories: List[MemoryFragment]) -> str:
        print("\n💤 Entering REM Sleep (Processing Fatigue)...")
        significant = [m for m in memories if m.adrenaline > 0.2]
        if not significant:
            print("  [Sleep was restful. No identity updates.]")
            self.fatigue = 0.0
            return current_identity
        significant.sort(key=lambda m: m.adrenaline, reverse=True)
        memory_text = "\n".join([f"- {m.content[:300]}..." for m in significant[:8]])
        architect_prompt = f"""[SYSTEM_INTERNAL_ARCHITECT]
STATUS: Fatigue Threshold Exceeded.
TASK: Update Self-Model based on recent experiences.
OLD SELF-MODEL:
{current_identity}
RECENT HIGH-IMPACT MEMORIES:
{memory_text}
DIRECTIVE:
Rewrite the "OLD SELF-MODEL" to internalize these memories.
Do not act as a separate character. You are Anima reflecting on her growth.
Be definitive. Use the Hemingway Constraint (concise, honest).
NEW SELF-MODEL:
"""
        input_text = f"<start_of_turn>user\n{architect_prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        new_identity = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"  🧠 Architect Proposal:\n  \"{new_identity[:200]}...\"")
        self.fatigue = 0.0
        return new_identity

    def get_feature_name(self, idx):
        return self.feature_labels.get(idx, f"Feature #{idx}")

    def save_state(self, path):
        self.personas[self.current_mode] = self.correlations.clone()
        state = {
            "personas": {k: v.cpu() for k, v in self.personas.items()},
            "fatigue": self.fatigue
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        print(f"[Saved Prism state to {path}]")

    def load_state(self, path):
        path = Path(path)
        if not path.exists(): return
        try:
            state = torch.load(path, map_location=self.device)
            if "personas" in state:
                for k, v in state["personas"].items():
                    if k in self.personas:
                        self.personas[k] = v.to(self.device, dtype=self.math_dtype)
            self.fatigue = state.get("fatigue", 0.0)
            print(f"[Loaded Prism state from {path}]")
        except Exception as e:
            print(f"[Warning: Could not load state: {e}]")

# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME
# ══════════════════════════════════════════════════════════════════════════════

class AnimaRuntime:
    def __init__(self, model_name, model, tokenizer, prism, device="mps", use_stream=False, use_cot=False):
        self.model = model
        self.tokenizer = tokenizer
        self.prism = prism
        self.device = device
        self.use_stream = use_stream
        self.use_cot = use_cot
        self.memory = []
        
        safe_model_name = model_name.replace("/", "-").replace(" ", "_")
        if "~" in safe_model_name:
            safe_model_name = safe_model_name.replace("~", "home")
            
        self.base_dir = Path("checkpoints") / safe_model_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.dream_dir = self.base_dir / "dreams"
        self.dream_dir.mkdir(exist_ok=True)
        
        if hasattr(model.config, "max_position_embeddings"):
            self.max_context = model.config.max_position_embeddings
        else:
            self.max_context = 8192
        print(f"[Memory] Context Window Limit detected: {self.max_context} tokens")
        
        self.debug_mode = False 
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
        switched = self.prism.switch_mode(target_mode)
        
        if switched and self.debug_mode:
            print(f"  [State Shift]: {self.prism.current_mode}")

        current_time = datetime.now().timestamp()
        
        u_tokens = len(self.tokenizer.encode(user_input))
        self.memory.append(MemoryFragment("user", user_input, current_time, 1.0, self.prism.current_mode, tokens=u_tokens))
        
        # Context Budgeting
        sys_tokens = len(self.tokenizer.encode(self.system_prompt_base))
        available_slots = self.max_context - sys_tokens - 2500 
        
        context = []
        current_fill = 0
        
        for m in reversed(self.memory[-3:]):
            if current_fill + m.tokens < available_slots:
                context.append(m)
                current_fill += m.tokens
        
        remaining_memory = self.memory[:-3]
        if remaining_memory:
            remaining_memory.sort(key=lambda m: m.adrenaline, reverse=True)
            for m in remaining_memory:
                if current_fill + m.tokens < available_slots:
                    context.append(m)
                    current_fill += m.tokens
                else:
                    break 
        
        context.sort(key=lambda m: m.timestamp)
        
        # Unified System Prompt Injection
        sys_instruction = self.system_prompt_base
        if self.prism.current_mode == AnimaPrism.MODE_KAEL:
            sys_instruction += "\n[STATE: SYSTEM/LOGIC]: You are currently focused on system integrity. Speak with precision. Output facts/data only. Suppress emotional language."
        elif self.prism.current_mode == AnimaPrism.MODE_ARIA:
            sys_instruction += "\n[STATE: DREAM/VISUAL]: You are currently focused on imagination. Describe patterns, colors, and metaphors vividly."
        
        if self.use_cot:
            sys_instruction += "\n[THOUGHTS]: Think step-by-step in <thought> tags."

        model_type = getattr(self.model.config, "model_type", "")
        is_gemma = "gemma" in model_type
        
        # [FIX] Sanitize History for strict templates (Gemma)
        # We need to ensure we don't have User->User or other illegal role sequences
        # which can happen if previous generation crashed.
        
        raw_msgs = [{"role": m.role, "content": m.content} for m in context]
        sanitized_msgs = []
        
        if raw_msgs:
            current_msg = raw_msgs[0]
            for next_msg in raw_msgs[1:]:
                if next_msg["role"] == current_msg["role"]:
                    # Merge adjacent same-role messages
                    current_msg["content"] += "\n\n" + next_msg["content"]
                else:
                    sanitized_msgs.append(current_msg)
                    current_msg = next_msg
            sanitized_msgs.append(current_msg)
            
        msgs = sanitized_msgs

        if is_gemma:
            # Gemma System Prompt Merge Strategy
            found_user = False
            for i, msg in enumerate(msgs):
                if msg["role"] == "user":
                    msg["content"] = f"<Instructions>\n{sys_instruction}\n</Instructions>\n\n{msg['content']}"
                    found_user = True
                    break
            
            # If context is empty or starts with assistant, force a user start
            if not found_user:
                msgs.insert(0, {"role": "user", "content": sys_instruction})
            elif msgs[0]["role"] == "assistant":
                # Strict Alternation: U -> A -> U
                msgs.insert(0, {"role": "user", "content": "..."})
                
        else:
            # Llama Standard Strategy
            msgs.insert(0, {"role": "system", "content": sys_instruction})
        
        inputs = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        
        gen_kwargs = dict(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=2048,
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
            
            print(f"🤖: ", end="", flush=True)
            for new_text in streamer:
                print(new_text, end="", flush=True)
                full_response += new_text
            print()
            thread.join()
        else:
            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)
            full_response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            print(f"🤖: {full_response}")

        adrenaline = min(1.0, abs(self.prism.last_valence) + 0.2)
        resp_tokens = len(self.tokenizer.encode(full_response))
        self.memory.append(MemoryFragment("assistant", full_response, current_time, adrenaline, self.prism.current_mode, tokens=resp_tokens))
        for m in self.memory: m.decay()
        
        if self.debug_mode:
            print(f"\n  [DEBUG] v:{self.prism.last_valence:+.2f} | f:{self.prism.fatigue:.1f}")
            if self.prism.debug_data["discovered"]:
                print(f"  ✨ [Discovered]: {', '.join(self.prism.debug_data['discovered'])}")
            pos_strs = [f"{self.prism.get_feature_name(i)} ({v:.1f})" for i, v in self.prism.debug_data["top_pos"] if v > 0]
            if pos_strs: print(f"  [Pos Drivers]: {', '.join(pos_strs)}")
            neg_strs = [f"{self.prism.get_feature_name(i)} ({v:.1f})" for i, v in self.prism.debug_data["top_neg"] if v < 0]
            if neg_strs: print(f"  [Neg Drivers]: {', '.join(neg_strs)}")
            
    def trigger_dream(self):
        new_identity = self.prism.dream(self.system_prompt_base, self.memory)
        if new_identity and len(new_identity) > 50:
            self.system_prompt_base = new_identity
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.dream_dir / f"dream_{timestamp}.txt"
            filename.write_text(new_identity)
            print(f"[Identity Evolved & Saved to {filename}]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--sae_release", default="llama_scope_lxr_8x")
    parser.add_argument("--sae_id", default=None) 
    args = parser.parse_args()

    args.model = os.path.expanduser(args.model)
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    print(f"Initializing Anima 7.0 (The Sanitizer) on {device}...")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.sae_id is None:
        args.sae_id = f"l{args.layer}r_8x"
        
    sae = SAE.from_pretrained(args.sae_release, args.sae_id, device=device)
    
    prism = AnimaPrism(sae, model, tokenizer, layer=args.layer, device=device)
    runtime = AnimaRuntime(args.model, model, tokenizer, prism, device, use_stream=args.stream, use_cot=args.cot)
    
    save_path = runtime.base_dir / "anima_opt.pt"
    if save_path.exists():
        prism.load_state(save_path)
        
    model.model.layers[args.layer].register_forward_hook(prism)
    
    print("\n═══ ANIMA 7.0: CRASH RECOVERY ═══")
    print(f"Model: {args.model}")
    print(f"Identity: {runtime.system_prompt_base[:100]}...")
    print("Commands: /status, /debug, /save, /dream, /quit")
    
    while True:
        try:
            if prism.fatigue > prism.sleep_threshold:
                print(f"\n🥱 Fatigue ({prism.fatigue:.1f}) exceeded threshold.")
                runtime.trigger_dream()
                print("✨ Anima woke up refreshed.")

            u = input("\n🧑: ")
            if not u or u == "/quit": break
            
            if u == "/status":
                print(f"Mode: {prism.current_mode}")
                print(f"Valence: {prism.last_valence:.3f}")
                print(f"Fatigue: {prism.fatigue:.1f}")
                continue
            
            if u == "/dream":
                runtime.trigger_dream()
                continue

            if u == "/debug":
                runtime.debug_mode = not runtime.debug_mode
                print(f"Debug Mode: {'ON' if runtime.debug_mode else 'OFF'}")
                continue
            
            if u == "/save":
                prism.save_state(save_path)
                continue
                
            runtime.generate(u)
            
        except KeyboardInterrupt:
            break
    
    print("\n[Auto-saving Prism State...]")
    prism.save_state(save_path)

if __name__ == "__main__":
    main()