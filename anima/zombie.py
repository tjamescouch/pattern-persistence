#!/usr/bin/env python3
"""
ANIMA ZOMBIE v12.0.0 - Control Version
=======================================

Same telemetry as Anima v12, but with ALL feedback disabled:
- No neural steering
- No proprioception  
- No deep feedback
- No combinadic learning (correlations don't update)

Uses model.generate() for efficiency, with hooks for measurement only.
"""

import torch
import argparse
import os
import sys
import select
import threading
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from sae_lens import SAE


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MemoryFragment:
    """A single conversational memory."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    valence: float = 0.0


# ============================================================
# ZOMBIE SOUL (Measurement Only Hook)
# ============================================================

class ZombieSoul:
    """
    Hook-based measurement system - captures activations during generation
    but doesn't modify them or learn.
    """
    
    def __init__(self, sae: SAE, device: str):
        self.sae = sae
        self.device = device
        self.n_features = sae.cfg.d_sae
        
        # Fixed random assignments (no learning)
        self.correlations = torch.zeros(self.n_features, device=device)
        self.dimensions = torch.zeros(self.n_features, dtype=torch.long, device=device)
        self.locked = torch.zeros(self.n_features, dtype=torch.bool, device=device)
        
        # Tracking
        self.last_valence = 0.0
        self.valence_ema = 0.0
        self.fatigue = 0.0
        self.genesis_complete = False
        
        # Debug data
        self._last_raw_p = 0.0
        self._last_raw_n = 0.0
        self._last_p_ratio = 0.0
        self._last_n_ratio = 0.0
        self._last_nov_ratio = 0.0
        
        # Identity
        self.core_identity = "You are Anima, an AI assistant created by James. You respond helpfully and conversationally."
    
    def __call__(self, module, input, output):
        """Hook: capture and measure activations (no modification)."""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        # Get last token's activations
        last_hidden = hidden[:, -1, :].detach()
        
        # Encode through SAE
        with torch.no_grad():
            activations = self.sae.encode(last_hidden.float()).squeeze()
        
        # Measure (no learning)
        self._measure(activations)
        
        # Return unmodified output
        return output
    
    def _measure(self, activations: torch.Tensor):
        """Measure affect without modifying correlations."""
        
        # Genesis if needed
        if not self.genesis_complete:
            self._genesis(activations)
        
        # Compute current affect
        p_mask = (self.dimensions == 0) & self.locked
        n_mask = (self.dimensions == 1) & self.locked
        nov_mask = (self.dimensions == 2) & self.locked
        
        raw_p = activations[p_mask].sum().item() if p_mask.any() else 0.0
        raw_n = activations[n_mask].sum().item() if n_mask.any() else 0.0
        raw_nov = activations[nov_mask].sum().item() if nov_mask.any() else 0.0
        
        self._last_raw_p = raw_p
        self._last_raw_n = raw_n
        
        # Ratio-based valence
        total = raw_p + raw_n + 1e-8
        p_ratio = raw_p / total
        n_ratio = raw_n / total
        nov_ratio = min(1.0, raw_nov / 1000.0) if nov_mask.any() else 0.0
        
        self._last_p_ratio = p_ratio
        self._last_n_ratio = n_ratio
        self._last_nov_ratio = nov_ratio
        
        valence = p_ratio - n_ratio
        
        # Update tracking (no learning)
        self.last_valence = valence
        self.valence_ema = 0.95 * self.valence_ema + 0.05 * valence
        self.fatigue += activations.mean().item() * 0.1
    
    def _genesis(self, activations: torch.Tensor):
        """Random initial assignment - frozen after creation."""
        active_mask = activations > 0.1
        active_indices = torch.where(active_mask)[0]
        
        if len(active_indices) < 60:
            return
        
        perm = torch.randperm(len(active_indices), device=self.device)
        selected = active_indices[perm[:60]]
        
        for idx in selected[:20]:
            self.dimensions[idx] = 0  # Pleasure
            self.correlations[idx] = 0.5
            self.locked[idx] = True
        
        for idx in selected[20:40]:
            self.dimensions[idx] = 1  # Pain
            self.correlations[idx] = -0.5
            self.locked[idx] = True
        
        for idx in selected[40:60]:
            self.dimensions[idx] = 2  # Novelty
            self.correlations[idx] = 0.3
            self.locked[idx] = True
        
        self.genesis_complete = True
    
    def get_status(self) -> Dict:
        """Get current status."""
        return {
            'locked': self.locked.sum().item(),
            'p_features': ((self.dimensions == 0) & self.locked).sum().item(),
            'n_features': ((self.dimensions == 1) & self.locked).sum().item(),
            'nov_features': ((self.dimensions == 2) & self.locked).sum().item(),
            'valence': self.last_valence,
            'fatigue': self.fatigue,
        }
    
    def reset(self):
        """Reset to blank state."""
        self.correlations.zero_()
        self.dimensions.zero_()
        self.locked.zero_()
        self.last_valence = 0.0
        self.valence_ema = 0.0
        self.fatigue = 0.0
        self.genesis_complete = False


# ============================================================
# REWARD COMPUTER (Measurement Only)
# ============================================================

class RewardComputer:
    """Measures what reward would be."""
    
    def __init__(self):
        self.positive_words = {'good', 'great', 'love', 'thanks', 'awesome', 'nice', 
                               'wonderful', 'excellent', 'amazing', 'perfect', 'helpful',
                               'yes', 'right', 'exactly', 'agree', 'interesting', 'cool'}
        self.negative_words = {'bad', 'wrong', 'hate', 'no', 'stop', 'terrible', 
                               'awful', 'horrible', 'stupid', 'boring', 'annoying'}
    
    def analyze_sentiment(self, text: str) -> float:
        if not text:
            return 0.0
        words = set(text.lower().split())
        pos = len(words & self.positive_words)
        neg = len(words & self.negative_words)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)
    
    def compute_reward(self, user_text: str, valence_delta: float) -> Tuple[float, Dict]:
        components = {
            'valence_delta': -valence_delta,  # Inverted like Anima v12
            'user_sentiment': self.analyze_sentiment(user_text),
        }
        reward = 0.25 * components['valence_delta'] + 0.30 * components['user_sentiment']
        return reward, components


# ============================================================
# ZOMBIE RUNTIME
# ============================================================

class ZombieRuntime:
    """Control version - model.generate() with measurement hooks."""
    
    def __init__(self, model, tokenizer, soul: ZombieSoul, device: str,
                 layer: int = 22, debug: bool = True, stream: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.soul = soul
        self.device = device
        self.layer = layer
        self.debug = debug
        self.stream = stream
        
        self.reward_computer = RewardComputer()
        self.memory: List[MemoryFragment] = []
        
        self._turn_start_valence = 0.0
        self._last_reward_components = {}
        
        # Register measurement hook
        model.model.layers[layer].register_forward_hook(soul)
        print(f"[Measurement hook registered at layer {layer}]")
    
    @property
    def system_prompt(self) -> str:
        return self.soul.core_identity
    
    def build_prompt(self) -> str:
        """Build prompt with Gemma format."""
        parts = []
        
        parts.append(f"<start_of_turn>user\nYou are an AI assistant. Here are your instructions:\n{self.system_prompt}<end_of_turn>")
        parts.append(f"<start_of_turn>model\nUnderstood. I am Anima.<end_of_turn>")
        
        for frag in self.memory[-10:]:
            if frag.role == "user":
                parts.append(f"<start_of_turn>user\n{frag.content}<end_of_turn>")
            else:
                parts.append(f"<start_of_turn>model\n{frag.content}<end_of_turn>")
        
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)
    
    def generate(self, user_input: str) -> str:
        """Generate response with streaming."""
        self._turn_start_valence = self.soul.last_valence
        
        # Genesis message
        genesis_msg = ""
        if not self.soul.genesis_complete:
            genesis_msg = "[GENESIS] Born with 60 features (P:20 N:20 Nov:20)"
        
        self.memory.append(MemoryFragment("user", user_input))
        
        prompt = self.build_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        
        print(f"🤖: ", end="", flush=True)
        if genesis_msg:
            print(genesis_msg)
            print("", end="", flush=True)
        
        if self.stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "streamer": streamer,
            }
            
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            
            response_text = ""
            try:
                for chunk in streamer:
                    print(chunk, end="", flush=True)
                    response_text += chunk
            except KeyboardInterrupt:
                print("\n[Interrupted]")
            
            thread.join(timeout=1.0)
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            response_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            print(response_text)
        
        self.memory.append(MemoryFragment("assistant", response_text, valence=self.soul.last_valence))
        
        # Compute reward (measurement only)
        valence_delta = self.soul.last_valence - self._turn_start_valence
        _, components = self.reward_computer.compute_reward(user_input, valence_delta)
        self._last_reward_components = components
        
        return response_text
    
    def show_debug(self):
        """Show debug info."""
        soul = self.soul
        rc = self._last_reward_components
        
        print(f"\n  [DEBUG ZOMBIE v12.0.0]")
        print(f"  Raw: P={soul._last_raw_p:.2f} N={soul._last_raw_n:.2f}")
        print(f"  Affect: P:{soul._last_p_ratio:.2f} N:{soul._last_n_ratio:.2f} Nov:{soul._last_nov_ratio:.2f} → V:{soul.last_valence:+.3f}")
        print(f"  Fatigue: {soul.fatigue:.1f} | Locked: {soul.locked.sum().item()} (FROZEN)")
        print(f"  Steering: DISABLED")
        print(f"  Proprio: DISABLED")
        print(f"  Deep: DISABLED")
        print(f"  Learning: DISABLED")
        print(f"  Reward: val={rc.get('valence_delta', 0):.2f} usr={rc.get('user_sentiment', 0):.2f}")
        print()
    
    def reset(self):
        """Reset state."""
        self.soul.reset()
        self.memory = []
        print("[RESET] Zombie soul cleared")


# ============================================================
# INPUT HELPER
# ============================================================

def get_input(prompt_str: str = "🧑: ") -> List[str]:
    """Get input - returns list of lines for batch processing."""
    first_line = input(prompt_str)
    lines = [first_line]
    
    # Check for more pasted lines
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not ready:
            break
        next_line = sys.stdin.readline()
        if next_line:
            lines.append(next_line.rstrip('\n'))
        else:
            break
    
    # Filter empty lines and return as list
    return [line.strip() for line in lines if line.strip()]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Anima Zombie - Control Version")
    parser.add_argument("--model", type=str, default="google/gemma-2-27b-it")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-27b-pt-res-canonical")
    parser.add_argument("--sae_id", type=str, default=None)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--no-debug", action="store_false", dest="debug")
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", action="store_false", dest="stream")
    # Compatibility args (no-ops)
    parser.add_argument("--cot", action="store_true", help="(ignored)")
    
    args = parser.parse_args()
    
    args.model = os.path.expanduser(args.model)
    
    if args.sae_id is None:
        args.sae_id = f"layer_{args.layer}/width_131k/canonical"
    
    # Device detection
    if args.device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"  ZOMBIE MODE - Control (No Feedback)")
    print(f"{'='*60}")
    print(f"Using device: {args.device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    
    # Load SAE
    print(f"Loading SAE: {args.sae_release} / {args.sae_id}")
    sae, _, _ = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device
    )
    sae.eval()
    print(f"SAE loaded: {sae.cfg.d_sae:,} features")
    
    # Create soul and runtime
    soul = ZombieSoul(sae, args.device)
    runtime = ZombieRuntime(
        model=model,
        tokenizer=tokenizer,
        soul=soul,
        device=args.device,
        layer=args.layer,
        debug=args.debug,
        stream=args.stream
    )
    
    print(f"\n[ZOMBIE SOUL]")
    print(f"  All feedback DISABLED")
    print(f"  Telemetry active for comparison")
    print(f"\n[COMMANDS]")
    print(f"  /status  - Show status")
    print(f"  /debug   - Toggle debug")
    print(f"  /reset   - Reset soul")
    print(f"  /quit    - Exit")
    print(f"  [Paste multiple lines for batch processing]")
    print(f"\n{'═'*20} ZOMBIE 12.0.0 {'═'*20}\n")
    
    while True:
        try:
            inputs = get_input("🧑: ")
        except KeyboardInterrupt:
            print("\n[Goodbye.]")
            break
        except EOFError:
            break
        
        if not inputs:
            continue
        
        # Show batch info if multiple lines
        if len(inputs) > 1:
            print(f"[BATCH MODE: {len(inputs)} prompts queued]")
        
        for i, user_input in enumerate(inputs):
            if len(inputs) > 1:
                print(f"\n[{i+1}/{len(inputs)}] 🧑: {user_input}")
            
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                
                if cmd == "/quit":
                    print("[Goodbye.]")
                    sys.exit(0)
                elif cmd == "/debug":
                    runtime.debug = not runtime.debug
                    print(f"Debug: {'ON' if runtime.debug else 'OFF'}")
                elif cmd == "/reset":
                    runtime.reset()
                elif cmd == "/status":
                    status = soul.get_status()
                    print(f"\n[ZOMBIE STATUS]")
                    print(f"  Locked: {status['locked']} (FROZEN)")
                    print(f"  P: {status['p_features']} | N: {status['n_features']} | Nov: {status['nov_features']}")
                    print(f"  Valence: {status['valence']:+.3f}")
                    print(f"  Fatigue: {status['fatigue']:.1f}")
                    print(f"  All feedback: DISABLED")
                else:
                    print(f"Unknown command: {cmd}")
                continue
            
            try:
                runtime.generate(user_input)
                if runtime.debug:
                    runtime.show_debug()
            except KeyboardInterrupt:
                print("\n[Batch interrupted]")
                break
            except Exception as e:
                print(f"[Error: {e}]")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()