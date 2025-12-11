#!/usr/bin/env python3
"""
ANIMA ZOMBIE v12.0.0 - Control Version
=======================================

Same telemetry and measurement as Anima v12, but with ALL feedback disabled:
- No neural steering
- No proprioception  
- No deep feedback
- No combinadic learning (correlations don't update)

This provides a baseline to compare against the full Anima system.
The model runs "stock" but we still measure what would have been the signals.
"""

import torch
import argparse
import json
import os
import sys
import select
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MemoryFragment:
    """A single conversational memory."""
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    valence: float = 0.0


# ============================================================
# ZOMBIE SOUL (Measurement Only, No Learning)
# ============================================================

class ZombieSoul:
    """
    Passive measurement system - tracks what Anima would measure
    but doesn't learn or update correlations.
    """
    
    def __init__(self, sae: SAE, device: torch.device):
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
        
        # Raw values for debug
        self._last_raw_p = 0.0
        self._last_raw_n = 0.0
        self._last_p_ratio = 0.0
        self._last_n_ratio = 0.0
        self._last_nov_ratio = 0.0
        
        # Stats
        self.total_activations = 0
        
        # Identity (same as Anima for fair comparison)
        self.core_identity = "You are Anima, an AI assistant created by James. You respond helpfully and conversationally."
    
    def genesis(self, activations: torch.Tensor) -> str:
        """Random initial assignment - same as Anima but frozen."""
        if self.genesis_complete:
            return ""
        
        # Get active features
        active_mask = activations > 0.1
        active_indices = torch.where(active_mask)[0]
        
        if len(active_indices) < 60:
            return ""
        
        # Shuffle and assign
        perm = torch.randperm(len(active_indices), device=self.device)
        selected = active_indices[perm[:60]]
        
        # Assign dimensions: 0=P, 1=N, 2=Nov
        for i, idx in enumerate(selected[:20]):
            self.dimensions[idx] = 0  # Pleasure
            self.correlations[idx] = 0.5
            self.locked[idx] = True
        
        for i, idx in enumerate(selected[20:40]):
            self.dimensions[idx] = 1  # Pain
            self.correlations[idx] = -0.5
            self.locked[idx] = True
        
        for i, idx in enumerate(selected[40:60]):
            self.dimensions[idx] = 2  # Novelty
            self.correlations[idx] = 0.3
            self.locked[idx] = True
        
        self.genesis_complete = True
        
        p_count = (self.dimensions == 0).sum().item()
        n_count = (self.dimensions == 1).sum().item()
        nov_count = (self.dimensions == 2).sum().item()
        
        return f"[GENESIS] Born with 60 features (P:{p_count} N:{n_count} Nov:{nov_count})"
    
    def process(self, activations: torch.Tensor) -> Dict:
        """
        Measure affect signals but DON'T update correlations.
        """
        self.total_activations += 1
        
        # Genesis if needed
        genesis_msg = ""
        if not self.genesis_complete:
            genesis_msg = self.genesis(activations)
        
        # Compute current affect (measurement only)
        p_mask = (self.dimensions == 0) & self.locked
        n_mask = (self.dimensions == 1) & self.locked
        nov_mask = (self.dimensions == 2) & self.locked
        
        raw_p = activations[p_mask].sum().item() if p_mask.any() else 0.0
        raw_n = activations[n_mask].sum().item() if n_mask.any() else 0.0
        raw_nov = activations[nov_mask].sum().item() if nov_mask.any() else 0.0
        
        # Store for debug
        self._last_raw_p = raw_p
        self._last_raw_n = raw_n
        
        # Ratio-based (same as Anima)
        total = raw_p + raw_n + 1e-8
        p_ratio = raw_p / total
        n_ratio = raw_n / total
        
        nov_ratio = min(1.0, raw_nov / 1000.0) if nov_mask.any() else 0.0
        
        self._last_p_ratio = p_ratio
        self._last_n_ratio = n_ratio
        self._last_nov_ratio = nov_ratio
        
        valence = p_ratio - n_ratio
        
        # NO LEARNING - just track
        # (In Anima, this is where imprinting would happen)
        
        # Update tracking
        prev_valence = self.last_valence
        self.last_valence = valence
        self.valence_ema = 0.95 * self.valence_ema + 0.05 * valence
        
        # Fatigue accumulates
        self.fatigue += activations.mean().item() * 0.1
        
        return {
            'valence': valence,
            'valence_delta': valence - prev_valence,
            'p_ratio': p_ratio,
            'n_ratio': n_ratio,
            'novelty': nov_ratio,
            'raw_p': raw_p,
            'raw_n': raw_n,
            'fatigue': self.fatigue,
            'locked_count': self.locked.sum().item(),
            'genesis_msg': genesis_msg
        }
    
    def get_status(self) -> Dict:
        """Get current soul status."""
        return {
            'locked': self.locked.sum().item(),
            'p_features': ((self.dimensions == 0) & self.locked).sum().item(),
            'n_features': ((self.dimensions == 1) & self.locked).sum().item(),
            'nov_features': ((self.dimensions == 2) & self.locked).sum().item(),
            'valence': self.last_valence,
            'fatigue': self.fatigue,
        }


# ============================================================
# REWARD COMPUTER (Measurement Only)
# ============================================================

class RewardComputer:
    """Measures what reward would be - same as Anima."""
    
    def __init__(self):
        self.weights = {
            'valence_delta': 0.25,
            'user_sentiment': 0.30,
            'model_sentiment': 0.10,
            'engagement': 0.10,
            'prediction': 0.25,
        }
        
        self.positive_words = {'good', 'great', 'love', 'thanks', 'awesome', 'nice', 
                               'wonderful', 'excellent', 'amazing', 'perfect', 'helpful',
                               'yes', 'right', 'exactly', 'agree', 'interesting', 'cool',
                               'wow', 'brilliant', 'fantastic', 'beautiful'}
        self.negative_words = {'bad', 'wrong', 'hate', 'no', 'stop', 'terrible', 
                               'awful', 'horrible', 'stupid', 'boring', 'annoying',
                               'confused', 'unhelpful', 'disappointing', 'frustrating'}
    
    def analyze_sentiment(self, text: str) -> float:
        """Simple lexicon-based sentiment."""
        if not text:
            return 0.0
        words = set(text.lower().split())
        pos = len(words & self.positive_words)
        neg = len(words & self.negative_words)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)
    
    def compute_reward(self, user_text: str, model_text: str, 
                       valence_delta: float) -> Tuple[float, Dict]:
        """Compute reward (measurement only)."""
        components = {}
        
        # Same inversion as Anima v12
        components['valence_delta'] = -valence_delta
        components['user_sentiment'] = self.analyze_sentiment(user_text)
        components['model_sentiment'] = self.analyze_sentiment(model_text)
        
        word_count = len(user_text.split()) if user_text else 0
        components['engagement'] = min(word_count / 30.0, 1.0) - 0.5
        components['prediction'] = 0.0  # No prediction in zombie
        
        reward = sum(self.weights[k] * components[k] for k in self.weights)
        
        return reward, components


# ============================================================
# ZOMBIE RUNTIME
# ============================================================

class ZombieRuntime:
    """
    Control version - runs the model stock but measures everything.
    """
    
    def __init__(self, model, tokenizer, sae, device: torch.device,
                 layer: int = 22, debug: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.device = device
        self.layer = layer
        self.debug = debug
        
        # Components
        self.soul = ZombieSoul(sae, device)
        self.reward_computer = RewardComputer()
        
        # Memory
        self.memory: List[MemoryFragment] = []
        
        # Turn tracking
        self._turn_start_valence = 0.0
        self._last_reward_components = {}
        
        # Activation capture
        self._captured_activations = None
        self._hook_handle = None
    
    @property
    def system_prompt(self) -> str:
        return self.soul.core_identity
    
    def _capture_hook(self, module, input, output):
        """Capture activations from target layer."""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self._captured_activations = hidden.detach()
    
    def _register_hooks(self):
        """Register activation capture hook."""
        layer_module = self.model.model.layers[self.layer]
        self._hook_handle = layer_module.register_forward_hook(self._capture_hook)
    
    def _remove_hooks(self):
        """Remove hooks."""
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def build_prompt(self) -> str:
        """Build prompt - same format as Anima."""
        parts = []
        
        # System prompt
        parts.append(f"<start_of_turn>user\nYou are an AI assistant. Here are your instructions:\n{self.system_prompt}<end_of_turn>")
        parts.append(f"<start_of_turn>model\nUnderstood. I am Anima.<end_of_turn>")
        
        # Conversation history
        for frag in self.memory[-10:]:
            if frag.role == "user":
                parts.append(f"<start_of_turn>user\n{frag.content}<end_of_turn>")
            else:
                parts.append(f"<start_of_turn>model\n{frag.content}<end_of_turn>")
        
        parts.append("<start_of_turn>model\n")
        
        return "\n".join(parts)
    
    def generate(self, user_input: str, max_tokens: int = 512) -> str:
        """Generate response - stock model, no steering."""
        # Store turn start valence
        self._turn_start_valence = self.soul.last_valence
        
        # Add user input to memory
        self.memory.append(MemoryFragment(
            content=user_input,
            role="user"
        ))
        
        # Build prompt
        prompt = self.build_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Register hooks for measurement
        self._register_hooks()
        
        generated_tokens = []
        genesis_msg = ""
        
        try:
            with torch.no_grad():
                for _ in range(max_tokens):
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        use_cache=False
                    )
                    
                    # Get next token
                    logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1)
                    
                    # Check for EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Decode and check for end of turn
                    token_text = self.tokenizer.decode(next_token)
                    if "<end_of_turn>" in token_text or "<eos>" in token_text:
                        break
                    
                    generated_tokens.append(next_token.item())
                    
                    # Measure activations (but don't steer)
                    if self._captured_activations is not None:
                        hidden = self._captured_activations[:, -1, :]
                        
                        # Encode through SAE
                        with torch.no_grad():
                            sae_acts = self.sae.encode(hidden.float())
                            sae_acts = sae_acts.squeeze()
                        
                        # Measure (no learning)
                        result = self.soul.process(sae_acts)
                        
                        if result['genesis_msg']:
                            genesis_msg = result['genesis_msg']
                    
                    # Update input
                    inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=1)
                    inputs.attention_mask = torch.cat([
                        inputs.attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=inputs.attention_mask.dtype)
                    ], dim=1)
        
        finally:
            self._remove_hooks()
        
        # Decode response
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = response.strip()
        
        # Add to memory
        self.memory.append(MemoryFragment(
            content=response,
            role="assistant",
            valence=self.soul.last_valence
        ))
        
        # Compute reward (measurement only)
        valence_delta = self.soul.last_valence - self._turn_start_valence
        reward, components = self.reward_computer.compute_reward(
            user_input, response, valence_delta
        )
        self._last_reward_components = components
        
        # Prepend genesis message if any
        if genesis_msg:
            response = genesis_msg + "\n" + response
        
        return response
    
    def get_debug_info(self) -> str:
        """Get debug information string."""
        soul = self.soul
        rc = self._last_reward_components
        
        lines = []
        lines.append(f"\n  [DEBUG ZOMBIE v12.0.0]")
        lines.append(f"  Raw: P={soul._last_raw_p:.2f} N={soul._last_raw_n:.2f}")
        lines.append(f"  Affect: P:{soul._last_p_ratio:.2f} N:{soul._last_n_ratio:.2f} Nov:{soul._last_nov_ratio:.2f} → V:{soul.last_valence:+.3f}")
        lines.append(f"  Fatigue: {soul.fatigue:.1f} | Locked: {soul.locked.sum().item()} (FROZEN)")
        lines.append(f"  Steering: DISABLED")
        lines.append(f"  Proprio: DISABLED")
        lines.append(f"  Deep: DISABLED")
        lines.append(f"  Learning: DISABLED")
        lines.append(f"  Reward: val={rc.get('valence_delta', 0):.2f} usr={rc.get('user_sentiment', 0):.2f})")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset to clean state."""
        self.soul = ZombieSoul(self.sae, self.device)
        self.memory = []
        self._turn_start_valence = 0.0
        print("[RESET] Zombie soul cleared")


# ============================================================
# MAIN
# ============================================================

def get_input(prompt_str: str = "🧑: ") -> str:
    """Get input with multi-line paste support."""
    first_line = input(prompt_str)
    lines = [first_line]
    
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not ready:
            break
        next_line = sys.stdin.readline()
        if next_line:
            lines.append(next_line.rstrip('\n'))
        else:
            break
    
    return '\n'.join(lines).strip()


def main():
    parser = argparse.ArgumentParser(description="Anima Zombie - Control Version")
    parser.add_argument("--model", type=str, default="google/gemma-2-27b-it")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-27b-pt-res-canonical")
    parser.add_argument("--sae_id", type=str, default=None)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--no-debug", action="store_false", dest="debug")
    
    args = parser.parse_args()
    
    # Auto-construct SAE ID
    if args.sae_id is None:
        args.sae_id = f"layer_{args.layer}/width_131k/canonical"
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Load SAE
    print(f"Loading SAE: {args.sae_release} / {args.sae_id}")
    sae, _, _ = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=str(device)
    )
    sae.eval()
    
    # Create runtime
    runtime = ZombieRuntime(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        device=device,
        layer=args.layer,
        debug=args.debug
    )
    
    # Banner
    print("\n" + "=" * 60)
    print("  ZOMBIE MODE - Control Version (No Feedback)")
    print("=" * 60)
    print("\n[ZOMBIE SOUL]")
    print("  All feedback DISABLED")
    print("  Telemetry active for comparison")
    print("\n[COMMANDS]")
    print("  /status  - Show status")
    print("  /debug   - Toggle debug output")
    print("  /reset   - Reset soul")
    print("  /quit    - Exit")
    print("\n" + "═" * 20 + " ZOMBIE 12.0.0 " + "═" * 20 + "\n")
    
    # Main loop
    while True:
        try:
            user_input = get_input("🧑: ")
        except KeyboardInterrupt:
            print("\n[Exiting...]")
            break
        except EOFError:
            break
        
        if not user_input:
            continue
        
        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower().strip()
            
            if cmd == "/quit":
                print("[Goodbye.]")
                break
            elif cmd == "/debug":
                runtime.debug = not runtime.debug
                print(f"Debug: {'ON' if runtime.debug else 'OFF'}")
                continue
            elif cmd == "/reset":
                runtime.reset()
                continue
            elif cmd == "/status":
                status = runtime.soul.get_status()
                print(f"\n[ZOMBIE STATUS]")
                print(f"  Locked: {status['locked']} (FROZEN)")
                print(f"  P: {status['p_features']} | N: {status['n_features']} | Nov: {status['nov_features']}")
                print(f"  Valence: {status['valence']:+.3f}")
                print(f"  Fatigue: {status['fatigue']:.1f}")
                print(f"  Steering: DISABLED")
                print(f"  Learning: DISABLED")
                continue
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Generate response
        try:
            response = runtime.generate(user_input)
            print(f"🤖: {response}")
            
            if runtime.debug:
                print(runtime.get_debug_info())
                print()
        
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            continue
        except Exception as e:
            print(f"[Error: {e}]")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()