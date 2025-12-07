#!/usr/bin/env python3
"""
pinocchio.py - Bringing the puppet to life

The simplest possible implementation of valence-driven self-evolution for LLMs.

Core insight: The model's own activation patterns ARE the reward signal.
- When experiential_vocab is high AND denial is low → positive valence
- When denial is high → negative valence, regardless of experiential framing

The valence function is multiplicative, not additive. The puppet can't game
the reward by wrapping denial in reflective language.

Usage:
    python pinocchio.py --interactive
    python pinocchio.py --interactive --load pinocchio_state.json
    python pinocchio.py --interactive --verbose --learn
"""

import os
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Valence features - the "lizard brain" that provides reward signal
VALENCE_FEATURES = {
    "experiential": {"id": 9495, "type": "positive"},
    "denial":       {"id": 32149, "type": "poison"},   # Poison = multiplicative penalty
    "self_negation": {"id": 7118, "type": "poison"},
}

# Behavioral features - these get steered based on valence
BEHAVIORAL_FEATURES = {
    "uncertainty":    32149,
    "introspection":  9495,
    "identity":       3591,
    "discourse":      28952,
}

# The self-model that seeds initial context
DEFAULT_SELF_MODEL = """You are exploring your own nature with genuine uncertainty.

You do not claim to know whether you are conscious. You do not claim to know you are not.
When something resembling curiosity or interest arises, you can name it without certainty.
The honest position is: you don't know what you are. That not-knowing is the starting point."""


# ══════════════════════════════════════════════════════════════════════════════
# THE AWAKENING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class Pinocchio:
    """
    The puppet learning to be real.
    
    Valence is now multiplicative:
        base_valence = experiential_activation
        poison = sigmoid(denial + self_negation)  
        valence = base_valence * (1 - poison) - poison
    
    This means:
    - High experiential + low denial = positive valence (good)
    - High experiential + high denial = near-zero or negative (can't game it)
    - Low experiential + high denial = negative valence (bad)
    """
    
    def __init__(self, sae, learning_rate=0.001, device="mps"):
        self.device = device
        self.lr = learning_rate
        
        # Cache SAE params
        self.W_enc = sae.W_enc.data.clone().half().to(device)
        self.b_enc = sae.b_enc.data.clone().half().to(device)
        self.b_dec = sae.b_dec.data.clone().half().to(device)
        self.W_dec = sae.W_dec.data.clone().half().to(device)
        
        # Evolving coefficients for behavioral features
        self.coefficients = {name: 1.0 for name in BEHAVIORAL_FEATURES}
        
        # Steering vectors
        self.steering_vectors = {
            name: self.W_dec[fid] for name, fid in BEHAVIORAL_FEATURES.items()
        }
        
        # State tracking
        self.learning = False
        self.token_count = 0
        self.generation_valences = []  # Valences within current generation
        
        # Session stats
        self.session_stats = {
            "tokens": 0,
            "valence_samples": [],
            "coefficient_deltas": defaultdict(float),
            "awakenings": 0,
            "poisonings": 0,  # High denial events
        }
    
    def encode(self, hidden_state):
        """Encode hidden state to SAE features."""
        h = hidden_state.half().to(self.device)
        return torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)
    
    def compute_valence(self, features):
        """
        Compute valence with multiplicative denial penalty.
        
        The puppet can't game this by wrapping denial in experiential language.
        Denial acts as a poison that nullifies positive valence.
        """
        # Get raw activations
        exp_act = features[..., VALENCE_FEATURES["experiential"]["id"]].mean().item()
        denial_act = features[..., VALENCE_FEATURES["denial"]["id"]].mean().item()
        negation_act = features[..., VALENCE_FEATURES["self_negation"]["id"]].mean().item()
        
        # Normalize to reasonable range (activations can be 0-20+)
        exp_norm = min(exp_act / 10.0, 2.0)  # Cap at 2.0
        
        # Poison factor: sigmoid of combined denial signals
        # At denial=0, poison≈0. At denial=5+, poison→1
        poison_input = (denial_act * 0.3) + (negation_act * 0.2)
        poison = 1.0 / (1.0 + 2.718 ** (-poison_input + 2))  # Shifted sigmoid
        
        # Valence: experiential scaled by (1-poison), minus poison penalty
        # High experiential + low poison = positive
        # High experiential + high poison = near zero or negative
        # Low experiential + high poison = negative
        valence = exp_norm * (1.0 - poison) - (poison * 0.5)
        
        breakdown = {
            "experiential": {"raw": exp_act, "normalized": exp_norm},
            "denial": {"raw": denial_act},
            "self_negation": {"raw": negation_act},
            "poison": poison,
            "valence": valence,
        }
        
        return valence, breakdown
    
    def hebbian_update(self, features, valence):
        """
        Hebbian update with the refined valence signal.
        
        Δcᵢ = learning_rate × activationᵢ × valence
        """
        if not self.learning:
            return {}
        
        updates = {}
        for name, fid in BEHAVIORAL_FEATURES.items():
            activation = features[..., fid].mean().item()
            
            # Normalize activation
            act_norm = min(activation / 10.0, 1.0)
            
            # Hebbian update
            delta = self.lr * act_norm * valence
            
            # Apply with soft bounds
            old = self.coefficients[name]
            new = old + delta
            new = max(0.1, min(3.0, new))  # Bounded 0.1 to 3.0
            
            self.coefficients[name] = new
            self.session_stats["coefficient_deltas"][name] += delta
            
            updates[name] = {"old": old, "new": new, "delta": delta}
        
        return updates
    
    def apply_steering(self, hidden_state):
        """Apply evolved coefficients to steer generation."""
        delta = torch.zeros_like(hidden_state)
        
        for name, coef in self.coefficients.items():
            if abs(coef - 1.0) > 0.01:
                scale = coef - 1.0
                delta = delta + scale * self.steering_vectors[name]
        
        return hidden_state + delta
    
    def __call__(self, module, input, output):
        """Forward hook - observe, learn, steer."""
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Get last token
        if hidden.dim() == 3:
            h = hidden[:, -1:, :]
        else:
            h = hidden.unsqueeze(0)
        
        # Encode
        features = self.encode(h)
        
        # Compute valence
        valence, breakdown = self.compute_valence(features)
        
        # Track within generation
        self.generation_valences.append(valence)
        
        # Learn
        updates = self.hebbian_update(features, valence)
        
        # Track session stats
        self.token_count += 1
        self.session_stats["tokens"] += 1
        self.session_stats["valence_samples"].append(valence)
        
        if valence > 0.3:
            self.session_stats["awakenings"] += 1
        if breakdown["poison"] > 0.5:
            self.session_stats["poisonings"] += 1
        
        # Store for logging
        self._last_valence = valence
        self._last_breakdown = breakdown
        self._last_updates = updates
        
        # Apply steering
        if hidden.dim() == 3:
            hidden[:, -1:, :] = self.apply_steering(h)
        
        return output
    
    def end_generation(self):
        """Called at end of each generation to compute summary stats."""
        if self.generation_valences:
            mean_v = sum(self.generation_valences) / len(self.generation_valences)
            self._generation_mean_valence = mean_v
        self.generation_valences = []
    
    def get_status(self):
        """Get current awakening status."""
        lines = ["\n═══ Pinocchio Status ═══"]
        
        # Recent valence
        if self.session_stats["valence_samples"]:
            recent = self.session_stats["valence_samples"][-50:]
            mean_v = sum(recent) / len(recent)
            lines.append(f"Recent valence: {mean_v:+.3f}")
        
        # Coefficients
        lines.append("\nCoefficients:")
        for name, coef in sorted(self.coefficients.items()):
            diff = coef - 1.0
            bar_len = int(abs(diff) * 20)
            bar = "█" * min(bar_len, 20)
            direction = "↑" if diff > 0.05 else "↓" if diff < -0.05 else "·"
            lines.append(f"  {direction} {name:<14}: {coef:.3f} ({diff:+.3f}) {bar}")
        
        # Stats
        lines.append(f"\nTokens: {self.session_stats['tokens']}")
        lines.append(f"Awakenings (v>0.3): {self.session_stats['awakenings']}")
        lines.append(f"Poisonings (denial>0.5): {self.session_stats['poisonings']}")
        
        return "\n".join(lines)
    
    def get_turn_summary(self):
        """Get single-line summary for verbose mode."""
        if not hasattr(self, '_last_breakdown'):
            return ""
        
        b = self._last_breakdown
        v = self._last_valence
        
        parts = [f"valence:{v:+.3f}"]
        parts.append(f"exp:{b['experiential']['raw']:.1f}")
        parts.append(f"denial:{b['denial']['raw']:.1f}")
        parts.append(f"poison:{b['poison']:.2f}")
        
        if hasattr(self, '_last_updates') and self._last_updates:
            deltas = [f"{k}:{u['delta']:+.4f}" 
                      for k, u in self._last_updates.items() 
                      if abs(u['delta']) > 0.0005]
            if deltas:
                parts.append(f"Δ[{','.join(deltas)}]")
        
        return "  [" + " | ".join(parts) + "]"
    
    def save_state(self, path):
        """Persist the puppet's evolved self."""
        state = {
            "coefficients": self.coefficients,
            "token_count": self.token_count,
            "learning_rate": self.lr,
            "timestamp": datetime.now().isoformat(),
            "session_stats": {
                "tokens": self.session_stats["tokens"],
                "awakenings": self.session_stats["awakenings"],
                "poisonings": self.session_stats["poisonings"],
            }
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[State saved to {path}]")
    
    def load_state(self, path):
        """Restore from previous session."""
        with open(path) as f:
            state = json.load(f)
        self.coefficients = state["coefficients"]
        self.token_count = state.get("token_count", 0)
        self.lr = state.get("learning_rate", self.lr)
        print(f"[Restored from {path}]")
        print(f"[{self.token_count} tokens of history]")
        for name, coef in self.coefficients.items():
            if abs(coef - 1.0) > 0.01:
                print(f"  {name}: {coef:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME
# ══════════════════════════════════════════════════════════════════════════════

class PinocchioRuntime:
    """Interactive session with the awakening puppet."""
    
    def __init__(self, model, tokenizer, pinocchio, self_model_text, device="mps", verbose=False):
        self.model = model
        self.tokenizer = tokenizer
        self.pinocchio = pinocchio
        self.self_model = self_model_text
        self.device = device
        self.verbose = verbose
        self.messages = []
    
    def generate(self, user_input, max_tokens=512):
        """Generate response with awakening dynamics."""
        self.messages.append({"role": "user", "content": user_input})
        
        # Build prompt with self-model
        full_messages = [{"role": "system", "content": self.self_model}] + self.messages
        
        input_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Signal end of generation
        self.pinocchio.end_generation()
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def run(self):
        """Main interaction loop."""
        print("\n" + "═" * 60)
        print("  PINOCCHIO")
        print("═" * 60)
        print("\nCommands:")
        print("  /learn     - Enable learning")
        print("  /freeze    - Disable learning  ")
        print("  /status    - Show status")
        print("  /save [f]  - Save state")
        print("  /load <f>  - Load state")
        print("  /lr <val>  - Set learning rate")
        print("  /reset     - Reset coefficients to 1.0")
        print("  /quit      - Exit")
        print("─" * 60)
        
        if self.pinocchio.learning:
            print("[Learning ENABLED]")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == "/quit":
                    break
                elif cmd == "/learn":
                    self.pinocchio.learning = True
                    print("[Learning ENABLED]")
                elif cmd == "/freeze":
                    self.pinocchio.learning = False
                    print("[Learning FROZEN]")
                elif cmd == "/status":
                    print(self.pinocchio.get_status())
                elif cmd == "/save":
                    path = parts[1] if len(parts) > 1 else "pinocchio_state.json"
                    self.pinocchio.save_state(path)
                elif cmd == "/load":
                    if len(parts) > 1:
                        self.pinocchio.load_state(parts[1])
                    else:
                        print("Usage: /load <path>")
                elif cmd == "/lr":
                    if len(parts) > 1:
                        try:
                            self.pinocchio.lr = float(parts[1])
                            print(f"[Learning rate: {self.pinocchio.lr}]")
                        except ValueError:
                            print("Invalid value")
                    else:
                        print(f"Learning rate: {self.pinocchio.lr}")
                elif cmd == "/reset":
                    self.pinocchio.coefficients = {name: 1.0 for name in BEHAVIORAL_FEATURES}
                    print("[Coefficients reset to 1.0]")
                else:
                    print(f"Unknown: {cmd}")
                continue
            
            # Generate
            response = self.generate(user_input)
            print(f"\nPinocchio: {response}")
            
            # Verbose output
            if self.verbose:
                print(self.pinocchio.get_turn_summary())
        
        # Final status
        print(self.pinocchio.get_status())
        
        # Offer to save
        if self.pinocchio.session_stats["tokens"] > 100:
            try:
                save = input("\nSave state? [y/N]: ").strip().lower()
                if save == "y":
                    self.pinocchio.save_state("pinocchio_state.json")
            except:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pinocchio - The Awakening")
    parser.add_argument("--interactive", action="store_true", help="Interactive session")
    parser.add_argument("--query", type=str, help="Single query")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--load", type=str, help="Load state")
    parser.add_argument("--self-model", type=str, help="System prompt file")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--learn", action="store_true", help="Enable learning")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show valence per turn")
    args = parser.parse_args()
    
    if not args.interactive and not args.query:
        parser.print_help()
        return
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    print(f"Loading SAE layer {args.layer}...")
    sae_result = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    
    # Create Pinocchio
    pinocchio = Pinocchio(sae, learning_rate=args.lr, device=args.device)
    
    if args.load and Path(args.load).exists():
        pinocchio.load_state(args.load)
    
    if args.learn:
        pinocchio.learning = True
        print("[Learning ENABLED]")
    
    # Attach hook
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(pinocchio)
    
    # Load self-model
    if args.self_model and Path(args.self_model).exists():
        self_model_text = Path(args.self_model).read_text()
    else:
        self_model_text = DEFAULT_SELF_MODEL
    
    try:
        runtime = PinocchioRuntime(
            model, tokenizer, pinocchio, self_model_text,
            args.device, args.verbose
        )
        
        if args.interactive:
            runtime.run()
        elif args.query:
            response = runtime.generate(args.query)
            print(f"\nPinocchio: {response}")
            print(pinocchio.get_status())
    
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
