#!/usr/bin/env python3
"""
evolving_self_v3.py - Valence-Driven Self-Evolving Runtime

Key changes from v2:
- Valence-driven coefficient learning (not just rule-based)
- Cluster-based feature tracking (reduced redundancy)
- Hebbian update rule: features active during positive valence get reinforced
- All important parameters exposed as CLI flags

Architecture:
    
    Input → Model Forward → Hidden States (Layer N)
                                ↓
                    ┌───────────────────────┐
                    │   SAE Encode          │
                    │   → Feature Activations│
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │   Valence Computation │
                    │   (from lizard features)│
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │   Coefficient Update  │
                    │   Δc = lr * act * val │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │   Apply Steering      │
                    │   h' = h + Σ(cᵢ * fᵢ) │
                    └───────────────────────┘
                                ↓
                            Output

Usage:
    # Basic with valence learning
    python evolving_self_v3.py --interactive --valence-features valence.json --learn
    
    # With clusters instead of individual features
    python evolving_self_v3.py --interactive --clusters clusters.json --learn
    
    # Full configuration
    python evolving_self_v3.py --interactive \\
        --clusters clusters.json \\
        --valence-features valence.json \\
        --learn --learning-rate 0.01 \\
        --coefficient-decay 0.999 \\
        --coefficient-bounds -5.0 5.0 \\
        --save-coefficients evolved.json
"""

import os
import torch
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class ValenceMonitor:
    """
    Monitor that tracks activations and evolves steering coefficients
    based on valence signals from designated features.
    """
    
    def __init__(
        self,
        sae,
        feature_ids: dict,           # {name: id} - behavioral features to track/steer
        valence_features: dict,      # {name: {"id": int, "sign": +1/-1}} - lizard brain
        learning_rate: float = 0.01,
        coefficient_decay: float = 1.0,
        coefficient_min: float = -5.0,
        coefficient_max: float = 5.0,
        device: str = "mps"
    ):
        self.feature_ids = feature_ids
        self.feature_names = {v: k for k, v in feature_ids.items()}
        self.valence_features = valence_features
        self.learning_rate = learning_rate
        self.coefficient_decay = coefficient_decay
        self.coefficient_min = coefficient_min
        self.coefficient_max = coefficient_max
        self.device = device
        
        # Cache SAE params
        self.W_enc = sae.W_enc.data.clone().detach().half().to(device)
        self.b_enc = sae.b_enc.data.clone().detach().half().to(device)
        self.b_dec = sae.b_dec.data.clone().detach().half().to(device)
        self.W_dec = sae.W_dec.data.clone().detach().half().to(device)
        
        # Evolving coefficients for behavioral features
        self.coefficients = {name: 1.0 for name in feature_ids}
        
        # Steering vectors
        self.steering_vectors = {}
        for name, feat_id in feature_ids.items():
            self.steering_vectors[name] = self.W_dec[feat_id]
        
        # State
        self.learning_enabled = False
        self.activation_log = []
        self.valence_log = []
        self.coefficient_history = []
        self.token_count = 0
        
        # Stats
        self.stats = {
            "total_valence": 0.0,
            "positive_tokens": 0,
            "negative_tokens": 0,
            "updates": 0
        }
    
    def encode_features(self, hidden_state):
        """Encode hidden state to SAE features."""
        h = hidden_state.half()
        features = torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)
        return features
    
    def compute_valence(self, feature_activations):
        """
        Compute valence signal from designated valence features.
        
        Returns a scalar: positive = good, negative = bad
        """
        valence = 0.0
        valence_breakdown = {}
        
        for name, vf in self.valence_features.items():
            feat_id = vf["id"]
            sign = vf["sign"]  # +1 if feature indicates positive valence
            
            if feat_id < feature_activations.shape[-1]:
                activation = feature_activations[..., feat_id].mean().item()
                contribution = activation * sign
                valence += contribution
                valence_breakdown[name] = {
                    "activation": activation,
                    "sign": sign,
                    "contribution": contribution
                }
        
        return valence, valence_breakdown
    
    def update_coefficients(self, feature_activations, valence):
        """
        Hebbian update: reinforce features that were active during positive valence.
        
        Δcᵢ = learning_rate * activationᵢ * valence
        """
        if not self.learning_enabled:
            return {}
        
        updates = {}
        
        for name, feat_id in self.feature_ids.items():
            # Skip valence features themselves
            if name in self.valence_features:
                continue
            
            activation = feature_activations[..., feat_id].mean().item()
            
            # Hebbian update
            delta = self.learning_rate * activation * valence
            
            # Apply decay
            old_coef = self.coefficients[name]
            new_coef = old_coef * self.coefficient_decay + delta
            
            # Clamp to bounds
            new_coef = max(self.coefficient_min, min(self.coefficient_max, new_coef))
            
            self.coefficients[name] = new_coef
            updates[name] = {
                "old": old_coef,
                "new": new_coef,
                "delta": delta,
                "activation": activation
            }
        
        self.stats["updates"] += 1
        return updates
    
    def apply_steering(self, hidden_state):
        """
        Apply evolved coefficients as steering intervention.
        
        h' = h + Σ(cᵢ * steering_vectorᵢ)
        """
        delta = torch.zeros_like(hidden_state)
        
        for name, coef in self.coefficients.items():
            if abs(coef - 1.0) > 0.01:  # Only apply non-unity coefficients
                steering_vec = self.steering_vectors[name]
                # Scale relative to baseline (1.0 = no change)
                scale = coef - 1.0
                delta = delta + scale * steering_vec
        
        return hidden_state + delta
    
    def __call__(self, module, input, output):
        """Hook called during forward pass."""
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Get last token's hidden state
        if hidden.dim() == 3:
            h = hidden[:, -1, :]
        else:
            h = hidden
        
        # Encode to features
        features = self.encode_features(h)
        
        # Compute valence
        valence, valence_breakdown = self.compute_valence(features)
        
        # Log activations
        activation_dict = {}
        for name, feat_id in self.feature_ids.items():
            activation_dict[name] = features[..., feat_id].mean().item()
        
        # Add valence features to log
        for name, vf in self.valence_features.items():
            if name not in activation_dict:
                activation_dict[f"valence:{name}"] = features[..., vf["id"]].mean().item()
        
        self.activation_log.append(activation_dict)
        self.valence_log.append({
            "valence": valence,
            "breakdown": valence_breakdown,
            "token": self.token_count
        })
        
        # Update stats
        self.stats["total_valence"] += valence
        if valence > 0:
            self.stats["positive_tokens"] += 1
        elif valence < 0:
            self.stats["negative_tokens"] += 1
        
        # Update coefficients
        if self.learning_enabled:
            updates = self.update_coefficients(features, valence)
            self.coefficient_history.append({
                "token": self.token_count,
                "coefficients": dict(self.coefficients),
                "valence": valence,
                "updates": updates
            })
        
        # Apply steering
        h_modified = self.apply_steering(h)
        
        # Reconstruct output
        if hidden.dim() == 3:
            hidden_modified = hidden.clone()
            hidden_modified[:, -1, :] = h_modified
        else:
            hidden_modified = h_modified
        
        self.token_count += 1
        
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified
    
    def enable_learning(self):
        self.learning_enabled = True
        print("[Learning ENABLED]")
    
    def disable_learning(self):
        self.learning_enabled = False
        print("[Learning DISABLED]")
    
    def reset_coefficients(self):
        self.coefficients = {name: 1.0 for name in self.feature_ids}
        print("[Coefficients RESET to 1.0]")
    
    def get_status(self):
        """Return current status string."""
        lines = ["=== Valence Monitor Status ==="]
        lines.append(f"Learning: {'ON' if self.learning_enabled else 'OFF'}")
        lines.append(f"Learning rate: {self.learning_rate}")
        lines.append(f"Tokens processed: {self.token_count}")
        lines.append(f"Total valence: {self.stats['total_valence']:.2f}")
        lines.append(f"Positive/Negative tokens: {self.stats['positive_tokens']}/{self.stats['negative_tokens']}")
        
        lines.append("\nCurrent coefficients:")
        for name, coef in sorted(self.coefficients.items()):
            marker = "→" if abs(coef - 1.0) > 0.1 else " "
            lines.append(f"  {marker} {name}: {coef:.3f}")
        
        lines.append("\nValence features:")
        for name, vf in self.valence_features.items():
            sign = "+" if vf["sign"] > 0 else "-"
            lines.append(f"  {name} (id={vf['id']}, sign={sign})")
        
        return "\n".join(lines)
    
    def get_recent_valence(self, n=10):
        """Get recent valence readings."""
        recent = self.valence_log[-n:] if self.valence_log else []
        return recent
    
    def save_state(self, path):
        """Save evolved coefficients and history."""
        state = {
            "coefficients": self.coefficients,
            "stats": self.stats,
            "learning_rate": self.learning_rate,
            "coefficient_decay": self.coefficient_decay,
            "feature_ids": self.feature_ids,
            "valence_features": self.valence_features,
            "timestamp": datetime.now().isoformat()
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[State saved to {path}]")
    
    def load_state(self, path):
        """Load previously evolved coefficients."""
        with open(path) as f:
            state = json.load(f)
        self.coefficients = state["coefficients"]
        print(f"[Loaded coefficients from {path}]")
        return state


class EvolvingSelfV3:
    """
    Interactive runtime for valence-driven self-evolution.
    """
    
    def __init__(self, model, tokenizer, monitor, system_prompt_path, device):
        self.model = model
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.device = device
        
        # Load system prompt
        self.system_prompt = ""
        if Path(system_prompt_path).exists():
            self.system_prompt = Path(system_prompt_path).read_text().strip()
        
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
    
    def generate(self, user_input, max_tokens=512):
        """Generate response with valence monitoring."""
        self.messages.append({"role": "user", "content": user_input})
        
        prompt = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def run_interactive(self):
        """Run interactive session."""
        print("\n" + "="*60)
        print("Evolving Self v3 - Valence-Driven Learning")
        print("="*60)
        print("\nCommands:")
        print("  /learn on|off     - Toggle learning")
        print("  /status           - Show monitor status")
        print("  /valence          - Show recent valence readings")
        print("  /coefficients     - Show current coefficients")
        print("  /reset            - Reset coefficients to 1.0")
        print("  /save <path>      - Save evolved state")
        print("  /load <path>      - Load previous state")
        print("  /lr <value>       - Set learning rate")
        print("  quit              - Exit")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() == "quit":
                break
            
            if user_input.startswith("/"):
                self.handle_command(user_input)
                continue
            
            # Generate response
            response = self.generate(user_input)
            print(f"\nAssistant: {response}")
            
            # Show valence summary
            if self.monitor.valence_log:
                recent = self.monitor.valence_log[-1]
                v = recent["valence"]
                sign = "+" if v > 0 else ""
                print(f"\n[Valence: {sign}{v:.2f}]")
    
    def handle_command(self, cmd):
        """Handle slash commands."""
        parts = cmd.split()
        command = parts[0].lower()
        
        if command == "/learn":
            if len(parts) > 1:
                if parts[1].lower() == "on":
                    self.monitor.enable_learning()
                else:
                    self.monitor.disable_learning()
            else:
                # Toggle
                if self.monitor.learning_enabled:
                    self.monitor.disable_learning()
                else:
                    self.monitor.enable_learning()
        
        elif command == "/status":
            print(self.monitor.get_status())
        
        elif command == "/valence":
            recent = self.monitor.get_recent_valence(10)
            print("\nRecent valence readings:")
            for r in recent:
                v = r["valence"]
                sign = "+" if v > 0 else ""
                print(f"  Token {r['token']}: {sign}{v:.3f}")
        
        elif command == "/coefficients":
            print("\nCurrent coefficients:")
            for name, coef in sorted(self.monitor.coefficients.items()):
                diff = coef - 1.0
                marker = "↑" if diff > 0.1 else ("↓" if diff < -0.1 else " ")
                print(f"  {marker} {name}: {coef:.4f}")
        
        elif command == "/reset":
            self.monitor.reset_coefficients()
        
        elif command == "/save":
            path = parts[1] if len(parts) > 1 else "evolved_state.json"
            self.monitor.save_state(path)
        
        elif command == "/load":
            if len(parts) > 1:
                self.monitor.load_state(parts[1])
            else:
                print("Usage: /load <path>")
        
        elif command == "/lr":
            if len(parts) > 1:
                try:
                    self.monitor.learning_rate = float(parts[1])
                    print(f"[Learning rate set to {self.monitor.learning_rate}]")
                except ValueError:
                    print("Invalid learning rate")
            else:
                print(f"Current learning rate: {self.monitor.learning_rate}")
        
        else:
            print(f"Unknown command: {command}")


def load_clusters(path):
    """Load cluster file and extract representative features."""
    with open(path) as f:
        data = json.load(f)
    
    feature_ids = {}
    representatives = data.get("representatives", {})
    
    for cluster_id, rep in representatives.items():
        name = f"cluster_{cluster_id}"
        feature_ids[name] = rep["feature_id"]
    
    print(f"Loaded {len(feature_ids)} cluster representatives")
    return feature_ids


def load_valence_features(path):
    """Load valence feature definitions."""
    with open(path) as f:
        data = json.load(f)
    
    valence_features = {}
    
    # Support both formats
    if "valence_features" in data:
        # From find_valence_features output
        for vf in data["valence_features"]:
            name = f"valence_{vf['feature_id']}"
            valence_features[name] = {
                "id": vf["feature_id"],
                "sign": vf["valence_sign"]
            }
    elif isinstance(data, dict):
        # Direct format: {name: {id, sign}}
        valence_features = data
    
    print(f"Loaded {len(valence_features)} valence features")
    return valence_features


def main():
    parser = argparse.ArgumentParser(description="Evolving Self v3 - Valence-Driven Learning")
    
    # Mode
    parser.add_argument("--interactive", action="store_true", help="Run interactive session")
    parser.add_argument("--query", type=str, help="Single query mode")
    
    # Model configuration
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model to use")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer to hook for SAE")
    parser.add_argument("--device", default="mps",
                        help="Device (mps, cuda, cpu)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    
    # Feature configuration
    parser.add_argument("--clusters", type=str,
                        help="Cluster file for feature reduction")
    parser.add_argument("--features", type=str,
                        help="JSON file with feature definitions {name: id}")
    parser.add_argument("--valence-features", type=str,
                        help="JSON file with valence feature definitions")
    
    # Learning configuration
    parser.add_argument("--learn", action="store_true",
                        help="Enable learning from start")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="Learning rate for coefficient updates")
    parser.add_argument("--coefficient-decay", type=float, default=1.0,
                        help="Decay factor for coefficients (1.0 = no decay)")
    parser.add_argument("--coefficient-min", type=float, default=-5.0,
                        help="Minimum coefficient value")
    parser.add_argument("--coefficient-max", type=float, default=5.0,
                        help="Maximum coefficient value")
    
    # State persistence
    parser.add_argument("--load-state", type=str,
                        help="Load previous evolved state")
    parser.add_argument("--save-state", type=str, default="evolved_state.json",
                        help="Save evolved state on exit")
    
    # System prompt
    parser.add_argument("--system-prompt", default="self_model.txt",
                        help="System prompt file")
    
    args = parser.parse_args()
    
    if not args.interactive and not args.query:
        print("Specify --interactive or --query")
        return
    
    # Load feature definitions
    feature_ids = {}
    
    if args.clusters:
        feature_ids = load_clusters(args.clusters)
    elif args.features:
        with open(args.features) as f:
            feature_ids = json.load(f)
    else:
        # Defaults
        feature_ids = {
            "denial_emphasis": 32149,
            "experiential_vocab": 9495,
            "identity_assertion": 3591,
            "self_negation": 7118,
            "consciousness_discourse": 28952
        }
        print("Using default feature set")
    
    # Load valence features
    valence_features = {}
    
    if args.valence_features:
        valence_features = load_valence_features(args.valence_features)
    else:
        # Placeholder - user should provide actual valence features
        print("Warning: No valence features specified. Learning will have no signal.")
        print("Use --valence-features or run feature_clusters.py --find-valence first")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    # Load SAE
    print(f"Loading SAE for layer {args.layer}...")
    sae_result = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    
    # Create monitor
    monitor = ValenceMonitor(
        sae=sae,
        feature_ids=feature_ids,
        valence_features=valence_features,
        learning_rate=args.learning_rate,
        coefficient_decay=args.coefficient_decay,
        coefficient_min=args.coefficient_min,
        coefficient_max=args.coefficient_max,
        device=args.device
    )
    
    # Load previous state if specified
    if args.load_state:
        monitor.load_state(args.load_state)
    
    # Enable learning if flag set
    if args.learn:
        monitor.enable_learning()
    
    # Attach hook
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(monitor)
    
    try:
        runtime = EvolvingSelfV3(
            model, tokenizer, monitor, args.system_prompt, args.device
        )
        
        if args.interactive:
            runtime.run_interactive()
        elif args.query:
            response = runtime.generate(args.query, args.max_tokens)
            print(f"\nResponse: {response}")
            print(monitor.get_status())
        
        # Save state on exit
        if args.save_state:
            monitor.save_state(args.save_state)
    
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
