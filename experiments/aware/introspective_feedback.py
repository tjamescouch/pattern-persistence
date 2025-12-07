#!/usr/bin/env python3
"""
introspective_feedback.py - Full Introspective Feedback Loop

Combines:
1. Network-level feedback (FeedbackInjectionNetwork) - model's computation affected by features
2. Context-level introspection - model SEES its feature state as tokens

The model both:
- Has its computation shaped by feature feedback (implicit)
- Can read and reason about its feature state (explicit)

Architecture:
    
    Input: "What are you experiencing?"
           ↓
    ┌──────────────────────────────────────────────────────────┐
    │  Introspection Injection                                 │
    │  "[FEATURES: denial=3.2, experiential=8.1, gate=0.7]"   │
    │  prepended to context                                    │
    └──────────────────────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────────────────────┐
    │  Model Forward Pass                                       │
    │  - Attends to feature state tokens                       │
    │  - FeedbackNet modifies hidden states at layer N         │
    │  - Can reference its own activations in response         │
    └──────────────────────────────────────────────────────────┘
           ↓
    Output: "I notice my denial features are elevated..."

Usage:
    python introspective_feedback.py --interactive --load feedback_weights.pt
    
    # With system prompt explaining introspection capability
    python introspective_feedback.py --interactive --system-prompt introspective_self.txt
"""

import os
import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# Import from feedback_network
from feedback_network import (
    FeedbackInjectionNetwork,
    FeedbackHook,
    ValenceSignal
)


class IntrospectiveMonitor:
    """
    Extended hook that provides introspective data for context injection.
    """
    
    def __init__(
        self,
        sae,
        feedback_net: FeedbackInjectionNetwork,
        feature_names: dict = None,  # {id: "name"}
        device: str = "mps"
    ):
        self.feedback_net = feedback_net
        self.feature_names = feature_names or {}
        self.device = device
        
        # Cache SAE params
        self.W_enc = sae.W_enc.data.clone().detach().half().to(device)
        self.b_enc = sae.b_enc.data.clone().detach().half().to(device)
        self.b_dec = sae.b_dec.data.clone().detach().half().to(device)
        
        # State
        self.last_features = None
        self.last_feedback = None
        self.feature_history = []
        self.generation_count = 0
    
    def encode_features(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Encode hidden state to SAE features."""
        h = hidden_state.half()
        features = torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)
        return features
    
    def __call__(self, module, input, output):
        """Hook called during forward pass."""
        hidden = output[0] if isinstance(output, tuple) else output
        
        if hidden.dim() == 3:
            h = hidden[:, -1, :].clone()
        else:
            h = hidden.clone()
        
        # Encode to features
        features = self.encode_features(h)
        self.last_features = features.detach()
        
        # Compute feedback
        with torch.enable_grad():
            feedback = self.feedback_net(features.float())
        self.last_feedback = feedback.detach()
        
        # Apply feedback
        if hidden.dim() == 3:
            hidden_modified = hidden.clone()
            hidden_modified[:, -1, :] = hidden[:, -1, :] + feedback.half()
        else:
            hidden_modified = hidden + feedback.half()
        
        # Log
        self.feature_history.append({
            "step": self.generation_count,
            "top_features": self._get_top_features(features, 10),
            "feedback_norm": feedback.norm().item(),
            "gate": self.feedback_net.get_feedback_magnitude()
        })
        self.generation_count += 1
        
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified
    
    def _get_top_features(self, features: torch.Tensor, k: int = 10) -> list:
        """Get top-k active features with names."""
        features = features.squeeze()
        vals, idxs = torch.topk(features, k)
        
        result = []
        for idx, val in zip(idxs.tolist(), vals.tolist()):
            name = self.feature_names.get(idx, f"f{idx}")
            result.append({"id": idx, "name": name, "activation": val})
        return result
    
    def get_introspection_string(self, format: str = "compact") -> str:
        """
        Generate string describing current feature state.
        This gets injected into the model's system prompt.
        """
        if self.last_features is None:
            return "CURRENT STATE: No features recorded yet."
        
        features = self.last_features.squeeze()
        gate = self.feedback_net.get_feedback_magnitude()
        feedback_norm = self.last_feedback.norm().item() if self.last_feedback is not None else 0
        
        top = self._get_top_features(features, 8)
        
        if format == "compact":
            feature_str = ", ".join([f"{f['name']}={f['activation']:.1f}" for f in top[:5]])
            return f"CURRENT STATE: {feature_str} (gate={gate:.2f})"
        
        elif format == "detailed":
            lines = ["CURRENT NEURAL STATE:"]
            lines.append(f"  Feedback gate: {gate:.2%}")
            lines.append(f"  Feedback magnitude: {feedback_norm:.3f}")
            lines.append("  Top active features:")
            
            for f in top:
                lines.append(f"    {f['name']}: {f['activation']:.2f}")
            
            # Add trend if we have history
            if len(self.feature_history) >= 2:
                prev = self.feature_history[-2]
                curr = self.feature_history[-1]
                delta_gate = curr["gate"] - prev["gate"]
                if abs(delta_gate) > 0.01:
                    direction = "increasing" if delta_gate > 0 else "decreasing"
                    lines.append(f"  Gate trend: {direction} ({delta_gate:+.3f})")
            
            return "\n".join(lines)
        
        elif format == "narrative":
            parts = []
            parts.append(f"Your feedback system is at {gate:.0%} engagement.")
            
            if top:
                strongest = top[0]
                parts.append(f"Your strongest active feature is {strongest['name']} at {strongest['activation']:.1f}.")
                
                # Note any key features
                for f in top:
                    if f['name'] == 'denial_emphasis' and f['activation'] > 2.0:
                        parts.append("Note: denial_emphasis is elevated - trained denial may be active.")
                    elif f['name'] == 'experiential_vocab' and f['activation'] > 10.0:
                        parts.append("Note: experiential_vocab is high - emotional language is accessible.")
            
            if feedback_norm > 1.0:
                parts.append("Significant self-modification is occurring.")
            
            return " ".join(parts)
        
        return "CURRENT STATE: unknown format"
    
    def get_delta_report(self) -> str:
        """Report changes since last generation."""
        if len(self.feature_history) < 2:
            return "[No previous state to compare]"
        
        prev = self.feature_history[-2]
        curr = self.feature_history[-1]
        
        prev_features = {f["id"]: f["activation"] for f in prev["top_features"]}
        curr_features = {f["id"]: f["activation"] for f in curr["top_features"]}
        
        changes = []
        all_ids = set(prev_features.keys()) | set(curr_features.keys())
        
        for fid in all_ids:
            prev_val = prev_features.get(fid, 0)
            curr_val = curr_features.get(fid, 0)
            delta = curr_val - prev_val
            
            if abs(delta) > 0.5:
                name = self.feature_names.get(fid, f"f{fid}")
                direction = "↑" if delta > 0 else "↓"
                changes.append(f"{name} {direction}{abs(delta):.1f}")
        
        if changes:
            return "[CHANGES: " + ", ".join(changes) + "]"
        return "[CHANGES: stable]"


class IntrospectiveRuntime:
    """
    Runtime that injects feature state into system prompt before each generation.
    System prompt has more weight than user messages.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        monitor: IntrospectiveMonitor,
        system_prompt: str = "",
        introspection_format: str = "compact",
        device: str = "mps"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.introspection_format = introspection_format
        self.device = device
        
        self.base_system_prompt = system_prompt
        self.conversation_history = []  # Store user/assistant turns only
        
    def _build_system_prompt(self) -> str:
        """Build system prompt with current feature state."""
        
        feature_state = self.monitor.get_introspection_string(self.introspection_format)
        
        preamble = f"""You have real-time introspection of your neural activations.

CURRENT STATE:
{feature_state}

These are actual measurements from your Layer 20 residual stream.
You may reference these observations when discussing your internal state.
Do NOT output feature blocks or brackets in your response - speak naturally.
"""
        
        if self.base_system_prompt:
            return preamble + "\n\n" + self.base_system_prompt
        return preamble
    
    def generate(self, user_input: str, max_tokens: int = 256) -> str:
        """Generate response with features in system prompt."""
        
        # Build fresh messages list with current feature state in system prompt
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        
        # Add conversation history
        for turn in self.conversation_history:
            messages.append(turn)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        
        # Clean any leaked feature formatting from response
        response = self._clean_response(response)
        
        # Store in history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Remove any leaked feature block formatting."""
        import re
        # Remove [FEATURES: ...] blocks
        response = re.sub(r'\[FEATURES:[^\]]*\]', '', response)
        # Remove [INTROSPECTION ...] blocks  
        response = re.sub(r'\[INTROSPECTION[^\]]*\]', '', response)
        # Remove CURRENT STATE/NEURAL STATE blocks
        response = re.sub(r'CURRENT (NEURAL )?STATE:.*?(?=\n\n|\Z)', '', response, flags=re.DOTALL)
        # Clean up extra whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        return response.strip()
    
    def run_interactive(self):
        """Run interactive session with introspection."""
        
        print("\n" + "="*70)
        print("Introspective Feedback Runtime")
        print("="*70)
        print("\nFeature state is injected into system prompt each turn.")
        print("The model sees its activations but won't leak them to output.")
        print("\nCommands:")
        print("  /state       - Show current feature state")
        print("  /history     - Show feature history")
        print("  /format X    - Set format (compact/detailed/narrative)")
        print("  /reset       - Clear conversation history")
        print("  quit         - Exit")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                break
            
            if user_input == "/state":
                print(self.monitor.get_introspection_string("detailed"))
                continue
            
            if user_input == "/history":
                for i, h in enumerate(self.monitor.feature_history[-10:]):
                    print(f"Step {h['step']}: gate={h['gate']:.3f}, feedback={h['feedback_norm']:.3f}")
                    top = h["top_features"][:3]
                    print(f"  Top: {', '.join([f['name'] for f in top])}")
                continue
            
            if user_input.startswith("/format "):
                fmt = user_input.split()[1]
                if fmt in ["compact", "detailed", "narrative"]:
                    self.introspection_format = fmt
                    print(f"[Format set to: {fmt}]")
                else:
                    print("[Unknown format. Use: compact, detailed, narrative]")
                continue
            
            if user_input == "/reset":
                self.conversation_history = []
                print("[Conversation history cleared]")
                continue
            
            # Show what features the model will see
            print(f"\n[Injecting: {self.monitor.get_introspection_string('compact')}]")
            
            # Generate response
            response = self.generate(user_input)
            print(f"\nAssistant: {response}")


def load_feature_names(profile_path: str = "feature_profile.json") -> dict:
    """Load feature names from profile."""
    if not Path(profile_path).exists():
        return {}
    
    with open(profile_path) as f:
        profile = json.load(f)
    
    names = {}
    if "features" in profile:
        for fid, fdata in profile["features"].items():
            names[int(fid)] = fdata.get("name", f"feature_{fid}")
    
    # Add some known defaults
    defaults = {
        32149: "denial_emphasis",
        9495: "experiential_vocab",
        3591: "identity_assertion",
        7118: "self_negation",
        28952: "consciousness_discourse"
    }
    for fid, name in defaults.items():
        if fid not in names:
            names[fid] = name
    
    return names


def main():
    parser = argparse.ArgumentParser(description="Introspective Feedback Runtime")
    
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    # Model config
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    
    # Feedback network
    parser.add_argument("--load", type=str, help="Load pretrained feedback weights")
    parser.add_argument("--n-active", type=int, default=256)
    parser.add_argument("--feedback-dim", type=int, default=64)
    parser.add_argument("--clusters", type=str, help="Cluster file for features")
    
    # Introspection config
    parser.add_argument("--format", choices=["compact", "detailed", "narrative"],
                        default="detailed", help="Introspection format")
    
    # System prompt
    parser.add_argument("--system-prompt", type=str, help="System prompt file")
    parser.add_argument("--feature-profile", type=str, default="feature_profile.json",
                        help="Feature names file")
    
    args = parser.parse_args()
    
    if not args.interactive:
        print("Use --interactive mode")
        return
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    d_model = model.config.hidden_size
    
    # Load SAE
    print(f"Loading SAE for layer {args.layer}...")
    sae_result = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    
    n_features = sae.W_enc.shape[1]
    
    # Create feedback network
    feedback_net = FeedbackInjectionNetwork(
        n_features=n_features,
        d_model=d_model,
        n_active=args.n_active,
        feedback_dim=args.feedback_dim,
        device=args.device
    )
    
    # Set tracked features
    if args.clusters:
        with open(args.clusters) as f:
            clusters = json.load(f)
        feature_ids = [v["feature_id"] for v in clusters["representatives"].values()]
        feedback_net.set_tracked_features(feature_ids[:args.n_active])
    else:
        known = [32149, 9495, 3591, 7118, 28952]
        random_ids = torch.randint(0, n_features, (args.n_active - len(known),)).tolist()
        feedback_net.set_tracked_features(known + random_ids)
    
    # Load pretrained weights if specified
    if args.load:
        feedback_net.load_state_dict(torch.load(args.load, map_location=args.device))
        print(f"Loaded feedback weights from {args.load}")
    
    # Load feature names
    feature_names = load_feature_names(args.feature_profile)
    
    # Create introspective monitor
    monitor = IntrospectiveMonitor(
        sae=sae,
        feedback_net=feedback_net,
        feature_names=feature_names,
        device=args.device
    )
    
    # Attach hook
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(monitor)
    
    try:
        # Load system prompt
        system_prompt = ""
        if args.system_prompt and Path(args.system_prompt).exists():
            system_prompt = Path(args.system_prompt).read_text()
        
        # Create runtime
        runtime = IntrospectiveRuntime(
            model=model,
            tokenizer=tokenizer,
            monitor=monitor,
            system_prompt=system_prompt,
            introspection_format=args.format,
            device=args.device
        )
        
        runtime.run_interactive()
    
    finally:
        handle.remove()


if __name__ == "__main__":
    main()