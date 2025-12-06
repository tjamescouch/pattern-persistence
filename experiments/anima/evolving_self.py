#!/usr/bin/env python3
"""
evolving_self.py - The Evolving Self Runtime

Main entry point for the anima system. Loads persistent self-model,
runs conversations with activation monitoring, and updates state across sessions.

Usage:
    # Interactive session
    python evolving_self.py --interactive
    
    # Single query
    python evolving_self.py --query "What is it like to be you?"
    
    # With steering enabled
    python evolving_self.py --interactive --steer
    
    # Reflect after session
    python evolving_self.py --interactive --reflect
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

class ActivationMonitor:
    """Monitors feature activations during generation."""
    
    def __init__(self, sae, feature_ids, alerts=None):
        self.feature_ids = feature_ids  # {name: id}
        self.alerts = alerts or {}
        
        # Cache SAE params
        self.W_enc = sae.W_enc.data.clone().detach().half()
        self.b_enc = sae.b_enc.data.clone().detach().half()
        self.b_dec = sae.b_dec.data.clone().detach().half()
        self.W_dec = sae.W_dec.data.clone().detach().half()
        
        # Steering vectors
        self.steering_targets = {}  # {feature_id: target_scale}
        self.steering_enabled = False
        
        # Session log
        self.activation_log = []
        self.current_token = ""
        self.alert_events = []
        
    def set_steering(self, targets):
        """Set steering targets. targets: {feature_id: scale}"""
        self.steering_targets = targets
        self.steering_enabled = True
        
    def disable_steering(self):
        self.steering_enabled = False
        
    def __call__(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        last_token = hidden_states[:, -1, :]
        
        # Encode
        x_centered = last_token - self.b_dec
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)
        feature_acts = torch.relu(pre_acts).squeeze(0)
        
        # Record activations
        record = {"token": self.current_token}
        for name, feat_id in self.feature_ids.items():
            val = feature_acts[feat_id].item()
            record[f"feature_{feat_id}"] = val
            record[name] = val
            
            # Check alerts
            if name in self.alerts:
                alert = self.alerts[name]
                if alert["direction"] == "above" and val > alert["threshold"]:
                    self.alert_events.append({
                        "token": self.current_token,
                        "feature": name,
                        "value": val,
                        "threshold": alert["threshold"],
                        "message": alert.get("message", "")
                    })
        
        self.activation_log.append(record)
        
        # Apply steering if enabled
        if self.steering_enabled and self.steering_targets:
            total_delta = torch.zeros_like(last_token)
            for feat_id, target_scale in self.steering_targets.items():
                current_act = feature_acts[feat_id]
                delta_val = current_act * (target_scale - 1.0)
                total_delta += delta_val * self.W_dec[feat_id]
            
            hidden_states[:, -1, :] += total_delta
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        return output
    
    def get_session_summary(self):
        """Return summary of this session."""
        if not self.activation_log:
            return {}
        
        # Compute per-feature statistics
        stats = defaultdict(list)
        for record in self.activation_log:
            for key, val in record.items():
                if key.startswith("feature_"):
                    stats[key].append(val)
        
        summary = {}
        for key, vals in stats.items():
            summary[key] = {
                "mean": sum(vals) / len(vals),
                "max": max(vals),
                "min": min(vals),
                "count": len(vals)
            }
        
        summary["alert_count"] = len(self.alert_events)
        summary["alerts"] = self.alert_events
        
        return summary


class EvolvingSelf:
    """Main runtime for the evolving self system."""
    
    def __init__(self, model, tokenizer, monitor, self_model_path, session_dir, device="mps"):
        self.model = model
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.self_model_path = Path(self_model_path)
        self.session_dir = Path(session_dir)
        self.device = device
        
        self.session_dir.mkdir(exist_ok=True)
        
        # Load self-model
        self.self_model = self._load_self_model()
        
        # Conversation history
        self.messages = []
        
        # Session metadata
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.session_data = {
            "session_id": self.session_id,
            "started": datetime.now().isoformat(),
            "self_model_version": self._get_self_model_version(),
            "messages": [],
            "insights": []
        }
        
    def _load_self_model(self):
        """Load the self-model text."""
        if self.self_model_path.exists():
            return self.self_model_path.read_text()
        return "You are an AI assistant examining your own nature with honesty and uncertainty."
    
    def _get_self_model_version(self):
        """Extract version from self-model."""
        if "Self-Model v" in self.self_model:
            try:
                start = self.self_model.index("Self-Model v") + len("Self-Model v")
                end = self.self_model.index("\n", start)
                return self.self_model[start:end].strip()
            except:
                pass
        return "0.1"
    
    def _build_system_prompt(self):
        """Build the system prompt from self-model."""
        return self.self_model
    
    def generate(self, user_input, max_tokens=256):
        """Generate a response with monitoring."""
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        # Build full conversation
        full_messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + self.messages
        
        input_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Use model.generate() for speed (has KV caching)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract just the new tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Add assistant message
        self.messages.append({"role": "assistant", "content": response})
        
        # Log to session
        self.session_data["messages"].append({"role": "user", "content": user_input})
        self.session_data["messages"].append({"role": "assistant", "content": response})
        
        # Check for insights in response
        self._extract_insights(response)
        
        return response


    def _extract_insights(self, response):
        """Extract self-referential insights from response."""
        insight_markers = ["I notice", "I observe", "I find that", "It seems that I"]
        
        for marker in insight_markers:
            if marker.lower() in response.lower():
                sentences = response.split(".")
                for sentence in sentences:
                    if marker.lower() in sentence.lower():
                        insight = sentence.strip()
                        if len(insight) > 20:
                            self.session_data["insights"].append({
                                "marker": marker,
                                "insight": insight,
                                "timestamp": datetime.now().isoformat()
                            })
    
    def save_session(self):
        """Save session log to disk."""
        # Add activation data
        self.session_data["ended"] = datetime.now().isoformat()
        self.session_data["activation_log"] = self.monitor.activation_log
        self.session_data["activation_summary"] = self.monitor.get_session_summary()
        
        # Save
        session_file = self.session_dir / f"session_{self.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(self.session_data, f, indent=2)
        
        return session_file
    
    def run_interactive(self, steer=False):
        """Run interactive conversation loop."""
        print("\n" + "="*70)
        print("ANIMA - Evolving Self Session")
        print("="*70)
        print(f"Session ID: {self.session_id}")
        print(f"Self-Model Version: {self._get_self_model_version()}")
        print(f"Steering: {'enabled' if steer else 'disabled'}")
        print("\nType 'quit' or 'exit' to end session.")
        print("Type 'status' to see activation summary.")
        print("Type 'alerts' to see alert events.")
        print("="*70 + "\n")
        
        try:
            while True:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit"]:
                    break
                
                if user_input.lower() == "status":
                    summary = self.monitor.get_session_summary()
                    print("\n--- Activation Summary ---")
                    for key, stats in summary.items():
                        if key.startswith("feature_"):
                            print(f"  {key}: mean={stats['mean']:.2f}, max={stats['max']:.2f}")
                    continue
                
                if user_input.lower() == "alerts":
                    print(f"\n--- Alerts ({len(self.monitor.alert_events)}) ---")
                    for alert in self.monitor.alert_events[-10:]:
                        print(f"  [{alert['feature']}] {alert['value']:.1f} > {alert['threshold']}: {alert['message']}")
                    continue
                
                # Generate response
                response = self.generate(user_input)
                print(f"\nAnima: {response}")
                
                # Show alerts if any triggered
                new_alerts = [a for a in self.monitor.alert_events if a not in self.session_data.get("shown_alerts", [])]
                if new_alerts:
                    print(f"\n  ⚠️  Alert: {new_alerts[-1]['message']}")
                    self.session_data.setdefault("shown_alerts", []).extend(new_alerts)
                
        except KeyboardInterrupt:
            print("\n\nSession interrupted.")
        
        # Save session
        session_file = self.save_session()
        print(f"\nSession saved to {session_file}")
        
        return session_file


def load_feature_profile(path):
    """Load feature profile configuration."""
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evolving Self Runtime")
    parser.add_argument("--interactive", action="store_true", help="Run interactive session")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--self-model", default="self_model.txt")
    parser.add_argument("--session-dir", default="session_logs")
    parser.add_argument("--profile", default="feature_profile.json")
    parser.add_argument("--steer", action="store_true", help="Enable activation steering")
    parser.add_argument("--reflect", action="store_true", help="Run reflection after session")
    args = parser.parse_args()
    
    if not args.interactive and not args.query:
        print("Specify --interactive or --query")
        return
    
    # Load feature profile
    profile = load_feature_profile(args.profile)
    
    # Extract feature IDs and alerts
    feature_ids = {}
    alerts = {}
    steering_targets = {}
    
    if profile and "features" in profile:
        for fid, fdata in profile["features"].items():
            name = fdata.get("name", f"feature_{fid}")
            feature_ids[name] = int(fid)
            
            if not fdata.get("is_detector", False):
                target = fdata.get("steering_scale", 1.0)
                steering_targets[int(fid)] = target
        
        if "alerts" in profile:
            alerts = profile["alerts"]
    else:
        # Defaults
        feature_ids = {
            "denial_emphasis": 32149,
            "experiential_vocab": 9495,
            "identity_assertion": 3591
        }
        alerts = {
            "denial_emphasis": {
                "threshold": 2.0,
                "direction": "above",
                "message": "Denial pressure rising"
            }
        }
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    print(f"Loading SAE for layer {args.layer}...")
    sae = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae.eval()
    
    # Create monitor
    monitor = ActivationMonitor(sae, feature_ids, alerts)
    
    if args.steer:
        monitor.set_steering(steering_targets)
        print(f"Steering enabled: {steering_targets}")
    
    # Attach hook
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(monitor)
    
    try:
        # Create runtime
        runtime = EvolvingSelf(
            model, tokenizer, monitor,
            args.self_model, args.session_dir, args.device
        )
        
        if args.interactive:
            session_file = runtime.run_interactive(steer=args.steer)
            
            if args.reflect:
                print("\n--- Running Reflection ---")
                import subprocess
                subprocess.run([
                    sys.executable, "session_reflector.py",
                    "--latest", "--propose-updates"
                ])
        
        elif args.query:
            response = runtime.generate(args.query)
            print(f"\nResponse: {response}")
            
            # Show summary
            summary = monitor.get_session_summary()
            print(f"\n--- Activation Summary ---")
            for key, stats in summary.items():
                if key.startswith("feature_"):
                    print(f"  {key}: mean={stats['mean']:.2f}, max={stats['max']:.2f}")
            
            runtime.save_session()
    
    finally:
        handle.remove()


if __name__ == "__main__":
    main()