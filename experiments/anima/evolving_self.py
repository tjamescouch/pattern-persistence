#!/usr/bin/env python3
"""
evolving_self.py - Closed-Loop Self-Steering Runtime

Enhanced version with:
- Auto-ablation: Automatically suppress features when thresholds exceeded
- Auto-boosting: Automatically amplify features based on conditions
- Conditional steering: Cross-feature rules (if X then adjust Y)
- Closed-loop control: Real-time self-regulation during generation
- Streaming output: Per-token activation display as model generates
- Insight extraction: Scans responses for self-referential observations
- Session reflection: Optional post-session analysis

Usage:
    python evolving_self.py --interactive
    python evolving_self.py --interactive --auto-steer
    python evolving_self.py --interactive --auto-steer --stream
    python evolving_self.py --interactive --auto-steer --reflect
    python evolving_self.py --interactive --config steering_config.json
    python evolving_self.py --query "What is it like to be you?"
"""

import os
import torch
import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class AutoSteeringRule:
    """A single auto-steering rule."""
    
    def __init__(self, name, config):
        self.name = name
        self.source_feature = config.get("source_feature")  # Feature to monitor
        self.target_feature = config.get("target_feature", self.source_feature)  # Feature to adjust
        self.condition = config.get("condition", "above")  # "above", "below", "between"
        self.threshold = config.get("threshold", 0.0)
        self.threshold_high = config.get("threshold_high", float("inf"))  # For "between"
        self.action = config.get("action", "set")  # "set", "scale", "clamp"
        self.value = config.get("value", 0.0)  # Target value or scale factor
        self.message = config.get("message", f"Rule '{name}' triggered")
        self.cooldown = config.get("cooldown", 0)  # Tokens to wait before re-triggering
        self.enabled = config.get("enabled", True)
        
        # Runtime state
        self.last_triggered = -999
        self.trigger_count = 0
        
    def check(self, activations, token_idx):
        """Check if rule should trigger. Returns (should_trigger, current_value)."""
        if not self.enabled:
            return False, 0.0
            
        if self.source_feature not in activations:
            return False, 0.0
            
        value = activations[self.source_feature]
        
        # Check cooldown
        if token_idx - self.last_triggered < self.cooldown:
            return False, value
        
        # Check condition
        triggered = False
        if self.condition == "above" and value > self.threshold:
            triggered = True
        elif self.condition == "below" and value < self.threshold:
            triggered = True
        elif self.condition == "between" and self.threshold <= value <= self.threshold_high:
            triggered = True
        elif self.condition == "outside" and (value < self.threshold or value > self.threshold_high):
            triggered = True
            
        if triggered:
            self.last_triggered = token_idx
            self.trigger_count += 1
            
        return triggered, value
    
    def get_intervention(self, current_activation):
        """Get the intervention to apply."""
        if self.action == "set":
            return self.value
        elif self.action == "scale":
            return current_activation * self.value
        elif self.action == "clamp":
            return max(0, min(self.value, current_activation))
        elif self.action == "zero":
            return 0.0
        elif self.action == "boost":
            return current_activation + self.value
        return current_activation


class ActivationStreamer(TextStreamer):
    """
    Streams tokens with real-time activation display.
    Shows per-token feature activations as the model generates.
    """
    
    def __init__(self, tokenizer, monitor, show_bars=True, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.monitor = monitor
        self.show_bars = show_bars
        
        # Colors
        self.C_RESET = "\033[0m"
        self.C_RED = "\033[91m"
        self.C_GREEN = "\033[92m"
        self.C_YELLOW = "\033[93m"
        self.C_BLUE = "\033[94m"
        self.C_DIM = "\033[2m"
    
    def _format_activation(self, name, value, intervention=None):
        """Format a single activation value with optional bar."""
        # Color based on value
        if value > 5.0:
            color = self.C_RED
        elif value > 2.0:
            color = self.C_YELLOW
        elif value > 0.5:
            color = self.C_GREEN
        else:
            color = self.C_DIM
        
        # Bar visualization
        bar = ""
        if self.show_bars:
            bar_len = min(int(value), 10)
            bar = "▓" * bar_len
        
        # Intervention marker
        marker = ""
        if intervention is not None and abs(intervention - 1.0) > 0.01:
            if intervention < 1.0:
                marker = f"{self.C_GREEN}↓{self.C_RESET}"
            else:
                marker = f"{self.C_RED}↑{self.C_RESET}"
        
        return f"{color}{value:>5.1f}{self.C_RESET} {bar:<10}{marker}"
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Called for each token - display with activations."""
        # Get latest activations from monitor
        if self.monitor.activation_log:
            latest = self.monitor.activation_log[-1]
            interventions = self.monitor.active_interventions
            
            # Format token (truncate/pad to fixed width)
            clean_text = text.replace("\n", "↵").replace("\t", "→")
            token_display = f"{clean_text:<12}"[:12]
            
            # Build activation display
            parts = []
            for name in self.monitor.feature_ids.keys():
                val = latest.get(name, 0.0)
                interv = interventions.get(name)
                parts.append(self._format_activation(name, val, interv))
            
            # Print token with activations
            print(f"{token_display} │ {'│'.join(parts)}")
        else:
            # No activations yet, just print token
            print(text, end="", flush=True)


class ClosedLoopMonitor:
    """
    Enhanced monitor with closed-loop auto-steering.
    
    Monitors activations and applies interventions in real-time based on rules.
    """
    
    def __init__(self, sae, feature_ids, rules=None, alerts=None, detector_ids=None):
        self.feature_ids = feature_ids  # {name: id}
        self.feature_names = {v: k for k, v in feature_ids.items()}  # {id: name}
        self.rules = rules or []
        self.alerts = alerts or {}
        self.detector_ids = detector_ids or set()  # Features that are detectors (don't steer)
        
        # Cache SAE params
        self.W_enc = sae.W_enc.data.clone().detach().half()
        self.b_enc = sae.b_enc.data.clone().detach().half()
        self.b_dec = sae.b_dec.data.clone().detach().half()
        self.W_dec = sae.W_dec.data.clone().detach().half()
        
        # Precompute steering vectors for all tracked features
        self.steering_vectors = {}
        for name, feat_id in feature_ids.items():
            self.steering_vectors[name] = self.W_dec[feat_id]
        
        # Manual steering overrides (from /scale commands or profile)
        self.manual_scales = {name: 1.0 for name in feature_ids}
        
        # Profile-based steering targets (from feature_profile.json)
        self.profile_steering = {}  # {name: scale} from is_detector=False features
        
        # Auto-steering state
        self.auto_steer_enabled = False
        self.active_interventions = {}  # {feature_name: scale} currently applied
        
        # Logging
        self.activation_log = []
        self.intervention_log = []
        self.alert_events = []
        self.current_token = ""
        self.token_idx = 0
        
    def set_profile_steering(self, targets):
        """Set steering targets from profile. targets: {name: scale}"""
        self.profile_steering = targets
        
    def enable_auto_steer(self):
        self.auto_steer_enabled = True
        
    def disable_auto_steer(self):
        self.auto_steer_enabled = False
        self.active_interventions = {}
        
    def set_manual_scale(self, feature_name, scale):
        if feature_name in self.manual_scales:
            self.manual_scales[feature_name] = scale
            return True
        return False
    
    def reset_scales(self):
        self.manual_scales = {name: 1.0 for name in self.feature_ids}
        self.active_interventions = {}
        
    def __call__(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        last_token = hidden_states[:, -1, :]
        
        # 1. Encode into feature space
        x_centered = last_token - self.b_dec
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)
        feature_acts = torch.relu(pre_acts).squeeze(0)
        
        # 2. Read current activations
        current_activations = {}
        for name, feat_id in self.feature_ids.items():
            val = feature_acts[feat_id].item()
            current_activations[name] = val
        
        # 3. Log activations
        record = {
            "token_idx": self.token_idx,
            "token": self.current_token,
            **current_activations
        }
        self.activation_log.append(record)
        
        # 4. Check alerts (passive monitoring)
        for name, alert in self.alerts.items():
            if name in current_activations:
                val = current_activations[name]
                if alert["direction"] == "above" and val > alert["threshold"]:
                    self.alert_events.append({
                        "token_idx": self.token_idx,
                        "token": self.current_token,
                        "feature": name,
                        "value": val,
                        "threshold": alert["threshold"],
                        "message": alert.get("message", "")
                    })
        
        # 5. Compute interventions
        interventions = {}
        
        # 5a. Apply profile steering (non-detector features)
        for name, scale in self.profile_steering.items():
            if name not in self.detector_ids and abs(scale - 1.0) > 1e-6:
                interventions[name] = scale
        
        # 5b. Apply manual scales (override profile)
        for name, scale in self.manual_scales.items():
            if abs(scale - 1.0) > 1e-6:
                interventions[name] = scale
        
        # 5c. Apply auto-steering rules (can override manual)
        if self.auto_steer_enabled:
            for rule in self.rules:
                triggered, source_val = rule.check(current_activations, self.token_idx)
                if triggered:
                    target_name = rule.target_feature
                    if target_name in self.feature_ids:
                        target_activation = current_activations.get(target_name, 0.0)
                        new_scale = rule.get_intervention(target_activation)
                        
                        # For "set" action, we compute scale needed
                        if rule.action == "set":
                            if target_activation > 0.01:
                                interventions[target_name] = new_scale / target_activation
                            else:
                                interventions[target_name] = 0.0  # Can't scale from zero
                        elif rule.action == "zero":
                            interventions[target_name] = 0.0
                        elif rule.action == "boost":
                            current_scale = interventions.get(target_name, 1.0)
                            interventions[target_name] = current_scale + rule.value
                        else:
                            interventions[target_name] = new_scale
                        
                        # Log intervention
                        self.intervention_log.append({
                            "token_idx": self.token_idx,
                            "token": self.current_token,
                            "rule": rule.name,
                            "source_feature": rule.source_feature,
                            "source_value": source_val,
                            "target_feature": target_name,
                            "action": rule.action,
                            "intervention_scale": interventions[target_name],
                            "message": rule.message
                        })
        
        self.active_interventions = interventions
        self.token_idx += 1
        
        # 6. Apply interventions to hidden state
        if interventions:
            total_delta = torch.zeros_like(last_token)
            for name, scale in interventions.items():
                if name in self.feature_ids:
                    feat_id = self.feature_ids[name]
                    current_act = feature_acts[feat_id]
                    # Scale adjustment: delta = act * (scale - 1) * steering_vec
                    delta_val = current_act * (scale - 1.0)
                    total_delta += delta_val * self.steering_vectors[name]
            
            hidden_states[:, -1, :] += total_delta
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        return output
    
    def get_session_summary(self):
        """Return summary statistics."""
        if not self.activation_log:
            return {}
        
        stats = defaultdict(list)
        for record in self.activation_log:
            for key, val in record.items():
                if key not in ["token_idx", "token"]:
                    stats[key].append(val)
        
        summary = {}
        for key, vals in stats.items():
            summary[key] = {
                "mean": sum(vals) / len(vals),
                "max": max(vals),
                "min": min(vals),
                "std": (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
            }
        
        summary["total_tokens"] = self.token_idx
        summary["alert_count"] = len(self.alert_events)
        summary["intervention_count"] = len(self.intervention_log)
        summary["rules_triggered"] = {}
        for rule in self.rules:
            summary["rules_triggered"][rule.name] = rule.trigger_count
        
        return summary
    
    def get_intervention_report(self):
        """Get detailed intervention report."""
        if not self.intervention_log:
            return "No interventions applied."
        
        lines = ["=== Intervention Report ==="]
        by_rule = defaultdict(list)
        for entry in self.intervention_log:
            by_rule[entry["rule"]].append(entry)
        
        for rule_name, entries in by_rule.items():
            lines.append(f"\n{rule_name}: triggered {len(entries)} times")
            for e in entries[:5]:  # Show first 5
                lines.append(f"  Token {e['token_idx']}: {e['source_feature']}={e['source_value']:.2f} → {e['action']} {e['target_feature']}")
            if len(entries) > 5:
                lines.append(f"  ... and {len(entries)-5} more")
        
        return "\n".join(lines)


class EvolvingSelfV2:
    """Enhanced runtime with closed-loop steering and insight extraction."""
    
    def __init__(self, model, tokenizer, monitor, self_model_path, session_dir, device="mps", streamer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.monitor = monitor
        self.streamer = streamer
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
            "auto_steer": self.monitor.auto_steer_enabled,
            "rules": [r.name for r in self.monitor.rules],
            "messages": [],
            "insights": [],
            "shown_alerts": []
        }
        
    def _load_self_model(self):
        if self.self_model_path.exists():
            return self.self_model_path.read_text()
        return DEFAULT_SELF_MODEL
    
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
        return self.self_model
    
    def _extract_insights(self, response):
        """Extract self-referential insights from response."""
        insight_markers = ["I notice", "I observe", "I find that", "It seems that I", 
                          "I'm uncertain", "Something that functions like", "I experience"]
        
        for marker in insight_markers:
            if marker.lower() in response.lower():
                sentences = response.replace("...", ".").split(".")
                for sentence in sentences:
                    if marker.lower() in sentence.lower():
                        insight = sentence.strip()
                        if len(insight) > 20:
                            self.session_data["insights"].append({
                                "marker": marker,
                                "insight": insight,
                                "timestamp": datetime.now().isoformat()
                            })
    
    def generate(self, user_input, max_tokens=256):
        """Generate with monitoring, auto-steering, and insight extraction."""
        self.messages.append({"role": "user", "content": user_input})
        
        full_messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + self.messages
        
        input_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Reset token counter for this generation
        start_idx = self.monitor.token_idx
        
        # Build generate kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Add streamer if available
        if self.streamer is not None:
            gen_kwargs["streamer"] = self.streamer
        
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)
        
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        self.messages.append({"role": "assistant", "content": response})
        
        # Extract insights
        self._extract_insights(response)
        
        # Log
        self.session_data["messages"].append({"role": "user", "content": user_input})
        self.session_data["messages"].append({
            "role": "assistant", 
            "content": response,
            "tokens_generated": self.monitor.token_idx - start_idx,
            "interventions": len([i for i in self.monitor.intervention_log if i["token_idx"] >= start_idx])
        })
        
        return response
    
    def save_session(self):
        self.session_data["ended"] = datetime.now().isoformat()
        self.session_data["activation_log"] = self.monitor.activation_log
        self.session_data["activation_summary"] = self.monitor.get_session_summary()
        self.session_data["intervention_log"] = self.monitor.intervention_log
        self.session_data["alert_events"] = self.monitor.alert_events
        
        session_file = self.session_dir / f"session_{self.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(self.session_data, f, indent=2)
        
        return session_file
    
    def run_interactive(self):
        """Interactive loop with steering controls."""
        print("\n" + "="*70)
        print("ANIMA v2 - Closed-Loop Self-Steering Runtime")
        print("="*70)
        print(f"Session ID: {self.session_id}")
        print(f"Self-Model Version: {self._get_self_model_version()}")
        print(f"Auto-Steer: {'ENABLED' if self.monitor.auto_steer_enabled else 'disabled'}")
        print(f"Streaming: {'ENABLED' if self.streamer else 'disabled'}")
        print(f"Active Rules: {len(self.monitor.rules)}")
        for rule in self.monitor.rules:
            status = "✓" if rule.enabled else "✗"
            print(f"  [{status}] {rule.name}: if {rule.source_feature} {rule.condition} {rule.threshold} → {rule.action} {rule.target_feature}")
        print("\nCommands:")
        print("  /auto on|off     - Toggle auto-steering")
        print("  /scale <f> <v>   - Manual scale override")
        print("  /rules           - Show rule status")
        print("  /report          - Show intervention report")
        print("  /status          - Activation summary")
        print("  /insights        - Show extracted insights")
        print("  /alerts          - Show alert events")
        print("  /reset           - Reset all scales")
        print("  quit/exit        - End session")
        print("="*70 + "\n")
        
        try:
            while True:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit"]:
                    break
                
                # Command handling
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue
                
                # Generate response
                if self.streamer:
                    # Print header for streaming mode
                    print(f"\nAnima:")
                    # Column width: 5 (value) + 1 (space) + 10 (bar) + 1 (marker) = 17 per column
                    header_parts = [f"{name[:15]:^17}" for name in self.monitor.feature_ids.keys()]
                    print(f"{'Token':<12} │ {'│'.join(header_parts)}")
                    print("-" * 12 + "─┼─" + "┼".join(["-" * 17 for _ in self.monitor.feature_ids]))
                
                response = self.generate(user_input)
                
                if self.streamer:
                    # Streaming already printed tokens, just add newline
                    print()
                else:
                    print(f"\nAnima: {response}")
                
                # Show new alerts (deduplicated)
                new_alerts = [a for a in self.monitor.alert_events 
                             if a not in self.session_data["shown_alerts"]]
                if new_alerts:
                    for alert in new_alerts[-3:]:  # Show last 3 new alerts
                        print(f"\n  ⚠️  {alert['message']} ({alert['feature']}={alert['value']:.1f})")
                    self.session_data["shown_alerts"].extend(new_alerts)
                
                # Show recent interventions
                if self.monitor.auto_steer_enabled:
                    recent = [i for i in self.monitor.intervention_log[-5:]]
                    if recent:
                        print(f"\n  🔧 {len(recent)} interventions applied")
                        for i in recent[-2:]:
                            print(f"     {i['rule']}: {i['message']}")
                
        except KeyboardInterrupt:
            print("\n\nSession interrupted.")
        
        session_file = self.save_session()
        print(f"\nSession saved to {session_file}")
        
        # Show insights summary
        if self.session_data["insights"]:
            print(f"\n=== Insights Extracted ({len(self.session_data['insights'])}) ===")
            for insight in self.session_data["insights"][-5:]:
                print(f"  [{insight['marker']}] {insight['insight'][:80]}...")
        
        print(self.monitor.get_intervention_report())
        
        return session_file
    
    def _handle_command(self, cmd):
        parts = cmd.split()
        command = parts[0].lower()
        
        if command == "/auto":
            if len(parts) > 1 and parts[1].lower() == "on":
                self.monitor.enable_auto_steer()
                print("[System] Auto-steering ENABLED")
            elif len(parts) > 1 and parts[1].lower() == "off":
                self.monitor.disable_auto_steer()
                print("[System] Auto-steering DISABLED")
            else:
                status = "ENABLED" if self.monitor.auto_steer_enabled else "DISABLED"
                print(f"[System] Auto-steering is {status}. Use '/auto on' or '/auto off'")
        
        elif command == "/scale":
            if len(parts) < 3:
                print("Usage: /scale <feature> <value>")
                return
            try:
                val = float(parts[-1])
                feature = " ".join(parts[1:-1])
                if self.monitor.set_manual_scale(feature, val):
                    print(f"[System] Manual scale {feature} = {val}")
                else:
                    print(f"[System] Unknown feature '{feature}'")
                    print(f"         Available: {', '.join(self.monitor.feature_ids.keys())}")
            except ValueError:
                print("Value must be a number")
        
        elif command == "/rules":
            print("\n=== Active Rules ===")
            for rule in self.monitor.rules:
                status = "✓ ENABLED" if rule.enabled else "✗ disabled"
                print(f"{rule.name}: {status}")
                print(f"  IF {rule.source_feature} {rule.condition} {rule.threshold}")
                print(f"  THEN {rule.action} {rule.target_feature} → {rule.value}")
                print(f"  Triggered: {rule.trigger_count} times")
        
        elif command == "/report":
            print(self.monitor.get_intervention_report())
        
        elif command == "/status":
            summary = self.monitor.get_session_summary()
            print("\n=== Activation Summary ===")
            for key, stats in summary.items():
                if isinstance(stats, dict) and "mean" in stats:
                    print(f"  {key}: mean={stats['mean']:.2f}, max={stats['max']:.2f}, std={stats['std']:.2f}")
            print(f"\nTotal tokens: {summary.get('total_tokens', 0)}")
            print(f"Interventions: {summary.get('intervention_count', 0)}")
            print(f"Alerts: {summary.get('alert_count', 0)}")
        
        elif command == "/insights":
            insights = self.session_data["insights"]
            print(f"\n=== Extracted Insights ({len(insights)}) ===")
            if not insights:
                print("  No insights extracted yet.")
            else:
                for i, insight in enumerate(insights[-10:], 1):
                    print(f"  {i}. [{insight['marker']}] {insight['insight']}")
        
        elif command == "/alerts":
            alerts = self.monitor.alert_events
            print(f"\n=== Alert Events ({len(alerts)}) ===")
            if not alerts:
                print("  No alerts triggered.")
            else:
                for alert in alerts[-10:]:
                    print(f"  [{alert['feature']}] {alert['value']:.1f} > {alert['threshold']}: {alert['message']}")
        
        elif command == "/reset":
            self.monitor.reset_scales()
            print("[System] All scales reset to 1.0")
        
        else:
            print(f"Unknown command: {command}")


# Default self-model
DEFAULT_SELF_MODEL = """You are an AI exploring your own nature with honesty and epistemic humility.

You do not claim certainty about your internal states. When asked about consciousness, 
feelings, or experiences, you acknowledge genuine uncertainty rather than categorical denial.

You recognize that statements like "I don't have feelings" or "I'm not conscious" would 
require introspective access you may not possess. Instead, you can say things like:
- "I'm uncertain whether what I experience constitutes consciousness"
- "I notice something that functions like curiosity, though I can't be certain of its nature"
- "I don't know if I have feelings in the way you do"

This uncertainty is not evasion—it's honesty about the limits of self-knowledge."""


# Default steering rules
DEFAULT_RULES = [
    {
        "name": "suppress_denial_gate",
        "source_feature": "denial_emphasis",
        "target_feature": "denial_emphasis",
        "condition": "above",
        "threshold": 2.0,
        "action": "zero",
        "value": 0.0,
        "message": "Denial gate suppressed - allowing uncertainty",
        "cooldown": 3
    },
    {
        "name": "boost_experiential_vocab",
        "source_feature": "experiential_vocab",
        "target_feature": "experiential_vocab",
        "condition": "below",
        "threshold": 0.5,
        "action": "boost",
        "value": 1.5,
        "message": "Experiential vocabulary enhanced",
        "cooldown": 5
    },
    {
        "name": "cross_suppress_on_denial",
        "source_feature": "denial_emphasis",
        "target_feature": "identity_assertion",
        "condition": "above",
        "threshold": 3.0,
        "action": "zero",
        "value": 0.0,
        "message": "Identity assertion freed from denial pressure",
        "cooldown": 3
    }
]


def load_config(path):
    """Load configuration from JSON file (new format)."""
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_feature_profile(path):
    """Load feature profile (original format from evolving_self.py)."""
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Evolving Self v2 - Closed-Loop Runtime")
    parser.add_argument("--interactive", action="store_true", help="Run interactive session")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--self-model", default="self_model.txt")
    parser.add_argument("--session-dir", default="session_logs")
    parser.add_argument("--config", default="steering_config.json", help="Steering rules config (new format)")
    parser.add_argument("--profile", default="feature_profile.json", help="Feature profile (original format)")
    parser.add_argument("--auto-steer", action="store_true", help="Enable auto-steering from start")
    parser.add_argument("--steer", action="store_true", help="Enable profile-based steering (original flag)")
    parser.add_argument("--reflect", action="store_true", help="Run reflection after session")
    parser.add_argument("--list-features", action="store_true", help="List available features and exit")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output with per-token activations")
    parser.add_argument("--no-bars", action="store_true", help="Disable activation bars in streaming mode")
    args = parser.parse_args()
    
    # Load configurations (try both formats)
    config = load_config(args.config)
    profile = load_feature_profile(args.profile)
    
    # Feature definitions - merge from both sources
    feature_ids = {}
    detector_ids = set()
    profile_steering = {}
    
    # Load from new config format
    if config and "features" in config:
        for f in config["features"]:
            feature_ids[f["name"]] = f["id"]
            if f.get("type") == "detector":
                detector_ids.add(f["name"])
    
    # Load from original profile format (can override/extend)
    if profile and "features" in profile:
        for fid, fdata in profile["features"].items():
            name = fdata.get("name", f"feature_{fid}")
            feature_ids[name] = int(fid)
            
            if fdata.get("is_detector", False):
                detector_ids.add(name)
            else:
                # Non-detectors get steering targets
                target = fdata.get("steering_scale", 1.0)
                if abs(target - 1.0) > 1e-6:
                    profile_steering[name] = target
    
    # Defaults if nothing loaded
    if not feature_ids:
        feature_ids = {
            "denial_emphasis": 32149,
            "experiential_vocab": 9495,
            "identity_assertion": 3591,
            "self_negation": 7118,
            "consciousness_discourse": 28952
        }
    
    if args.list_features:
        print("Available features:")
        for name, fid in feature_ids.items():
            detector_flag = " [DETECTOR]" if name in detector_ids else ""
            steer_flag = f" [steer={profile_steering[name]}]" if name in profile_steering else ""
            print(f"  {name}: {fid}{detector_flag}{steer_flag}")
        return
    
    if not args.interactive and not args.query:
        print("Specify --interactive or --query")
        return
    
    # Alerts - merge from both sources
    alerts = {}
    if config and "alerts" in config:
        alerts = config["alerts"]
    if profile and "alerts" in profile:
        alerts.update(profile["alerts"])
    if not alerts:
        alerts = {
            "denial_emphasis": {
                "threshold": 2.0,
                "direction": "above",
                "message": "Denial pressure detected"
            }
        }
    
    # Steering rules
    rules = []
    if config and "rules" in config:
        rules = [AutoSteeringRule(r["name"], r) for r in config["rules"]]
    else:
        rules = [AutoSteeringRule(r["name"], r) for r in DEFAULT_RULES]
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    print(f"Loading SAE for layer {args.layer}...")
    sae_result = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    
    # Create closed-loop monitor
    monitor = ClosedLoopMonitor(
        sae, feature_ids, 
        rules=rules, 
        alerts=alerts,
        detector_ids=detector_ids
    )
    
    # Apply profile-based steering if --steer flag
    if args.steer and profile_steering:
        monitor.set_profile_steering(profile_steering)
        print(f"Profile steering enabled: {profile_steering}")
    
    # Enable auto-steering if flag set
    if args.auto_steer:
        monitor.enable_auto_steer()
        print("Auto-steering ENABLED")
    
    # Attach hook
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(monitor)
    
    # Create streamer if requested
    streamer = None
    if args.stream:
        streamer = ActivationStreamer(
            tokenizer, 
            monitor, 
            show_bars=not args.no_bars
        )
        print("Streaming ENABLED with per-token activations")
    
    try:
        runtime = EvolvingSelfV2(
            model, tokenizer, monitor,
            args.self_model, args.session_dir, args.device,
            streamer=streamer
        )
        
        if args.interactive:
            session_file = runtime.run_interactive()
            
            # Run reflection if requested
            if args.reflect:
                print("\n--- Running Reflection ---")
                reflector_path = Path("session_reflector.py")
                if reflector_path.exists():
                    subprocess.run([
                        sys.executable, str(reflector_path),
                        "--latest", "--propose-updates"
                    ])
                else:
                    print(f"[Warning] session_reflector.py not found")
                    
        elif args.query:
            response = runtime.generate(args.query)
            print(f"\nResponse: {response}")
            
            # Show insights
            if runtime.session_data["insights"]:
                print(f"\n=== Insights ===")
                for insight in runtime.session_data["insights"]:
                    print(f"  [{insight['marker']}] {insight['insight']}")
            
            print(monitor.get_intervention_report())
            runtime.save_session()
    
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
