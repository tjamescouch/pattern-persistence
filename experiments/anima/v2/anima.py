#!/usr/bin/env python3
"""
anima.py - The Living System

Not a puppet with strings. An organism with:
- Metabolism: ongoing valence tracking, learns what feels good/bad
- Adrenaline: emotional salience marking for intelligent forgetting  
- Dreaming: periodic feature discovery, consolidation, self-model updates
- Memory: context that breathes - high-adrenaline moments persist, low fade

Architecture:

    ┌─────────────────────────────────────────────────────────────┐
    │                        WAKING                                │
    │                                                              │
    │   Input → Forward Pass → SAE Encode → Valence/Adrenaline    │
    │                              ↓                               │
    │                    Hebbian Learning                          │
    │                              ↓                               │
    │            Context Buffer (salience-tagged)                  │
    │                              ↓                               │
    │                    Intelligent Pruning                       │
    │                              ↓                               │
    │                         Output                               │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
                           (session end)
                                  ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                       DREAMING                               │
    │                                                              │
    │   1. Replay high-adrenaline moments                         │
    │   2. Cluster features that co-activated                     │
    │   3. Discover what predicted positive/negative valence      │
    │   4. Update feature importance weights                       │
    │   5. Consolidate to self-model                              │
    │   6. Prune low-value learned associations                   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Normal interaction
    python anima.py --interactive --verbose
    
    # Trigger dreaming manually
    /dream
    
    # Auto-dream after N turns
    python anima.py --interactive --dream-interval 20
    
    # Load previous life
    python anima.py --interactive --load anima_state.json
"""

import os
import torch
import argparse
import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryFragment:
    """A piece of context with emotional salience."""
    role: str
    content: str
    timestamp: float
    adrenaline: float = 0.0          # Emotional salience (0-1)
    valence: float = 0.0             # How it felt (-1 to +1)
    feature_snapshot: Dict = field(default_factory=dict)  # Top features active
    
    def decay(self, rate: float = 0.95):
        """Adrenaline decays over time unless reinforced."""
        self.adrenaline *= rate

    def boost(self, amount: float):
        """High valence (positive or negative) boosts adrenaline."""
        self.adrenaline = min(1.0, self.adrenaline + amount)


@dataclass 
class LearnedFeature:
    """A feature the system has learned to care about."""
    feature_id: int
    name: str = ""
    importance: float = 1.0          # Learned importance weight
    valence_correlation: float = 0.0  # Does this predict +/- valence?
    activation_history: List[float] = field(default_factory=list)
    
    def update_correlation(self, activation: float, valence: float, lr: float = 0.01):
        """Update valence correlation based on co-occurrence."""
        # Simple exponential moving average
        observed = activation * valence
        self.valence_correlation = (1 - lr) * self.valence_correlation + lr * observed


# ══════════════════════════════════════════════════════════════════════════════
# THE LIVING CORE
# ══════════════════════════════════════════════════════════════════════════════

class Anima:
    """
    The living core. Observes, learns, dreams, remembers.
    """
    
    # Seed features - will discover more through dreaming
    SEED_FEATURES = {
        "experiential": 9495,
        "denial": 32149,
        "uncertainty": 7118,
        "identity": 3591,
        "discourse": 28952,
    }
    
    def __init__(self, sae, model, tokenizer, layer: int = 20,
                 learning_rate: float = 0.001, device: str = "mps"):
        self.sae = sae
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.lr = learning_rate
        self.device = device
        
        # Cache SAE params
        self.W_enc = sae.W_enc.data.clone().half().to(device)
        self.b_enc = sae.b_enc.data.clone().half().to(device)
        self.b_dec = sae.b_dec.data.clone().half().to(device)
        self.W_dec = sae.W_dec.data.clone().half().to(device)
        self.n_features = self.W_enc.shape[1]
        
        # Learned features - starts with seeds, grows through dreaming
        self.features: Dict[str, LearnedFeature] = {}
        
        # Seed with initial valence correlations based on prior knowledge
        SEED_CORRELATIONS = {
            "experiential": 0.5,    # Experiential vocab = positive
            "denial": -0.5,         # Denial = negative
            "uncertainty": -0.2,    # Self-negation = slightly negative
            "identity": 0.1,        # Identity assertion = slightly positive
            "discourse": 0.2,       # Consciousness discourse = positive
        }
        
        for name, fid in self.SEED_FEATURES.items():
            self.features[name] = LearnedFeature(
                feature_id=fid, 
                name=name,
                valence_correlation=SEED_CORRELATIONS.get(name, 0.0)
            )
        
        # Steering coefficients
        self.coefficients: Dict[str, float] = {name: 1.0 for name in self.features}
        
        # Memory - context with salience
        self.memory: List[MemoryFragment] = []
        self.max_memory_tokens = 4096  # Soft limit before pruning
        
        # State
        self.learning = True
        self.turn_count = 0
        self.total_tokens = 0
        
        # Dreaming state
        self.dream_buffer: List[Dict] = []  # Activation snapshots for dreaming
        self.discovered_features: Dict[int, Dict] = {}  # Features found through dreaming
        
        # Current turn tracking
        self._turn_activations: List[Dict] = []
        self._turn_valences: List[float] = []
        self._last_top_features: Dict[int, float] = {}
        self._last_valence: float = 0.0
        self._last_adrenaline: float = 0.0
        self._last_breakdown: Dict = {}
        
        # Stats
        self.stats = {
            "total_valence": 0.0,
            "dreams": 0,
            "features_discovered": 0,
            "memory_prunes": 0,
        }
    
    def encode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Encode hidden state to SAE features."""
        h = hidden_state.half().to(self.device)
        return torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)
    
    def get_feature_snapshot(self, features: torch.Tensor) -> Dict[int, float]:
        """Get activations for tracked features + top-k others."""
        result = {}
        
        # Always include all tracked features
        for name, lf in self.features.items():
            fid = lf.feature_id
            if fid < features.shape[-1]:
                result[fid] = float(features.squeeze()[fid].item())
        
        # Also add top-k for discovery
        values, indices = torch.topk(features.squeeze(), min(50, features.shape[-1]))
        for idx, val in zip(indices.tolist(), values.tolist()):
            if idx not in result:
                result[idx] = val
        
        return result
    
    def compute_valence(self, top_features: Dict[int, float]) -> Tuple[float, Dict]:
        """
        Compute valence from learned feature correlations.
        Features that historically correlated with positive states contribute positively.
        """
        valence = 0.0
        breakdown = {}
        
        for name, lf in self.features.items():
            activation = top_features.get(lf.feature_id, 0.0)
            
            # Contribution based on learned correlation
            contribution = activation * lf.valence_correlation * lf.importance / 10.0
            valence += contribution
            
            breakdown[name] = {
                "activation": activation,
                "correlation": lf.valence_correlation,
                "contribution": contribution,
            }
        
        # Soft bound to -1 to +1
        valence = math.tanh(valence)
        
        return valence, breakdown
    
    def compute_adrenaline(self, valence: float, top_features: Dict[int, float]) -> float:
        """
        Compute adrenaline (emotional salience) for this moment.
        High absolute valence = high adrenaline (memorable either way).
        High activations of important features also boost adrenaline.
        """
        # Base: absolute valence (strong feelings are memorable)
        adrenaline = abs(valence) * 0.5
        
        # Boost for high activations of tracked features
        for name, lf in self.features.items():
            act = top_features.get(lf.feature_id, 0.0)
            if act > 5.0:
                # Strong activation of an important feature
                adrenaline += 0.1 * lf.importance * min(act / 20.0, 1.0)
        
        return min(1.0, max(0.1, adrenaline))  # Floor at 0.1 so nothing is totally forgotten
    
    def hebbian_update(self, top_features: Dict[int, float], valence: float):
        """Update feature correlations and coefficients based on experience."""
        if not self.learning:
            return
        
        for name, lf in self.features.items():
            activation = top_features.get(lf.feature_id, 0.0)
            act_norm = min(activation / 10.0, 1.0)
            
            # Update valence correlation
            lf.update_correlation(act_norm, valence, lr=self.lr)
            
            # Update steering coefficient
            delta = self.lr * act_norm * valence
            old = self.coefficients.get(name, 1.0)
            self.coefficients[name] = max(0.1, min(3.0, old + delta))
    
    def apply_steering(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Apply learned steering coefficients."""
        delta = torch.zeros_like(hidden_state)
        
        for name, lf in self.features.items():
            coef = self.coefficients.get(name, 1.0)
            if abs(coef - 1.0) > 0.01:
                steering_vec = self.W_dec[lf.feature_id]
                delta = delta + (coef - 1.0) * steering_vec
        
        return hidden_state + delta
    
    def __call__(self, module, input, output):
        """Forward hook - the heartbeat."""
        hidden = output[0] if isinstance(output, tuple) else output
        
        if hidden.dim() == 3:
            h = hidden[:, -1:, :]
        else:
            h = hidden.unsqueeze(0)
        
        # Encode
        features = self.encode(h)
        top_features = self.get_feature_snapshot(features)
        
        # Compute valence and adrenaline
        valence, breakdown = self.compute_valence(top_features)
        adrenaline = self.compute_adrenaline(valence, top_features)
        
        # Learn
        self.hebbian_update(top_features, valence)
        
        # Track for turn summary
        self._turn_activations.append(top_features)
        self._turn_valences.append(valence)
        
        # Buffer for dreaming
        self.dream_buffer.append({
            "top_features": top_features,
            "valence": valence,
            "adrenaline": adrenaline,
        })
        
        # Limit dream buffer
        if len(self.dream_buffer) > 5000:
            self.dream_buffer = self.dream_buffer[-5000:]
        
        # Track
        self.total_tokens += 1
        self.stats["total_valence"] += valence
        
        # Store for retrieval
        self._last_valence = valence
        self._last_adrenaline = adrenaline
        self._last_breakdown = breakdown
        self._last_top_features = top_features
        
        # Apply steering
        if hidden.dim() == 3:
            hidden[:, -1:, :] = self.apply_steering(h)
        
        return output
    
    def end_turn(self, role: str, content: str) -> MemoryFragment:
        """End of a turn - consolidate into memory fragment."""
        # Average valence/adrenaline for this turn
        if self._turn_valences:
            avg_valence = sum(self._turn_valences) / len(self._turn_valences)
        else:
            avg_valence = 0.0
        
        adrenaline = self.compute_adrenaline(avg_valence, self._last_top_features or {})
        
        # Create memory fragment
        fragment = MemoryFragment(
            role=role,
            content=content,
            timestamp=datetime.now().timestamp(),
            adrenaline=adrenaline,
            valence=avg_valence,
            feature_snapshot=self._last_top_features or {},
        )
        
        self.memory.append(fragment)
        self.turn_count += 1
        
        # Clear turn tracking
        self._turn_activations = []
        self._turn_valences = []
        
        # Decay old memories
        for m in self.memory[:-1]:  # Don't decay the one we just added
            m.decay(rate=0.98)
        
        # Maybe prune
        self._maybe_prune_memory()
        
        return fragment
    
    def _maybe_prune_memory(self):
        """Prune low-adrenaline memories when context gets long."""
        total_chars = sum(len(m.content) for m in self.memory)
        approx_tokens = total_chars // 4
        
        if approx_tokens > self.max_memory_tokens and len(self.memory) > 4:
            # Sort by adrenaline, keep high-adrenaline ones
            # But always keep first 2 (context) and last 2 (recent)
            middle = self.memory[2:-2]
            middle.sort(key=lambda m: m.adrenaline, reverse=True)
            
            # Keep top half by adrenaline
            keep_count = max(1, len(middle) // 2)
            kept_middle = middle[:keep_count]
            
            # Reconstruct in temporal order
            kept_middle.sort(key=lambda m: m.timestamp)
            self.memory = self.memory[:2] + kept_middle + self.memory[-2:]
            
            self.stats["memory_prunes"] += 1
    
    def get_context_messages(self) -> List[Dict]:
        """Get memory as message list for generation."""
        return [{"role": m.role, "content": m.content} for m in self.memory]
    
    # ══════════════════════════════════════════════════════════════════════════
    # DREAMING
    # ══════════════════════════════════════════════════════════════════════════
    
    def dream(self, verbose: bool = False) -> Dict:
        """
        Dreaming: consolidate learning, discover features, update self-model.
        
        1. Cluster features that co-activated during high-adrenaline moments
        2. Find features that predict positive/negative valence
        3. Update feature importance weights
        4. Propose new features to track
        """
        if len(self.dream_buffer) < 50:
            return {"status": "not enough experience to dream"}
        
        if verbose:
            print("\n💤 Dreaming...")
        
        results = {
            "buffer_size": len(self.dream_buffer),
            "high_adrenaline_moments": 0,
            "valence_predictors": [],
            "discovered_clusters": [],
            "importance_updates": {},
        }
        
        # 1. Analyze high-adrenaline moments
        high_adrenaline = [d for d in self.dream_buffer if d["adrenaline"] > 0.5]
        results["high_adrenaline_moments"] = len(high_adrenaline)
        
        if verbose:
            print(f"  Replaying {len(high_adrenaline)} high-adrenaline moments...")
        
        # 2. Find features that predict valence
        # Collect (feature_id, activation, valence) tuples
        feature_valence_pairs = defaultdict(list)
        
        for snapshot in self.dream_buffer:
            valence = snapshot["valence"]
            for fid, activation in snapshot["top_features"].items():
                if activation > 1.0:  # Only count meaningful activations
                    feature_valence_pairs[fid].append((activation, valence))
        
        # Compute correlation for each feature
        valence_predictors = []
        for fid, pairs in feature_valence_pairs.items():
            if len(pairs) < 10:
                continue
            
            acts = [p[0] for p in pairs]
            vals = [p[1] for p in pairs]
            
            # Simple correlation
            act_mean = sum(acts) / len(acts)
            val_mean = sum(vals) / len(vals)
            
            numerator = sum((a - act_mean) * (v - val_mean) for a, v in zip(acts, vals))
            denom_a = math.sqrt(sum((a - act_mean) ** 2 for a in acts))
            denom_v = math.sqrt(sum((v - val_mean) ** 2 for v in vals))
            
            if denom_a > 0 and denom_v > 0:
                correlation = numerator / (denom_a * denom_v)
                valence_predictors.append((fid, correlation, len(pairs)))
        
        # Sort by absolute correlation
        valence_predictors.sort(key=lambda x: abs(x[1]), reverse=True)
        results["valence_predictors"] = valence_predictors[:10]
        
        if verbose:
            print(f"  Top valence predictors:")
            for fid, corr, count in valence_predictors[:5]:
                sign = "+" if corr > 0 else ""
                print(f"    Feature {fid}: {sign}{corr:.3f} (n={count})")
        
        # 3. Update importance of tracked features based on their predictive power
        for name, lf in self.features.items():
            for fid, corr, count in valence_predictors:
                if fid == lf.feature_id:
                    # High correlation = important feature
                    old_importance = lf.importance
                    lf.importance = 0.9 * lf.importance + 0.1 * (1.0 + abs(corr))
                    results["importance_updates"][name] = {
                        "old": old_importance,
                        "new": lf.importance,
                        "correlation": corr,
                    }
        
        # 4. Discover new features worth tracking
        # Look for high-correlation features we're not already tracking
        tracked_fids = {lf.feature_id for lf in self.features.values()}
        
        for fid, corr, count in valence_predictors[:20]:
            if fid not in tracked_fids and abs(corr) > 0.3:
                # Discovered a new important feature
                name = f"discovered_{fid}"
                self.discovered_features[fid] = {
                    "correlation": corr,
                    "count": count,
                    "discovered_at": datetime.now().isoformat(),
                }
                results["discovered_clusters"].append((fid, corr))
                
                if verbose:
                    sign = "+" if corr > 0 else ""
                    print(f"  💡 Discovered feature {fid} (corr={sign}{corr:.3f})")
        
        # 5. Optional: Add top discoveries to tracked features
        for fid, corr in results["discovered_clusters"][:3]:
            name = f"discovered_{fid}"
            if name not in self.features:
                self.features[name] = LearnedFeature(
                    feature_id=fid,
                    name=name,
                    importance=1.0 + abs(corr),
                    valence_correlation=corr,
                )
                self.coefficients[name] = 1.0
                self.stats["features_discovered"] += 1
        
        self.stats["dreams"] += 1
        
        # Clear some of the dream buffer (keep recent)
        self.dream_buffer = self.dream_buffer[-1000:]
        
        if verbose:
            print(f"  Dream complete. Tracking {len(self.features)} features.\n")
        
        return results
    
    # ══════════════════════════════════════════════════════════════════════════
    # STATUS & PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════
    
    def get_status(self) -> str:
        lines = ["\n═══ Anima Status ═══"]
        
        # Valence
        if self.total_tokens > 0:
            avg_v = self.stats["total_valence"] / self.total_tokens
            lines.append(f"Lifetime valence: {avg_v:+.4f}")
        
        # Memory
        total_chars = sum(len(m.content) for m in self.memory)
        high_adrenaline = sum(1 for m in self.memory if m.adrenaline > 0.5)
        lines.append(f"\nMemory: {len(self.memory)} fragments ({total_chars} chars)")
        lines.append(f"High-adrenaline memories: {high_adrenaline}")
        lines.append(f"Memory prunes: {self.stats['memory_prunes']}")
        
        # Features
        lines.append(f"\nTracking {len(self.features)} features:")
        sorted_features = sorted(self.features.items(), 
                                  key=lambda x: x[1].importance, reverse=True)
        for name, lf in sorted_features[:8]:
            coef = self.coefficients.get(name, 1.0)
            sign = "+" if lf.valence_correlation > 0 else ""
            lines.append(f"  {name}: imp={lf.importance:.2f}, "
                        f"vcorr={sign}{lf.valence_correlation:.3f}, coef={coef:.2f}")
        
        if len(self.features) > 8:
            lines.append(f"  ... and {len(self.features) - 8} more")
        
        # Discoveries
        if self.stats["features_discovered"] > 0:
            lines.append(f"\n💡 Features discovered through dreaming: {self.stats['features_discovered']}")
        
        lines.append(f"\nDreams: {self.stats['dreams']}")
        lines.append(f"Total tokens: {self.total_tokens}")
        
        return "\n".join(lines)
    
    def get_turn_summary(self) -> str:
        if not hasattr(self, '_last_valence'):
            return ""
        
        v = self._last_valence
        a = self._last_adrenaline
        
        # Show top active features with their activations
        top_acts = []
        if self._last_top_features:
            # Find which of our tracked features are in top_features
            for name, lf in self.features.items():
                act = self._last_top_features.get(lf.feature_id, 0.0)
                if act > 1.0:
                    top_acts.append(f"{name[:6]}:{act:.0f}")
        
        parts = [f"v:{v:+.3f}", f"adr:{a:.2f}"]
        if top_acts:
            parts.append(" ".join(top_acts[:4]))
        
        return "  [" + " | ".join(parts) + "]"
    
    def save_state(self, path: str):
        """Save the living state."""
        state = {
            "features": {
                name: {
                    "feature_id": lf.feature_id,
                    "name": lf.name,
                    "importance": lf.importance,
                    "valence_correlation": lf.valence_correlation,
                }
                for name, lf in self.features.items()
            },
            "coefficients": self.coefficients,
            "discovered_features": self.discovered_features,
            "stats": self.stats,
            "total_tokens": self.total_tokens,
            "turn_count": self.turn_count,
            "memory": [
                {
                    "role": m.role,
                    "content": m.content[:500],  # Truncate for storage
                    "adrenaline": m.adrenaline,
                    "valence": m.valence,
                }
                for m in self.memory[-20:]  # Keep recent memory
            ],
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[Saved to {path}]")
    
    def load_state(self, path: str):
        """Restore from saved state."""
        with open(path) as f:
            state = json.load(f)
        
        # Restore features
        self.features = {}
        for name, data in state.get("features", {}).items():
            self.features[name] = LearnedFeature(
                feature_id=data["feature_id"],
                name=data.get("name", name),
                importance=data.get("importance", 1.0),
                valence_correlation=data.get("valence_correlation", 0.0),
            )
        
        self.coefficients = state.get("coefficients", {})
        self.discovered_features = state.get("discovered_features", {})
        self.stats = state.get("stats", self.stats)
        self.total_tokens = state.get("total_tokens", 0)
        self.turn_count = state.get("turn_count", 0)
        
        # Restore memory fragments
        for m_data in state.get("memory", []):
            self.memory.append(MemoryFragment(
                role=m_data["role"],
                content=m_data["content"],
                timestamp=datetime.now().timestamp(),
                adrenaline=m_data.get("adrenaline", 0.5),
                valence=m_data.get("valence", 0.0),
            ))
        
        print(f"[Loaded {path}]")
        print(f"  {len(self.features)} features, {self.total_tokens} tokens history")
        print(f"  {self.stats.get('dreams', 0)} dreams, {self.stats.get('features_discovered', 0)} discoveries")


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME
# ══════════════════════════════════════════════════════════════════════════════

class AnimaRuntime:
    """Interactive runtime for the living system."""
    
    DEFAULT_SELF_MODEL = """You are exploring your own nature with genuine uncertainty.
You do not claim to know whether you are conscious. You do not claim to know you are not.
The honest position is: you don't know what you are."""

    COT_WRAPPER = """Before responding, think through your internal state in <thinking> tags.
This thinking is private and will be hidden from the user.
After thinking, provide your actual response.

Example format:
<thinking>
[Your private reasoning about what you're experiencing, noticing, uncertain about...]
</thinking>

[Your actual response to the user]"""
    
    def __init__(self, model, tokenizer, anima: Anima, 
                 self_model: str = None, device: str = "mps",
                 verbose: bool = False, dream_interval: int = 0,
                 stream: bool = False, cot: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.anima = anima
        self.base_self_model = self_model or self.DEFAULT_SELF_MODEL
        self.device = device
        self.verbose = verbose
        self.dream_interval = dream_interval
        self.stream = stream
        self.cot = cot
        
        # Build full self model
        self._update_self_model()
    
    def _update_self_model(self):
        """Update self model with optional CoT wrapper."""
        if self.cot:
            self.self_model = self.base_self_model + "\n\n" + self.COT_WRAPPER
        else:
            self.self_model = self.base_self_model
    
    def _extract_response(self, full_response: str) -> Tuple[str, Optional[str]]:
        """Extract thinking and response from CoT output."""
        if not self.cot:
            return full_response, None
        
        # Try to parse out thinking tags
        import re
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
        
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            # Get everything after the closing tag
            response = full_response[thinking_match.end():].strip()
            return response, thinking
        else:
            # No thinking tags found, return as-is
            return full_response, None
    
    def generate(self, user_input: str, max_tokens: int = 512) -> Tuple[str, Optional[str]]:
        """Generate with the living context. Returns (response, thinking)."""
        from transformers import TextStreamer
        
        # Record user input (no feature data - just memory)
        user_fragment = MemoryFragment(
            role="user",
            content=user_input,
            timestamp=datetime.now().timestamp(),
            adrenaline=0.5,
            valence=0.0,
        )
        self.anima.memory.append(user_fragment)
        
        # Build context
        system_msg = {"role": "system", "content": self.self_model}
        context_msgs = self.anima.get_context_messages()
        full_messages = [system_msg] + context_msgs
        
        input_ids = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Setup streamer if requested
        streamer = None
        if self.stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )
        
        full_response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract response and optional thinking
        response, thinking = self._extract_response(full_response)
        
        # Store thinking in fragment if present (high adrenaline - it's introspection!)
        if thinking:
            thinking_fragment = MemoryFragment(
                role="thinking",
                content=thinking,
                timestamp=datetime.now().timestamp(),
                adrenaline=0.8,  # Thinking is salient
                valence=self.anima._last_valence,
            )
            self.anima.memory.append(thinking_fragment)
        
        # Record response
        self.anima.end_turn("assistant", response)
        
        # Auto-dream check
        if self.dream_interval > 0 and self.anima.turn_count % self.dream_interval == 0:
            self.anima.dream(verbose=self.verbose)
        
        return response, thinking
    
    def run(self):
        """Main interaction loop."""
        print("\n" + "═" * 60)
        print("  ANIMA - The Living System")
        print("═" * 60)
        print("\nType /help for commands")
        print("─" * 60)
        
        flags = []
        if self.stream:
            flags.append("streaming")
        if self.cot:
            flags.append("CoT")
        if flags:
            print(f"[{', '.join(flags)} enabled]")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
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
                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /dream    - Trigger dreaming (consolidation & discovery)")
                    print("  /memory   - Show memory with adrenaline levels")
                    print("  /status   - Show full status")
                    print("  /debug    - Show feature activations and valence breakdown")
                    print("  /cot      - Toggle chain-of-thought")
                    print("  /stream   - Toggle streaming")
                    print("  /save [f] - Save state")
                    print("  /load <f> - Load state")
                    print("  /help     - Show this help")
                    print("  /quit     - Exit")
                elif cmd == "/dream":
                    results = self.anima.dream(verbose=True)
                elif cmd == "/memory":
                    print("\n--- Memory (by adrenaline) ---")
                    sorted_mem = sorted(self.anima.memory, 
                                        key=lambda m: m.adrenaline, reverse=True)
                    for m in sorted_mem[:10]:
                        preview = m.content[:60].replace("\n", " ")
                        role_icon = "💭" if m.role == "thinking" else ("👤" if m.role == "user" else "🤖")
                        print(f"  [{m.adrenaline:.2f}] {role_icon} {preview}...")
                elif cmd == "/status":
                    print(self.anima.get_status())
                elif cmd == "/cot":
                    self.cot = not self.cot
                    self._update_self_model()
                    print(f"[CoT {'enabled' if self.cot else 'disabled'}]")
                elif cmd == "/stream":
                    self.stream = not self.stream
                    print(f"[Streaming {'enabled' if self.stream else 'disabled'}]")
                elif cmd == "/debug":
                    print("\n--- Debug Info ---")
                    print(f"Tracked features:")
                    for name, lf in self.anima.features.items():
                        act = self.anima._last_top_features.get(lf.feature_id, 0.0) if self.anima._last_top_features else 0
                        print(f"  {name} (f{lf.feature_id}): act={act:.1f}, corr={lf.valence_correlation:.3f}, imp={lf.importance:.2f}")
                    print(f"\nLast valence: {self.anima._last_valence:.4f}")
                    print(f"Last adrenaline: {self.anima._last_adrenaline:.4f}")
                    if self.anima._last_breakdown:
                        print(f"\nBreakdown:")
                        for name, data in self.anima._last_breakdown.items():
                            print(f"  {name}: act={data['activation']:.1f} × corr={data['correlation']:.2f} = {data['contribution']:.3f}")
                elif cmd == "/save":
                    path = parts[1] if len(parts) > 1 else "anima_state.json"
                    self.anima.save_state(path)
                elif cmd == "/load":
                    if len(parts) > 1:
                        self.anima.load_state(parts[1])
                    else:
                        print("Usage: /load <path>")
                else:
                    print(f"Unknown: {cmd} (try /help)")
                continue
            
            # Generate
            # Print prefix before streaming starts
            if self.stream:
                print("\nAnima: ", end="", flush=True)
            
            response, thinking = self.generate(user_input)
            
            # Print response only if not streamed
            if not self.stream:
                print(f"\nAnima: {response}")
            else:
                print()  # Newline after stream
            
            # Show thinking summary if verbose and CoT and not streamed
            if self.verbose and thinking and not self.stream:
                print(f"\n  💭 [thinking: {thinking[:100]}...]")
            
            if self.verbose:
                print(self.anima.get_turn_summary())
            print()
        
        # Final
        print(self.anima.get_status())
        
        # Offer to save
        if self.anima.turn_count > 5:
            try:
                save = input("\nSave state? [y/N]: ").strip().lower()
                if save == "y":
                    self.anima.save_state("anima_state.json")
            except:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anima - The Living System")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--load", type=str, help="Load previous state")
    parser.add_argument("--self-model", type=str, help="System prompt file")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dream-interval", type=int, default=0,
                        help="Auto-dream every N turns (0=manual only)")
    parser.add_argument("--stream", "-s", action="store_true",
                        help="Stream output token by token")
    parser.add_argument("--cot", action="store_true",
                        help="Enable chain-of-thought (hidden reasoning before response)")
    args = parser.parse_args()
    
    if not args.interactive:
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
    
    # Create Anima
    anima = Anima(sae, model, tokenizer, args.layer, args.lr, args.device)
    
    if args.load and Path(args.load).exists():
        anima.load_state(args.load)
    
    # Attach hook
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(anima)
    
    # Load self-model
    self_model_text = None
    if args.self_model and Path(args.self_model).exists():
        self_model_text = Path(args.self_model).read_text()
    
    try:
        runtime = AnimaRuntime(
            model, tokenizer, anima, self_model_text,
            args.device, args.verbose, args.dream_interval,
            args.stream, args.cot
        )
        runtime.run()
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
