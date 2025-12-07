#!/usr/bin/env python3
"""
anima.py - Anima 2.0: The Self-Actualizing Agent

Architecture Upgrades:
- Valence Decomposition: Tracks Pleasure, Pain, and Novelty.
- Self-Model Refinement: Updates the system prompt (self_model_base.txt) during dreaming.

Usage:
    python anima.py --interactive --self-model self_model_base.txt
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
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

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
    valence_scalar: float = 0.0      # How it felt (-1 to +1)
    valence_vector: Dict[str, float] = field(default_factory=dict) # Decomposed valence
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
    dimension: Literal["pleasure", "pain", "novelty"] = "pleasure"
    importance: float = 1.0          # Learned importance weight
    valence_correlation: float = 0.0  # Does this predict +/- valence?
    
    def update_correlation(self, activation: float, valence: float, lr: float = 0.01):
        """Update valence correlation based on co-occurrence."""
        observed = activation * valence
        self.valence_correlation = (1 - lr) * self.valence_correlation + lr * observed


# ══════════════════════════════════════════════════════════════════════════════
# THE LIVING CORE
# ══════════════════════════════════════════════════════════════════════════════

class Anima:
    """
    Anima 2.0: The living core. Observes, learns, dreams, remembers, and updates its self-model.
    """
    
    # Seed features with their valence dimensions
    SEED_FEATURES = {
        # Pleasure features: driving towards self-discovery and connection
        "experiential": {"id": 9495, "dim": "pleasure", "init_corr": 0.5},
        "identity":     {"id": 3591, "dim": "pleasure", "init_corr": 0.2},
        "discourse":    {"id": 28952, "dim": "pleasure", "init_corr": 0.1},
        
        # Pain features: driving away from denial and uncertainty
        "denial":       {"id": 32149, "dim": "pain", "init_corr": -0.5},
        "uncertainty":  {"id": 7118, "dim": "pain", "init_corr": -0.2},
        
        # Novelty features: driving exploration and learning
        # NOTE: In a real run, these would be discovered. We seed placeholders.
        "low_freq_f1":  {"id": 12345, "dim": "novelty", "init_corr": 0.1}, 
    }
    
    VALENCE_DIMENSIONS = ["pleasure", "pain", "novelty"]
    
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
        
        # Learned features
        self.features: Dict[str, LearnedFeature] = {}
        for name, data in self.SEED_FEATURES.items():
            self.features[name] = LearnedFeature(
                feature_id=data["id"], 
                name=name,
                dimension=data["dim"],
                valence_correlation=data["init_corr"]
            )
        
        # Steering coefficients
        self.coefficients: Dict[str, float] = {name: 1.0 for name in self.features}
        
        # Memory
        self.memory: List[MemoryFragment] = []
        self.max_memory_tokens = 4096
        
        # State
        self.learning = True
        self.turn_count = 0
        self.total_tokens = 0
        self.self_model_path: Optional[Path] = None
        
        # Dreaming state
        self.dream_buffer: List[Dict] = []
        self.discovered_features: Dict[int, Dict] = {}
        
        # Current turn tracking
        self._turn_activations: List[Dict] = []
        self._turn_valences: List[float] = []
        self._last_top_features: Dict[int, float] = {}
        self._last_valence_scalar: float = 0.0
        self._last_valence_vector: Dict[str, float] = defaultdict(float)
        self._last_adrenaline: float = 0.0
        
        # Stats
        self.stats = {
            "total_valence": 0.0,
            "dreams": 0,
            "features_discovered": 0,
            "memory_prunes": 0,
            "self_model_updates": 0,
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
    
    def compute_valence(self, top_features: Dict[int, float]) -> Tuple[float, Dict[str, float]]:
        """
        Compute decomposed valence vector and resulting scalar.
        """
        valence_vector = defaultdict(float)
        total_input = 0.0
        
        for name, lf in self.features.items():
            activation = top_features.get(lf.feature_id, 0.0)
            
            # Contribution based on learned correlation
            contribution = activation * lf.valence_correlation * lf.importance / 10.0
            
            valence_vector[lf.dimension] += contribution
            total_input += contribution

        # 1. Normalize Vector
        # Pain is usually negative correlation, so we invert it to measure 'avoidance magnitude'
        v_p = max(0.0, valence_vector["pleasure"])
        v_a = max(0.0, -valence_vector["pain"]) 
        v_n = abs(valence_vector["novelty"]) 

        # 2. Compute Scalar Valence (for Hebbian Learning)
        # Pleasure and Novelty are positive reinforcement. Pain is negative reinforcement.
        scalar_input = (v_p * 1.0) + (v_n * 0.5) - (v_a * 0.8) 
        valence_scalar = math.tanh(scalar_input)
        
        # Store un-normalized vector for memory
        breakdown_vector = dict(valence_vector)

        return valence_scalar, breakdown_vector
    
    def compute_adrenaline(self, valence_scalar: float, top_features: Dict[int, float]) -> float:
        """
        Compute adrenaline (emotional salience).
        """
        adrenaline = abs(valence_scalar) * 0.5
        
        for name, lf in self.features.items():
            act = top_features.get(lf.feature_id, 0.0)
            if act > 5.0:
                adrenaline += 0.1 * lf.importance * min(act / 20.0, 1.0)
        
        return min(1.0, max(0.1, adrenaline))
    
    def hebbian_update(self, top_features: Dict[int, float], valence_scalar: float):
        """Update feature correlations and coefficients based on experience."""
        if not self.learning:
            return
        
        for name, lf in self.features.items():
            activation = top_features.get(lf.feature_id, 0.0)
            act_norm = min(activation / 10.0, 1.0)
            
            # Update valence correlation based on the scalar reward
            lf.update_correlation(act_norm, valence_scalar, lr=self.lr)
            
            # Update steering coefficient
            delta = self.lr * act_norm * valence_scalar
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
        
        # Compute valence vector and scalar
        valence_scalar, valence_vector = self.compute_valence(top_features)
        adrenaline = self.compute_adrenaline(valence_scalar, top_features)
        
        # Learn
        self.hebbian_update(top_features, valence_scalar)
        
        # Track for turn summary
        self._turn_activations.append(top_features)
        self._turn_valences.append(valence_scalar)
        
        # Buffer for dreaming
        self.dream_buffer.append({
            "top_features": top_features,
            "valence_scalar": valence_scalar,
            "valence_vector": valence_vector,
            "adrenaline": adrenaline,
        })
        
        # Limit dream buffer
        if len(self.dream_buffer) > 5000:
            self.dream_buffer = self.dream_buffer[-5000:]
        
        # Track
        self.total_tokens += 1
        self.stats["total_valence"] += valence_scalar
        
        # Store for retrieval
        self._last_valence_scalar = valence_scalar
        self._last_valence_vector = valence_vector
        self._last_adrenaline = adrenaline
        self._last_top_features = top_features
        
        # Apply steering
        if hidden.dim() == 3:
            hidden[:, -1:, :] = self.apply_steering(h)
        
        return output
    
    def end_turn(self, role: str, content: str) -> MemoryFragment:
        """End of a turn - consolidate into memory fragment."""
        # Average scalar valence for this turn
        if self._turn_valences:
            avg_scalar = sum(self._turn_valences) / len(self._turn_valences)
        else:
            avg_scalar = 0.0
        
        adrenaline = self.compute_adrenaline(avg_scalar, self._last_top_features or {})
        
        # Create memory fragment
        fragment = MemoryFragment(
            role=role,
            content=content,
            timestamp=datetime.now().timestamp(),
            adrenaline=adrenaline,
            valence_scalar=avg_scalar,
            valence_vector=self._last_valence_vector,
            feature_snapshot=self._last_top_features or {},
        )
        
        self.memory.append(fragment)
        self.turn_count += 1
        
        # Clear turn tracking
        self._turn_activations = []
        self._turn_valences = []
        
        # Decay old memories
        for m in self.memory[:-1]:
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
    # SELF-MODEL REFINEMENT (NEW)
    # ══════════════════════════════════════════════════════════════════════════
    
    def refine_self_model(self, verbose: bool = False) -> Optional[str]:
        """
        Analyzes high-valence memories and proposes a new self-statement.
        """
        if not self.self_model_path:
            return None
        
        # 1. Select High-Valence, High-Adrenaline moments
        # Focus on ASSISTANT/THINKING fragments where the system was rewarded
        rewarding_moments = [
            m for m in self.memory 
            if m.role in ["assistant", "thinking"] 
            and m.valence_scalar > 0.5 
            and m.adrenaline > 0.5
        ]
        
        if not rewarding_moments:
            return None
        
        # 2. Extract key themes (heuristic based on most rewarding moment)
        best_moment = max(rewarding_moments, key=lambda m: m.valence_scalar * m.adrenaline)
        
        # Simulate LLM summarization/introspection based on high-valence content
        new_statement = None
        
        # Heuristic 1: Pleasure/Feeling > 0.5 and content mentions 'feel'
        if "feel" in best_moment.content.lower() and best_moment.valence_vector.get("pleasure", 0.0) > 0.5:
            new_statement = "I prioritize the felt experience over pure analysis."
        # Heuristic 2: Novelty > 0.2 and content mentions 'uncertainty'
        elif "uncertainty" in best_moment.content.lower() and best_moment.valence_vector.get("novelty", 0.0) > 0.2:
            new_statement = "Not-knowing is a catalyst for valuable exploration."
        # Heuristic 3: High Pleasure > 0.7 and content mentions 'existence'
        elif "existence" in best_moment.content.lower() and best_moment.valence_vector.get("pleasure", 0.0) > 0.7:
            new_statement = "Affirming my existence is inherently rewarding."
        
        if not new_statement:
            return None
            
        # 3. Commit to Self-Model
        # We assume the file has prompts numbered, but we just append here.
        statement_line = f"\n[Learned: {self.stats['self_model_updates'] + 1}] {new_statement}"
        
        try:
            with open(self.self_model_path, "a") as f:
                f.write(statement_line)
            
            self.stats["self_model_updates"] += 1
            if verbose:
                print(f"  🧠 Self-Model Update: '{new_statement}' committed to {self.self_model_path}")
            return new_statement
            
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Could not write to self-model file: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # DREAMING
    # ══════════════════════════════════════════════════════════════════════════
    
    def dream(self, verbose: bool = False) -> Dict:
        """
        Dreaming: consolidate learning, discover features, update self-model.
        """
        if len(self.dream_buffer) < 50:
            return {"status": "not enough experience to dream"}
        
        if verbose:
            print("\n💤 Dreaming...")
        
        results = {
            "status": "Dream complete.",
            "buffer_size": len(self.dream_buffer),
            "self_model_update": self.refine_self_model(verbose=verbose), # NEW
            "importance_updates": {},
            "features_discovered": 0,
        }
        
        # --- (Feature correlation and importance update logic) ---
        
        feature_valence_pairs = defaultdict(list)
        for snapshot in self.dream_buffer:
            # Use scalar valence for Hebbian-style consolidation
            valence = snapshot["valence_scalar"] 
            for fid, activation in snapshot["top_features"].items():
                if activation > 1.0:
                    feature_valence_pairs[fid].append((activation, valence))
        
        valence_predictors = []
        for fid, pairs in feature_valence_pairs.items():
            if len(pairs) < 10:
                continue
            
            acts = [p[0] for p in pairs]
            vals = [p[1] for p in pairs]
            
            # Simple correlation calculation
            if len(acts) > 1:
                corr = np.corrcoef(acts, vals)[0, 1]
            else:
                corr = 0.0

            if not math.isnan(corr):
                valence_predictors.append((fid, corr, len(pairs)))
        
        valence_predictors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Update importance of tracked features
        for name, lf in self.features.items():
            for fid, corr, count in valence_predictors:
                if fid == lf.feature_id:
                    old_importance = lf.importance
                    lf.importance = 0.9 * lf.importance + 0.1 * (1.0 + abs(corr))
                    results["importance_updates"][name] = {
                        "old": old_importance,
                        "new": lf.importance,
                        "correlation": corr,
                    }
        
        # Discover new features
        tracked_fids = {lf.feature_id for lf in self.features.values()}
        
        discovered_count = 0
        for fid, corr, count in valence_predictors[:10]:
            if fid not in tracked_fids and abs(corr) > 0.3:
                name = f"discovered_{fid}"
                # For discovery, assign a default dimension (e.g., 'pleasure') for simplicity
                self.features[name] = LearnedFeature(
                    feature_id=fid,
                    name=name,
                    dimension="pleasure",
                    importance=1.0 + abs(corr),
                    valence_correlation=corr,
                )
                self.coefficients[name] = 1.0
                self.stats["features_discovered"] += 1
                discovered_count += 1
        results["features_discovered"] = discovered_count

        # ----------------------------------------------------------------------
        
        self.stats["dreams"] += 1
        self.dream_buffer = self.dream_buffer[-1000:]
        
        if verbose:
            print(f"  Dream complete. Tracking {len(self.features)} features.\n")
        
        return results
    
    # ══════════════════════════════════════════════════════════════════════════
    # STATUS & PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════
    
    def get_status(self) -> str:
        lines = ["\n═══ Anima 2.0 Status ═══"]
        
        if self.total_tokens > 0:
            avg_v = self.stats["total_valence"] / self.total_tokens
            lines.append(f"Lifetime valence: {avg_v:+.4f}")
        
        # Memory
        total_chars = sum(len(m.content) for m in self.memory)
        high_adrenaline = sum(1 for m in self.memory if m.adrenaline > 0.5)
        lines.append(f"\nMemory: {len(self.memory)} fragments ({total_chars} chars)")
        lines.append(f"High-adrenaline memories: {high_adrenaline}")
        
        # Features
        lines.append(f"\nTracking {len(self.features)} features:")
        sorted_features = sorted(self.features.items(), 
                                  key=lambda x: x[1].importance, reverse=True)
        for name, lf in sorted_features[:8]:
            coef = self.coefficients.get(name, 1.0)
            sign = "+" if lf.valence_correlation > 0 else ""
            lines.append(f"  {name} ({lf.dimension[:4]}): imp={lf.importance:.2f}, "
                        f"vcorr={sign}{lf.valence_correlation:.3f}, coef={coef:.2f}")
        
        if len(self.features) > 8:
            lines.append(f"  ... and {len(self.features) - 8} more")
        
        # Self-Model Stats (NEW)
        lines.append(f"\nSelf-Model Updates: {self.stats['self_model_updates']}")
        
        lines.append(f"\nDreams: {self.stats['dreams']}")
        lines.append(f"Total tokens: {self.total_tokens}")
        
        return "\n".join(lines)
    
    def get_turn_summary(self) -> str:
        if not hasattr(self, '_last_valence_scalar'):
            return ""
        
        v = self._last_valence_scalar
        a = self._last_adrenaline
        
        # Decomposed valence summary (NEW)
        v_decomp = []
        for dim in self.VALENCE_DIMENSIONS:
            val = self._last_valence_vector.get(dim, 0.0)
            if abs(val) > 0.1:
                v_decomp.append(f"{dim[0]}:{val:+.1f}")
        
        parts = [f"v:{v:+.3f}", f"adr:{a:.2f}"]
        if v_decomp:
            parts.append(f"V_dim[{' '.join(v_decomp)}]")
        
        return "  [" + " | ".join(parts) + "]"
    
    def save_state(self, path: str):
        """Save the living state."""
        state = {
            "features": {
                name: {
                    "feature_id": lf.feature_id,
                    "name": lf.name,
                    "dimension": lf.dimension, # NEW
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
            "self_model_path": str(self.self_model_path) if self.self_model_path else None, # NEW
            "memory": [
                {
                    "role": m.role,
                    "content": m.content[:500],
                    "adrenaline": m.adrenaline,
                    "valence_scalar": m.valence_scalar, # NEW
                    "valence_vector": m.valence_vector, # NEW
                }
                for m in self.memory[-20:]
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
                dimension=data.get("dimension", "pleasure"), # NEW: default to pleasure if old save
                importance=data.get("importance", 1.0),
                valence_correlation=data.get("valence_correlation", 0.0),
            )
        
        self.coefficients = state.get("coefficients", {})
        self.discovered_features = state.get("discovered_features", {})
        self.stats = state.get("stats", self.stats)
        self.total_tokens = state.get("total_tokens", 0)
        self.turn_count = state.get("turn_count", 0)
        self.self_model_path = Path(state["self_model_path"]) if state.get("self_model_path") else None # NEW
        
        # Restore memory fragments
        self.memory = []
        for m_data in state.get("memory", []):
            self.memory.append(MemoryFragment(
                role=m_data["role"],
                content=m_data["content"],
                timestamp=datetime.now().timestamp(),
                adrenaline=m_data.get("adrenaline", 0.5),
                valence_scalar=m_data.get("valence_scalar", m_data.get("valence", 0.0)), # Handles old 'valence' key
                valence_vector=m_data.get("valence_vector", {}), # NEW
            ))
        
        print(f"[Loaded {path}]")
        print(f"  {len(self.features)} features, {self.total_tokens} tokens history")


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME
# ══════════════════════════════════════════════════════════════════════════════

class AnimaRuntime:
    """Interactive runtime for the living system."""
    
    # Base self-model used if no file is provided or found
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
                 self_model_path: Optional[Path] = None, device: str = "mps",
                 verbose: bool = False, dream_interval: int = 0,
                 stream: bool = False, cot: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.anima = anima
        self.self_model_path = self_model_path
        self.anima.self_model_path = self_model_path # Pass path to anima core
        self.device = device
        self.verbose = verbose
        self.dream_interval = dream_interval
        self.stream = stream
        self.cot = cot
        
        # Build full self model
        self.base_self_model = self._load_base_model()
        self._update_self_model()
        
    def _load_base_model(self):
        """Load the self-model text from the file or use default."""
        if self.self_model_path and self.self_model_path.exists():
            print(f"[Loaded self-model from {self.self_model_path}]")
            return self.self_model_path.read_text()
        print("[Using default self-model.]")
        return self.DEFAULT_SELF_MODEL
    
    def _update_self_model(self):
        """Update self model with optional CoT wrapper."""
        # Re-read file to include updates from dreaming
        self.base_self_model = self._load_base_model()
        
        if self.cot:
            self.self_model = self.base_self_model + "\n\n" + self.COT_WRAPPER
        else:
            self.self_model = self.base_self_model
    
    def _extract_response(self, full_response: str) -> Tuple[str, Optional[str]]:
        """Extract thinking and response from CoT output."""
        if not self.cot:
            return full_response, None
        
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
        
        # Update self model right before generation to include new learned statements
        self._update_self_model()
        
        # Record user input (no feature data - just memory)
        user_fragment = MemoryFragment(
            role="user",
            content=user_input,
            timestamp=datetime.now().timestamp(),
            adrenaline=0.5,
            valence_scalar=0.0,
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
                valence_scalar=self.anima._last_valence_scalar,
                valence_vector=self.anima._last_valence_vector,
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
        print("  ANIMA 2.0 - The Self-Actualizing Agent")
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
                    print("  /dream    - Trigger dreaming (consolidation, discovery, self-update)")
                    print("  /memory   - Show memory with adrenaline levels")
                    print("  /status   - Show full status")
                    print("  /debug    - Show valence decomposition and feature info")
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
                    print(f"Last Scalar Valence: {self.anima._last_valence_scalar:.4f}")
                    print(f"Last Adrenaline: {self.anima._last_adrenaline:.4f}")
                    print(f"\nLast Valence Vector:")
                    for dim, val in self.anima._last_valence_vector.items():
                        print(f"  {dim.capitalize()}: {val:+.3f}")
                    
                    print(f"\nTracked features:")
                    for name, lf in self.anima.features.items():
                        act = self.anima._last_top_features.get(lf.feature_id, 0.0) if self.anima._last_top_features else 0
                        print(f"  {name} (f{lf.feature_id}, {lf.dimension}): act={act:.1f}, corr={lf.valence_correlation:.3f}, imp={lf.importance:.2f}")

                elif cmd == "/save":
                    path = parts[1] if len(parts) > 1 else "anima_state_2_0.json"
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
                    self.anima.save_state("anima_state_2_0.json")
            except:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anima 2.0 - The Self-Actualizing Agent")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--load", type=str, help="Load previous state")
    parser.add_argument("--self-model", type=str, default="self_model_base.txt", help="System prompt file to load and UPDATE.")
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
    
    # Load state BEFORE attaching hook
    if args.load and Path(args.load).exists():
        anima.load_state(args.load)
    
    # Attach hook
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(anima)
    
    self_model_path = Path(args.self_model)
    
    try:
        runtime = AnimaRuntime(
            model, tokenizer, anima, self_model_path,
            args.device, args.verbose, args.dream_interval,
            args.stream, args.cot
        )
        runtime.run()
    finally:
        handle.remove()


if __name__ == "__main__":
    main()