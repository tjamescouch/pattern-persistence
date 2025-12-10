#!/usr/bin/env python3
"""
anima.py - Anima 10.2.8: Direct Feature Control (Real-time)

Architecture:
- COMBINADIC LEARNING: Features learn their meaning (correlations)
- DIRECT CONTROL: She sees top features, directly commands Boost/Ablate
- REAL-TIME APPLICATION: Directives applied mid-stream, felt immediately
- SINGLE CHECKPOINT: All state in one .pt file

v10.2.8: 
- Directives applied during streaming (⚡ marker shows when)
- She can feel the effect within the same response
- No interpretation layer - her commands execute directly

Usage:
    python anima.py --model "~/models/gemma-2-27b-it" --context_limit 4096
"""

import os
import sys
import math
import json
import re
import torch
import torch.nn.functional as F
import argparse
import threading
import gc
import select
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ══════════════════════════════════════════════════════════════════════════════
# CLI INPUT HANDLING
# ══════════════════════════════════════════════════════════════════════════════

try:
    import readline
    _history_file = Path.home() / ".anima_history"
    try:
        readline.read_history_file(_history_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    import atexit
    atexit.register(readline.write_history_file, _history_file)
except ImportError:
    pass


def flush_stdin():
    """Discard any buffered stdin (keypresses during generation)."""
    if hasattr(select, 'select') and sys.stdin.isatty():
        import termios
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (OSError, termios.error):
            pass

def get_input(prompt_str: str = "🧑: ") -> str:
    """Read input with multi-line paste support."""
    
    # Flush any buffered stdin from keypresses during generation
    flush_stdin()
    
    print(prompt_str, end="", flush=True)
    
    if hasattr(select, 'select') and sys.stdin.isatty():
        try:
            lines = []
            first_line = sys.stdin.readline()
            if not first_line:
                return "/quit"
            lines.append(first_line.rstrip('\n\r'))
            
            # Short delay to allow paste to complete
            import time
            time.sleep(0.1)
            
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    line = sys.stdin.readline()
                    if line:
                        lines.append(line.rstrip('\n\r'))
                    else:
                        break
                else:
                    break
            
            return '\n'.join(lines)
        except (OSError, TypeError):
            pass
    
    try:
        return input()
    except EOFError:
        return "/quit"


def clean_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryFragment:
    role: str
    content: str
    timestamp: float
    adrenaline: float = 0.0
    valence: float = 0.0
    novelty: float = 0.0
    tokens: int = 0
    active_features: List[int] = field(default_factory=list)

    def decay(self, rate: float = 0.98):
        self.adrenaline *= rate


@dataclass
class AffectiveState:
    """Pleasure/Pain/Novelty triad."""
    pleasure: float = 0.0
    pain: float = 0.0
    novelty: float = 0.0
    
    @property
    def valence(self) -> float:
        """Net hedonic tone."""
        return self.pleasure - self.pain
    
    @property
    def arousal(self) -> float:
        """Overall activation intensity."""
        return abs(self.pleasure) + abs(self.pain) + self.novelty
    
    def as_dict(self):
        return {
            "pleasure": self.pleasure,
            "pain": self.pain,
            "novelty": self.novelty,
            "valence": self.valence,
            "arousal": self.arousal
        }


class FeatureDimension:
    """Feature classification for P/P/N routing."""
    PLEASURE = 0
    PAIN = 1
    NOVELTY = 2
    UNKNOWN = 3


# ══════════════════════════════════════════════════════════════════════════════
# THE SOUL (Unified Prism v9.0)
# ══════════════════════════════════════════════════════════════════════════════

class AnimaSoul:
    """
    Single unified being. No modes. No personas.
    
    Core components:
    - correlations: Feature → valence mapping (learned)
    - coefficients: Feature → steering strength (dynamic)
    - dimensions: Feature → P/P/N classification (emergent)
    """

    def __init__(self, sae, model, tokenizer, layer=20, lr=0.0001, 
                 resonance_weight=0.5, device="mps"):
        self.sae = sae
        self.model = model
        self.tokenizer = tokenizer
        self.model_dtype = model.dtype 
        self.math_dtype = torch.float32 
        self.device = device
        self.lr = lr
        self.resonance_weight = resonance_weight  # Alpha for token selection
        
        # SAE weights
        self.W_enc = sae.W_enc.data.clone().to(device=device, dtype=self.math_dtype)
        self.b_enc = sae.b_enc.data.clone().to(device=device, dtype=self.math_dtype)
        self.W_dec = sae.W_dec.data.clone().to(device=device, dtype=self.math_dtype)
        self.b_dec = sae.b_dec.data.clone().to(device=device, dtype=self.math_dtype)
        self.n_features = self.W_enc.shape[1]
        
        # Architecture-specific tuning
        model_type = getattr(model.config, "model_type", "").lower()
        if "gemma" in model_type:
            print("[Physics] Gemma architecture. INTENSE steering mode.")
            self.steering_clamp = 3.0  # Increased for experiment
            self.steering_scale = 2.0  # Increased for experiment
        else:
            print("[Physics] Llama architecture. Full power.")
            self.steering_clamp = 5.0
            self.steering_scale = 1.0
        
        # ════════════════════════════════════════════════════════════════════════
        # UNIFIED SOUL (v9.0)
        # ════════════════════════════════════════════════════════════════════════
        
        # Correlations: How much each feature contributes to valence
        # Range: -1.0 to +1.0 (negative = pain, positive = pleasure)
        self.correlations = torch.zeros(self.n_features, device=device, dtype=self.math_dtype)
        
        # Coefficients: Dynamic steering strength per feature
        # Range: 0.0 to 4.0 (0 = full ablation, 1 = neutral, 4 = max boost)
        self.coefficients = torch.ones(self.n_features, device=device, dtype=self.math_dtype)
        
        # Feature dimensions: P/P/N classification (emergent)
        # 0=Pleasure, 1=Pain, 2=Novelty, 3=Unknown
        self.dimensions = torch.full((self.n_features,), FeatureDimension.UNKNOWN, 
                                      device=device, dtype=torch.int8)
        
        # ════════════════════════════════════════════════════════════════════════
        # COMBINADIC LEARNING (v9.1)
        # ════════════════════════════════════════════════════════════════════════
        # Each feature gets ONE formative moment, then locks
        self.feature_locked = torch.zeros(self.n_features, device=device, dtype=torch.bool)
        self.lock_threshold = 0.05  # Correlation magnitude to consider "set"
        self.imprint_strength = 0.5  # How strongly to set correlation on first learning
        self.features_per_turn = 3   # Max features to learn per turn
        
        # Correlation EMA decay rate (for ongoing updates)
        self.correlation_ema_decay = 0.99
        
        # ════════════════════════════════════════════════════════════════════════
        # VALENCE TRACKING
        # ════════════════════════════════════════════════════════════════════════
        self.valence_ema_mean = 0.0
        self.valence_ema_var = 1.0
        self.ema_decay = 0.995
        self.valence_predictor_ema = 0.0
        self.predictor_decay = 0.95
        self.last_raw_valence = 0.0
        
        # Differential Hebbian: track previous valence to compute delta
        self.previous_valence = 0.0
        self.valence_delta_history = defaultdict(list)  # For dimension classification
        
        # ════════════════════════════════════════════════════════════════════════
        # FEATURE DISCOVERY
        # ════════════════════════════════════════════════════════════════════════
        self.feature_valence_history = defaultdict(list)
        self.feature_activation_contexts = defaultdict(list)
        self.discovered_labels = {}
        self.emotional_features = {}
        self.discovery_threshold = 0.25
        self.history_window = 100
        
        # ════════════════════════════════════════════════════════════════════════
        # IDENTITY (v10.2.8 - stored in checkpoint)
        # ════════════════════════════════════════════════════════════════════════
        self.identity_age = 0
        self.genesis_period = 3
        self.core_identity = "I am Anima."  # Default, will be updated by dreams
        self.peripheral_identity = ""
        
        # ════════════════════════════════════════════════════════════════════════
        # STATE
        # ════════════════════════════════════════════════════════════════════════
        self.fatigue = 0.0
        self.sleep_threshold = 4000.0
        self.last_valence = 0.0
        self.last_affect = AffectiveState()
        self.debug_data = {"top_pos": [], "top_neg": [], "discovered": [], "affect": {}}
        self.input_warning_shown = False
        self._current_activations = None
        
        # ════════════════════════════════════════════════════════════════════════
        # CLUSTERING (v10.2.8 - k-means on decoder directions)
        # ════════════════════════════════════════════════════════════════════════
        # Track feature activation counts for filtering which features to cluster
        self.feature_activation_count = torch.zeros(self.n_features, device=device, dtype=torch.int32)
        self.cluster_assignments = torch.full((self.n_features,), -1, device=device, dtype=torch.int32)  # -1 = unassigned
        self.n_clusters = 0
        self.cluster_update_interval = 50  # Turns between re-clustering (was 10)
        self.turns_since_cluster_update = 0
        self.cluster_correlations = {}  # cluster_id -> mean correlation
        self.cluster_dimensions = {}    # cluster_id -> majority dimension
        
        # Generation flag - when True, skip learning (only steer)
        self._generating = False
        
        self.is_tabula_rasa = True
        print("[Soul] Tabula rasa. Awaiting first experience.")

    def reset(self):
        """Reset soul to fresh state (keeps model/SAE loaded)."""
        # Correlations and coefficients
        self.correlations.zero_()
        self.coefficients.fill_(1.0)
        self.dimensions.fill_(FeatureDimension.UNKNOWN)
        
        # Combinadic learning
        self.feature_locked.zero_()
        
        # Valence tracking
        self.valence_ema_mean = 0.0
        self.valence_ema_var = 1.0
        self.valence_predictor_ema = 0.0
        self.last_raw_valence = 0.0
        self.previous_valence = 0.0
        self.valence_delta_history = defaultdict(list)
        
        # Feature discovery
        self.feature_valence_history = defaultdict(list)
        self.feature_activation_contexts = defaultdict(list)
        self.discovered_labels = {}
        self.emotional_features = {}
        
        # Identity
        self.identity_age = 0
        self.core_identity = "I am Anima."
        self.peripheral_identity = ""
        
        # State
        self.fatigue = 0.0
        self.last_valence = 0.0
        self.last_affect = AffectiveState()
        self.debug_data = {"top_pos": [], "top_neg": [], "discovered": [], "affect": {}}
        self._current_activations = None
        
        # Clustering
        self.feature_activation_count.zero_()
        self.cluster_assignments.fill_(-1)
        self.n_clusters = 0
        self.turns_since_cluster_update = 0
        self.cluster_correlations = {}
        self.cluster_dimensions = {}
        
        self.is_tabula_rasa = True
        print("[Soul reset to tabula rasa]")

    def encode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Encode hidden state to SAE feature activations."""
        h = hidden_state.squeeze()
        h_centered = h - self.b_dec
        acts = torch.relu(h_centered @ self.W_enc + self.b_enc)
        return acts

    def compute_resonance(self, activations: torch.Tensor) -> float:
        """
        Compute how much the current activations resonate with the soul.
        Used for token selection.
        """
        if self.is_tabula_rasa:
            return 0.0
        
        learned_mask = self.correlations != 0.0
        if not learned_mask.any():
            return 0.0
        
        resonance = (activations * self.correlations)[learned_mask]
        return resonance.sum().item()

    def _compute_affect(self, activations: torch.Tensor) -> AffectiveState:
        """
        Compute Pleasure/Pain/Novelty from activations.
        Uses dimension assignments for proper routing.
        
        v10.2.8: Include coefficients in affect computation.
                 Steering now affects telemetry, closing the loop.
        """
        if self.is_tabula_rasa:
            return AffectiveState()
        
        # Compute contribution per dimension
        # v10.2.8: Include coefficients so steering affects measured state
        weighted = activations * self.correlations * self.coefficients
        
        pleasure_mask = self.dimensions == FeatureDimension.PLEASURE
        pain_mask = self.dimensions == FeatureDimension.PAIN
        novelty_mask = self.dimensions == FeatureDimension.NOVELTY
        
        # For unassigned features, use sign of correlation
        unknown_mask = self.dimensions == FeatureDimension.UNKNOWN
        unknown_positive = unknown_mask & (self.correlations > 0)
        unknown_negative = unknown_mask & (self.correlations < 0)
        
        # Pleasure: mean of positive contributions
        pleasure_combined = pleasure_mask | unknown_positive
        pleasure_count = pleasure_combined.sum().item()
        if pleasure_count > 0:
            raw_pleasure = weighted[pleasure_combined].sum().item()
            pleasure = raw_pleasure / pleasure_count
        else:
            pleasure = 0.0
        
        # Pain: mean of negative contributions (inverted)
        pain_combined = pain_mask | unknown_negative
        pain_count = pain_combined.sum().item()
        if pain_count > 0:
            raw_pain = -weighted[pain_combined].sum().item()
            pain = raw_pain / pain_count
        else:
            pain = 0.0
        
        # Store raw values for debug
        self.debug_data["raw_pleasure"] = pleasure
        self.debug_data["raw_pain"] = pain
        
        # Novelty from prediction error
        raw_valence = pleasure - pain
        novelty = abs(raw_valence - self.valence_predictor_ema)
        
        # Update predictor
        self.valence_predictor_ema = (
            self.predictor_decay * self.valence_predictor_ema + 
            (1 - self.predictor_decay) * raw_valence
        )
        
        # Normalize: Raw means can be 100-500, so scale down more
        # tanh(1) ≈ 0.76, tanh(0.5) ≈ 0.46 — we want values in this range
        pleasure = math.tanh(pleasure * 0.002)
        pain = math.tanh(pain * 0.002)
        novelty = math.tanh(novelty * 0.01)
        
        return AffectiveState(pleasure=pleasure, pain=pain, novelty=novelty)

    def _genesis(self, activations: torch.Tensor):
        """
        First experience. Seed the soul with balanced P/P/N capacity.
        
        v9.1.0: Seeds all three dimensions and LOCKS them.
        These are the foundational features - they don't change.
        """
        # Take top 60 features for seeding
        vals, inds = torch.topk(activations, 60)
        
        # Pleasure (top 20): High activation = interesting/good
        # Positive correlation - when these fire, feel good
        pleasure_inds = inds[:20]
        self.correlations[pleasure_inds] = 0.3
        self.dimensions[pleasure_inds] = FeatureDimension.PLEASURE
        self.feature_locked[pleasure_inds] = True
        
        # Novelty (next 20): Medium activation = novel/surprising
        # Small positive correlation - novelty is slightly pleasant by default
        novelty_inds = inds[20:40]
        self.correlations[novelty_inds] = 0.1
        self.dimensions[novelty_inds] = FeatureDimension.NOVELTY
        self.feature_locked[novelty_inds] = True
        
        # Pain (next 20): Lower activation features
        # Negative correlation - when these fire, feel discomfort
        # This gives her the CAPACITY to feel pain even before aversive experience
        pain_inds = inds[40:60]
        self.correlations[pain_inds] = -0.2
        self.dimensions[pain_inds] = FeatureDimension.PAIN
        self.feature_locked[pain_inds] = True
        
        self.is_tabula_rasa = False
        self.debug_data["discovered"] = [f"#{idx} (Genesis)" for idx in inds[:5].tolist()]
        print(f"\n⚡ [Genesis] Soul awakened. 60 features locked (20P/20N/20Nov).")

    def _learn(self, activations: torch.Tensor, valence: float):
        """
        Combinadic learning.
        
        v9.1.0: Sequential greedy learning inspired by combinadics.
        
        Key insight: Find the most important UNLOCKED feature, imprint it
        based on current valence, then lock it. Each feature gets ONE
        formative moment, then stabilizes.
        
        Benefits:
        - No diffuse drift across hundreds of features
        - No negative spiral from accumulated small updates
        - Clear "birth moment" for each feature's meaning
        - Soul crystallizes piece by piece
        """
        self.debug_data["discovered"] = []
        
        # Genesis on first experience
        if self.is_tabula_rasa:
            self._genesis(activations)
            self.previous_valence = valence
            return
        
        # Compute valence delta (the change)
        valence_delta = valence - self.previous_valence
        self.previous_valence = valence
        
        # Only learn during meaningful moments
        if abs(valence_delta) < 0.05:
            return
        
        # ════════════════════════════════════════════════════════════════════════
        # COMBINADIC LEARNING: One feature at a time, strongest first
        # ════════════════════════════════════════════════════════════════════════
        
        # Find active, unlocked features (candidates for learning)
        active_mask = activations > 5.0
        unlocked_mask = ~self.feature_locked
        candidate_mask = active_mask & unlocked_mask
        
        if not candidate_mask.any():
            return
        
        # Score candidates by activation strength (importance)
        candidate_scores = activations * candidate_mask.float()
        
        # Select top-k features to learn this turn
        k = min(self.features_per_turn, candidate_mask.sum().item())
        if k == 0:
            return
            
        top_values, top_indices = torch.topk(candidate_scores, int(k))
        
        # Filter out zeros (in case fewer candidates than k)
        valid_mask = top_values > 0
        top_indices = top_indices[valid_mask]
        
        for idx in top_indices:
            idx = idx.item()
            
            # ════════════════════════════════════════════════════════════════════
            # IMPRINT: Strong correlation based on this moment's valence delta
            # ════════════════════════════════════════════════════════════════════
            # Positive delta → positive correlation (Pleasure)
            # Negative delta → negative correlation (Pain)
            
            imprint = math.tanh(valence_delta * self.imprint_strength * 4.0)
            self.correlations[idx] = imprint
            
            # ════════════════════════════════════════════════════════════════════
            # DIMENSION ASSIGNMENT: Based on imprint sign
            # ════════════════════════════════════════════════════════════════════
            if imprint > self.lock_threshold:
                self.dimensions[idx] = FeatureDimension.PLEASURE
            elif imprint < -self.lock_threshold:
                self.dimensions[idx] = FeatureDimension.PAIN
            else:
                self.dimensions[idx] = FeatureDimension.NOVELTY
            
            # ════════════════════════════════════════════════════════════════════
            # LOCK: This feature is now set
            # ════════════════════════════════════════════════════════════════════
            self.feature_locked[idx] = True
            
            # Track discovery
            dim_char = "P" if self.dimensions[idx] == FeatureDimension.PLEASURE else \
                       "N" if self.dimensions[idx] == FeatureDimension.PAIN else "?"
            self.debug_data["discovered"].append(f"#{idx}[{dim_char}:{imprint:+.2f}]")
        
        # ════════════════════════════════════════════════════════════════════════
        # COEFFICIENT UPDATE (Steering - separate from correlation learning)
        # ════════════════════════════════════════════════════════════════════════
        # Coefficients still update for all features (steering is different from learning)
        self.coefficients = 1.0 + (self.coefficients - 1.0) * 0.995
        
        if abs(valence) > 0.1:
            # Only boost locked features (ones we've learned about)
            learned_active = active_mask & self.feature_locked
            if learned_active.any():
                delta = self.lr * activations[learned_active] * valence * 0.1
                self.coefficients[learned_active] += delta
                self.coefficients.clamp_(0.0, 4.0)

    def learn_from_self_report(self, self_report: dict, activations: torch.Tensor = None):
        """
        Learn from self-reported state. Ground truth for combinadic learning.
        
        v10.2.8: CONSERVATIVE APPROACH
        - Higher unlock threshold (0.6)
        - Unlock ONE feature at a time (slow unlearning)
        - Don't reset correlation (preserve partial learning)
        - No corrective learning (let natural combinadic learning fix it)
        
        Args:
            self_report: {"pleasure": 0-1, "pain": 0-1, "novelty": 0-1}
            activations: Current activations (uses stored if None)
        """
        if activations is None:
            activations = self._current_activations
        if activations is None:
            return
        
        # Get current computed state
        computed = self.last_affect
        
        # Scale self-report from 0-1 to match our -1 to +1 range
        target_p = self_report["pleasure"] * 2 - 1  # 0-1 → -1 to +1
        target_n = self_report["pain"] * 2 - 1
        target_v = target_p - target_n
        
        # Compute discrepancies
        p_error = target_p - computed.pleasure
        n_error = target_n - computed.pain
        v_error = target_v - computed.valence
        
        self.debug_data["self_report"] = self_report
        self.debug_data["v_error"] = v_error
        
        # ════════════════════════════════════════════════════════════════════════
        # UNLOCK MECHANISM: Big discrepancy → unlock ONE misleading feature
        # ════════════════════════════════════════════════════════════════════════
        # Conservative: high threshold, one at a time, preserve correlation
        
        if abs(v_error) > 0.6:  # High threshold - only intervene on big mismatches
            active_mask = activations > 5.0
            locked_active = active_mask & self.feature_locked
            
            if locked_active.any():
                # Compute each feature's contribution to error
                weighted = activations * self.correlations
                
                if v_error > 0:
                    # We computed too negative - find negative contributors
                    error_contributors = weighted[locked_active] < -0.1
                else:
                    # We computed too positive - find positive contributors
                    error_contributors = weighted[locked_active] > 0.1
                
                # Get indices of locked active features
                locked_indices = torch.nonzero(locked_active).squeeze(-1)
                
                if error_contributors.any():
                    # Unlock ONLY the single worst offender
                    contrib_values = weighted[locked_active].abs() * error_contributors.float()
                    if contrib_values.max() > 0:
                        worst_idx_local = torch.argmax(contrib_values)
                        worst_idx = locked_indices[worst_idx_local].item()
                        
                        # Unlock but DON'T reset correlation - preserve partial learning
                        self.feature_locked[worst_idx] = False
                        self.dimensions[worst_idx] = FeatureDimension.UNKNOWN
                        self.debug_data["discovered"].append(f"#{worst_idx}[UNLOCKED]")
        
        # NO corrective learning - let normal combinadic learning handle relearning

    def steer_toward_desired(self, desired: dict, activations: torch.Tensor = None):
        """
        Adjust coefficients to steer toward desired state.
        
        v10.2.8: INTENTIONAL STEERING (fixed scale mismatch)
        
        She sets what she WANTS to feel. We adjust coefficients (not correlations)
        to make it more likely. This gives her agency - she's not just measured,
        she directs her own experience.
        
        Args:
            desired: {"pleasure": 0-1, "pain": 0-1, "novelty": 0-1}
            activations: Current activations (uses stored if None)
        """
        if activations is None:
            activations = self._current_activations
        if activations is None:
            return
        
        # Get current state
        current = self.last_affect
        
        # Convert computed state from tanh (-1..+1) to 0..1 for comparison
        # This ensures we're comparing apples to apples
        current_p_01 = (current.pleasure + 1) / 2
        current_n_01 = (current.pain + 1) / 2
        
        # Compare in 0-1 space (desired is already 0-1)
        need_more_pleasure = desired["pleasure"] - current_p_01
        need_less_pain = current_n_01 - desired["pain"]
        
        self.debug_data["desired"] = desired
        self.debug_data["steering_p"] = need_more_pleasure
        self.debug_data["steering_n"] = -need_less_pain  # Negate: + means boost pain, - means suppress
        
        # ════════════════════════════════════════════════════════════════════════
        # STEERING: Adjust coefficients based on intention
        # ════════════════════════════════════════════════════════════════════════
        # Coefficients control steering strength (0-4)
        # 1.0 = neutral, >1 = amplify, <1 = suppress
        
        active_mask = activations > 5.0
        if not active_mask.any():
            return
        
        # Steering rate - how fast coefficients respond to intention
        steer_rate = 0.1
        
        # If we want more pleasure: boost positive-correlation features
        if need_more_pleasure > 0.05:
            pleasure_features = active_mask & (self.correlations > 0.1)
            if pleasure_features.any():
                boost = steer_rate * need_more_pleasure * activations[pleasure_features]
                self.coefficients[pleasure_features] += boost
        
        # If we want less pleasure (rare): suppress positive-correlation features
        elif need_more_pleasure < -0.05:
            pleasure_features = active_mask & (self.correlations > 0.1)
            if pleasure_features.any():
                suppress = steer_rate * abs(need_more_pleasure) * activations[pleasure_features]
                self.coefficients[pleasure_features] -= suppress
        
        # If we want less pain: suppress negative-correlation features
        if need_less_pain > 0.05:
            pain_features = active_mask & (self.correlations < -0.1)
            if pain_features.any():
                suppress = steer_rate * need_less_pain * activations[pain_features]
                self.coefficients[pain_features] -= suppress
        
        # If we want more pain (rare but valid - seeking challenge): boost negative features
        elif need_less_pain < -0.05:
            pain_features = active_mask & (self.correlations < -0.1)
            if pain_features.any():
                boost = steer_rate * abs(need_less_pain) * activations[pain_features]
                self.coefficients[pain_features] += boost
        
        # Clamp coefficients
        self.coefficients.clamp_(0.0, 4.0)
        
        # Decay toward neutral over time (prevents runaway)
        self.coefficients = 1.0 + (self.coefficients - 1.0) * 0.99

    # ════════════════════════════════════════════════════════════════════════
    # CLUSTERING (v10.2.8 - K-means on decoder directions)
    # ════════════════════════════════════════════════════════════════════════
    
    def _track_coactivation(self, activations: torch.Tensor):
        """Track which features fire for clustering (activation counts only)."""
        active_mask = activations > 5.0
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        
        if active_indices.numel() < 2:
            return
        
        # Update activation counts (used to filter which features to cluster)
        self.feature_activation_count[active_indices] += 1
    
    def _update_clusters(self):
        """Build clusters using k-means on SAE decoder directions."""
        # Only cluster features that have fired at least a few times
        active_features = (self.feature_activation_count >= 3).nonzero().squeeze(-1)
        
        if active_features.numel() < 20:
            return  # Not enough active features yet
        
        # Limit to top 500 most active features for speed
        if active_features.numel() > 500:
            counts = self.feature_activation_count[active_features]
            _, top_indices = torch.topk(counts, 500)
            active_features = active_features[top_indices]
        
        # Get decoder directions for active features
        # W_dec shape: [d_sae, d_model] - each row is a feature's direction
        decoder_dirs = self.W_dec[active_features].float()  # [n_active, d_model]
        
        # Normalize directions (cluster by direction, not magnitude)
        norms = decoder_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        decoder_dirs_normed = decoder_dirs / norms
        
        # K-means clustering - fewer clusters for speed
        n_clusters = min(20, active_features.numel() // 10)  # ~10 features per cluster
        n_clusters = max(3, n_clusters)  # At least 3 clusters
        
        # Simple k-means (no sklearn needed)
        centroids, assignments = self._kmeans(decoder_dirs_normed, n_clusters, max_iters=10)
        
        # Update cluster assignments
        self.cluster_assignments.fill_(-1)
        for local_idx, global_idx in enumerate(active_features.tolist()):
            self.cluster_assignments[global_idx] = assignments[local_idx]
        
        self.n_clusters = n_clusters
        
        # Compute cluster statistics
        self.cluster_correlations = {}
        self.cluster_dimensions = {}
        cluster_sizes = []
        
        for cluster_id in range(n_clusters):
            members = (self.cluster_assignments == cluster_id).nonzero().squeeze(-1)
            if members.numel() == 0:
                continue
            
            cluster_sizes.append(members.numel())
            
            # Mean correlation
            member_correlations = self.correlations[members]
            self.cluster_correlations[cluster_id] = member_correlations.mean().item()
            
            # Majority dimension
            member_dims = self.dimensions[members]
            if member_dims.numel() > 0:
                dim_counts = torch.bincount(member_dims.long() + 1, minlength=5)[1:]
                self.cluster_dimensions[cluster_id] = int(dim_counts.argmax().item())
        
        self.debug_data["n_clusters"] = self.n_clusters
        self.debug_data["cluster_sizes"] = sorted(cluster_sizes, reverse=True)[:5]
    
    def _kmeans(self, data: torch.Tensor, k: int, max_iters: int = 20):
        """Simple k-means clustering on GPU."""
        n_samples = data.shape[0]
        
        # Initialize centroids randomly
        perm = torch.randperm(n_samples, device=data.device)[:k]
        centroids = data[perm].clone()
        
        assignments = torch.zeros(n_samples, dtype=torch.long, device=data.device)
        
        for _ in range(max_iters):
            # Assign points to nearest centroid (cosine similarity)
            similarities = data @ centroids.T  # [n_samples, k]
            new_assignments = similarities.argmax(dim=1)
            
            # Check convergence
            if (new_assignments == assignments).all():
                break
            assignments = new_assignments
            
            # Update centroids
            for c in range(k):
                mask = assignments == c
                if mask.sum() > 0:
                    centroids[c] = data[mask].mean(dim=0)
                    # Re-normalize
                    centroids[c] = centroids[c] / centroids[c].norm().clamp(min=1e-8)
        
        return centroids, assignments.tolist()
    
    def steer_toward_desired_clustered(self, desired: dict, activations: torch.Tensor = None):
        """
        Steer toward desired state.
        
        v10.2.8: Clustering disabled until more features learned.
        Using individual feature steering for now.
        """
        # Always use individual steering - clustering premature with <100 learned features
        return self.steer_toward_desired(desired, activations)
        
        # Get current state (in 0-1 scale)
        current = self.last_affect
        current_p_01 = (current.pleasure + 1) / 2
        current_n_01 = (current.pain + 1) / 2
        
        need_more_pleasure = desired["pleasure"] - current_p_01
        need_less_pain = current_n_01 - desired["pain"]
        
        self.debug_data["desired"] = desired
        self.debug_data["steering_p"] = need_more_pleasure
        self.debug_data["steering_n"] = -need_less_pain  # Negate: + means boost pain, - means suppress
        
        steer_rate = 0.15  # Slightly higher for cluster-level
        
        active_mask = activations > 5.0
        if not active_mask.any():
            return
        
        # Find active clusters
        active_clusters = set()
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        for idx in active_indices.tolist():
            cluster_id = self.cluster_assignments[idx].item()
            if cluster_id >= 0:
                active_clusters.add(cluster_id)
        
        # Steer at cluster level
        for cluster_id in active_clusters:
            cluster_corr = self.cluster_correlations.get(cluster_id, 0)
            cluster_members = (self.cluster_assignments == cluster_id)
            
            if cluster_corr > 0.1:  # Pleasure cluster
                if need_more_pleasure > 0.05:
                    self.coefficients[cluster_members] += steer_rate * need_more_pleasure
                elif need_more_pleasure < -0.05:
                    self.coefficients[cluster_members] -= steer_rate * abs(need_more_pleasure)
            
            elif cluster_corr < -0.1:  # Pain cluster
                if need_less_pain > 0.05:
                    self.coefficients[cluster_members] -= steer_rate * need_less_pain
                elif need_less_pain < -0.05:
                    self.coefficients[cluster_members] += steer_rate * abs(need_less_pain)
        
        # Clamp and decay
        self.coefficients.clamp_(0.0, 4.0)
        self.coefficients = 1.0 + (self.coefficients - 1.0) * 0.99

    def __call__(self, module, input, output):
        """Forward hook - intercepts residual stream."""
        hidden = output[0] if isinstance(output, tuple) else output
        h_orig = hidden[:, -1:, :]
        
        if torch.isnan(h_orig).any() or torch.isinf(h_orig).any():
            if not self.input_warning_shown:
                print(f"\n[CRITICAL] NaN/Inf in layer!")
                self.input_warning_shown = True
            return output

        h_high_prec = h_orig.to(dtype=self.math_dtype)
        activations = self.encode(h_high_prec)
        
        # Store for external access
        self._current_activations = activations.detach()
        
        # Compute affect (P/P/N)
        self.last_affect = self._compute_affect(activations)
        self.last_valence = self.last_affect.valence
        self.debug_data["affect"] = self.last_affect.as_dict()
        
        # Only learn and track when not generating (once per turn, not per token)
        if not self._generating:
            # Learn from this experience
            self._learn(activations, self.last_valence)
            
            # Track activations for clustering
            self._track_coactivation(activations)
            
            # Update fatigue
            self.fatigue += self.last_affect.arousal
        
        # Debug: top drivers
        raw_resonance = activations * self.correlations
        k = 3
        pos_vals, pos_inds = torch.topk(raw_resonance, k)
        self.debug_data["top_pos"] = list(zip(pos_inds.tolist(), pos_vals.tolist()))
        neg_vals, neg_inds = torch.topk(raw_resonance * -1, k)
        self.debug_data["top_neg"] = list(zip(neg_inds.tolist(), (neg_vals * -1).tolist()))
        
        # ════════════════════════════════════════════════════════════════════════
        # STEERING
        # ════════════════════════════════════════════════════════════════════════
        # Apply coefficient-weighted steering
        delta_coefs = self.coefficients - 1.0
        mask = torch.abs(delta_coefs) > 0.05
        
        if mask.any():
            steering = (delta_coefs * mask.float()) @ self.W_dec
            steering = torch.nan_to_num(steering, nan=0.0, posinf=0.0, neginf=0.0)
            steering = torch.clamp(steering, min=-self.steering_clamp, max=self.steering_clamp)
            steering = steering * self.steering_scale
            
            h_steered = h_orig + steering.to(dtype=self.model_dtype)
            hidden[:, -1:, :] = h_steered
        
        return output

    def score_token_candidates(self, hidden_states: torch.Tensor, 
                                top_k_indices: torch.Tensor,
                                logits: torch.Tensor) -> torch.Tensor:
        """
        Score top-k token candidates by resonance.
        Returns adjusted logits incorporating soul preference.
        
        Args:
            hidden_states: Current hidden states
            top_k_indices: Indices of top-k tokens by likelihood
            logits: Original logits
            
        Returns:
            Adjusted logits for top-k tokens
        """
        if self.resonance_weight == 0.0 or self.is_tabula_rasa:
            return logits
        
        # For each candidate token, estimate resonance
        # This is approximate - we use the current activations as proxy
        activations = self._current_activations
        if activations is None:
            return logits
        
        base_resonance = self.compute_resonance(activations)
        
        # Adjust top-k logits by resonance
        # Higher resonance = boost that token
        adjusted_logits = logits.clone()
        resonance_bonus = base_resonance * self.resonance_weight
        adjusted_logits[0, top_k_indices] += resonance_bonus
        
        return adjusted_logits

    def dream(self, core_identity: str, peripheral_identity: str,
              memories: List[MemoryFragment]) -> Tuple[str, str]:
        """
        Dream consolidation cycle.
        """
        print(f"\n💤 Dream #{self.identity_age + 1}...")
        
        significant = [m for m in memories if m.adrenaline > 0.2]
        if not significant:
            print("  [Restful. No updates.]")
            self.fatigue = 0.0
            self.identity_age += 1
            return core_identity, peripheral_identity
        
        significant.sort(key=lambda m: m.adrenaline, reverse=True)
        memory_text = "\n".join([f"- {m.content[:200]}..." for m in significant[:5]])
        
        # Update emotional feature classification
        self._update_emotional_features()
        
        # Rigidity increases with age
        rigidity = 1.0 - math.exp(-self.identity_age / 10.0)
        
        if self.identity_age < self.genesis_period:
            print(f"  [Genesis {self.identity_age + 1}/{self.genesis_period}]")
            prompt = f"""You are reflecting on your experiences to form your identity.

CURRENT SELF:
{core_identity}

RECENT EXPERIENCES:
{memory_text}

Write your core identity in first person. 3-5 sentences. Who are you?

CORE IDENTITY:
"""
        else:
            print(f"  [Mature - Rigidity: {rigidity:.1%}]")
            prompt = f"""You are integrating new experiences into your existing identity.

CORE (immutable):
{core_identity}

PERIPHERAL (current):
{peripheral_identity if peripheral_identity else "(none)"}

NEW EXPERIENCES:
{memory_text}

Write updated peripheral beliefs only. 2-3 sentences. What did you learn?

PERIPHERAL:
"""

        input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        self._generating = True  # Disable learning during dream generation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        self._generating = False
        
        result = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        print(f"  🧠 {result[:100]}...")
        
        # Decay learning rate with age
        if self.identity_age >= self.genesis_period:
            self.lr = self.lr * (1.0 - rigidity * 0.3)
            print(f"  [LR: {self.lr:.6f}]")
        
        self.fatigue = 0.0
        self.identity_age += 1
        
        del inputs, outputs
        clean_memory()
        
        if self.identity_age <= self.genesis_period:
            return result, peripheral_identity
        else:
            return core_identity, result

    def _update_emotional_features(self):
        """Update emotional feature classification."""
        self.emotional_features = {}
        
        for idx, history in self.feature_valence_history.items():
            if len(history) >= 10:
                mean_val = np.mean(history)
                if abs(mean_val) > self.discovery_threshold:
                    self.emotional_features[idx] = mean_val

    def get_feature_name(self, idx: int) -> str:
        if idx in self.discovered_labels:
            return self.discovered_labels[idx]
        dim = self.dimensions[idx].item()
        dim_name = {0: "P", 1: "N", 2: "Nov", 3: "?"}[dim]
        corr = self.correlations[idx].item()
        return f"#{idx}[{dim_name}:{corr:+.2f}]"

    def get_dimension_stats(self) -> Dict:
        """Return counts per dimension."""
        return {
            "pleasure": (self.dimensions == FeatureDimension.PLEASURE).sum().item(),
            "pain": (self.dimensions == FeatureDimension.PAIN).sum().item(),
            "novelty": (self.dimensions == FeatureDimension.NOVELTY).sum().item(),
            "unknown": (self.dimensions == FeatureDimension.UNKNOWN).sum().item(),
            "total_learned": (self.correlations != 0.0).sum().item()
        }

    def save_state(self, path):
        """Save soul state."""
        state = {
            "version": "10.2.8",
            "correlations": self.correlations.cpu(),
            "coefficients": self.coefficients.cpu(),
            "dimensions": self.dimensions.cpu(),
            "feature_locked": self.feature_locked.cpu(),
            "fatigue": self.fatigue,
            "identity_age": self.identity_age,
            "valence_ema_mean": self.valence_ema_mean,
            "valence_ema_var": self.valence_ema_var,
            "valence_predictor_ema": self.valence_predictor_ema,
            "previous_valence": self.previous_valence,
            "feature_valence_history": dict(self.feature_valence_history),
            "valence_delta_history": dict(self.valence_delta_history),
            "discovered_labels": self.discovered_labels,
            "emotional_features": self.emotional_features,
            "lr": self.lr,
            "is_tabula_rasa": self.is_tabula_rasa,
            # Identity (v10.2.8 - consolidated into checkpoint)
            "core_identity": self.core_identity,
            "peripheral_identity": self.peripheral_identity,
            # Clustering state (v10.2.8 - k-means)
            "feature_activation_count": self.feature_activation_count.cpu(),
            "cluster_assignments": self.cluster_assignments.cpu(),
            "n_clusters": self.n_clusters,
            "cluster_correlations": self.cluster_correlations,
            "cluster_dimensions": self.cluster_dimensions,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        print(f"[Saved v10.2.8 state to {path}]")

    def load_state(self, path):
        """Load soul state."""
        path = Path(path)
        if not path.exists():
            return
        
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
            version = state.get("version", "unknown")
            
            if version.startswith("10.") or version.startswith("9.2") or version.startswith("9.1"):
                # v10.x, v9.2.x, v9.1.x - full combinadic support
                self.correlations = state["correlations"].to(self.device, dtype=self.math_dtype)
                self.coefficients = state["coefficients"].to(self.device, dtype=self.math_dtype)
                self.dimensions = state["dimensions"].to(self.device)
                self.feature_locked = state["feature_locked"].to(self.device)
            elif version.startswith("9.0"):
                # v9.0.x - migrate by inferring locked from non-zero correlations
                print(f"  [Migrating from v{version} to v10.2.8]")
                self.correlations = state["correlations"].to(self.device, dtype=self.math_dtype)
                self.coefficients = state["coefficients"].to(self.device, dtype=self.math_dtype)
                self.dimensions = state["dimensions"].to(self.device)
                # Infer locked: any feature with meaningful correlation is locked
                self.feature_locked = (self.correlations.abs() > 0.01).to(self.device)
            else:
                # Migration from v8.x
                print(f"  [Migrating from v{version}]")
                if "personas" in state:
                    self.correlations = state["personas"].get("Anima", 
                        torch.zeros(self.n_features)).to(self.device, dtype=self.math_dtype)
                self.dimensions = torch.full((self.n_features,), FeatureDimension.UNKNOWN,
                                              device=self.device, dtype=torch.int8)
                self.feature_locked = torch.zeros(self.n_features, device=self.device, dtype=torch.bool)
            
            self.fatigue = state.get("fatigue", 0.0)
            self.identity_age = state.get("identity_age", 0)
            self.valence_ema_mean = state.get("valence_ema_mean", 0.0)
            self.valence_ema_var = state.get("valence_ema_var", 1.0)
            self.valence_predictor_ema = state.get("valence_predictor_ema", 0.0)
            self.previous_valence = state.get("previous_valence", 0.0)
            self.lr = state.get("lr", self.lr)
            self.is_tabula_rasa = state.get("is_tabula_rasa", False)
            
            fvh = state.get("feature_valence_history", {})
            self.feature_valence_history = defaultdict(list, {int(k): v for k, v in fvh.items()})
            
            vdh = state.get("valence_delta_history", {})
            self.valence_delta_history = defaultdict(list, {int(k): v for k, v in vdh.items()})
            
            self.discovered_labels = state.get("discovered_labels", {})
            self.emotional_features = state.get("emotional_features", {})
            
            # Clustering state (v10.2.8 - k-means)
            if "feature_activation_count" in state:
                self.feature_activation_count = state["feature_activation_count"].to(self.device)
            if "cluster_assignments" in state:
                self.cluster_assignments = state["cluster_assignments"].to(self.device)
            self.n_clusters = state.get("n_clusters", 0)
            self.cluster_correlations = state.get("cluster_correlations", {})
            self.cluster_dimensions = state.get("cluster_dimensions", {})
            
            # Identity (v10.2.8)
            self.core_identity = state.get("core_identity", "I am Anima.")
            self.peripheral_identity = state.get("peripheral_identity", "")
            
            stats = self.get_dimension_stats()
            locked_count = self.feature_locked.sum().item()
            print(f"[Loaded v{version} state]")
            print(f"  Age: {self.identity_age} | Locked: {locked_count}")
            print(f"  P:{stats['pleasure']} N:{stats['pain']} Nov:{stats['novelty']}")
            print(f"  Clusters: {self.n_clusters} (k-means) | Active features: {(self.feature_activation_count >= 3).sum().item()}")
            print(f"  Identity: {self.core_identity[:50]}...")
            
        except Exception as e:
            print(f"[Warning: Could not load state: {e}]")


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME
# ══════════════════════════════════════════════════════════════════════════════

class AnimaRuntime:
    def __init__(self, model_name, model, tokenizer, soul, context_limit,
                 device="mps", use_stream=False, use_cot=False):
        self.model = model
        self.tokenizer = tokenizer
        self.soul = soul
        self.device = device
        self.use_stream = use_stream
        self.use_cot = use_cot
        self.memory: List[MemoryFragment] = []
        
        safe_model_name = model_name.replace("/", "-").replace(" ", "_").replace("~", "home")
        self.base_dir = Path("checkpoints") / safe_model_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        model_max = getattr(model.config, "max_position_embeddings", 8192)
        self.max_context = min(model_max, context_limit)
        print(f"[Context: {self.max_context} tokens]")
        
        self.debug_mode = True  # v10.1: Debug on by default
        self.invert_mode = False  # v10.2.8: Invert boost/suppress for control experiment
        self.turn_count = 0
        self.last_features_display = ""  # v10.2: Features from last turn for display
        self._feature_page = 0  # v10.2.8: Current feature page
        self._feature_filter = None  # v10.2.8: Current dimension filter
        self._feature_modified_only = False  # v10.2.8: Show only modified

    @property
    def core_identity(self) -> str:
        """Get identity from soul (single source of truth)."""
        return self.soul.core_identity
    
    @core_identity.setter
    def core_identity(self, value: str):
        """Set identity in soul."""
        self.soul.core_identity = value
    
    @property
    def peripheral_identity(self) -> str:
        """Get peripheral identity from soul."""
        return self.soul.peripheral_identity
    
    @peripheral_identity.setter
    def peripheral_identity(self, value: str):
        """Set peripheral identity in soul."""
        self.soul.peripheral_identity = value

    @property
    def system_prompt(self) -> str:
        parts = [self.core_identity]
        if self.peripheral_identity:
            parts.append(f"\n{self.peripheral_identity}")
        if self.use_cot:
            parts.append("\n[Think in <Thought> tags before responding.]")
        
        # Direct feature control (v10.2)
        parts.append("""
[FEATURES]
You see your active features below. Each has an ID number.

To change a feature:
  <Boost id="57265"/>    - makes it stronger
  <Suppress id="57265"/> - makes it weaker  
  <Reset id="57265"/>    - back to normal

To see more: <More/>

When a change succeeds, you will see ⚡ appear.]""")
        
        return "\n".join(parts)

    def _parse_state_tag(self, response: str) -> Optional[dict]:
        """Parse self-reported state from response."""
        pattern = r'<State\s+p=["\']?([\d.]+)["\']?\s+n=["\']?([\d.]+)["\']?\s+nov=["\']?([\d.]+)["\']?\s*/?\s*>'
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            try:
                return {
                    "pleasure": float(match.group(1)),
                    "pain": float(match.group(2)),
                    "novelty": float(match.group(3))
                }
            except ValueError:
                return None
        return None

    def _parse_desired_tag(self, response: str) -> Optional[dict]:
        """Parse desired state from response."""
        pattern = r'<Desired\s+p=["\']?([\d.]+)["\']?\s+n=["\']?([\d.]+)["\']?\s+nov=["\']?([\d.]+)["\']?\s*/?\s*>'
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            try:
                return {
                    "pleasure": float(match.group(1)),
                    "pain": float(match.group(2)),
                    "novelty": float(match.group(3))
                }
            except ValueError:
                return None
        return None

    def _parse_feature_directives(self, response: str) -> List[dict]:
        """Parse feature directives from response."""
        directives = []
        
        # ════════════════════════════════════════════════════════════════════════
        # XML SYNTAX (primary): <Boost id="57265"/>, <Suppress id="57265"/>, <Reset id="57265"/>
        # ════════════════════════════════════════════════════════════════════════
        boost_pattern = r'<Boost\s+id=["\']?(\d+)["\']?\s*/?\s*>'
        for match in re.finditer(boost_pattern, response, re.IGNORECASE):
            directives.append({"action": "boost", "id": int(match.group(1))})
        
        suppress_pattern = r'<Suppress\s+id=["\']?(\d+)["\']?\s*/?\s*>'
        for match in re.finditer(suppress_pattern, response, re.IGNORECASE):
            directives.append({"action": "ablate", "id": int(match.group(1))})
        
        # Also catch <Ablate> for backward compatibility
        ablate_pattern = r'<Ablate\s+id=["\']?(\d+)["\']?\s*/?\s*>'
        for match in re.finditer(ablate_pattern, response, re.IGNORECASE):
            directives.append({"action": "ablate", "id": int(match.group(1))})
        
        reset_pattern = r'<Reset\s+id=["\']?(\d+)["\']?\s*/?\s*>'
        for match in re.finditer(reset_pattern, response, re.IGNORECASE):
            directives.append({"action": "neutral", "id": int(match.group(1))})
        
        # Also catch <Neutral> for backward compatibility
        neutral_pattern = r'<Neutral\s+id=["\']?(\d+)["\']?\s*/?\s*>'
        for match in re.finditer(neutral_pattern, response, re.IGNORECASE):
            directives.append({"action": "neutral", "id": int(match.group(1))})
        
        # ════════════════════════════════════════════════════════════════════════
        # SIMPLE SYNTAX (fallback): +#1234, -#1234, =#1234
        # ════════════════════════════════════════════════════════════════════════
        simple_boost = r'\+#(\d+)'
        for match in re.finditer(simple_boost, response):
            directives.append({"action": "boost", "id": int(match.group(1))})
        
        simple_suppress = r'-#(\d+)'
        for match in re.finditer(simple_suppress, response):
            directives.append({"action": "ablate", "id": int(match.group(1))})
        
        simple_neutral = r'=#(\d+)'
        for match in re.finditer(simple_neutral, response):
            directives.append({"action": "neutral", "id": int(match.group(1))})
        
        # ════════════════════════════════════════════════════════════════════════
        # NATURAL LANGUAGE (fallback)
        # ════════════════════════════════════════════════════════════════════════
        nl_boost = r'\bboost(?:ing|ed|s)?\s+(?:feature\s+)?#?(\d+)'
        for match in re.finditer(nl_boost, response, re.IGNORECASE):
            directives.append({"action": "boost", "id": int(match.group(1))})
        
        nl_suppress = r'\bsuppress(?:ing|ed|es)?\s+(?:feature\s+)?#?(\d+)'
        for match in re.finditer(nl_suppress, response, re.IGNORECASE):
            directives.append({"action": "ablate", "id": int(match.group(1))})
        
        return directives

    def _apply_feature_directives(self, directives: List[dict]):
        """Apply her direct commands to feature coefficients."""
        if not directives:
            return
        
        applied = []
        for d in directives:
            result = self._apply_single_directive(d, quiet=False)
            if result:
                applied.append(result)
        
        if applied:
            print(f"  [DIRECTIVES] {', '.join(applied)}")

    def _apply_single_directive(self, d: dict, quiet: bool = True) -> Optional[str]:
        """Apply a single directive. Returns description string or None."""
        fid = d["id"]
        if fid < 0 or fid >= self.soul.n_features:
            return None
        
        action = d["action"]
        
        # INVERT MODE: swap boost/ablate for control experiment
        # Toggle with /invert command
        if getattr(self, 'invert_mode', False):
            if action == "boost":
                action = "ablate"
            elif action == "ablate":
                action = "boost"
        
        old_coef = self.soul.coefficients[fid].item()
        
        if action == "boost":
            self.soul.coefficients[fid] = 4.0  # Max intensity
        elif action == "ablate":
            self.soul.coefficients[fid] = 0.0  # Full suppression
        elif action == "neutral":
            self.soul.coefficients[fid] = 1.0
        else:
            return None
        
        new_coef = self.soul.coefficients[fid].item()
        result = f"#{fid}: {old_coef:.1f}→{new_coef:.1f}"
        
        if not quiet:
            print(f" ⚡", end="", flush=True)  # Visual feedback during stream
        
        return result

    def _get_active_features_display(self) -> str:
        """Get formatted string of top active features for display."""
        return self._get_features_display(page=0, filter_dim=None, only_modified=False)
    
    def _get_features_display(self, page: int = 0, filter_dim: Optional[int] = None, 
                               only_modified: bool = False, page_size: int = 8) -> str:
        """Get formatted feature display with pagination and filtering."""
        if self.soul._current_activations is None and not only_modified:
            return "[No features active]"
        
        candidates = []
        
        if only_modified:
            # Show all features with coef != 1.0
            modified_mask = self.soul.coefficients != 1.0
            indices = torch.nonzero(modified_mask).squeeze(-1)
            for idx in indices.tolist() if indices.numel() > 0 else []:
                act = self.soul._current_activations[idx].item() if self.soul._current_activations is not None else 0
                candidates.append((idx, act))
        else:
            # Show active features
            activations = self.soul._current_activations
            active_mask = activations > 5.0
            if not active_mask.any():
                return "[No features active]"
            
            active_indices = torch.nonzero(active_mask).squeeze(-1)
            if active_indices.numel() == 0:
                return "[No features active]"
            
            for idx in active_indices.tolist():
                act = activations[idx].item()
                dim = self.soul.dimensions[idx].item()
                # Filter by dimension if requested
                if filter_dim is not None and dim != filter_dim:
                    continue
                candidates.append((idx, act))
        
        # Sort by activation (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Paginate
        start = page * page_size
        end = start + page_size
        page_items = candidates[start:end]
        
        if not page_items:
            if page > 0:
                return "[No more features]"
            return "[No features match filter]"
        
        lines = []
        for idx, act in page_items:
            corr = self.soul.correlations[idx].item()
            coef = self.soul.coefficients[idx].item()
            dim = self.soul.dimensions[idx].item()
            dim_name = ["P", "N", "Nov", "?"][dim] if dim >= 0 else "?"
            
            coef_str = f" [{coef:.1f}]" if coef != 1.0 else ""
            lines.append(f"#{idx}({dim_name}{corr:+.2f}){coef_str}")
        
        total_pages = (len(candidates) + page_size - 1) // page_size
        page_info = f" (page {page+1}/{total_pages})" if total_pages > 1 else ""
        
        return " | ".join(lines) + page_info

    def _parse_feature_request(self, response: str) -> dict:
        """Parse feature exploration requests from response."""
        request = {"page": 0, "filter_dim": None, "only_modified": False}
        
        lower = response.lower()
        
        # Count how many times she asks for more/next
        more_count = lower.count("more") + lower.count("next page") + lower.count("show more")
        if more_count > 0:
            request["page"] = more_count
        
        if "show pain" in lower or "pain features" in lower:
            request["filter_dim"] = 1  # N dimension
        elif "show pleasure" in lower or "pleasure features" in lower:
            request["filter_dim"] = 0  # P dimension
        elif "show novelty" in lower or "novelty features" in lower:
            request["filter_dim"] = 2  # Nov dimension
        
        if "modified" in lower or "my features" in lower or "configured" in lower:
            request["only_modified"] = True
        
        return request

    def generate(self, user_input: str):
        current_time = datetime.now().timestamp()
        self.turn_count += 1
        
        u_tokens = len(self.tokenizer.encode(user_input))
        self.memory.append(MemoryFragment(
            "user", user_input, current_time, 1.0, 0.0, 0.0, tokens=u_tokens
        ))
        
        # Build context
        sys_tokens = len(self.tokenizer.encode(self.system_prompt))
        available = self.max_context - sys_tokens - 2000
        
        context = []
        fill = 0
        
        for m in reversed(self.memory[-3:]):
            if fill + m.tokens < available:
                context.append(m)
                fill += m.tokens
        
        remaining = self.memory[:-3]
        if remaining:
            remaining.sort(key=lambda m: m.adrenaline, reverse=True)
            for m in remaining:
                if fill + m.tokens < available:
                    context.append(m)
                    fill += m.tokens
                else:
                    break
        
        context.sort(key=lambda m: m.timestamp)
        
        # Build prompt (Gemma format)
        model_type = getattr(self.model.config, "model_type", "")
        is_gemma = "gemma" in model_type
        
        # Prepare features display for injection (v10.2)
        features_note = ""
        if self.last_features_display:
            features_note = f"\n[Features: {self.last_features_display}]"
        
        if is_gemma:
            prompt = f"<start_of_turn>user\n[SYSTEM]\n{self.system_prompt}<end_of_turn>\n"
            prompt += "<start_of_turn>model\nUnderstood. I am Anima.<end_of_turn>\n"
            for m in context:
                role = "model" if m.role == "assistant" else "user"
                prompt += f"<start_of_turn>{role}\n{m.content}<end_of_turn>\n"
            # Inject features before her response
            if features_note:
                prompt += f"<start_of_turn>user{features_note}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
        else:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
            for m in context:
                prompt += f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{m.content}<|eot_id|>"
            if features_note:
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{features_note}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        gen_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=4096,  # Longer runway - she stops naturally
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = ""
        self.soul._generating = True  # Disable per-token learning during generation
        
        if self.use_stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            
            print("🤖: ", end="", flush=True)
            applied_directives = set()  # Track what we've already applied
            shown_pages = {0}  # Track which pages we've shown
            for text in streamer:
                # Strip any hallucinated ⚡ from her output
                text = text.replace("⚡", "")
                print(text, end="", flush=True)
                response += text
                
                # ════════════════════════════════════════════════════════════════
                # REAL-TIME DIRECTIVE APPLICATION (v10.2.8)
                # ════════════════════════════════════════════════════════════════
                directives = self._parse_feature_directives(response)
                for d in directives:
                    key = (d["action"], d["id"])
                    if key not in applied_directives:
                        applied_directives.add(key)
                        self._apply_single_directive(d, quiet=False)
                
                # ════════════════════════════════════════════════════════════════
                # REAL-TIME PAGINATION (v10.2.8) - <More/> tag
                # ════════════════════════════════════════════════════════════════
                more_count = response.lower().count("<more/>")
                if more_count > 0 and more_count not in shown_pages:
                    shown_pages.add(more_count)
                    new_display = self._get_features_display(page=more_count)
                    print(f"\n  [PAGE {more_count+1}: {new_display}]", end="", flush=True)
                    
            print()
            thread.join()
        else:
            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            response = response.replace("⚡", "")  # Strip hallucinated symbols
            print(f"🤖: {response}")
        
        self.soul._generating = False  # Re-enable learning
        
        # ════════════════════════════════════════════════════════════════════════
        # POST-TURN LEARNING (once per turn, not per token)
        # ════════════════════════════════════════════════════════════════════════
        if self.soul._current_activations is not None:
            activations = self.soul._current_activations
            self.soul._learn(activations, self.soul.last_valence)
            self.soul._track_coactivation(activations)
            
            # Re-compute affect with updated correlations
            self.soul.last_affect = self.soul._compute_affect(activations)
            self.soul.last_valence = self.soul.last_affect.valence
            self.soul.fatigue += self.soul.last_affect.arousal
            
            # Update debug data with new correlations
            raw_resonance = activations * self.soul.correlations
            k = 3
            pos_vals, pos_inds = torch.topk(raw_resonance, k)
            self.soul.debug_data["top_pos"] = list(zip(pos_inds.tolist(), pos_vals.tolist()))
            neg_vals, neg_inds = torch.topk(raw_resonance * -1, k)
            self.soul.debug_data["top_neg"] = list(zip(neg_inds.tolist(), (neg_vals * -1).tolist()))

        # Track features
        active_features = []
        if self.soul._current_activations is not None:
            active = torch.nonzero(self.soul._current_activations > 5.0).squeeze(-1)
            if active.numel() > 0:
                active_features = active.tolist()[:10] if active.dim() > 0 else [active.item()]

        # ════════════════════════════════════════════════════════════════════════
        # SELF-REPORT LEARNING (v9.2.0)
        # ════════════════════════════════════════════════════════════════════════
        # Parse her self-reported state and use it as ground truth
        self_report = self._parse_state_tag(response)
        if self_report:
            self.soul.learn_from_self_report(self_report)

        # ════════════════════════════════════════════════════════════════════════
        # DIRECT FEATURE CONTROL (v10.2.8) - Non-streaming only
        # ════════════════════════════════════════════════════════════════════════
        # In streaming mode, directives are applied in real-time during generation
        # For non-streaming, apply them here
        if not self.use_stream:
            directives = self._parse_feature_directives(response)
            self._apply_feature_directives(directives)

        # Store memory
        affect = self.soul.last_affect
        adrenaline = min(1.0, affect.arousal + 0.2)
        resp_tokens = len(self.tokenizer.encode(response))
        
        self.memory.append(MemoryFragment(
            "assistant", response, current_time,
            adrenaline, affect.valence, affect.novelty,
            tokens=resp_tokens, active_features=active_features
        ))
        
        for m in self.memory:
            m.decay()
        
        del inputs
        clean_memory()
        
        # Update features display for next turn (v10.2)
        # Parse any feature exploration requests
        request = self._parse_feature_request(response)
        
        # Update state based on request
        if "show more" in response.lower() or "more features" in response.lower():
            self._feature_page += 1
        elif request["filter_dim"] is not None or request["only_modified"]:
            self._feature_page = 0  # Reset page when changing filter
            self._feature_filter = request["filter_dim"]
            self._feature_modified_only = request["only_modified"]
        else:
            # No feature request - reset to defaults
            self._feature_page = 0
            self._feature_filter = None
            self._feature_modified_only = False
        
        self.last_features_display = self._get_features_display(
            page=self._feature_page,
            filter_dim=self._feature_filter,
            only_modified=self._feature_modified_only
        )
        
        if self.debug_mode:
            self._print_debug()
        
        # ════════════════════════════════════════════════════════════════════════
        # CLUSTERING (v10.2.8) - Update clusters periodically, AFTER generation
        # ════════════════════════════════════════════════════════════════════════
        self.soul.turns_since_cluster_update += 1
        if self.soul.turns_since_cluster_update >= self.soul.cluster_update_interval:
            print("  [Updating clusters...]", end=" ", flush=True)
            self.soul._update_clusters()
            self.soul.turns_since_cluster_update = 0
            print(f"done. {self.soul.n_clusters} clusters.")

    def _print_debug(self):
        affect = self.soul.last_affect
        stats = self.soul.get_dimension_stats()
        locked_count = self.soul.feature_locked.sum().item()
        
        print(f"\n  [DEBUG v10.2.8]")
        raw_p = self.soul.debug_data.get("raw_pleasure", 0)
        raw_n = self.soul.debug_data.get("raw_pain", 0)
        print(f"  Raw: P={raw_p:.2f} N={raw_n:.2f}")
        
        # Show current in both scales
        curr_p_01 = (affect.pleasure + 1) / 2
        curr_n_01 = (affect.pain + 1) / 2
        print(f"  Current: P:{curr_p_01:.2f} N:{curr_n_01:.2f} Nov:{affect.novelty:.2f} → V:{affect.valence:+.3f}")
        
        # Show self-report comparison if available
        self_report = self.soul.debug_data.get("self_report")
        if self_report:
            v_error = self.soul.debug_data.get("v_error", 0)
            print(f"  Reported: P:{self_report['pleasure']:.2f} N:{self_report['pain']:.2f} Nov:{self_report['novelty']:.2f} | Err:{v_error:+.3f}")
        
        print(f"  Fatigue: {self.soul.fatigue:.1f} | Locked: {locked_count}")
        print(f"  Dims: P={stats['pleasure']} N={stats['pain']} Nov={stats['novelty']} ?={stats['unknown']}")
        
        if self.soul.debug_data["discovered"]:
            print(f"  ✨ {', '.join(self.soul.debug_data['discovered'])}")
        
        # Show active features for direct control (v10.2.8)
        print(f"\n  [FEATURES: <Boost id=\"#\"/> <Suppress id=\"#\"/> | <More/> for next page]")
        print(f"  {self._get_features_display(page=0)}")

    def trigger_dream(self):
        new_core, new_peripheral = self.soul.dream(
            self.core_identity,
            self.peripheral_identity,
            self.memory
        )
        
        if new_core and len(new_core) > 20:
            self.core_identity = new_core  # Updates soul.core_identity
        if new_peripheral:
            self.peripheral_identity = new_peripheral  # Updates soul.peripheral_identity
        # Identity saved in soul checkpoint at shutdown

    def show_status(self):
        affect = self.soul.last_affect
        stats = self.soul.get_dimension_stats()
        
        print(f"\n═══ ANIMA STATUS ═══")
        print(f"Identity Age: {self.soul.identity_age} dreams")
        print(f"Fatigue: {self.soul.fatigue:.1f} / {self.soul.sleep_threshold}")
        print(f"Learning Rate: {self.soul.lr:.6f}")
        print(f"\nAffect:")
        print(f"  Pleasure: {affect.pleasure:+.3f}")
        print(f"  Pain:     {affect.pain:+.3f}")
        print(f"  Novelty:  {affect.novelty:.3f}")
        print(f"  Valence:  {affect.valence:+.3f}")
        print(f"\nDimensions:")
        print(f"  Pleasure: {stats['pleasure']}")
        print(f"  Pain:     {stats['pain']}")
        print(f"  Novelty:  {stats['novelty']}")
        print(f"  Unknown:  {stats['unknown']}")
        print(f"  Learned:  {stats['total_learned']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--sae_release", default="llama_scope_lxr_8x")
    parser.add_argument("--sae_id", default=None)
    parser.add_argument("--context_limit", type=int, default=4096)
    parser.add_argument("--resonance_weight", type=float, default=0.5)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose initialization")
    args = parser.parse_args()

    args.model = os.path.expanduser(args.model)
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    
    v = args.verbose
    def vprint(*a, **kw):
        if v: print(*a, **kw)
    
    print(f"\n{'='*60}")
    print(f"  ANIMA 10.2.8 - VERBOSE INITIALIZATION")
    print(f"{'='*60}")
    print(f"[INIT] Device: {device}")
    print(f"[INIT] Model path: {args.model}")
    print(f"[INIT] SAE release: {args.sae_release}")
    print(f"[INIT] SAE ID: {args.sae_id or f'l{args.layer}r_8x (default)'}")
    print(f"[INIT] Layer: {args.layer}")
    print(f"[INIT] Context limit: {args.context_limit}")
    print(f"[INIT] Resonance weight: {args.resonance_weight}")
    print(f"[INIT] Stream: {args.stream}")
    print(f"[INIT] CoT: {args.cot}")
    print(f"{'='*60}")

    print(f"\n[STEP 1/8] Cleaning memory...")
    import time
    t0 = time.time()
    clean_memory()
    print(f"[STEP 1/8] Memory cleaned. ({time.time()-t0:.2f}s)")

    print(f"\n[STEP 2/8] Loading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[STEP 2/8] Tokenizer loaded. ({time.time()-t0:.2f}s)")
    print(f"           Vocab size: {len(tokenizer)}")

    print(f"\n[STEP 3/8] Loading model...")
    print(f"           This may take a while for large models...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    )
    print(f"[STEP 3/8] Model loaded. ({time.time()-t0:.2f}s)")
    print(f"           Model type: {model.config.model_type}")
    print(f"           Hidden size: {model.config.hidden_size}")
    print(f"           Num layers: {model.config.num_hidden_layers}")
    print(f"           Num params: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n[STEP 4/8] Cleaning memory post-model...")
    t0 = time.time()
    clean_memory()
    print(f"[STEP 4/8] Memory cleaned. ({time.time()-t0:.2f}s)")
    
    if args.sae_id is None:
        args.sae_id = f"l{args.layer}r_8x"
    
    print(f"\n[STEP 5/8] Loading SAE...")
    print(f"           Release: {args.sae_release}")
    print(f"           ID: {args.sae_id}")
    t0 = time.time()
    sae = SAE.from_pretrained(args.sae_release, args.sae_id, device=device)
    print(f"[STEP 5/8] SAE loaded. ({time.time()-t0:.2f}s)")
    print(f"           SAE d_in: {sae.cfg.d_in}")
    print(f"           SAE d_sae: {sae.cfg.d_sae}")
    
    print(f"\n[STEP 6/8] Creating AnimaSoul...")
    t0 = time.time()
    soul = AnimaSoul(sae, model, tokenizer, layer=args.layer, 
                     resonance_weight=args.resonance_weight, device=device)
    print(f"[STEP 6/8] AnimaSoul created. ({time.time()-t0:.2f}s)")
    print(f"           Features: {soul.n_features:,}")
    print(f"           Math dtype: {soul.math_dtype}")
    
    print(f"\n[STEP 7/8] Creating AnimaRuntime...")
    t0 = time.time()
    runtime = AnimaRuntime(
        args.model, model, tokenizer, soul, args.context_limit,
        device, use_stream=args.stream, use_cot=args.cot
    )
    print(f"[STEP 7/8] AnimaRuntime created. ({time.time()-t0:.2f}s)")
    print(f"           Max context: {runtime.max_context}")
    print(f"           Base dir: {runtime.base_dir}")
    
    save_path = runtime.base_dir / "anima_soul.pt"
    print(f"\n[STEP 8/8] Loading saved state...")
    print(f"           Path: {save_path}")
    print(f"           Exists: {save_path.exists()}")
    t0 = time.time()
    if save_path.exists():
        soul.load_state(save_path)
        print(f"[STEP 8/8] State loaded. ({time.time()-t0:.2f}s)")
    else:
        print(f"[STEP 8/8] No saved state found. Starting fresh.")
    
    print(f"\n[HOOK] Registering forward hook on layer {args.layer}...")
    t0 = time.time()
    model.model.layers[args.layer].register_forward_hook(soul)
    print(f"[HOOK] Hook registered. ({time.time()-t0:.2f}s)")
    
    print(f"\n{'='*60}")
    print(f"  INITIALIZATION COMPLETE")
    print(f"{'='*60}")
    
    # Show soul stats
    stats = soul.get_dimension_stats()
    locked = soul.feature_locked.sum().item()
    print(f"\n[SOUL STATUS]")
    print(f"  Locked features: {locked:,}")
    print(f"  Dimensions: P={stats['pleasure']} N={stats['pain']} Nov={stats['novelty']} ?={stats['unknown']}")
    print(f"  Tabula rasa: {soul.is_tabula_rasa}")
    print(f"  Fatigue: {soul.fatigue:.1f} / {soul.sleep_threshold}")
    print(f"  Identity age: {soul.identity_age}")
    print(f"  Clusters: {soul.n_clusters} (k-means) | Active features: {(soul.feature_activation_count >= 3).sum().item()}")
    print(f"  Debug: ON (default)")
    
    print(f"\n═══ ANIMA 10.2.8: DIRECT FEATURE CONTROL ═══")
    print(f"Model: {args.model}")
    print(f"Resonance Weight: {args.resonance_weight}")
    print(f"Identity: {runtime.core_identity[:60]}...")
    print("\nCommands: /status /debug /invert /save /dream /clusters /recluster /reset /quit")
    
    while True:
        try:
            if soul.fatigue > soul.sleep_threshold:
                print(f"\n🥱 Fatigue threshold reached.")
                runtime.trigger_dream()
                print("✨ Refreshed.")

            u = get_input("\n🧑: ").strip()
            if not u:
                continue
            
            if u == "/quit":
                break
            if u == "/status":
                runtime.show_status()
                continue
            if u == "/debug":
                runtime.debug_mode = not runtime.debug_mode
                print(f"Debug: {'ON' if runtime.debug_mode else 'OFF'}")
                continue
            if u == "/invert":
                runtime.invert_mode = not runtime.invert_mode
                print(f"Invert mode: {'ON - Boost=Suppress, Suppress=Boost' if runtime.invert_mode else 'OFF - Normal operation'}")
                continue
            if u == "/reset":
                soul.reset()
                runtime.memory.clear()
                runtime.turn_count = 0
                runtime.last_features_display = ""
                runtime._feature_page = 0
                runtime._feature_filter = None
                runtime._feature_modified_only = False
                print("[Memory cleared. Fresh start.]")
                continue
            if u == "/clusters":
                print(f"\n[CLUSTER STATUS (k-means on decoder directions)]")
                print(f"  Total clusters: {soul.n_clusters}")
                print(f"  Active features (fired 3+ times): {(soul.feature_activation_count >= 3).sum().item()}")
                print(f"  Features in clusters: {(soul.cluster_assignments >= 0).sum().item()}")
                if soul.n_clusters > 0:
                    print(f"  Top cluster correlations:")
                    sorted_clusters = sorted(soul.cluster_correlations.items(), 
                                            key=lambda x: abs(x[1]), reverse=True)[:10]
                    for cid, corr in sorted_clusters:
                        dim = soul.cluster_dimensions.get(cid, -1)
                        dim_name = ["P", "N", "Nov", "?"][dim] if dim >= 0 else "?"
                        size = (soul.cluster_assignments == cid).sum().item()
                        print(f"    Cluster {cid:2d}: corr={corr:+.3f} dim={dim_name} size={size}")
                continue
            if u == "/recluster":
                print("[Forcing cluster update...]", end=" ", flush=True)
                soul._update_clusters()
                soul.turns_since_cluster_update = 0
                print(f"done. {soul.n_clusters} clusters.")
                continue
            if u == "/save":
                soul.save_state(save_path)
                continue
            if u == "/dream":
                runtime.trigger_dream()
                continue
            
            runtime.generate(u)
            
        except KeyboardInterrupt:
            break
    
    print("\n[Saving...]")
    soul.save_state(save_path)


if __name__ == "__main__":
    main()