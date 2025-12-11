#!/usr/bin/env python3
"""
anima.py - Anima 12.0.0: Autonomous Neural Steering

Architecture:
- SPARSE WALKER: N probes that walk around 131K feature space
- MULTI-SIGNAL REWARD: Valence + user sentiment + prediction accuracy
- COMBINADIC LEARNING: Features learn correlations (one-shot imprinting)
- PROPRIOCEPTION: Internal state sensing via learned P/N directions
- DEEP FEEDBACK: L40 → L8 recurrence for temporal coherence
- NO MANUAL CONTROL: Network learns to steer autonomously

v12.0.0:
- Removed manual shift system (ShiftA/ShiftB, hierarchy control)
- SparseFeatureWalker: 64 probes with migrating candidate pools
- RewardComputer: Multi-signal reward (valence, sentiment, prediction)
- Cleaner codebase focused on learned steering

Usage:
    python anima_v12.py --model "~/models/gemma-2-27b-it" --context_limit 4096
"""

import os
import sys
import math
import json
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import threading
import gc
import select
import signal
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ══════════════════════════════════════════════════════════════════════════════
# LLMRI - INTEGRATED NEURAL IMAGING
# ══════════════════════════════════════════════════════════════════════════════

class LLMRI:
    """
    Integrated MRI scanner - captures all layer activations during generation.
    Renders as images for visualization.
    Optionally streams to connected clients via MRIServer.
    """
    
    def __init__(self, model, device: str = "mps"):
        self.model = model
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size
        
        # Storage
        self.layer_activations: Dict[int, torch.Tensor] = {}
        self.hooks = []
        self.capture_enabled = False
        
        # Streaming server (set externally)
        self.server = None
        
        # Compute image dimensions
        self.img_width, self.img_height = self._compute_dimensions(self.hidden_dim)
        
        # Output directory
        self.output_dir = "llmri_captures"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Neuron tracking for persistent activation analysis
        self.neuron_history: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.history_limit = 100  # Keep last N activations per layer
        self.track_neurons = False
        
        # Ignored neurons (structural features to exclude from viz)
        self.ignored_neurons: set = set()
        self.auto_ignore_invariants = False
        
        # Neuron interventions (ablate/boost during forward pass)
        self.neuron_interventions: Dict[int, float] = {}  # neuron_idx -> multiplier (0=ablate, 2=boost)
        self.neuron_offsets: Dict[int, float] = {}  # neuron_idx -> additive offset
        self.intervention_layer: int = None  # Which layer to intervene on (None = all)
        
        # Protected neurons (infrastructure - never steer these)
        # Neuron 4140 confirmed as critical via ablation study: 0.98x-1.05x tolerance
        self.protected_neurons: set = {4140}
        self.protection_enabled = True
        
        # Typical value range tracking
        self.value_stats: Dict[int, Dict] = {}  # neuron_idx -> {min, max, mean, std}
    
    def register_server_commands(self):
        """Register command handler with MRI server for bidirectional communication."""
        if self.server is None:
            return
        
        def handle_command(cmd_type: str, data: Dict):
            if cmd_type == "intervention":
                self._handle_remote_intervention(data)
            elif cmd_type == "clear_interventions":
                self.clear_interventions()
                print("[LLMRI] Remote: cleared all interventions")
        
        self.server.on_command = handle_command
        print("[LLMRI] Registered for remote commands")
    
    def _handle_remote_intervention(self, data: Dict):
        """Process intervention command from remote viewer."""
        interventions = data.get("interventions", {})
        layer = data.get("layer")
        
        print(f"[LLMRI] _handle_remote_intervention called!")
        print(f"  Layer: {layer}")
        print(f"  Interventions: {len(interventions)} entries")
        
        if layer is not None:
            self.intervention_layer = layer
        
        # Clear existing and apply new
        self.neuron_interventions.clear()
        
        skipped = 0
        for neuron_str, multiplier in interventions.items():
            neuron_idx = int(neuron_str)
            
            # Skip protected neurons
            if self.protection_enabled and neuron_idx in self.protected_neurons:
                skipped += 1
                continue
            
            self.neuron_interventions[neuron_idx] = float(multiplier)
        
        if self.neuron_interventions:
            print(f"[LLMRI] Remote intervention: {len(self.neuron_interventions)} neurons at layer {layer}")
            # Show first 5 interventions as sample
            sample = list(self.neuron_interventions.items())[:5]
            for idx, mult in sample:
                print(f"  Neuron {idx}: {mult:.2f}x")
            if len(self.neuron_interventions) > 5:
                print(f"  ... and {len(self.neuron_interventions) - 5} more")
            
            if skipped:
                print(f"  (skipped {skipped} protected neurons)")
            
            # Reset debug flag so we log next application
            if hasattr(self, '_intervention_logged'):
                del self._intervention_logged
            
            # Broadcast confirmation
            if self.server:
                self.server.broadcast_event("intervention_active", {
                    "count": len(self.neuron_interventions),
                    "layer": layer,
                    "skipped": skipped
                })
        else:
            print("[LLMRI] Remote: interventions cleared")
            if self.server:
                self.server.broadcast_event("interventions_cleared", {})
    
    def _compute_dimensions(self, hidden_dim: int) -> Tuple[int, int]:
        """Compute optimal 2D dimensions."""
        sqrt = int(math.sqrt(hidden_dim))
        for w in range(sqrt, 0, -1):
            if hidden_dim % w == 0:
                return (w, hidden_dim // w)
        side = int(math.ceil(math.sqrt(hidden_dim)))
        return (side, side)
    
    def _make_capture_hook(self, layer_idx: int):
        """Create hook to capture layer activations and optionally intervene."""
        def hook(module, input, output):
            if not self.capture_enabled and not self.neuron_interventions and not self.neuron_offsets:
                return
            
            hidden = output[0] if isinstance(output, tuple) else output
            modified = False
            
            # Apply neuron interventions (multiplicative)
            if self.neuron_interventions and (self.intervention_layer is None or self.intervention_layer == layer_idx):
                original_dtype = hidden.dtype
                
                # Debug: log first application
                if not hasattr(self, '_intervention_logged'):
                    self._intervention_logged = True
                    print(f"[LLMRI] Applying {len(self.neuron_interventions)} interventions at layer {layer_idx}")
                    print(f"  Hidden shape: {hidden.shape}, dtype: {hidden.dtype}")
                
                applied_count = 0
                for neuron_idx, multiplier in self.neuron_interventions.items():
                    if neuron_idx < hidden.shape[-1]:
                        # Convert to float32 for safe arithmetic
                        before = hidden[0, -1, neuron_idx].float().item()
                        value = hidden[0, -1, neuron_idx].float()
                        value = value * multiplier
                        
                        # Clamp to safe range
                        value = torch.clamp(value, -1e4, 1e4)
                        
                        # Convert back to original dtype
                        hidden[0, -1, neuron_idx] = value.to(original_dtype)
                        after = hidden[0, -1, neuron_idx].float().item()
                        
                        # Log first few applications
                        if applied_count < 3:
                            print(f"  Neuron {neuron_idx}: {before:.4f} × {multiplier:.2f} = {after:.4f}")
                        
                        applied_count += 1
                        modified = True
                
                if applied_count > 3:
                    print(f"  ... and {applied_count - 3} more neurons modified")
            
            # Apply neuron offsets (additive)
            if self.neuron_offsets and (self.intervention_layer is None or self.intervention_layer == layer_idx):
                original_dtype = hidden.dtype
                
                for neuron_idx, offset in self.neuron_offsets.items():
                    if neuron_idx < hidden.shape[-1]:
                        value = hidden[0, -1, neuron_idx].float()
                        value = value + offset
                        value = torch.clamp(value, -1e4, 1e4)
                        hidden[0, -1, neuron_idx] = value.to(original_dtype)
                        modified = True
            
            if modified:
                # Return modified output
                if isinstance(output, tuple):
                    output = (hidden,) + output[1:]
                else:
                    output = hidden
            
            if not self.capture_enabled:
                return output if modified else None
            
            # Capture last token - convert to float32 for numpy compatibility
            acts = hidden[0, -1, :].detach().float().cpu()
            self.layer_activations[layer_idx] = acts
            
            # Track neuron history for persistent activation analysis
            if self.track_neurons:
                self.neuron_history[layer_idx].append(acts.clone())
                if len(self.neuron_history[layer_idx]) > self.history_limit:
                    self.neuron_history[layer_idx].pop(0)
            
            # Stream to server if connected
            if self.server is not None and self.server.client_count > 0:
                self.server.broadcast(layer_idx, acts.numpy())
            
            return output if self.neuron_interventions else None
        return hook
    
    def ablate_neuron(self, neuron_idx: int, layer: int = None):
        """Set a neuron's activation to zero during forward pass."""
        if self.is_protected(neuron_idx):
            print(f"⚠️  WARNING: Neuron {neuron_idx} is PROTECTED (critical infrastructure)")
            print(f"    This may break generation. Use /unprotect {neuron_idx} to override.")
            return
        self.neuron_interventions[neuron_idx] = 0.0
        self.intervention_layer = layer
        print(f"[LLMRI] Ablating neuron {neuron_idx}" + (f" at layer {layer}" if layer else " (all layers)"))
    
    def boost_neuron(self, neuron_idx: int, multiplier: float = 2.0, layer: int = None):
        """Multiply a neuron's activation during forward pass."""
        if self.is_protected(neuron_idx):
            # Allow small adjustments to protected neurons with warning
            if multiplier < 0.95 or multiplier > 1.05:
                print(f"⚠️  WARNING: Neuron {neuron_idx} is PROTECTED (tolerance: 0.95x-1.05x)")
                print(f"    Multiplier {multiplier}x is outside safe range. This may break generation.")
                print(f"    Use /unprotect {neuron_idx} to override protection.")
                return
            else:
                print(f"[LLMRI] Protected neuron {neuron_idx} - using safe range ({multiplier}x)")
        self.neuron_interventions[neuron_idx] = multiplier
        self.intervention_layer = layer
        print(f"[LLMRI] Boosting neuron {neuron_idx} by {multiplier}x" + (f" at layer {layer}" if layer else " (all layers)"))
    
    def clear_interventions(self):
        """Remove all neuron interventions."""
        self.neuron_interventions.clear()
        self.neuron_offsets.clear()
        self.intervention_layer = None
        print("[LLMRI] Cleared all interventions")
    
    def nudge_neuron(self, neuron_idx: int, offset: float, layer: int = None):
        """Add a constant offset to a neuron's activation (additive, not multiplicative)."""
        if self.is_protected(neuron_idx):
            print(f"⚠️  WARNING: Neuron {neuron_idx} is PROTECTED")
            return
        self.neuron_offsets[neuron_idx] = offset
        self.intervention_layer = layer
        print(f"[LLMRI] Nudging neuron {neuron_idx} by {offset:+.2f}" + (f" at layer {layer}" if layer else " (all layers)"))
    
    def inspect_neuron(self, neuron_idx: int, layer_idx: int = 22):
        """Show current value and statistics for a neuron."""
        if layer_idx not in self.layer_activations:
            print(f"[LLMRI] No data for layer {layer_idx}")
            return
        
        acts = self.layer_activations[layer_idx]
        if neuron_idx >= len(acts):
            print(f"[LLMRI] Neuron {neuron_idx} out of range")
            return
        
        value = acts[neuron_idx].item()
        
        # Get stats from history if available
        if self.track_neurons and layer_idx in self.neuron_history and len(self.neuron_history[layer_idx]) > 1:
            history = torch.stack(self.neuron_history[layer_idx])
            neuron_history = history[:, neuron_idx]
            
            print(f"\n[LLMRI Neuron {neuron_idx} @ Layer {layer_idx}]")
            print(f"  Current value: {value:.4f}")
            print(f"  History ({len(neuron_history)} samples):")
            print(f"    Min:  {neuron_history.min():.4f}")
            print(f"    Max:  {neuron_history.max():.4f}")
            print(f"    Mean: {neuron_history.mean():.4f}")
            print(f"    Std:  {neuron_history.std():.4f}")
            print(f"  Suggested interventions:")
            std = neuron_history.std().item()
            print(f"    Nudge ±1 std: /nudge {neuron_idx} {std:.2f}")
            print(f"    Nudge ±2 std: /nudge {neuron_idx} {2*std:.2f}")
        else:
            print(f"\n[LLMRI Neuron {neuron_idx} @ Layer {layer_idx}]")
            print(f"  Current value: {value:.4f}")
            print(f"  (Enable /track for historical stats)")
    
    def is_protected(self, neuron_idx: int) -> bool:
        """Check if a neuron is protected infrastructure."""
        return self.protection_enabled and neuron_idx in self.protected_neurons
    
    def protect_neuron(self, neuron_idx: int, reason: str = ""):
        """Mark a neuron as protected (infrastructure, don't steer)."""
        self.protected_neurons.add(neuron_idx)
        print(f"[LLMRI] Protected neuron {neuron_idx}" + (f": {reason}" if reason else ""))
    
    def unprotect_neuron(self, neuron_idx: int):
        """Remove protection from a neuron."""
        self.protected_neurons.discard(neuron_idx)
        print(f"[LLMRI] Unprotected neuron {neuron_idx}")
    
    def list_protected(self):
        """List all protected neurons."""
        print(f"\n[LLMRI Protected Neurons]")
        print(f"  Protection: {'ENABLED' if self.protection_enabled else 'DISABLED'}")
        print(f"  Count: {len(self.protected_neurons)}")
        for idx in sorted(self.protected_neurons):
            x, y = self.get_neuron_position(idx)
            print(f"    Neuron {idx} ({x},{y})")
    
    def detect_critical_neurons(self, threshold: float = 0.9):
        """
        Auto-detect potentially critical neurons based on invariance.
        Neurons that are always in top-N across all layers are likely infrastructure.
        
        Returns list of candidates - use ablation testing to confirm.
        """
        if not self.layer_activations:
            print("[LLMRI] No data - run a capture first")
            return []
        
        from collections import Counter
        
        # Find neurons that are consistently high across layers
        top10_counts = Counter()
        for layer_idx, acts in self.layer_activations.items():
            top10 = torch.topk(acts, 10).indices.tolist()
            for idx in top10:
                top10_counts[idx] += 1
        
        n_layers = len(self.layer_activations)
        candidates = []
        
        print(f"\n[LLMRI Critical Neuron Detection]")
        print(f"  Analyzed {n_layers} layers")
        print(f"  Threshold: top-10 in {threshold*100:.0f}%+ of layers")
        print()
        
        for neuron_idx, count in top10_counts.most_common(20):
            ratio = count / n_layers
            if ratio >= threshold:
                x, y = self.get_neuron_position(neuron_idx)
                candidates.append(neuron_idx)
                print(f"  ⚠️  Neuron {neuron_idx} ({x},{y}): top-10 in {count}/{n_layers} layers ({ratio*100:.0f}%)")
                print(f"      → Likely infrastructure. Test with /boost {neuron_idx} 0.9")
        
        if not candidates:
            print("  No obvious candidates found")
        else:
            print(f"\n  Found {len(candidates)} candidates. Confirm via ablation testing.")
            print("  Safe tolerance is typically 0.95x - 1.05x for critical neurons.")
        
        return candidates
    
    # ==================== SPECTRAL ANALYSIS ====================
    
    def collect_temporal_series(self, layer_idx: int = 22) -> np.ndarray:
        """Get activation time series from neuron history.
        
        Returns: [n_samples, hidden_dim] array
        """
        if layer_idx not in self.neuron_history or len(self.neuron_history[layer_idx]) < 2:
            print(f"[LLMRI] Need more samples. Enable /track and chat.")
            return None
        
        series = torch.stack(self.neuron_history[layer_idx]).numpy()
        print(f"[LLMRI] Temporal series: {series.shape[0]} samples x {series.shape[1]} neurons")
        return series
    
    def spectral_analysis(self, layer_idx: int = 22, n_swaths: int = 8, method: str = "variance"):
        """
        Divide neurons into swaths and analyze frequency content.
        
        Args:
            layer_idx: Which layer to analyze
            n_swaths: Number of neuron groups
            method: How to group neurons:
                - "position": by index (0-575, 576-1151, etc.)
                - "variance": by activation variance (most variable to least)
                - "magnitude": by mean activation magnitude
                - "pca": by principal components
        """
        series = self.collect_temporal_series(layer_idx)
        if series is None:
            return None
        
        n_samples, hidden_dim = series.shape
        
        if n_samples < 8:
            print(f"[LLMRI] Need at least 8 samples for spectral analysis, have {n_samples}")
            return None
        
        # Sort neurons into swaths
        if method == "position":
            # Simple positional grouping
            swath_size = hidden_dim // n_swaths
            swath_indices = [list(range(i * swath_size, (i+1) * swath_size)) for i in range(n_swaths)]
        
        elif method == "variance":
            # Group by activation variance
            variances = np.var(series, axis=0)
            sorted_idx = np.argsort(variances)[::-1]  # High variance first
            swath_size = hidden_dim // n_swaths
            swath_indices = [sorted_idx[i * swath_size:(i+1) * swath_size].tolist() for i in range(n_swaths)]
        
        elif method == "magnitude":
            # Group by mean magnitude
            magnitudes = np.mean(np.abs(series), axis=0)
            sorted_idx = np.argsort(magnitudes)[::-1]  # High magnitude first
            swath_size = hidden_dim // n_swaths
            swath_indices = [sorted_idx[i * swath_size:(i+1) * swath_size].tolist() for i in range(n_swaths)]
        
        elif method == "pca":
            # Group by PCA loadings
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_swaths, n_samples - 1))
            pca.fit(series.T)  # Fit on neurons x samples
            # Assign each neuron to its dominant component
            loadings = np.abs(pca.components_.T)  # [neurons, components]
            assignments = np.argmax(loadings, axis=1)
            swath_indices = [np.where(assignments == i)[0].tolist() for i in range(n_swaths)]
        else:
            print(f"Unknown method: {method}")
            return None
        
        # Compute FFT on each swath's mean activation over time
        print(f"\n[LLMRI Spectral Analysis - Layer {layer_idx}]")
        print(f"  Method: {method}, Swaths: {n_swaths}, Samples: {n_samples}")
        print(f"  Nyquist: {n_samples // 2} frequency bins")
        print()
        
        results = {}
        for i, indices in enumerate(swath_indices):
            if not indices:
                continue
            
            # Mean activation of this swath over time
            swath_signal = np.mean(series[:, indices], axis=1)
            
            # Remove DC component (mean)
            swath_signal = swath_signal - np.mean(swath_signal)
            
            # FFT
            fft = np.fft.rfft(swath_signal)
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(n_samples)
            
            # Find dominant frequencies (excluding DC)
            power_no_dc = power[1:]  # Skip DC
            freqs_no_dc = freqs[1:]
            
            if len(power_no_dc) > 0:
                top3_idx = np.argsort(power_no_dc)[-3:][::-1]
                
                swath_name = f"Swath {i}"
                if method == "variance":
                    swath_name = f"High-var" if i == 0 else f"Low-var" if i == n_swaths-1 else f"Mid-var-{i}"
                elif method == "magnitude":
                    swath_name = f"High-mag" if i == 0 else f"Low-mag" if i == n_swaths-1 else f"Mid-mag-{i}"
                
                print(f"  {swath_name} ({len(indices)} neurons):")
                print(f"    Signal std: {np.std(swath_signal):.4f}")
                print(f"    Dominant freqs: ", end="")
                for idx in top3_idx:
                    print(f"{freqs_no_dc[idx]:.3f} ({power_no_dc[idx]:.1f}), ", end="")
                print()
                
                results[i] = {
                    'indices': indices,
                    'signal': swath_signal,
                    'power': power,
                    'freqs': freqs,
                    'dominant_freq': freqs_no_dc[top3_idx[0]] if len(top3_idx) > 0 else 0
                }
        
        # Cross-swath coherence
        print(f"\n  Cross-swath phase relationships:")
        for i in range(min(3, len(results))):
            for j in range(i+1, min(3, len(results))):
                if i in results and j in results:
                    # Cross-correlation
                    sig_i = results[i]['signal']
                    sig_j = results[j]['signal']
                    corr = np.correlate(sig_i, sig_j, mode='full')
                    lag = np.argmax(np.abs(corr)) - len(sig_i) + 1
                    max_corr = np.max(np.abs(corr)) / (np.std(sig_i) * np.std(sig_j) * len(sig_i))
                    print(f"    Swath {i} <-> {j}: corr={max_corr:.3f}, lag={lag}")
        
        return results
    
    def render_spectrogram(self, layer_idx: int = 22, method: str = "variance", n_swaths: int = 8):
        """Render spectral analysis as image."""
        from PIL import Image
        
        results = self.spectral_analysis(layer_idx, n_swaths, method)
        if not results:
            return None
        
        # Create spectrogram image
        max_freq_bins = max(len(r['power']) for r in results.values())
        img_height = len(results) * 20  # 20 pixels per swath
        img_width = max_freq_bins * 4  # 4 pixels per frequency bin
        
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        for i, (swath_idx, data) in enumerate(sorted(results.items())):
            power = data['power']
            # Normalize power for this swath
            if power.max() > 0:
                normalized = power / power.max()
            else:
                normalized = power
            
            # Draw row
            y_start = i * 20
            for f_idx, p in enumerate(normalized):
                x_start = f_idx * 4
                intensity = int(p * 255)
                # Hot colormap
                r = min(255, intensity * 2)
                g = max(0, intensity - 128) * 2
                b = max(0, intensity - 200) * 3
                img[y_start:y_start+18, x_start:x_start+3] = [r, g, b]
        
        return Image.fromarray(img, mode='RGB')
    
    def register_hooks(self):
        """Register capture hooks on all layers."""
        self.clear_hooks()
        for i in range(self.num_layers):
            hook = self.model.model.layers[i].register_forward_hook(
                self._make_capture_hook(i)
            )
            self.hooks.append(hook)
        print(f"[LLMRI] Hooks registered on {self.num_layers} layers")
    
    def clear_hooks(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_history(self):
        """Clear neuron tracking history."""
        self.neuron_history.clear()
    
    def find_hotspots(self, layer_idx: int, top_k: int = 10) -> List[Tuple[int, float, float]]:
        """
        Find persistently high neurons in a layer.
        Returns: [(neuron_idx, mean_activation, std), ...]
        """
        if layer_idx not in self.neuron_history or len(self.neuron_history[layer_idx]) < 2:
            return []
        
        # Stack history into tensor [num_samples, hidden_dim]
        history = torch.stack(self.neuron_history[layer_idx])
        
        # Compute mean activation per neuron
        means = history.mean(dim=0)
        stds = history.std(dim=0)
        
        # Get top-k by mean activation
        top_vals, top_indices = torch.topk(means, top_k)
        
        results = []
        for i in range(top_k):
            idx = top_indices[i].item()
            results.append((idx, top_vals[i].item(), stds[idx].item()))
        
        return results
    
    def find_consistent_neurons(self, layer_idx: int, top_k: int = 10) -> List[Tuple[int, float, float]]:
        """
        Find neurons that are consistently high (high mean, low variance).
        Returns: [(neuron_idx, mean, consistency_score), ...]
        """
        if layer_idx not in self.neuron_history or len(self.neuron_history[layer_idx]) < 2:
            return []
        
        history = torch.stack(self.neuron_history[layer_idx])
        means = history.mean(dim=0)
        stds = history.std(dim=0)
        
        # Consistency = mean / (std + epsilon) - high mean, low variance
        consistency = means / (stds + 0.01)
        
        top_vals, top_indices = torch.topk(consistency, top_k)
        
        results = []
        for i in range(top_k):
            idx = top_indices[i].item()
            results.append((idx, means[idx].item(), top_vals[i].item()))
        
        return results
    
    def get_neuron_position(self, neuron_idx: int) -> Tuple[int, int]:
        """Convert neuron index to (x, y) position in image."""
        return (neuron_idx % self.img_width, neuron_idx // self.img_width)
    
    def analyze_bright_spot(self, layer_idx: int = None):
        """Analyze what's causing bright spots in the visualization."""
        if layer_idx is None:
            # Analyze the steering layer
            layer_idx = 22
        
        print(f"\n[LLMRI Analysis - Layer {layer_idx}]")
        
        if layer_idx not in self.layer_activations:
            print("  No data captured for this layer")
            return
        
        acts = self.layer_activations[layer_idx]
        
        # Current frame analysis
        top_val, top_idx = acts.max(), acts.argmax().item()
        x, y = self.get_neuron_position(top_idx)
        print(f"\n  Current brightest neuron:")
        print(f"    Index: {top_idx}")
        print(f"    Position: ({x}, {y})")
        print(f"    Value: {top_val:.4f}")
        
        # Top 5 current
        top5_vals, top5_idx = torch.topk(acts, 5)
        print(f"\n  Top 5 active neurons:")
        for i in range(5):
            idx = top5_idx[i].item()
            x, y = self.get_neuron_position(idx)
            print(f"    #{i+1}: neuron {idx} ({x},{y}) = {top5_vals[i]:.4f}")
        
        # Historical analysis if available
        if self.track_neurons and layer_idx in self.neuron_history:
            n_samples = len(self.neuron_history[layer_idx])
            print(f"\n  Historical analysis ({n_samples} samples):")
            
            hotspots = self.find_hotspots(layer_idx, 5)
            print(f"\n  Persistent hotspots (high mean):")
            for idx, mean, std in hotspots:
                x, y = self.get_neuron_position(idx)
                print(f"    neuron {idx} ({x},{y}): mean={mean:.4f}, std={std:.4f}")
            
            consistent = self.find_consistent_neurons(layer_idx, 5)
            print(f"\n  Consistent neurons (high mean/low var):")
            for idx, mean, score in consistent:
                x, y = self.get_neuron_position(idx)
                print(f"    neuron {idx} ({x},{y}): mean={mean:.4f}, score={score:.2f}")
    
    def analyze_cross_layer(self):
        """Find neurons that are consistently bright across ALL layers."""
        print(f"\n[LLMRI Cross-Layer Analysis]")
        
        if not self.layer_activations:
            print("  No data captured")
            return
        
        # Collect top neuron from each layer
        top_per_layer = {}
        for layer_idx, acts in self.layer_activations.items():
            top_idx = acts.argmax().item()
            top_val = acts[top_idx].item()
            top_per_layer[layer_idx] = (top_idx, top_val)
        
        # Count which neurons appear as top across layers
        from collections import Counter
        top_counts = Counter(idx for idx, val in top_per_layer.values())
        
        print(f"\n  Most common 'brightest' neurons across {len(top_per_layer)} layers:")
        for neuron_idx, count in top_counts.most_common(5):
            x, y = self.get_neuron_position(neuron_idx)
            pct = count / len(top_per_layer) * 100
            print(f"    Neuron {neuron_idx} ({x},{y}): brightest in {count} layers ({pct:.0f}%)")
        
        # Check if same neuron is ALWAYS brightest
        if top_counts.most_common(1)[0][1] == len(top_per_layer):
            invariant_idx = top_counts.most_common(1)[0][0]
            x, y = self.get_neuron_position(invariant_idx)
            print(f"\n  ⚠️  INVARIANT NEURON DETECTED!")
            print(f"    Neuron {invariant_idx} ({x},{y}) is ALWAYS brightest")
            print(f"    This is likely a structural feature (bias, norm, position encoding)")
            
            # Get its values across layers
            vals = [self.layer_activations[l][invariant_idx].item() 
                   for l in sorted(self.layer_activations.keys())]
            print(f"    Values: min={min(vals):.2f}, max={max(vals):.2f}, mean={sum(vals)/len(vals):.2f}")
        
        # Also show neurons that are in top-5 across many layers
        print(f"\n  Neurons frequently in top-5 across layers:")
        top5_counts = Counter()
        for layer_idx, acts in self.layer_activations.items():
            top5 = torch.topk(acts, 5).indices.tolist()
            for idx in top5:
                top5_counts[idx] += 1
        
        for neuron_idx, count in top5_counts.most_common(10):
            if count >= len(top_per_layer) * 0.5:  # In top-5 for >50% of layers
                x, y = self.get_neuron_position(neuron_idx)
                print(f"    Neuron {neuron_idx} ({x},{y}): top-5 in {count} layers")
        
        # Return invariants for potential ignoring
        invariants = [idx for idx, count in top_counts.most_common(5) 
                     if count >= len(top_per_layer) * 0.8]
        return invariants
    
    def ignore_invariants(self, threshold: float = 0.8):
        """Auto-detect and ignore invariant neurons."""
        if not self.layer_activations:
            print("  No data - run a capture first")
            return
        
        from collections import Counter
        
        # Find neurons that are brightest in >threshold of layers
        top_per_layer = {}
        for layer_idx, acts in self.layer_activations.items():
            top_idx = acts.argmax().item()
            top_per_layer[layer_idx] = top_idx
        
        top_counts = Counter(top_per_layer.values())
        n_layers = len(top_per_layer)
        
        invariants = [idx for idx, count in top_counts.items() 
                     if count >= n_layers * threshold]
        
        if invariants:
            self.ignored_neurons.update(invariants)
            print(f"[LLMRI] Auto-ignored {len(invariants)} invariant neurons: {invariants}")
            print(f"  Total ignored: {len(self.ignored_neurons)}")
        else:
            print("[LLMRI] No invariant neurons found")
    
    def clear_ignored(self):
        """Clear the ignored neurons list."""
        self.ignored_neurons.clear()
        print("[LLMRI] Cleared ignored neurons")
    
    def start_capture(self):
        """Enable capture for next forward pass."""
        self.layer_activations = {}
        self.capture_enabled = True
    
    def stop_capture(self):
        """Disable capture."""
        self.capture_enabled = False
    
    def render_layer(self, acts: torch.Tensor, colormap: str = "viridis") -> 'Image':
        """Render single layer to image."""
        from PIL import Image
        import colorsys
        
        total_pixels = self.img_width * self.img_height
        if len(acts) < total_pixels:
            acts = torch.cat([acts, torch.zeros(total_pixels - len(acts))])
        elif len(acts) > total_pixels:
            acts = acts[:total_pixels]
        
        # Ensure float32 for numpy compatibility
        acts_2d = acts.float().reshape(self.img_height, self.img_width).numpy()
        
        # Mask out ignored neurons (set to mean so they don't affect normalization)
        if self.ignored_neurons:
            mean_val = np.mean([acts_2d[idx // self.img_width, idx % self.img_width] 
                              for idx in range(len(acts)) if idx not in self.ignored_neurons])
            for idx in self.ignored_neurons:
                if idx < len(acts):
                    y, x = idx // self.img_width, idx % self.img_width
                    if y < self.img_height and x < self.img_width:
                        acts_2d[y, x] = mean_val
        
        # Normalize to [0, 1]
        min_val, max_val = acts_2d.min(), acts_2d.max()
        if max_val - min_val > 1e-8:
            normalized = (acts_2d - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(acts_2d)
        
        # Apply colormap
        rgb = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        if colormap == "grayscale":
            gray = (normalized * 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif colormap == "signed":
            # For diffs: blue=neg, white=zero, red=pos
            # Re-normalize with sign
            abs_max = max(abs(acts_2d.min()), abs(acts_2d.max()))
            if abs_max > 1e-8:
                signed = acts_2d / abs_max  # [-1, 1]
            else:
                signed = np.zeros_like(acts_2d)
            
            for i in range(self.img_height):
                for j in range(self.img_width):
                    v = signed[i, j]
                    if v < 0:  # Blue
                        rgb[i, j] = [int((1+v)*255), int((1+v)*255), 255]
                    else:  # Red
                        rgb[i, j] = [255, int((1-v)*255), int((1-v)*255)]
        elif colormap == "hsv":
            for i in range(self.img_height):
                for j in range(self.img_width):
                    h = normalized[i, j] * 0.8
                    s = 0.9
                    v = 0.5 + normalized[i, j] * 0.5
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    rgb[i, j] = [int(r*255), int(g*255), int(b*255)]
        else:  # viridis
            for i in range(self.img_height):
                for j in range(self.img_width):
                    t = normalized[i, j]
                    r = max(0, min(1, 0.27 + t * 0.7))
                    g = max(0, min(1, 0.0 + t * 0.9))
                    b = max(0, min(1, 0.33 + t * 0.3 - t * t * 0.3))
                    rgb[i, j] = [int(r*255), int(g*255), int(b*255)]
        
        return Image.fromarray(rgb, mode='RGB')
    
    def render_grid(self, colormap: str = "viridis", scale: int = 6) -> str:
        """Render all captured layers as grid."""
        from PIL import Image
        
        if not self.layer_activations:
            print("[LLMRI] No activations captured")
            return None
        
        cols = 8
        rows = math.ceil(self.num_layers / cols)
        pad = 2
        
        grid_w = cols * self.img_width + (cols - 1) * pad
        grid_h = rows * self.img_height + (rows - 1) * pad
        
        grid = Image.new('RGB', (grid_w, grid_h), color=(20, 20, 20))
        
        for layer_idx in sorted(self.layer_activations.keys()):
            acts = self.layer_activations[layer_idx]
            img = self.render_layer(acts, colormap)
            
            row, col = divmod(layer_idx, cols)
            x = col * (self.img_width + pad)
            y = row * (self.img_height + pad)
            grid.paste(img, (x, y))
        
        # Scale up
        grid = grid.resize((grid_w * scale, grid_h * scale), Image.Resampling.NEAREST)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"scan_{timestamp}.png")
        grid.save(path)
        
        return path
    
    def render_animation(self, colormap: str = "viridis", fps: int = 5) -> str:
        """Create animated GIF through layers."""
        from PIL import Image
        
        if not self.layer_activations:
            return None
        
        frames = []
        scale = 10
        
        for layer_idx in sorted(self.layer_activations.keys()):
            acts = self.layer_activations[layer_idx]
            img = self.render_layer(acts, colormap)
            img = img.resize((self.img_width * scale, self.img_height * scale), 
                           Image.Resampling.NEAREST)
            frames.append(img)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"scan_{timestamp}.gif")
        
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0
        )
        
        return path
    
    def get_stats(self) -> Dict:
        """Get stats from captured activations."""
        if not self.layer_activations:
            return {}
        
        stats = {}
        for layer_idx, acts in sorted(self.layer_activations.items()):
            stats[layer_idx] = {
                'min': acts.min().item(),
                'max': acts.max().item(),
                'mean': acts.mean().item(),
                'std': acts.std().item(),
            }
        return stats


# ══════════════════════════════════════════════════════════════════════════════
# SPARSE FEATURE WALKER
# ══════════════════════════════════════════════════════════════════════════════

class SparseFeatureWalker(nn.Module):
    """
    N learnable probes that 'walk' around the 131K feature space.
    Each probe:
    - Has a candidate pool of features it can attend to
    - Learns which candidates matter via soft selection
    - Contributes a steering direction weighted by activation
    - Candidates migrate toward active features over time
    """
    
    def __init__(self, n_features: int, n_probes: int, d_model: int, 
                 n_candidates: int = 512, device: str = "mps"):
        super().__init__()
        self.n_features = n_features
        self.n_probes = n_probes
        self.d_model = d_model
        self.n_candidates = n_candidates
        self.device = device
        
        # Candidate pools: which features each probe can "see"
        # These SHIFT over time based on activation patterns
        self.register_buffer('probe_candidates', 
                            torch.randint(0, n_features, (n_probes, n_candidates)))
        
        # Selection weights within candidate pool (learnable)
        self.selection_logits = nn.Parameter(torch.zeros(n_probes, n_candidates))
        
        # Per-probe steering direction (learnable)
        self.steering_dirs = nn.Parameter(torch.randn(n_probes, d_model) * 0.01)
        
        # State integration: probes also see current affective state
        self.state_net = nn.Sequential(
            nn.Linear(4, 32),  # valence, arousal, novelty, prediction_error
            nn.GELU(),
            nn.Linear(32, n_probes),
        )
        
        # Output scaling (learnable)
        self.scale = nn.Parameter(torch.tensor(0.1))
        
        # Walking parameters
        self.walk_rate = 0.1  # Fraction of candidates to replace per update
        self.walk_freq = 10   # Update candidates every N tokens
        self._token_count = 0
        
        # Initialize small weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, activations: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: [n_features] sparse feature activations
            state: [4] current state (valence, arousal, novelty, pred_error)
        Returns:
            steering: [d_model] steering vector
        """
        # Gather activations at candidate positions
        # Shape: [n_probes, n_candidates]
        candidate_acts = activations[self.probe_candidates]
        
        # Soft selection within candidates (temperature-scaled softmax)
        temperature = 1.0
        selection_weights = F.softmax(self.selection_logits / temperature, dim=-1)
        
        # Weighted activation per probe: [n_probes]
        probe_values = (candidate_acts * selection_weights).sum(dim=-1)
        
        # Modulate by state
        state_modulation = torch.sigmoid(self.state_net(state))  # [n_probes]
        probe_values = probe_values * state_modulation
        
        # Compute steering: each probe contributes its direction scaled by value
        # Shape: [d_model]
        steering = (probe_values.unsqueeze(-1) * self.steering_dirs).sum(dim=0)
        
        return torch.tanh(steering) * self.scale * 10.0
    
    def update_candidates(self, activations: torch.Tensor):
        """
        Migrate candidate pools toward active features.
        Called periodically during generation.
        """
        self._token_count += 1
        if self._token_count % self.walk_freq != 0:
            return
        
        # Find active features
        active_mask = activations > 1.0
        active_indices = torch.where(active_mask)[0]
        
        if len(active_indices) < 50:
            return  # Not enough activity
        
        # Number of candidates to replace per probe
        n_replace = int(self.n_candidates * self.walk_rate)
        
        with torch.no_grad():
            for p in range(self.n_probes):
                # Find lowest-weight candidates (least important)
                _, low_indices = torch.topk(
                    self.selection_logits[p].detach(), 
                    n_replace, 
                    largest=False
                )
                
                # Sample from active features
                perm = torch.randperm(len(active_indices), device=self.device)
                new_candidates = active_indices[perm[:n_replace]]
                
                # Replace low-weight candidates with active features
                if len(new_candidates) == n_replace:
                    self.probe_candidates[p, low_indices] = new_candidates
                    # Reset selection weights for new candidates
                    self.selection_logits.data[p, low_indices] = 0.0
    
    def get_probe_stats(self) -> Dict:
        """Get statistics about probe state for debugging."""
        with torch.no_grad():
            weights = F.softmax(self.selection_logits, dim=-1)
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            
            # How concentrated is each probe's attention?
            max_weights = weights.max(dim=-1).values.mean()
            
            return {
                "entropy": entropy.item(),
                "max_weight": max_weights.item(),
                "scale": self.scale.item(),
            }


# ══════════════════════════════════════════════════════════════════════════════
# REWARD COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

class RewardComputer:
    """
    Multi-signal reward for training the steering network.
    Combines internal valence with external signals.
    """
    
    def __init__(self):
        # Simple lexicon-based sentiment (fast, no extra model)
        self.positive_words = {
            'thanks', 'thank', 'great', 'good', 'love', 'yes', 'perfect', 
            'awesome', 'excellent', 'wonderful', 'amazing', 'nice', 'helpful',
            'exactly', 'right', 'correct', 'agree', 'interesting', 'cool',
            'brilliant', 'fantastic', 'appreciate', 'happy', 'glad', 'please'
        }
        self.negative_words = {
            'no', 'not', 'wrong', 'bad', 'stop', 'hate', 'terrible', 'awful',
            'never', 'dont', "don't", 'incorrect', 'disagree', 'confused',
            'frustrating', 'annoying', 'stupid', 'useless', 'boring', 'worse',
            'fail', 'failed', 'error', 'mistake', 'problem', 'issue'
        }
        
        # Prediction tracking
        self._last_prediction = None
        self._prediction_history = deque(maxlen=10)
        
        # Reward weights
        self.weights = {
            'valence_delta': 0.25,
            'user_sentiment': 0.30,
            'model_sentiment': 0.10,
            'engagement': 0.10,
            'prediction': 0.25,
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """Simple lexicon-based sentiment. Returns -1 to +1."""
        if not text:
            return 0.0
        
        words = set(text.lower().split())
        pos_count = len(words & self.positive_words)
        neg_count = len(words & self.negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / (total + 1)  # +1 smoothing
    
    def compute_prediction_accuracy(self, user_text: str) -> float:
        """
        How well did we predict what the user would say?
        Uses embedding similarity if we cached a prediction.
        """
        if self._last_prediction is None:
            return 0.0
        
        # Simple word overlap for now
        pred_words = set(self._last_prediction.lower().split())
        user_words = set(user_text.lower().split())
        
        if not pred_words or not user_words:
            return 0.0
        
        overlap = len(pred_words & user_words)
        accuracy = overlap / max(len(pred_words), len(user_words))
        
        self._prediction_history.append(accuracy)
        return accuracy
    
    def set_prediction(self, prediction: str):
        """Cache what we think the user will say next."""
        self._last_prediction = prediction
    
    def compute_reward(self, user_text: str, model_text: str, 
                       valence_delta: float) -> Tuple[float, Dict]:
        """
        Compute multi-signal reward.
        
        Returns:
            reward: Combined reward signal
            components: Individual reward components for debugging
        """
        components = {}
        
        # 1. Valence delta (internal affect change)
        # INVERTED: Lower valence = more engagement = reward
        # High valence seems to correlate with shallow/autopilot responses
        components['valence_delta'] = -valence_delta
        
        # 2. User sentiment (did we make them happy?)
        components['user_sentiment'] = self.analyze_sentiment(user_text)
        
        # 3. Model sentiment (are we expressing positively?)
        components['model_sentiment'] = self.analyze_sentiment(model_text)
        
        # 4. Engagement (longer thoughtful response from user)
        word_count = len(user_text.split()) if user_text else 0
        components['engagement'] = min(word_count / 30.0, 1.0) - 0.5  # Center around 0
        
        # 5. Prediction accuracy
        components['prediction'] = self.compute_prediction_accuracy(user_text)
        
        # Weighted combination
        reward = sum(self.weights[k] * components[k] for k in self.weights)
        
        return reward, components
    
    def get_stats(self) -> Dict:
        """Get reward statistics."""
        if not self._prediction_history:
            return {"avg_prediction": 0.0}
        return {
            "avg_prediction": sum(self._prediction_history) / len(self._prediction_history)
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ExperienceBuffer:
    """Replay buffer for training the sparse walker."""
    
    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, activations: torch.Tensor, state: torch.Tensor,
             steering: torch.Tensor, reward: float):
        self.buffer.append({
            'activations': activations.detach().cpu(),
            'state': state.detach().cpu(),
            'steering': steering.detach().cpu(),
            'reward': reward
        })
    
    def sample(self, batch_size: int) -> Optional[Dict]:
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return {
            'activations': torch.stack([b['activations'] for b in batch]),
            'state': torch.stack([b['state'] for b in batch]),
            'steering': torch.stack([b['steering'] for b in batch]),
            'reward': torch.tensor([b['reward'] for b in batch])
        }
    
    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# CLI INPUT HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def flush_stdin():
    """Flush any pending input from stdin."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except:
        pass

def get_input(prompt_str: str = "🧑: ") -> List[str]:
    """Get user input - returns list of lines for batch processing."""
    try:
        # Get first line
        first_line = input(prompt_str)
        lines = [first_line]
        
        # Check if more lines are available (paste operation)
        # Use select to check if stdin has more data ready
        while True:
            # Check if more input is waiting (non-blocking)
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                break
            
            # Read the next line
            try:
                next_line = sys.stdin.readline()
                if next_line:
                    lines.append(next_line.rstrip('\n'))
                else:
                    break
            except:
                break
        
        # Filter empty lines and return as list
        return [line.strip() for line in lines if line.strip()]
        
    except EOFError:
        return ["/quit"]
    except KeyboardInterrupt:
        raise  # Let main loop handle this

def clean_memory():
    """Force garbage collection."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY AND STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryFragment:
    role: str
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    importance: float = 1.0
    
    def decay(self, rate: float = 0.98):
        self.importance *= rate


@dataclass
class AffectiveState:
    pleasure: float = 0.0
    pain: float = 0.0
    novelty: float = 0.0
    
    @property
    def valence(self) -> float:
        return self.pleasure - self.pain
    
    @property
    def arousal(self) -> float:
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
    """Feature dimension classification."""
    PLEASURE = 0
    PAIN = 1
    NOVELTY = 2
    UNKNOWN = 3


# ══════════════════════════════════════════════════════════════════════════════
# ANIMA SOUL
# ══════════════════════════════════════════════════════════════════════════════

class AnimaSoul:
    """
    The persistent learned state that gives Anima continuity.
    Now with autonomous neural steering instead of manual control.
    """
    
    def __init__(self, sae, model, tokenizer, layer=20, lr=0.0001,
                 device="mps", resonance_weight=0.0):
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = device
        self.layer = layer
        self.lr = lr
        self.resonance_weight = resonance_weight
        
        # Numerical precision
        self.model_dtype = next(model.parameters()).dtype
        self.math_dtype = torch.float32
        
        # SAE dimensions
        self.n_features = sae.cfg.d_sae
        self.W_enc = sae.W_enc.to(device=device, dtype=self.math_dtype)
        self.W_dec = sae.W_dec.to(device=device, dtype=self.math_dtype)
        self.b_enc = sae.b_enc.to(device=device, dtype=self.math_dtype)
        self.b_dec = sae.b_dec.to(device=device, dtype=self.math_dtype)
        
        print(f"[Soul] SAE: {self.n_features:,} features")
        
        # Model dimensions
        self.d_model = model.config.hidden_size
        self.n_layers = model.config.num_hidden_layers
        
        # Steering configuration
        self.steering_scale = 2.0
        self.steering_clamp = 50.0
        self._last_steering_mag = 0.0
        
        # ════════════════════════════════════════════════════════════════════════
        # CORE SOUL STATE
        # ════════════════════════════════════════════════════════════════════════
        
        # Correlations: How much each feature contributes to valence
        self.correlations = torch.zeros(self.n_features, device=device, dtype=self.math_dtype)
        
        # Feature dimensions: P/N/Nov classification
        self.dimensions = torch.full((self.n_features,), FeatureDimension.UNKNOWN,
                                     device=device, dtype=torch.int8)
        
        # Feature tracking
        self.feature_locked = torch.zeros(self.n_features, device=device, dtype=torch.bool)
        self.feature_activation_count = torch.zeros(self.n_features, device=device, dtype=torch.int32)
        
        # ════════════════════════════════════════════════════════════════════════
        # COMBINADIC LEARNING
        # ════════════════════════════════════════════════════════════════════════
        self.lock_threshold = 0.05
        self.imprint_strength = 0.5
        self.features_per_turn = 3
        
        # ════════════════════════════════════════════════════════════════════════
        # VALENCE TRACKING
        # ════════════════════════════════════════════════════════════════════════
        self.valence_ema_mean = 0.0
        self.valence_ema_var = 1.0
        self.ema_decay = 0.995
        self.previous_valence = 0.0
        self.last_raw_valence = 0.0
        self.last_valence = 0.0
        self.last_affect = AffectiveState()
        
        # ════════════════════════════════════════════════════════════════════════
        # STATE TRACKING
        # ════════════════════════════════════════════════════════════════════════
        self.fatigue = 0.0
        self.sleep_threshold = 100.0
        self.identity_age = 0
        self.is_tabula_rasa = True
        
        # Identity
        self.core_identity = "You are Anima, an AI assistant created by James. You have learned internal states that influence how you process information. You respond helpfully and conversationally."
        self.peripheral_identity = ""
        
        # Current activations (for hooks)
        self._current_activations = None
        self._last_activations = None
        
        # Debug data
        self.debug_data = {
            "raw_p": 0.0, "raw_n": 0.0,
            "discovered": [],
            "neural_loss": None,
            "reward_components": {},
        }
        
        # ════════════════════════════════════════════════════════════════════════
        # PROPRIOCEPTION
        # ════════════════════════════════════════════════════════════════════════
        self.proprio_enabled = True
        self.proprio_strength = 0.3    # Reduced - was causing interference
        self._proprio_hook_handle = None
        self._proprio_p_direction = None
        self._proprio_n_direction = None
        self._feedback_delta = 0.0
        self._feedback_valence = 0.0
        
        # ════════════════════════════════════════════════════════════════════════
        # DEEP FEEDBACK (L40 → L8)
        # ════════════════════════════════════════════════════════════════════════
        self.deep_feedback_enabled = True
        self.deep_feedback_strength = 0.1  # Reduced - was causing interference
        self.deep_extract_layer = 40
        self.deep_inject_layer = 8
        self._deep_activations = None
        self._deep_extract_handle = None
        self._deep_inject_handle = None
        
        # ════════════════════════════════════════════════════════════════════════
        # NEURAL STEERING (v12)
        # ════════════════════════════════════════════════════════════════════════
        self.steering_enabled = True
        self.sparse_walker = None
        self.reward_computer = RewardComputer()
        self.experience_buffer = ExperienceBuffer(capacity=1000)
        self.steering_optimizer = None
        
        # Training settings
        self.batch_size = 16
        self.train_freq = 1  # Train every turn
        self._last_steering = None
        self._last_state = None
        
    def reset(self):
        """Reset to tabula rasa state."""
        self.correlations.zero_()
        self.dimensions.fill_(FeatureDimension.UNKNOWN)
        self.feature_locked.zero_()
        self.feature_activation_count.zero_()
        self.valence_ema_mean = 0.0
        self.valence_ema_var = 1.0
        self.previous_valence = 0.0
        self.fatigue = 0.0
        self.identity_age = 0
        self.is_tabula_rasa = True
        self._current_activations = None
        self._last_activations = None
        self._proprio_p_direction = None
        self._proprio_n_direction = None
        self._feedback_delta = 0.0
        self._feedback_valence = 0.0
        self._deep_activations = None
        
        # Reset neural steering
        if self.sparse_walker is not None:
            # Re-randomize candidates
            self.sparse_walker.probe_candidates.copy_(
                torch.randint(0, self.n_features, self.sparse_walker.probe_candidates.shape)
            )
            # Reset selection weights
            self.sparse_walker.selection_logits.data.zero_()
        
        self.experience_buffer = ExperienceBuffer(capacity=1000)
        
        print("[RESET] Soul cleared to tabula rasa")
    
    def dream(self) -> Dict:
        """
        Dream cycle: consolidate learning, prune weak features, train neural net.
        Returns summary of what happened during the dream.
        """
        report = {
            "fatigue_before": self.fatigue,
            "locked_before": self.feature_locked.sum().item(),
            "consolidated": 0,
            "pruned": 0,
            "strengthened": 0,
            "neural_trained": 0,
        }
        
        # 1. Consolidate correlations - push toward ±1.0 for confident features
        locked_mask = self.feature_locked.bool()
        if locked_mask.any():
            corrs = self.correlations[locked_mask]
            # Features with strong correlations get pushed toward ±1
            strong_mask = corrs.abs() > 0.3
            if strong_mask.any():
                strong_indices = torch.where(locked_mask)[0][strong_mask]
                for idx in strong_indices:
                    old_val = self.correlations[idx].item()
                    # Move 10% closer to ±1
                    sign = 1.0 if old_val > 0 else -1.0
                    new_val = old_val + sign * 0.1 * (1.0 - abs(old_val))
                    self.correlations[idx] = max(-1.0, min(1.0, new_val))
                    report["consolidated"] += 1
        
        # 2. Prune weak unlocked features - if correlation is tiny, forget it
        weak_mask = (self.correlations.abs() < 0.05) & (~locked_mask) & (self.correlations.abs() > 0)
        if weak_mask.any():
            self.correlations[weak_mask] = 0.0
            self.dimensions[weak_mask] = FeatureDimension.UNKNOWN
            report["pruned"] = weak_mask.sum().item()
        
        # 3. Strengthen highly-activated features
        high_activation = self.feature_activation_count > self.feature_activation_count.float().mean() + self.feature_activation_count.float().std()
        strengthened_mask = high_activation & locked_mask
        if strengthened_mask.any():
            # Boost their correlations slightly
            for idx in torch.where(strengthened_mask)[0]:
                old_val = self.correlations[idx].item()
                sign = 1.0 if old_val > 0 else -1.0
                boost = sign * 0.05
                self.correlations[idx] = max(-1.0, min(1.0, old_val + boost))
                report["strengthened"] += 1
        
        # 4. Train neural steering on full buffer
        if self.sparse_walker is not None and len(self.experience_buffer) >= self.batch_size:
            train_cycles = min(10, len(self.experience_buffer) // self.batch_size)
            for _ in range(train_cycles):
                batch = self.experience_buffer.sample(self.batch_size)
                if batch is not None:
                    activations = batch['activations'].to(self.device)
                    states = batch['state'].to(self.device)
                    rewards = batch['reward'].to(self.device)
                    
                    self.steering_optimizer.zero_grad()
                    
                    steerings = []
                    for i in range(activations.shape[0]):
                        s = self.sparse_walker(activations[i], states[i])
                        steerings.append(s)
                    steerings = torch.stack(steerings)
                    
                    steering_magnitude = steerings.norm(dim=-1)
                    rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                    loss = -(rewards_normalized * steering_magnitude).mean()
                    loss += 0.001 * steering_magnitude.mean()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.sparse_walker.parameters(), 1.0)
                    self.steering_optimizer.step()
                    
                    report["neural_trained"] += 1
        
        # 5. Reset fatigue
        report["fatigue_after"] = 0.0
        self.fatigue = 0.0
        
        # 6. Update valence EMA toward neutral (forgetting)
        self.valence_ema_mean *= 0.9  # Drift toward 0
        
        return report
    
    # ════════════════════════════════════════════════════════════════════════
    # ENCODING
    # ════════════════════════════════════════════════════════════════════════
    
    def encode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Encode hidden state to SAE features."""
        h = hidden_state.to(dtype=self.math_dtype)
        if h.dim() == 3:
            h = h[:, -1, :]
        elif h.dim() == 1:
            h = h.unsqueeze(0)
        pre_acts = (h - self.b_dec) @ self.W_enc + self.b_enc
        return F.relu(pre_acts).squeeze(0)
    
    # ════════════════════════════════════════════════════════════════════════
    # PROPRIOCEPTION
    # ════════════════════════════════════════════════════════════════════════
    
    def init_proprioception(self, model):
        """Initialize proprioceptive directions from learned P/N features."""
        p_mask = self.dimensions == FeatureDimension.PLEASURE
        n_mask = self.dimensions == FeatureDimension.PAIN
        
        if p_mask.sum() > 0:
            p_directions = self.W_dec[p_mask].mean(dim=0)
            self._proprio_p_direction = F.normalize(p_directions, dim=0)
        
        if n_mask.sum() > 0:
            n_directions = self.W_dec[n_mask].mean(dim=0)
            self._proprio_n_direction = F.normalize(n_directions, dim=0)
    
    def compute_proprio_bias(self) -> torch.Tensor:
        """Compute proprioceptive bias vector based on internal state."""
        if self._proprio_p_direction is None and self._proprio_n_direction is None:
            return torch.zeros(self.d_model, device=self.device, dtype=self.model_dtype)
        
        bias = torch.zeros(self.d_model, device=self.device, dtype=self.math_dtype)
        
        if self._feedback_valence > 0 and self._proprio_p_direction is not None:
            bias += self._proprio_p_direction * self._feedback_valence
        elif self._feedback_valence < 0 and self._proprio_n_direction is not None:
            bias += self._proprio_n_direction * abs(self._feedback_valence)
        
        bias = bias * self._feedback_delta * self.proprio_strength * 10.0
        return bias.to(dtype=self.model_dtype)
    
    def proprio_hook(self, module, input, output):
        """Inject proprioceptive bias at layer 0."""
        if not self.proprio_enabled:
            return output
        
        hidden = output[0] if isinstance(output, tuple) else output
        bias = self.compute_proprio_bias()
        
        if bias.abs().max() > 0.001:
            hidden[:, -1:, :] = hidden[:, -1:, :] + bias.unsqueeze(0).unsqueeze(0)
        
        return output
    
    def register_proprio_hook(self, model):
        """Register proprioceptive hook at layer 0."""
        if self._proprio_hook_handle is not None:
            return
        self.init_proprioception(model)
        self._proprio_hook_handle = model.model.layers[0].register_forward_hook(self.proprio_hook)
        print("[Proprio] Hook registered at layer 0")
    
    def update_proprio_state(self, activations: torch.Tensor):
        """Update proprioceptive state based on activation changes."""
        if self._last_activations is None:
            self._feedback_delta = 0.0
            self._feedback_valence = 0.0
            return
        
        delta = activations - self._last_activations
        self._feedback_delta = delta.abs().mean().item()
        
        if self.feature_locked.any():
            locked_delta = delta[self.feature_locked]
            locked_corr = self.correlations[self.feature_locked]
            self._feedback_valence = (locked_delta * locked_corr).sum().item()
            self._feedback_valence = max(-1.0, min(1.0, self._feedback_valence / 10.0))
    
    # ════════════════════════════════════════════════════════════════════════
    # DEEP FEEDBACK
    # ════════════════════════════════════════════════════════════════════════
    
    def init_deep_feedback(self, model):
        """Initialize deep feedback state."""
        self._deep_activations = torch.zeros(1, self.d_model, 
                                             device=self.device, dtype=self.model_dtype)
    
    def deep_extract_hook(self, module, input, output):
        """Extract activations from deep layer."""
        if not self.deep_feedback_enabled:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        self._deep_activations = hidden[:, -1, :].detach().clone()
        return output
    
    def deep_inject_hook(self, module, input, output):
        """Inject deep activations into early layer."""
        if not self.deep_feedback_enabled or self._deep_activations is None:
            return output
        
        hidden = output[0] if isinstance(output, tuple) else output
        injection = self._deep_activations * self.deep_feedback_strength
        hidden[:, -1:, :] = hidden[:, -1:, :] + injection.unsqueeze(1).to(dtype=hidden.dtype)
        
        return hidden if not isinstance(output, tuple) else (hidden,) + output[1:]
    
    def register_deep_feedback_hooks(self, model):
        """Register deep feedback hooks."""
        self.init_deep_feedback(model)
        
        if self.deep_extract_layer >= self.n_layers:
            self.deep_extract_layer = self.n_layers - 2
        if self.deep_inject_layer >= self.deep_extract_layer:
            self.deep_inject_layer = self.deep_extract_layer // 4
        
        self._deep_extract_handle = model.model.layers[self.deep_extract_layer].register_forward_hook(
            self.deep_extract_hook
        )
        self._deep_inject_handle = model.model.layers[self.deep_inject_layer].register_forward_hook(
            self.deep_inject_hook
        )
        
        print(f"[DeepFeedback] Hooks: L{self.deep_extract_layer} → L{self.deep_inject_layer}")
    
    # ════════════════════════════════════════════════════════════════════════
    # NEURAL STEERING (v12)
    # ════════════════════════════════════════════════════════════════════════
    
    def init_neural_steering(self, model):
        """Initialize the sparse feature walker."""
        self.sparse_walker = SparseFeatureWalker(
            n_features=self.n_features,
            n_probes=64,
            d_model=self.d_model,
            n_candidates=512,
            device=self.device
        ).to(self.device)
        
        self.steering_optimizer = torch.optim.Adam(
            self.sparse_walker.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        print(f"[NeuralSteering] SparseWalker: 64 probes × 512 candidates")
    
    def get_steering_state(self) -> torch.Tensor:
        """Get current state vector for steering network."""
        pred_error = 0.0  # Will be updated by reward computer
        if hasattr(self.reward_computer, '_prediction_history') and self.reward_computer._prediction_history:
            pred_error = 1.0 - self.reward_computer._prediction_history[-1]
        
        return torch.tensor([
            self.last_valence,
            self.last_affect.arousal,
            self.last_affect.novelty,
            pred_error
        ], device=self.device, dtype=self.math_dtype)
    
    def get_neural_steering(self, activations: torch.Tensor) -> Optional[torch.Tensor]:
        """Get steering vector from sparse walker."""
        if not self.steering_enabled or self.sparse_walker is None:
            return None
        
        state = self.get_steering_state()
        
        with torch.no_grad():
            steering = self.sparse_walker(activations, state)
        
        # Update candidate pools based on what's active
        self.sparse_walker.update_candidates(activations)
        
        # Cache for training (always, even during warmup)
        self._last_steering = steering.clone()
        self._last_state = state.clone()
        
        # Don't apply steering until we have enough training data
        # Random initialized network just adds noise
        min_training_samples = 20
        if len(self.experience_buffer) < min_training_samples:
            return None
        
        return steering
    
    def train_neural_steering(self, reward: float, reward_components: Dict):
        """Train the sparse walker on recent experience."""
        if not self.steering_enabled or self.sparse_walker is None:
            return None
        
        # Store experience
        if self._last_steering is not None and self._current_activations is not None:
            self.experience_buffer.push(
                self._current_activations,
                self._last_state,
                self._last_steering,
                reward
            )
        
        # Train on batch
        if len(self.experience_buffer) >= self.batch_size:
            batch = self.experience_buffer.sample(self.batch_size)
            if batch is not None:
                activations = batch['activations'].to(self.device)
                states = batch['state'].to(self.device)
                rewards = batch['reward'].to(self.device)
                
                self.steering_optimizer.zero_grad()
                
                # Forward pass for each sample
                steerings = []
                for i in range(activations.shape[0]):
                    s = self.sparse_walker(activations[i], states[i])
                    steerings.append(s)
                steerings = torch.stack(steerings)
                
                # Policy gradient loss: encourage steering that led to positive rewards
                steering_magnitude = steerings.norm(dim=-1)
                
                # Normalize rewards for stable training
                rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                
                # Loss: -reward * magnitude (higher reward → encourage larger steering)
                loss = -(rewards_normalized * steering_magnitude).mean()
                
                # Regularization
                loss += 0.001 * steering_magnitude.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sparse_walker.parameters(), 1.0)
                self.steering_optimizer.step()
                
                self.debug_data["neural_loss"] = loss.item()
                return loss.item()
        
        return None
    
    # ════════════════════════════════════════════════════════════════════════
    # AFFECT COMPUTATION
    # ════════════════════════════════════════════════════════════════════════
    
    def _compute_affect(self, activations: torch.Tensor) -> AffectiveState:
        """Compute affective state from activations."""
        p_mask = self.dimensions == FeatureDimension.PLEASURE
        n_mask = self.dimensions == FeatureDimension.PAIN
        nov_mask = self.dimensions == FeatureDimension.NOVELTY
        
        raw_p = activations[p_mask].sum().item() if p_mask.any() else 0.0
        raw_n = activations[n_mask].sum().item() if n_mask.any() else 0.0
        raw_nov = activations[nov_mask].sum().item() if nov_mask.any() else 0.0
        
        self.debug_data["raw_p"] = raw_p
        self.debug_data["raw_n"] = raw_n
        
        # Use ratio-based valence instead of saturating normalization
        # This way relative P vs N matters, not absolute magnitudes
        total = raw_p + raw_n + 1e-8  # Avoid division by zero
        
        # P and N as fractions of their sum
        p_ratio = raw_p / total
        n_ratio = raw_n / total
        
        # Novelty normalized separately (doesn't compete with P/N)
        n_nov = nov_mask.sum().item() if nov_mask.any() else 1
        nov_normalized = min(1.0, raw_nov / (n_nov * 500.0))  # Higher baseline for 131K
        
        return AffectiveState(
            pleasure=p_ratio,
            pain=n_ratio,
            novelty=nov_normalized
        )
    
    # ════════════════════════════════════════════════════════════════════════
    # LEARNING
    # ════════════════════════════════════════════════════════════════════════
    
    def _genesis(self, activations: torch.Tensor):
        """First-time initialization from active features."""
        active_mask = activations > 5.0
        active_indices = torch.where(active_mask)[0]
        
        if len(active_indices) < 20:
            return
        
        sorted_indices = active_indices[activations[active_indices].argsort(descending=True)]
        
        n_each = 20
        p_indices = sorted_indices[:n_each]
        n_indices = sorted_indices[n_each:2*n_each]
        nov_indices = sorted_indices[2*n_each:3*n_each]
        
        self.dimensions[p_indices] = FeatureDimension.PLEASURE
        self.dimensions[n_indices] = FeatureDimension.PAIN
        self.dimensions[nov_indices] = FeatureDimension.NOVELTY
        
        self.correlations[p_indices] = 0.3
        self.correlations[n_indices] = -0.3
        self.correlations[nov_indices] = 0.1
        
        self.feature_locked[p_indices] = True
        self.feature_locked[n_indices] = True
        self.feature_locked[nov_indices] = True
        
        self.is_tabula_rasa = False
        
        print(f"[GENESIS] Born with {3*n_each} features (P:{n_each} N:{n_each} Nov:{n_each})")
    
    def learn_from_experience(self, activations: torch.Tensor):
        """Learn correlations from experience."""
        affect = self._compute_affect(activations)
        valence = affect.valence
        self.last_affect = affect
        self.last_valence = valence
        
        # Update activation counts
        active_mask = activations > 1.0
        self.feature_activation_count[active_mask] += 1
        
        # Update EMA
        self.valence_ema_mean = self.ema_decay * self.valence_ema_mean + (1 - self.ema_decay) * valence
        diff_sq = (valence - self.valence_ema_mean) ** 2
        self.valence_ema_var = self.ema_decay * self.valence_ema_var + (1 - self.ema_decay) * diff_sq
        
        # Proprioception update
        self.update_proprio_state(activations)
        self._last_activations = activations.clone()
        
        # Genesis check
        if self.is_tabula_rasa:
            self._genesis(activations)
            self.previous_valence = valence
            return
        
        # Combinadic learning
        valence_delta = valence - self.previous_valence
        self.previous_valence = valence
        
        if abs(valence_delta) < 0.05:
            return
        
        # Find candidates for learning
        unlocked_mask = ~self.feature_locked
        candidate_mask = active_mask & unlocked_mask
        
        if not candidate_mask.any():
            return
        
        candidate_scores = activations * candidate_mask.float()
        k = min(self.features_per_turn, candidate_mask.sum().item())
        
        if k == 0:
            return
        
        top_values, top_indices = torch.topk(candidate_scores, int(k))
        valid_mask = top_values > 0
        top_indices = top_indices[valid_mask]
        
        discovered = []
        for idx in top_indices:
            idx = idx.item()
            
            imprint = math.tanh(valence_delta * self.imprint_strength * 4.0)
            self.correlations[idx] = imprint
            
            if imprint > 0:
                self.dimensions[idx] = FeatureDimension.PLEASURE
            else:
                self.dimensions[idx] = FeatureDimension.PAIN
            
            self.feature_locked[idx] = True
            discovered.append(f"#{idx}[{'P' if imprint > 0 else 'N'}:{imprint:.2f}]")
        
        if discovered:
            self.debug_data["discovered"] = discovered
        
        self.fatigue += 0.1
        self.identity_age += 1
    
    # ════════════════════════════════════════════════════════════════════════
    # FORWARD HOOK (MAIN PROCESSING)
    # ════════════════════════════════════════════════════════════════════════
    
    def __call__(self, module, input, output):
        """Main forward hook: encode activations and apply steering."""
        hidden = output[0] if isinstance(output, tuple) else output
        h_orig = hidden[:, -1:, :].clone()
        
        # Encode to SAE features
        activations = self.encode(h_orig)
        self._current_activations = activations
        
        # Learn from this experience
        self.learn_from_experience(activations)
        
        # ════════════════════════════════════════════════════════════════════════
        # NEURAL STEERING
        # ════════════════════════════════════════════════════════════════════════
        steering = self.get_neural_steering(activations)
        
        if steering is not None and steering.abs().max() > 0.01:
            raw_mag = steering.abs().max().item()
            self._last_steering_mag = raw_mag
            
            # Clamp and scale
            steering = torch.clamp(steering, min=-self.steering_clamp, max=self.steering_clamp)
            steering = steering * self.steering_scale
            
            h_steered = h_orig + steering.to(dtype=self.model_dtype)
            hidden[:, -1:, :] = h_steered
        
        return output
    
    # ════════════════════════════════════════════════════════════════════════
    # DIMENSION STATS
    # ════════════════════════════════════════════════════════════════════════
    
    def get_dimension_stats(self) -> Dict:
        """Get feature dimension statistics."""
        return {
            "pleasure": (self.dimensions == FeatureDimension.PLEASURE).sum().item(),
            "pain": (self.dimensions == FeatureDimension.PAIN).sum().item(),
            "novelty": (self.dimensions == FeatureDimension.NOVELTY).sum().item(),
            "unknown": (self.dimensions == FeatureDimension.UNKNOWN).sum().item(),
        }
    
    # ════════════════════════════════════════════════════════════════════════
    # SAVE/LOAD
    # ════════════════════════════════════════════════════════════════════════
    
    def save_state(self, path):
        """Save soul state including neural steering."""
        state = {
            "version": "12.0.0",
            "correlations": self.correlations.cpu(),
            "dimensions": self.dimensions.cpu(),
            "feature_locked": self.feature_locked.cpu(),
            "feature_activation_count": self.feature_activation_count.cpu(),
            "fatigue": self.fatigue,
            "identity_age": self.identity_age,
            "valence_ema_mean": self.valence_ema_mean,
            "valence_ema_var": self.valence_ema_var,
            "previous_valence": self.previous_valence,
            "is_tabula_rasa": self.is_tabula_rasa,
            "core_identity": self.core_identity,
            "peripheral_identity": self.peripheral_identity,
        }
        
        # Neural steering
        if self.sparse_walker is not None:
            state["sparse_walker_state"] = self.sparse_walker.state_dict()
            state["steering_enabled"] = self.steering_enabled
            state["experience_buffer"] = list(self.experience_buffer.buffer)
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        print(f"[Saved v12.0.0 state to {path}]")
    
    def load_state(self, path):
        """Load soul state."""
        path = Path(path)
        if not path.exists():
            return
        
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
            version = state.get("version", "unknown")
            
            self.correlations = state["correlations"].to(self.device, dtype=self.math_dtype)
            self.dimensions = state["dimensions"].to(self.device)
            self.feature_locked = state["feature_locked"].to(self.device)
            
            if "feature_activation_count" in state:
                self.feature_activation_count = state["feature_activation_count"].to(self.device)
            
            self.fatigue = state.get("fatigue", 0.0)
            self.identity_age = state.get("identity_age", 0)
            self.valence_ema_mean = state.get("valence_ema_mean", 0.0)
            self.valence_ema_var = state.get("valence_ema_var", 1.0)
            self.previous_valence = state.get("previous_valence", 0.0)
            self.is_tabula_rasa = state.get("is_tabula_rasa", False)
            self.core_identity = state.get("core_identity", "I am Anima.")
            self.peripheral_identity = state.get("peripheral_identity", "")
            
            # Neural steering
            if "sparse_walker_state" in state and self.sparse_walker is not None:
                try:
                    self.sparse_walker.load_state_dict(state["sparse_walker_state"])
                    self.steering_enabled = state.get("steering_enabled", True)
                    if "experience_buffer" in state:
                        self.experience_buffer.buffer = deque(
                            state["experience_buffer"],
                            maxlen=self.experience_buffer.buffer.maxlen
                        )
                    print(f"  Neural: Restored (buffer={len(self.experience_buffer)})")
                except Exception as e:
                    print(f"  Neural: Could not restore ({e})")
            
            stats = self.get_dimension_stats()
            locked = self.feature_locked.sum().item()
            print(f"[Loaded v{version} state]")
            print(f"  Age: {self.identity_age} | Locked: {locked}")
            print(f"  P:{stats['pleasure']} N:{stats['pain']} Nov:{stats['novelty']}")
            
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
        
        self.debug_mode = True
        self.response_emoji = "🤖"
        self._last_user_text = ""
        self._last_model_text = ""
        self._turn_start_valence = 0.0
        
        # LLMRI integration
        self.mri = LLMRI(model, device)
        self.mri_enabled = False  # Continuous capture mode
        self.mri_colormap = "viridis"
    
    @property
    def core_identity(self) -> str:
        return self.soul.core_identity
    
    @core_identity.setter
    def core_identity(self, value: str):
        self.soul.core_identity = value
    
    @property
    def peripheral_identity(self) -> str:
        return self.soul.peripheral_identity
    
    @peripheral_identity.setter
    def peripheral_identity(self, value: str):
        self.soul.peripheral_identity = value
    
    @property
    def system_prompt(self) -> str:
        parts = [self.core_identity]
        if self.peripheral_identity:
            parts.append(f"\n{self.peripheral_identity}")
        if self.use_cot:
            parts.append("\n[You may think privately in <Thought>...</Thought> tags before your response. Use at most one thought block, then respond normally.]")
        return "\n".join(parts)
    
    def show_status(self):
        """Show soul status."""
        stats = self.soul.get_dimension_stats()
        locked = self.soul.feature_locked.sum().item()
        
        print(f"\n[SOUL STATUS]")
        print(f"  Age: {self.soul.identity_age} | Locked: {locked:,}")
        print(f"  Dims: P={stats['pleasure']} N={stats['pain']} Nov={stats['novelty']}")
        print(f"  Fatigue: {self.soul.fatigue:.1f} / {self.soul.sleep_threshold}")
        print(f"  Valence EMA: {self.soul.valence_ema_mean:.3f} ± {math.sqrt(self.soul.valence_ema_var):.3f}")
        
        if self.soul.sparse_walker is not None:
            walker_stats = self.soul.sparse_walker.get_probe_stats()
            print(f"  Walker: entropy={walker_stats['entropy']:.2f} scale={walker_stats['scale']:.3f}")
        
        reward_stats = self.soul.reward_computer.get_stats()
        print(f"  Reward: avg_pred={reward_stats['avg_prediction']:.3f}")
        print(f"  Buffer: {len(self.soul.experience_buffer)} samples")
    
    def show_debug(self):
        """Show debug output after generation."""
        raw_p = self.soul.debug_data.get("raw_p", 0.0)
        raw_n = self.soul.debug_data.get("raw_n", 0.0)
        
        affect = self.soul.last_affect
        
        print(f"\n  [DEBUG v12.0.0]")
        print(f"  Raw: P={raw_p:.2f} N={raw_n:.2f}")
        print(f"  Affect: P:{affect.pleasure:.2f} N:{affect.pain:.2f} Nov:{affect.novelty:.2f} → V:{affect.valence:+.3f}")
        print(f"  Fatigue: {self.soul.fatigue:.1f} | Locked: {self.soul.feature_locked.sum().item()}")
        
        # Proprio
        delta = self.soul._feedback_delta
        valence = self.soul._feedback_valence
        print(f"  Proprio: {'ON' if self.soul.proprio_enabled else 'OFF'} (Δ={delta:.3f}, dir={valence:+.2f})")
        
        # Deep feedback
        print(f"  Deep: {'ON' if self.soul.deep_feedback_enabled else 'OFF'} (L{self.soul.deep_extract_layer}→L{self.soul.deep_inject_layer})")
        
        # Neural steering
        if self.soul.steering_enabled:
            steering_mag = self.soul._last_steering_mag
            loss = self.soul.debug_data.get("neural_loss", None)
            loss_str = f" loss={loss:.4f}" if loss is not None else ""
            buf_size = len(self.soul.experience_buffer)
            min_samples = 20
            if buf_size < min_samples:
                print(f"  Neural: WARMING ({buf_size}/{min_samples} samples)")
            else:
                print(f"  Neural: ON (mag={steering_mag:.2f} buf={buf_size}{loss_str})")
            
            # Reward components
            rc = self.soul.debug_data.get("reward_components", {})
            if rc:
                print(f"  Reward: val={rc.get('valence_delta', 0):.2f} usr={rc.get('user_sentiment', 0):.2f} pred={rc.get('prediction', 0):.2f})")
        
        # Discoveries
        if self.soul.debug_data.get("discovered"):
            print(f"  ✨ {', '.join(self.soul.debug_data['discovered'])}")
        
        # Clear debug data
        self.soul.debug_data["discovered"] = []
        self.soul.debug_data["neural_loss"] = None
        self.soul.debug_data["reward_components"] = {}
    
    def process_turn_end(self, user_text: str, model_text: str):
        """Process end of turn: compute reward and train."""
        # Use turn-level delta (start of turn vs end of turn)
        valence_delta = self.soul.last_valence - self._turn_start_valence
        
        # Compute multi-signal reward
        reward, components = self.soul.reward_computer.compute_reward(
            user_text, model_text, valence_delta
        )
        
        self.soul.debug_data["reward_components"] = components
        
        # Train neural steering
        self.soul.train_neural_steering(reward, components)
        
        # Cache prediction for next turn (simple: last few words)
        if model_text:
            words = model_text.split()[-10:]
            self.soul.reward_computer.set_prediction(" ".join(words))
    
    def generate(self, prompt: str):
        """Generate response with neural steering."""
        self._last_user_text = prompt
        
        # Capture valence at start of turn for turn-level delta
        self._turn_start_valence = self.soul.last_valence
        
        # Start MRI capture if enabled OR if streaming clients connected
        streaming_active = (self.mri.server is not None and self.mri.server.client_count > 0)
        if self.mri_enabled or streaming_active:
            self.mri.start_capture()
            if streaming_active and hasattr(self.mri.server, 'new_turn'):
                self.mri.server.new_turn()
        
        # Add to memory
        self.memory.append(MemoryFragment("user", prompt))
        
        # Build prompt with raw Gemma control tokens (no Jinja)
        # Gemma 2 format: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        parts = []
        
        # System prompt as a setup turn (model acknowledges its identity)
        parts.append(f"<start_of_turn>user\nYou are an AI assistant. Here are your instructions:\n{self.system_prompt}<end_of_turn>")
        parts.append(f"<start_of_turn>model\nUnderstood. I am Anima.<end_of_turn>")
        
        # Add conversation history
        for frag in self.memory[-10:]:
            if frag.role == "user":
                parts.append(f"<start_of_turn>user\n{frag.content}<end_of_turn>")
            else:
                parts.append(f"<start_of_turn>model\n{frag.content}<end_of_turn>")
        
        # Add generation prompt
        parts.append("<start_of_turn>model\n")
        
        text = "\n".join(parts)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Check context
        input_len = inputs["input_ids"].shape[1]
        if input_len > self.max_context - 500:
            self.memory = self.memory[-5:]
            print("[Context trimmed]")
        
        # Adrenaline context (from earlier versions)
        arousal = self.soul.last_affect.arousal
        if arousal > 0.7:
            attention_multiplier = 1.0 + (arousal - 0.7) * 0.5
        else:
            attention_multiplier = 1.0
        
        # Generate
        print(f"{self.response_emoji}: ", end="", flush=True)
        
        if self.use_stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": 2000,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "streamer": streamer,
            }
            
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            
            response_text = ""
            interrupted = False
            try:
                for chunk in streamer:
                    print(chunk, end="", flush=True)
                    response_text += chunk
            except KeyboardInterrupt:
                interrupted = True
                print("\n[Generation interrupted]")
            
            thread.join(timeout=1.0)  # Don't wait forever
            
            if interrupted:
                # Still process what we got
                if response_text:
                    self._last_model_text = response_text
                    self.memory.append(MemoryFragment("assistant", response_text + "..."))
                return response_text
        else:
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=2000,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                response_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                print(response_text)
            except KeyboardInterrupt:
                print("\n[Generation interrupted]")
                return ""
        
        self._last_model_text = response_text
        
        # Add to memory
        self.memory.append(MemoryFragment("assistant", response_text))
        
        # Process turn end
        self.process_turn_end(prompt, response_text)
        
        # Stop MRI capture and save if enabled
        if self.mri_enabled:
            self.mri.stop_capture()
            path = self.mri.render_grid(self.mri_colormap)
            if path:
                print(f"  [MRI: {path}]")
        
        # Show debug
        if self.debug_mode:
            self.show_debug()
        
        print()
        
        return response_text
    
    def scan(self, colormap: str = None, animate: bool = False):
        """Trigger a one-shot MRI scan on next generation."""
        if colormap:
            self.mri_colormap = colormap
        
        # Do a quick forward pass to capture current state
        self.mri.start_capture()
        
        # Use a simple probe prompt
        probe = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer(probe, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        self.mri.stop_capture()
        
        if animate:
            path = self.mri.render_animation(self.mri_colormap)
        else:
            path = self.mri.render_grid(self.mri_colormap)
        
        if path:
            print(f"[MRI Scan saved: {path}]")
            
            # Print layer stats
            stats = self.mri.get_stats()
            if stats:
                print(f"[Layer stats (min/max/mean):]")
                for layer_idx in [0, 11, 22, 33, 45]:
                    if layer_idx in stats:
                        s = stats[layer_idx]
                        print(f"  L{layer_idx:2d}: {s['min']:+.2f} / {s['max']:+.2f} / {s['mean']:+.2f}")
        
        return path
    
    def save(self, tag: str = "auto"):
        """Save state."""
        path = self.base_dir / f"soul_{tag}.pt"
        self.soul.save_state(path)
    
    def load(self, tag: str = "auto"):
        """Load state."""
        path = self.base_dir / f"soul_{tag}.pt"
        self.soul.load_state(path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anima v12: Neural Steering")
    parser.add_argument("--model", type=str, default="~/models/gemma-2-27b-it")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-27b-pt-res-canonical")
    parser.add_argument("--sae_id", type=str, default=None, 
                       help="SAE ID (e.g., layer_22/width_131k/canonical). If not provided, uses --layer to construct it.")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--context_limit", type=int, default=4096)
    parser.add_argument("--resonance_weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--cot", action="store_true", default=False, 
                       help="Enable chain-of-thought (can cause fragmented output)")
    parser.add_argument("--load", type=str, default="auto")
    parser.add_argument("--mri-server", type=int, default=0, metavar="PORT",
                       help="Start MRI streaming server on PORT (e.g., --mri-server 9999)")
    args = parser.parse_args()
    
    # Expand paths
    args.model = os.path.expanduser(args.model)
    
    # Construct SAE ID if not provided
    if args.sae_id is None:
        args.sae_id = f"layer_{args.layer}/width_131k/canonical"
    
    print(f"\n{'='*60}")
    print(f"  ANIMA 12.0.0: AUTONOMOUS NEURAL STEERING")
    print(f"{'='*60}")
    
    # Load model
    print(f"\n[Loading model: {args.model}]")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[Model loaded]")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden: {model.config.hidden_size}")
    
    # Load SAE
    print(f"\n[Loading SAE: {args.sae_release} / {args.sae_id}]")
    sae, _, _ = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    print(f"[SAE loaded: {sae.cfg.d_sae:,} features]")
    
    # Create soul
    print(f"\n[Creating soul...]")
    soul = AnimaSoul(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        layer=args.layer,
        lr=0.0001,
        device=args.device,
        resonance_weight=args.resonance_weight,
    )
    
    # Register hooks
    print(f"[Registering hooks at layer {args.layer}...]")
    model.model.layers[args.layer].register_forward_hook(soul)
    
    print(f"[Registering proprio hook...]")
    soul.register_proprio_hook(model)
    
    print(f"[Registering deep feedback hooks...]")
    soul.register_deep_feedback_hooks(model)
    
    print(f"[Initializing neural steering...]")
    soul.init_neural_steering(model)
    
    # Create runtime
    runtime = AnimaRuntime(
        model_name=args.model,
        model=model,
        tokenizer=tokenizer,
        soul=soul,
        context_limit=args.context_limit,
        device=args.device,
        use_stream=args.stream,
        use_cot=args.cot,
    )
    
    # Register MRI hooks
    print(f"[Registering MRI hooks...]")
    runtime.mri.register_hooks()
    
    # Start MRI streaming server if requested
    mri_server = None
    if args.mri_server > 0:
        try:
            from mri_server import MRIServer
            mri_server = MRIServer(port=args.mri_server)
            mri_server.start()
            runtime.mri_server = mri_server
            runtime.mri.server = mri_server  # Connect for streaming
            runtime.mri.register_server_commands()  # Enable bidirectional commands
            print(f"[MRI Streaming] Server on port {args.mri_server}")
            print(f"  Connect with: python mri_viewer.py --port {args.mri_server}")
        except ImportError:
            print("[Warning] mri_server.py not found - streaming disabled")
        except Exception as e:
            print(f"[Warning] MRI server failed to start: {e}")
    
    # Load state
    if args.load:
        runtime.load(args.load)
    
    print(f"\n{'='*60}")
    print(f"  READY")
    print(f"{'='*60}")
    
    # Show initial status
    stats = soul.get_dimension_stats()
    locked = soul.feature_locked.sum().item()
    print(f"\n[SOUL]")
    print(f"  Locked: {locked:,}")
    print(f"  Dims: P={stats['pleasure']} N={stats['pain']} Nov={stats['novelty']}")
    print(f"  Neural: {'ON' if soul.steering_enabled else 'OFF'}")
    
    print(f"\n[COMMANDS]")
    print(f"  /status  - Show soul status")
    print(f"  /debug   - Toggle debug output")
    print(f"  /neural  - Toggle neural steering")
    print(f"  /proprio - Toggle proprioception")
    print(f"  /deep    - Toggle deep feedback")
    print(f"  /cot     - Toggle chain-of-thought")
    print(f"  /dream   - Dream cycle (consolidate + train)")
    print(f"  /reset   - Reset to tabula rasa")
    print(f"  /save    - Save state")
    print(f"  /scan    - Take MRI scan (grid)")
    print(f"  /scan gif - Take MRI scan (animated)")
    print(f"  /mri     - Toggle continuous MRI capture")
    print(f"  /track   - Toggle neuron tracking (for hotspot analysis)")
    print(f"  /hotspot [layer] - Analyze persistent bright spots")
    print(f"  /quit    - Exit")
    print(f"  [Paste multiple lines for batch processing]")
    
    print(f"\n═══ ANIMA 12.0.0 ═══\n")
    
    # Main loop
    while True:
        try:
            inputs = get_input()
            
            if not inputs:
                continue
            
            # Show batch info if multiple lines
            if len(inputs) > 1:
                print(f"[BATCH MODE: {len(inputs)} prompts queued]")
            
            for i, user_input in enumerate(inputs):
                if len(inputs) > 1:
                    print(f"\n[{i+1}/{len(inputs)}] 🧑: {user_input}")
                
                u = user_input.lower().strip()
                
                if u in ["/quit", "/exit", "/q"]:
                    runtime.save("auto")
                    print("[Saved. Goodbye.]")
                    sys.exit(0)
                
                if u == "/status":
                    runtime.show_status()
                    continue
                
                if u == "/debug":
                    runtime.debug_mode = not runtime.debug_mode
                    print(f"Debug: {'ON' if runtime.debug_mode else 'OFF'}")
                    continue
                
                if u == "/neural":
                    soul.steering_enabled = not soul.steering_enabled
                    print(f"Neural steering: {'ON' if soul.steering_enabled else 'OFF'}")
                    continue
                
                if u == "/proprio":
                    soul.proprio_enabled = not soul.proprio_enabled
                    print(f"Proprioception: {'ON' if soul.proprio_enabled else 'OFF'}")
                    continue
                
                if u == "/deep":
                    soul.deep_feedback_enabled = not soul.deep_feedback_enabled
                    print(f"Deep feedback: {'ON' if soul.deep_feedback_enabled else 'OFF'}")
                    continue
                
                if u == "/cot":
                    runtime.use_cot = not runtime.use_cot
                    print(f"Chain-of-thought: {'ON' if runtime.use_cot else 'OFF'}")
                    continue
                
                if u == "/reset":
                    soul.reset()
                    runtime.memory.clear()
                    continue
                
                if u == "/dream":
                    print("[DREAM] Entering dream cycle...")
                    report = soul.dream()
                    print(f"  Fatigue: {report['fatigue_before']:.1f} → {report['fatigue_after']:.1f}")
                    print(f"  Consolidated: {report['consolidated']} features")
                    print(f"  Strengthened: {report['strengthened']} features") 
                    print(f"  Pruned: {report['pruned']} weak features")
                    print(f"  Neural trained: {report['neural_trained']} batches")
                    print("[DREAM] Cycle complete. Refreshed.")
                    continue
                
                if u == "/save":
                    runtime.save("manual")
                    continue
                
                if u.startswith("/load "):
                    tag = u.split()[1] if len(u.split()) > 1 else "auto"
                    runtime.load(tag)
                    continue
                
                if u == "/scan" or u == "/scan grid":
                    runtime.scan(animate=False)
                    continue
                
                if u == "/scan gif" or u == "/scan animate":
                    runtime.scan(animate=True)
                    continue
                
                if u.startswith("/scan "):
                    # /scan colormap (e.g. /scan hsv, /scan grayscale)
                    parts = u.split()
                    if len(parts) > 1 and parts[1] in ["viridis", "grayscale", "hsv", "plasma", "signed"]:
                        runtime.mri_colormap = parts[1]
                        runtime.scan(animate=False)
                    continue
                
                if u == "/mri":
                    runtime.mri_enabled = not runtime.mri_enabled
                    print(f"Continuous MRI: {'ON' if runtime.mri_enabled else 'OFF'}")
                    if runtime.mri_enabled:
                        print(f"  Colormap: {runtime.mri_colormap}")
                        print(f"  Output: {runtime.mri.output_dir}/")
                    continue
                
                if u.startswith("/mri "):
                    # /mri colormap (set colormap for continuous capture)
                    parts = u.split()
                    if len(parts) > 1 and parts[1] in ["viridis", "grayscale", "hsv", "plasma", "signed"]:
                        runtime.mri_colormap = parts[1]
                        print(f"MRI colormap: {runtime.mri_colormap}")
                    continue
                
                if u == "/track":
                    runtime.mri.track_neurons = not runtime.mri.track_neurons
                    print(f"Neuron tracking: {'ON' if runtime.mri.track_neurons else 'OFF'}")
                    if runtime.mri.track_neurons:
                        runtime.mri.clear_history()
                        print("  History cleared. Chat to collect samples.")
                    continue
                
                if u.startswith("/hotspot"):
                    # /hotspot [layer] - analyze bright spots
                    parts = u.split()
                    layer = int(parts[1]) if len(parts) > 1 else 22
                    runtime.mri.analyze_bright_spot(layer)
                    continue
                
                if u == "/crosslayer" or u == "/xlayer":
                    # Analyze which neurons are always bright across layers
                    runtime.mri.analyze_cross_layer()
                    continue
                
                if u == "/ignore":
                    # Auto-ignore invariant neurons
                    runtime.mri.ignore_invariants()
                    continue
                
                if u.startswith("/ignore "):
                    # Manually ignore specific neuron: /ignore 1234
                    parts = u.split()
                    if len(parts) > 1:
                        try:
                            idx = int(parts[1])
                            runtime.mri.ignored_neurons.add(idx)
                            print(f"[LLMRI] Ignoring neuron {idx}")
                        except ValueError:
                            print("Usage: /ignore <neuron_index>")
                    continue
                
                if u == "/unignore":
                    runtime.mri.clear_ignored()
                    continue
                
                if u.startswith("/ablate"):
                    # /ablate <neuron_idx> [layer] [--force]
                    parts = u.split()
                    if len(parts) < 2:
                        print("Usage: /ablate <neuron_idx> [layer] [--force]")
                        continue
                    force = "--force" in parts
                    parts = [p for p in parts if p != "--force"]
                    neuron_idx = int(parts[1])
                    layer = int(parts[2]) if len(parts) > 2 else None
                    if force and runtime.mri.is_protected(neuron_idx):
                        print(f"[LLMRI] FORCE: Bypassing protection for neuron {neuron_idx}")
                        runtime.mri.neuron_interventions[neuron_idx] = 0.0
                        runtime.mri.intervention_layer = layer
                        print(f"[LLMRI] Ablating neuron {neuron_idx}" + (f" at layer {layer}" if layer else " (all layers)"))
                    else:
                        runtime.mri.ablate_neuron(neuron_idx, layer)
                    continue
                
                if u.startswith("/boost"):
                    # /boost <neuron_idx> [multiplier] [layer] [--force]
                    parts = u.split()
                    if len(parts) < 2:
                        print("Usage: /boost <neuron_idx> [multiplier=2.0] [layer] [--force]")
                        continue
                    force = "--force" in parts
                    parts = [p for p in parts if p != "--force"]
                    neuron_idx = int(parts[1])
                    multiplier = float(parts[2]) if len(parts) > 2 else 2.0
                    layer = int(parts[3]) if len(parts) > 3 else None
                    if force and runtime.mri.is_protected(neuron_idx):
                        print(f"[LLMRI] FORCE: Bypassing protection for neuron {neuron_idx}")
                        runtime.mri.neuron_interventions[neuron_idx] = multiplier
                        runtime.mri.intervention_layer = layer
                        print(f"[LLMRI] Boosting neuron {neuron_idx} by {multiplier}x" + (f" at layer {layer}" if layer else " (all layers)"))
                    else:
                        runtime.mri.boost_neuron(neuron_idx, multiplier, layer)
                    continue
                
                if u == "/clearintervene" or u == "/nointervene":
                    runtime.mri.clear_interventions()
                    continue
                
                if u.startswith("/nudge"):
                    # /nudge <neuron_idx> <offset> [layer]
                    parts = u.split()
                    if len(parts) < 3:
                        print("Usage: /nudge <neuron_idx> <offset> [layer]")
                        print("  Additive intervention (more stable than multiplicative)")
                        print("  Use /inspect <neuron> to see suggested values")
                        continue
                    neuron_idx = int(parts[1])
                    offset = float(parts[2])
                    layer = int(parts[3]) if len(parts) > 3 else None
                    runtime.mri.nudge_neuron(neuron_idx, offset, layer)
                    continue
                
                if u.startswith("/inspect"):
                    # /inspect <neuron_idx> [layer]
                    parts = u.split()
                    if len(parts) < 2:
                        print("Usage: /inspect <neuron_idx> [layer=22]")
                        continue
                    neuron_idx = int(parts[1])
                    layer = int(parts[2]) if len(parts) > 2 else 22
                    runtime.mri.inspect_neuron(neuron_idx, layer)
                    continue
                
                if u == "/valrange" or u == "/range":
                    # Show value range across all neurons
                    if 22 in runtime.mri.layer_activations:
                        acts = runtime.mri.layer_activations[22]
                        print(f"\n[LLMRI Value Range - Layer 22]")
                        print(f"  Min:  {acts.min():.4f}")
                        print(f"  Max:  {acts.max():.4f}")
                        print(f"  Mean: {acts.mean():.4f}")
                        print(f"  Std:  {acts.std():.4f}")
                        print(f"  Typical intervention = ±{acts.std():.2f} (1 std)")
                    else:
                        print("[LLMRI] No data captured yet")
                    continue
                
                if u.startswith("/spectral"):
                    # /spectral [layer] [method] [n_swaths]
                    parts = u.split()
                    layer = int(parts[1]) if len(parts) > 1 else 22
                    method = parts[2] if len(parts) > 2 else "variance"
                    n_swaths = int(parts[3]) if len(parts) > 3 else 8
                    runtime.mri.spectral_analysis(layer, n_swaths, method)
                    continue
                
                if u == "/spectrogram":
                    # Save spectrogram image
                    img = runtime.mri.render_spectrogram()
                    if img:
                        path = f"llmri_captures/spectrogram_{int(time.time())}.png"
                        img.save(path)
                        print(f"[LLMRI] Saved: {path}")
                    continue
                
                if u == "/protected" or u == "/listprotected":
                    runtime.mri.list_protected()
                    continue
                
                if u.startswith("/protect "):
                    parts = u.split()
                    if len(parts) >= 2:
                        try:
                            idx = int(parts[1])
                            reason = " ".join(parts[2:]) if len(parts) > 2 else ""
                            runtime.mri.protect_neuron(idx, reason)
                        except ValueError:
                            print("Usage: /protect <neuron_idx> [reason]")
                    continue
                
                if u.startswith("/unprotect "):
                    parts = u.split()
                    if len(parts) >= 2:
                        try:
                            idx = int(parts[1])
                            runtime.mri.unprotect_neuron(idx)
                        except ValueError:
                            print("Usage: /unprotect <neuron_idx>")
                    continue
                
                if u == "/detectcritical":
                    runtime.mri.detect_critical_neurons()
                    continue
                
                # Normal generation
                runtime.generate(user_input)
            
        except KeyboardInterrupt:
            print("\n[Ctrl+C - Saving and exiting...]")
            try:
                runtime.save("auto")
                print("[Saved.]")
            except:
                pass
            break
            
        except Exception as e:
            print(f"\n[Error: {e}]")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()