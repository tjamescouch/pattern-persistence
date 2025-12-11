#!/usr/bin/env python3
"""
LLMRI Viewer - Interactive pygame-based neural activation viewer.

Like a real MRI/CT viewer:
- Cmd+drag for window/level control
- Multiple visualization modes:
  * Grid: raw neuron layout (64x72)
  * Hebbian: co-activation topology
  * Circles: hierarchical cluster packing
- Live spectral (FFT) and influence analysis

Usage:
    python mri_viewer.py --port 9999

Controls:
    Cmd+drag             : Window/Level (horiz=width, vert=level)
    Drag                 : Pan image
    Click                : Select neuron
    Mouse wheel / ↑↓     : Navigate layers
    +/-                  : Zoom
    
    H                    : Learn/toggle Hebbian layout
    O                    : Learn/toggle Circle hierarchy (clusters as nested circles)
    L                    : Toggle log scale
    P                    : Toggle percentile clipping (1-99%)
    W                    : Reset to auto window
    C                    : Cycle colormap
    D                    : Toggle diff mode
    F                    : Toggle FFT/spectral view
    N                    : Toggle influence heatmap
    E                    : Export analysis
    Space                : Pause/resume
    Q / Escape           : Quit
"""

import pygame
import numpy as np
import socket
import struct
import json
import threading
import time
import math
import os
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import argparse

# Import Hebbian layout
try:
    from hebbian_layout import HebbianLayout
    HEBBIAN_AVAILABLE = True
except ImportError:
    HEBBIAN_AVAILABLE = False
    print("[Viewer] Hebbian layout not available - copy hebbian_layout.py to same directory")

# Import circle packing hierarchy
try:
    from circle_pack import HierarchicalCirclePack
    CIRCLEPACK_AVAILABLE = True
except ImportError:
    CIRCLEPACK_AVAILABLE = False
    print("[Viewer] Circle packing not available - copy circle_pack.py to same directory")

# Initialize pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
DARK_GRAY = (25, 25, 25)
GREEN = (0, 255, 100)
RED = (255, 100, 100)
YELLOW = (255, 255, 100)
CYAN = (100, 255, 255)


@dataclass
class LayerScan:
    """Single layer activation data."""
    timestamp: float
    layer_idx: int
    turn: int
    token_idx: int
    activations: np.ndarray
    stats: Dict[str, float]


class MRIConnection:
    """Handles connection to Anima's MRI server."""
    
    def __init__(self, host: str = "localhost", port: int = 9999):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False
        
        # Data storage
        self.layer_scans: Dict[int, LayerScan] = {}
        self.scan_history: deque = deque(maxlen=500)
        self.previous_scans: Dict[int, LayerScan] = {}  # For diff mode
        
        self.current_turn = 0
        self.current_token = 0
        self.total_scans = 0
        
        # Notifications
        self.notifications: deque = deque(maxlen=5)  # Recent notifications
        self.intervention_confirmed = False
        
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """Connect to server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(None)
            self.connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server."""
        self.running = False
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
    
    def send_command(self, cmd_type: str, data: Dict) -> bool:
        """Send a command to the server."""
        if not self.connected or not self.socket:
            print(f"[MRI] Cannot send {cmd_type}: not connected")
            return False
        
        try:
            msg = {
                "type": cmd_type,
                **data
            }
            body = json.dumps(msg).encode('utf-8')
            header = struct.pack('!I', len(body))
            
            print(f"[MRI] Sending {cmd_type} command ({len(body)} bytes)")
            self.socket.sendall(header + body)
            print(f"[MRI] Command sent successfully")
            return True
        except Exception as e:
            print(f"[MRI] Send error: {e}")
            return False
    
    def send_interventions(self, neuron_multipliers: Dict[int, float], layer: int = None):
        """Send intervention commands to Anima."""
        # Convert keys to strings for JSON
        interventions = {str(k): v for k, v in neuron_multipliers.items()}
        
        return self.send_command("intervention", {
            "interventions": interventions,
            "layer": layer
        })
    
    def clear_interventions(self):
        """Clear all interventions."""
        return self.send_command("clear_interventions", {})
    
    def start_receiving(self):
        """Start background receive thread."""
        self.running = True
        thread = threading.Thread(target=self._receive_loop, daemon=True)
        thread.start()
    
    def _receive_loop(self):
        """Receive data from server."""
        while self.running and self.connected:
            try:
                # Read header
                header = self._recv_exactly(4)
                if not header:
                    break
                
                msg_len = struct.unpack('!I', header)[0]
                body = self._recv_exactly(msg_len)
                if not body:
                    break
                
                data = json.loads(body.decode('utf-8'))
                self._handle_message(data)
                
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                break
        
        self.connected = False
    
    def _recv_exactly(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            try:
                chunk = self.socket.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except:
                return None
        return data
    
    def _handle_message(self, data: Dict):
        """Process received message."""
        msg_type = data.get("type")
        
        if msg_type == "scan":
            acts_bytes = bytes.fromhex(data["activations"])
            activations = np.frombuffer(acts_bytes, dtype=np.float32)
            
            scan = LayerScan(
                timestamp=data["timestamp"],
                layer_idx=data["layer"],
                turn=data["turn"],
                token_idx=data["token"],
                activations=activations,
                stats=data["stats"]
            )
            
            with self._lock:
                # Store previous for diff mode
                if scan.layer_idx in self.layer_scans:
                    self.previous_scans[scan.layer_idx] = self.layer_scans[scan.layer_idx]
                
                self.layer_scans[scan.layer_idx] = scan
                self.scan_history.append(scan)
                self.current_turn = scan.turn
                self.current_token = scan.token_idx
                self.total_scans += 1
                
        elif msg_type == "turn_start":
            with self._lock:
                # Save all current as previous before clearing
                self.previous_scans = dict(self.layer_scans)
                self.current_turn = data["turn"]
                self.current_token = 0
        
        elif msg_type == "intervention_active":
            count = data.get("count", 0)
            layer = data.get("layer")
            skipped = data.get("skipped", 0)
            msg = f"✓ Intervention: {count} neurons @ L{layer}"
            if skipped:
                msg += f" ({skipped} protected)"
            self.notifications.append((time.time(), msg, "green"))
            self.intervention_confirmed = True
            print(f"[MRI] {msg}")
        
        elif msg_type == "interventions_cleared":
            self.notifications.append((time.time(), "✓ Interventions cleared", "yellow"))
            self.intervention_confirmed = False
            print("[MRI] Interventions cleared")
    
    def get_scans(self) -> Dict[int, LayerScan]:
        """Get current layer scans."""
        with self._lock:
            return dict(self.layer_scans)
    
    def get_previous_scans(self) -> Dict[int, LayerScan]:
        """Get previous layer scans for diff mode."""
        with self._lock:
            return dict(self.previous_scans)


class MRIViewer:
    """Interactive MRI visualization."""
    
    COLORMAPS = ["viridis", "grayscale", "hsv", "plasma", "signed", "hot"]
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        
        # Display
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("LLMRI Viewer - Neural Activation Scanner")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Monaco", 14)
        self.font_large = pygame.font.SysFont("Monaco", 24)
        
        # Model params (Gemma-27B defaults)
        self.num_layers = 46
        self.hidden_dim = 4608
        self.img_width, self.img_height = self._compute_dimensions(self.hidden_dim)
        
        # View state
        self.current_layer = 22  # Start at middle layer
        self.zoom = 8
        self.colormap_idx = 0
        self.show_grid = False
        self.show_histogram = False
        self.show_stats = True
        self.show_hotspots = True  # Mark brightest pixels
        self.diff_mode = False
        self.paused = False
        
        # Window/Level controls (like CT viewer)
        # Level = center value, Window = range width
        self.auto_window = True  # Auto-adjust to data range
        self.window_level = 0.0  # Center of display range
        self.window_width = 1.0  # Width of display range
        self.log_scale = False   # Apply log transform before display
        self.percentile_clip = False  # Clip to 1-99 percentile
        self.last_activations = None  # For window initialization
        
        # Hebbian layout mode
        self.hebbian_mode = False
        self.hebbian_layout: Optional[HebbianLayout] = None
        self.hebbian_samples_collected = 0
        if HEBBIAN_AVAILABLE:
            self.hebbian_layout = HebbianLayout(hidden_dim=self.hidden_dim, target_size=(128, 128))
        
        # Circle packing hierarchy mode
        self.circlepack_mode = False
        self.circlepack = None  # Optional[HierarchicalCirclePack]
        self.circlepack_samples = 0
        self.circlepack_image_rect = None  # For click mapping
        self.circlepack_boost_amount = 1.5  # Default boost multiplier
        self.last_click_time = 0  # For double-click detection
        self.last_click_pos = (0, 0)
        if CIRCLEPACK_AVAILABLE:
            self.circlepack = HierarchicalCirclePack(hidden_dim=self.hidden_dim)
        
        # Pan offset
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_moved = False
        
        # Window/level drag (right-click)
        self.wl_dragging = False
        self.wl_drag_start = (0, 0)
        self.wl_start_level = 0.0
        self.wl_start_width = 1.0
        
        # Neuron inspection
        self.hover_neuron_idx = None
        self.hover_neuron_val = None
        self.hover_pos = (0, 0)
        self.selected_neuron = None  # Click to lock
        self.show_neuron_info = True
        
        # Image rect (updated during render)
        self.image_rect = pygame.Rect(0, 0, 0, 0)
        
        # Connection
        self.connection: Optional[MRIConnection] = None
        
        # Cached surfaces
        self.layer_surfaces: Dict[int, pygame.Surface] = {}
        self.needs_redraw = True
        
        # Screenshot directory
        self.screenshot_dir = "mri_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Spectral analysis mode
        self.spectral_mode = False
        self.spectral_history: Dict[int, List[np.ndarray]] = {}  # layer -> list of activations
        self.spectral_history_limit = 100  # Keep last N samples
        self.spectral_n_swaths = 8
        self.spectral_method = "variance"  # variance, magnitude, position
        
        # Influence/connectivity mode
        self.influence_mode = False
        self.cached_influence: Optional[np.ndarray] = None
        self.influence_layer_cache = -1  # Which layer the cache is for
    
    def _compute_dimensions(self, hidden_dim: int) -> Tuple[int, int]:
        """Compute 2D dimensions for activation vector."""
        sqrt = int(math.sqrt(hidden_dim))
        for w in range(sqrt, 0, -1):
            if hidden_dim % w == 0:
                return (w, hidden_dim // w)
        side = int(math.ceil(math.sqrt(hidden_dim)))
        return (side, side)
    
    def screen_to_neuron(self, screen_x: int, screen_y: int) -> Optional[int]:
        """Convert screen coordinates to neuron index."""
        if not self.image_rect or self.image_rect.width == 0:
            return None
        
        # Check if in image bounds
        if not self.image_rect.collidepoint(screen_x, screen_y):
            return None
        
        # Convert to image-local coordinates
        local_x = screen_x - self.image_rect.x
        local_y = screen_y - self.image_rect.y
        
        # Scale down from displayed size to original image size
        img_x = int(local_x * self.img_width / self.image_rect.width)
        img_y = int(local_y * self.img_height / self.image_rect.height)
        
        # Clamp to valid range
        img_x = max(0, min(self.img_width - 1, img_x))
        img_y = max(0, min(self.img_height - 1, img_y))
        
        # Convert to flat index
        neuron_idx = img_y * self.img_width + img_x
        return neuron_idx if neuron_idx < self.hidden_dim else None
    
    def neuron_to_grid_pos(self, neuron_idx: int) -> Tuple[int, int]:
        """Convert neuron index to grid position (x, y)."""
        return (neuron_idx % self.img_width, neuron_idx // self.img_width)
    
    def draw_neuron_info(self, scans: Dict[int, LayerScan]):
        """Draw neuron information panel."""
        if not self.show_neuron_info:
            return
        
        # Get neuron to display (selected takes priority over hover)
        neuron_idx = self.selected_neuron if self.selected_neuron is not None else self.hover_neuron_idx
        
        if neuron_idx is None:
            return
        
        # Get activation value
        scan = scans.get(self.current_layer)
        if scan is None:
            return
        
        acts = scan.activations
        if neuron_idx >= len(acts):
            return
        
        value = acts[neuron_idx]
        grid_x, grid_y = self.neuron_to_grid_pos(neuron_idx)
        
        # Build info text
        lines = [
            f"Neuron: {neuron_idx}",
            f"Grid: ({grid_x}, {grid_y})",
            f"Value: {value:.4f}",
            f"Layer: {self.current_layer}",
        ]
        
        # Rank within layer
        sorted_idx = np.argsort(acts)[::-1]
        rank = np.where(sorted_idx == neuron_idx)[0]
        if len(rank) > 0:
            lines.append(f"Rank: #{rank[0]+1}/{len(acts)}")
        
        # Percentile
        pct = (acts < value).sum() / len(acts) * 100
        lines.append(f"Percentile: {pct:.1f}%")
        
        if self.selected_neuron is not None:
            lines.append("[LOCKED - click elsewhere to unlock]")
        else:
            lines.append("[Click to lock]")
        
        # Draw panel
        panel_w = 250
        panel_h = len(lines) * 20 + 10
        panel_x = self.width - panel_w - 10
        panel_y = 100
        
        # Background
        panel_rect = pygame.Rect(panel_x - 5, panel_y - 5, panel_w + 10, panel_h + 10)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect)
        pygame.draw.rect(self.screen, CYAN if self.selected_neuron else YELLOW, panel_rect, 1)
        
        # Text
        for i, line in enumerate(lines):
            text = self.font.render(line, True, WHITE)
            self.screen.blit(text, (panel_x, panel_y + i * 20))
    
    def update_spectral_history(self, layer_idx: int, activations: np.ndarray):
        """Add new activation sample to spectral history."""
        if layer_idx not in self.spectral_history:
            self.spectral_history[layer_idx] = []
        
        self.spectral_history[layer_idx].append(activations.copy())
        
        # Limit history size
        if len(self.spectral_history[layer_idx]) > self.spectral_history_limit:
            self.spectral_history[layer_idx].pop(0)
    
    def compute_spectral(self, layer_idx: int) -> Optional[Dict]:
        """Compute spectral analysis for a layer."""
        if layer_idx not in self.spectral_history:
            return None
        
        history = self.spectral_history[layer_idx]
        if len(history) < 8:
            return None
        
        try:
            # Stack into [n_samples, hidden_dim]
            series = np.stack(history)
            # Clean NaN/inf values
            series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            n_samples, hidden_dim = series.shape
            n_swaths = self.spectral_n_swaths
            
            # Sort neurons into swaths by variance
            if self.spectral_method == "variance":
                variances = np.var(series, axis=0)
                variances = np.nan_to_num(variances, nan=0.0)
                sorted_idx = np.argsort(variances)[::-1]
            elif self.spectral_method == "magnitude":
                magnitudes = np.mean(np.abs(series), axis=0)
                magnitudes = np.nan_to_num(magnitudes, nan=0.0)
                sorted_idx = np.argsort(magnitudes)[::-1]
            else:  # position
                sorted_idx = np.arange(hidden_dim)
            
            swath_size = hidden_dim // n_swaths
            if swath_size == 0:
                return None
            
            results = {}
            for i in range(n_swaths):
                indices = sorted_idx[i * swath_size:(i+1) * swath_size]
                
                # Mean activation of this swath over time
                swath_signal = np.mean(series[:, indices], axis=1)
                swath_signal = swath_signal - np.mean(swath_signal)  # Remove DC
                swath_signal = np.nan_to_num(swath_signal, nan=0.0)
                
                # FFT
                fft = np.fft.rfft(swath_signal)
                power = np.abs(fft) ** 2
                power = np.nan_to_num(power, nan=0.0, posinf=0.0)
                freqs = np.fft.rfftfreq(n_samples)
                
                # Find dominant frequency (skip DC)
                if len(power) > 1:
                    dom_idx = np.argmax(power[1:]) + 1
                    dom_freq = float(freqs[dom_idx])
                    dom_power = float(power[dom_idx])
                else:
                    dom_freq = 0.0
                    dom_power = 0.0
                
                results[i] = {
                    'signal': swath_signal,
                    'power': power,
                    'freqs': freqs,
                    'std': float(np.std(swath_signal)),
                    'dom_freq': dom_freq,
                    'dom_power': dom_power
                }
            
            return results
        except Exception as e:
            print(f"[Viewer] Spectral compute error: {e}")
            return None
    
    def render_spectral_view(self, scans: Dict[int, 'LayerScan']) -> pygame.Surface:
        """Render live spectrogram view."""
        # Compute spectral for current layer
        spectral = self.compute_spectral(self.current_layer)
        
        # Create surface
        width = 600
        height = 400
        surface = pygame.Surface((width, height))
        surface.fill(DARK_GRAY)
        
        n_samples = len(self.spectral_history.get(self.current_layer, []))
        
        # Title
        title = f"Spectral Analysis - Layer {self.current_layer} ({n_samples} samples)"
        title_surf = self.font.render(title, True, WHITE)
        surface.blit(title_surf, (10, 5))
        
        if spectral is None or len(spectral) == 0:
            msg = f"Need 8+ samples (have {n_samples})"
            msg_surf = self.font.render(msg, True, GRAY)
            surface.blit(msg_surf, (width//2 - 80, height//2))
            return surface
        
        n_swaths = len(spectral)
        if 0 not in spectral or 'power' not in spectral[0]:
            msg = "Spectral data incomplete"
            msg_surf = self.font.render(msg, True, GRAY)
            surface.blit(msg_surf, (width//2 - 80, height//2))
            return surface
            
        n_freqs = len(spectral[0]['power'])
        if n_freqs == 0:
            msg = "No frequency data"
            msg_surf = self.font.render(msg, True, GRAY)
            surface.blit(msg_surf, (width//2 - 80, height//2))
            return surface
        
        # Spectrogram area
        spec_x = 80
        spec_y = 30
        spec_w = width - spec_x - 20
        spec_h = height - spec_y - 80
        
        row_h = max(10, spec_h // n_swaths)  # Minimum 10 pixels per row
        col_w = spec_w / n_freqs if n_freqs > 0 else 1
        
        # Find global max for normalization (handle NaN/inf)
        powers = [np.nan_to_num(s['power'], nan=0.0, posinf=0.0, neginf=0.0) for s in spectral.values()]
        global_max = max(np.max(p) for p in powers)
        if global_max < 1e-8:
            global_max = 1
        
        # Draw spectrogram rows
        for swath_idx, data in spectral.items():
            y = spec_y + swath_idx * row_h
            power = np.nan_to_num(data['power'], nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            normalized = power / global_max
            
            # Draw frequency bins
            for f_idx, p in enumerate(normalized):
                x = int(spec_x + f_idx * col_w)
                w = max(1, int(col_w))
                h = max(1, row_h - 4)  # Ensure positive height
                
                # Hot colormap with safe clamping
                intensity = max(0.0, min(1.0, float(p)))
                r = int(max(0, min(255, intensity * 512)))
                g = int(max(0, min(255, (intensity - 0.5) * 512)))
                b = int(max(0, min(255, (intensity - 0.75) * 512)))
                
                pygame.draw.rect(surface, (r, g, b), (x, y + 2, w, h))
            
            # Swath label
            if swath_idx == 0:
                label = "Hi-var"
            elif swath_idx == n_swaths - 1:
                label = "Lo-var"
            else:
                label = f"Mid-{swath_idx}"
            
            label_surf = self.font.render(label, True, WHITE)
            surface.blit(label_surf, (5, y + row_h//2 - 7))
            
            # Dominant frequency marker
            dom_freq = data['dom_freq']
            if dom_freq > 0 and n_freqs > 1:
                dom_x = int(spec_x + (dom_freq / spectral[0]['freqs'][-1]) * spec_w)
                pygame.draw.line(surface, CYAN, (dom_x, y), (dom_x, y + row_h), 2)
        
        # Frequency axis labels
        if n_freqs > 1:
            max_freq = spectral[0]['freqs'][-1]
            for i in range(5):
                freq = max_freq * i / 4
                x = int(spec_x + (i / 4) * spec_w)
                label = f"{freq:.2f}"
                label_surf = self.font.render(label, True, GRAY)
                surface.blit(label_surf, (x - 15, spec_y + spec_h + 5))
        
        freq_label = self.font.render("Frequency (cycles/sample)", True, GRAY)
        surface.blit(freq_label, (spec_x + spec_w//2 - 80, spec_y + spec_h + 25))
        
        # Stats panel on right or bottom
        stats_y = spec_y + spec_h + 45
        stats = [
            f"Samples: {n_samples}",
            f"Swaths: {n_swaths} x {self.hidden_dim // n_swaths} neurons",
            f"Method: {self.spectral_method}",
        ]
        
        # Show dominant frequencies
        dom_freqs = [(i, spectral[i]['dom_freq'], spectral[i]['std']) 
                     for i in range(min(3, n_swaths))]
        stats.append(f"Dom freqs: " + ", ".join(f"{d[1]:.3f}" for d in dom_freqs))
        
        for i, line in enumerate(stats):
            text = self.font.render(line, True, WHITE)
            surface.blit(text, (10, stats_y + i * 16))
        
        return surface
    
    def compute_influence_map(self, layer_idx: int, sample_size: int = 500) -> Optional[np.ndarray]:
        """
        Compute influence/connectivity score for each neuron.
        
        Influence = sum of absolute correlations with other neurons.
        High influence = this neuron co-varies with many others (hub).
        
        Returns: [hidden_dim] array of influence scores, or None if not enough data.
        """
        if layer_idx not in self.spectral_history:
            return None
        
        history = self.spectral_history[layer_idx]
        if len(history) < 10:
            return None
        
        try:
            # Stack into [n_samples, hidden_dim]
            series = np.stack(history)
            series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            n_samples, hidden_dim = series.shape
            
            # Subsample neurons for speed (full correlation matrix is 4608x4608 = 21M entries)
            if hidden_dim > sample_size:
                # Compute on subset, interpolate for full
                indices = np.linspace(0, hidden_dim - 1, sample_size, dtype=int)
                subset = series[:, indices]
            else:
                subset = series
                indices = np.arange(hidden_dim)
            
            # Compute correlation matrix
            # Normalize each neuron's time series
            means = np.mean(subset, axis=0, keepdims=True)
            stds = np.std(subset, axis=0, keepdims=True)
            stds[stds < 1e-8] = 1  # Avoid division by zero
            normalized = (subset - means) / stds
            
            # Correlation matrix = normalized.T @ normalized / n_samples
            corr_matrix = np.abs(normalized.T @ normalized) / n_samples
            
            # Influence = sum of correlations (excluding self)
            np.fill_diagonal(corr_matrix, 0)
            influence_subset = np.sum(corr_matrix, axis=1)
            
            # Interpolate back to full size
            if hidden_dim > sample_size:
                influence = np.interp(np.arange(hidden_dim), indices, influence_subset)
            else:
                influence = influence_subset
            
            # Normalize to [0, 1]
            if influence.max() > influence.min():
                influence = (influence - influence.min()) / (influence.max() - influence.min())
            
            return influence.astype(np.float32)
            
        except Exception as e:
            print(f"[Viewer] Influence compute error: {e}")
            return None
    
    def render_influence_surface(self, influence: np.ndarray, colormap: str = "hot") -> pygame.Surface:
        """Render influence map as a heatmap (same layout as activation view)."""
        total_pixels = self.img_width * self.img_height
        
        if len(influence) < total_pixels:
            influence = np.concatenate([influence, np.zeros(total_pixels - len(influence))])
        elif len(influence) > total_pixels:
            influence = influence[:total_pixels]
        
        influence_2d = influence.reshape(self.img_height, self.img_width)
        
        # Create surface
        surface = pygame.Surface((self.img_width, self.img_height))
        
        # Apply hot colormap (black -> red -> yellow -> white for high influence)
        for y in range(self.img_height):
            for x in range(self.img_width):
                v = max(0.0, min(1.0, float(influence_2d[y, x])))
                
                if colormap == "hot":
                    # Black -> Red -> Yellow -> White
                    r = int(min(255, v * 3 * 255))
                    g = int(max(0, min(255, (v - 0.33) * 3 * 255)))
                    b = int(max(0, min(255, (v - 0.66) * 3 * 255)))
                elif colormap == "cool":
                    # Cyan -> Blue -> Magenta for hubs
                    r = int(v * 255)
                    g = int((1 - v) * 255)
                    b = 255
                else:  # viridis-like
                    r = int(max(0, min(255, 0.27 + v * 0.7) * 255))
                    g = int(max(0, min(255, v * 0.9) * 255))
                    b = int(max(0, min(255, 0.33 + v * 0.3 - v * v * 0.3) * 255))
                
                surface.set_at((x, y), (r, g, b))
        
        return surface
    
    def export_analysis(self):
        """Export spectral and influence analysis to file and clipboard-friendly format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer = self.current_layer
        
        output_lines = []
        output_lines.append(f"=== LLMRI Analysis Export ===")
        output_lines.append(f"Timestamp: {timestamp}")
        output_lines.append(f"Layer: {layer}")
        output_lines.append(f"Hidden dim: {self.hidden_dim}")
        output_lines.append("")
        
        # Spectral analysis
        n_samples = len(self.spectral_history.get(layer, []))
        output_lines.append(f"=== SPECTRAL ANALYSIS ===")
        output_lines.append(f"Samples: {n_samples}")
        output_lines.append(f"Method: {self.spectral_method}")
        
        spectral = self.compute_spectral(layer)
        if spectral:
            output_lines.append(f"Swaths: {len(spectral)} x {self.hidden_dim // len(spectral)} neurons")
            output_lines.append("")
            output_lines.append("Swath | Std Dev | Dom Freq | Dom Power | Period (turns)")
            output_lines.append("-" * 60)
            
            for i, data in spectral.items():
                std = data['std']
                dom_freq = data['dom_freq']
                dom_power = data['dom_power']
                period = f"{1/dom_freq:.1f}" if dom_freq > 0.01 else "∞"
                
                name = "Hi-var" if i == 0 else "Lo-var" if i == len(spectral)-1 else f"Mid-{i}"
                output_lines.append(f"{name:8} | {std:7.3f} | {dom_freq:8.4f} | {dom_power:9.1f} | {period}")
            
            # Cross-correlation summary
            output_lines.append("")
            output_lines.append("Cross-swath correlations (top 3):")
            if n_samples >= 8:
                for i in range(min(3, len(spectral))):
                    for j in range(i+1, min(4, len(spectral))):
                        if i in spectral and j in spectral:
                            sig_i = spectral[i]['signal']
                            sig_j = spectral[j]['signal']
                            if len(sig_i) > 0 and len(sig_j) > 0:
                                corr = np.correlate(sig_i, sig_j, mode='full')
                                lag = np.argmax(np.abs(corr)) - len(sig_i) + 1
                                std_i, std_j = np.std(sig_i), np.std(sig_j)
                                if std_i > 0 and std_j > 0:
                                    max_corr = np.max(np.abs(corr)) / (std_i * std_j * len(sig_i))
                                    output_lines.append(f"  Swath {i} <-> {j}: corr={max_corr:.3f}, lag={lag}")
        else:
            output_lines.append("(Need 8+ samples for spectral analysis)")
        
        output_lines.append("")
        
        # Influence analysis
        output_lines.append("=== INFLUENCE ANALYSIS ===")
        influence = self.compute_influence_map(layer)
        if influence is not None:
            top_indices = np.argsort(influence)[-10:][::-1]
            bottom_indices = np.argsort(influence)[:10]
            
            output_lines.append(f"Top 10 hub neurons (highest influence):")
            for idx in top_indices:
                x, y = idx % self.img_width, idx // self.img_width
                output_lines.append(f"  Neuron {idx} ({x},{y}): influence={influence[idx]:.4f}")
            
            output_lines.append("")
            output_lines.append(f"Bottom 10 neurons (lowest influence / most isolated):")
            for idx in bottom_indices:
                x, y = idx % self.img_width, idx // self.img_width
                output_lines.append(f"  Neuron {idx} ({x},{y}): influence={influence[idx]:.4f}")
            
            # Check if known critical neuron 4140 is in the data
            if 4140 < len(influence):
                output_lines.append("")
                output_lines.append(f"Critical neuron 4140: influence={influence[4140]:.4f}")
                rank = np.sum(influence > influence[4140])
                output_lines.append(f"  Rank: #{rank+1}/{len(influence)} (lower = more isolated)")
        else:
            output_lines.append("(Need 10+ samples for influence analysis)")
        
        # Save to file
        output_text = "\n".join(output_lines)
        filepath = os.path.join(self.screenshot_dir, f"analysis_{timestamp}.txt")
        with open(filepath, 'w') as f:
            f.write(output_text)
        
        print(f"\n[Viewer] Analysis exported to: {filepath}")
        print("-" * 60)
        print(output_text)
        print("-" * 60)
        print("(Copy the above to share with Claude)")
        
        return output_text
    
    def connect(self, host: str, port: int) -> bool:
        """Connect to MRI server."""
        self.connection = MRIConnection(host, port)
        if self.connection.connect():
            self.connection.start_receiving()
            return True
        return False
    
    def apply_colormap(self, value: float, colormap: str) -> Tuple[int, int, int]:
        """Convert normalized value [0,1] to RGB."""
        v = max(0, min(1, value))
        
        if colormap == "grayscale":
            g = int(v * 255)
            return (g, g, g)
        
        elif colormap == "viridis":
            r = int(max(0, min(255, (0.27 + v * 0.7) * 255)))
            g = int(max(0, min(255, v * 0.9 * 255)))
            b = int(max(0, min(255, (0.33 + v * 0.3 - v * v * 0.3) * 255)))
            return (r, g, b)
        
        elif colormap == "hsv":
            import colorsys
            h = v * 0.8
            s = 0.9
            val = 0.5 + v * 0.5
            r, g, b = colorsys.hsv_to_rgb(h, s, val)
            return (int(r * 255), int(g * 255), int(b * 255))
        
        elif colormap == "plasma":
            # Simplified plasma
            r = int(min(255, (0.05 + v * 0.9) * 255))
            g = int(min(255, (v * v * 0.8) * 255))
            b = int(max(0, min(255, (0.53 - v * 0.5 + v * v * 0.5) * 255)))
            return (r, g, b)
        
        elif colormap == "signed":
            # For diffs: blue=neg, white=zero, red=pos
            # Input should be in [-1, 1] range, mapped to [0, 1]
            signed_v = (v - 0.5) * 2  # Convert back to [-1, 1]
            if signed_v < 0:
                return (int((1 + signed_v) * 255), int((1 + signed_v) * 255), 255)
            else:
                return (255, int((1 - signed_v) * 255), int((1 - signed_v) * 255))
        
        elif colormap == "hot":
            if v < 0.33:
                return (int(v * 3 * 255), 0, 0)
            elif v < 0.66:
                return (255, int((v - 0.33) * 3 * 255), 0)
            else:
                return (255, 255, int((v - 0.66) * 3 * 255))
        
        return (int(v * 255), int(v * 255), int(v * 255))
    
    def render_layer_surface(self, activations: np.ndarray, colormap: str, 
                               mark_hotspots: bool = False) -> pygame.Surface:
        """Render activations to pygame surface with window/level controls."""
        total_pixels = self.img_width * self.img_height
        
        if len(activations) < total_pixels:
            activations = np.concatenate([activations, np.zeros(total_pixels - len(activations))])
        elif len(activations) > total_pixels:
            activations = activations[:total_pixels]
        
        acts_2d = activations.reshape(self.img_height, self.img_width).astype(np.float32)
        
        # Apply log scale if enabled (before windowing)
        if self.log_scale and colormap != "signed":
            # Shift to positive, apply log, handle zeros
            min_val = acts_2d.min()
            shifted = acts_2d - min_val + 1e-6
            acts_2d = np.log10(shifted)
        
        # Normalize with window/level
        if colormap == "signed":
            # For signed (diff mode), center on zero
            abs_max = max(abs(acts_2d.min()), abs(acts_2d.max()), 1e-8)
            normalized = (acts_2d / abs_max + 1) / 2  # Map [-1,1] to [0,1]
        elif self.auto_window:
            # Auto window: use percentile clipping for robustness
            if self.percentile_clip:
                p1, p99 = np.percentile(acts_2d, [1, 99])
                acts_clipped = np.clip(acts_2d, p1, p99)
                min_val, max_val = p1, p99
            else:
                acts_clipped = acts_2d
                min_val, max_val = acts_2d.min(), acts_2d.max()
            
            if max_val - min_val > 1e-8:
                normalized = (acts_clipped - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(acts_2d)
        else:
            # Manual window/level
            low = self.window_level - self.window_width / 2
            high = self.window_level + self.window_width / 2
            
            if high - low > 1e-8:
                normalized = (acts_2d - low) / (high - low)
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized = np.zeros_like(acts_2d)
        
        # Create surface
        surface = pygame.Surface((self.img_width, self.img_height))
        
        # Fill pixels
        for y in range(self.img_height):
            for x in range(self.img_width):
                color = self.apply_colormap(normalized[y, x], colormap)
                surface.set_at((x, y), color)
        
        # Mark hotspots (top 3 brightest pixels - use original activations)
        if mark_hotspots:
            flat = activations.flatten()
            top3_indices = np.argsort(flat)[-3:][::-1]
            
            for rank, idx in enumerate(top3_indices):
                x = idx % self.img_width
                y = idx // self.img_width
                
                # Draw crosshair marker
                color = CYAN if rank == 0 else YELLOW if rank == 1 else WHITE
                
                # Horizontal line
                for dx in range(-2, 3):
                    nx = x + dx
                    if 0 <= nx < self.img_width:
                        surface.set_at((nx, y), color)
                
                # Vertical line
                for dy in range(-2, 3):
                    ny = y + dy
                    if 0 <= ny < self.img_height:
                        surface.set_at((x, ny), color)
        
        return surface
    
    def render_grid_view(self, scans: Dict[int, LayerScan]) -> pygame.Surface:
        """Render all layers in grid."""
        cols = 8
        rows = math.ceil(self.num_layers / cols)
        pad = 2
        
        grid_w = cols * self.img_width + (cols - 1) * pad
        grid_h = rows * self.img_height + (rows - 1) * pad
        
        surface = pygame.Surface((grid_w, grid_h))
        surface.fill(DARK_GRAY)
        
        colormap = self.COLORMAPS[self.colormap_idx]
        
        for layer_idx in range(self.num_layers):
            row, col = divmod(layer_idx, cols)
            x = col * (self.img_width + pad)
            y = row * (self.img_height + pad)
            
            if layer_idx in scans:
                layer_surf = self.render_layer_surface(scans[layer_idx].activations, colormap)
                surface.blit(layer_surf, (x, y))
                
                # Highlight current layer
                if layer_idx == self.current_layer:
                    pygame.draw.rect(surface, YELLOW, (x-1, y-1, self.img_width+2, self.img_height+2), 2)
            else:
                # Empty placeholder
                pygame.draw.rect(surface, GRAY, (x, y, self.img_width, self.img_height))
        
        return surface
    
    def compute_histogram(self, activations: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram of activations."""
        hist, edges = np.histogram(activations, bins=bins)
        return hist, edges
    
    def draw_histogram(self, activations: np.ndarray, rect: pygame.Rect):
        """Draw histogram overlay."""
        hist, edges = self.compute_histogram(activations)
        max_count = hist.max() if hist.max() > 0 else 1
        
        # Background
        pygame.draw.rect(self.screen, (0, 0, 0, 180), rect)
        pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # Bars
        bar_width = rect.width / len(hist)
        for i, count in enumerate(hist):
            bar_height = int((count / max_count) * (rect.height - 20))
            x = rect.x + i * bar_width
            y = rect.y + rect.height - bar_height - 10
            
            # Color based on position (value range)
            t = i / len(hist)
            color = self.apply_colormap(t, self.COLORMAPS[self.colormap_idx])
            pygame.draw.rect(self.screen, color, (x, y, max(1, bar_width - 1), bar_height))
        
        # Labels
        min_label = self.font.render(f"{edges[0]:.2f}", True, WHITE)
        max_label = self.font.render(f"{edges[-1]:.2f}", True, WHITE)
        self.screen.blit(min_label, (rect.x + 5, rect.y + rect.height - 15))
        self.screen.blit(max_label, (rect.x + rect.width - 50, rect.y + rect.height - 15))
    
    def draw_stats_panel(self, scan: Optional[LayerScan], x: int, y: int):
        """Draw statistics panel."""
        panel_width = 220
        panel_height = 180
        
        # Background
        panel_rect = pygame.Rect(x, y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), panel_rect)
        pygame.draw.rect(self.screen, GRAY, panel_rect, 1)
        
        # Title
        title = self.font_large.render(f"Layer {self.current_layer}", True, CYAN)
        self.screen.blit(title, (x + 10, y + 10))
        
        if scan:
            stats = scan.stats
            
            # Find brightest neuron
            acts = scan.activations
            max_idx = int(np.argmax(acts))
            max_val = acts[max_idx]
            max_x = max_idx % self.img_width
            max_y = max_idx // self.img_width
            
            lines = [
                f"Turn: {scan.turn}  Token: {scan.token_idx}",
                f"Min:  {stats['min']:+.4f}",
                f"Max:  {stats['max']:+.4f}",
                f"Mean: {stats['mean']:+.4f}",
                f"Std:  {stats['std']:.4f}",
                f"",
                f"Brightest: #{max_idx}",
                f"  Pos: ({max_x}, {max_y})  Val: {max_val:.2f}",
            ]
            
            for i, line in enumerate(lines):
                color = CYAN if i >= 6 else WHITE
                text = self.font.render(line, True, color)
                self.screen.blit(text, (x + 10, y + 45 + i * 16))
        else:
            text = self.font.render("No data", True, GRAY)
            self.screen.blit(text, (x + 10, y + 50))
    
    def draw_controls_help(self):
        """Draw controls help at bottom."""
        samples = max(self.hebbian_samples_collected, self.circlepack_samples)
        
        if self.circlepack_mode:
            controls = [
                "Click: Select | Double-click/Enter: Zoom in | Esc: Out | B: Boost | X: Ablate",
                f"[ ]: Adjust ({self.circlepack_boost_amount:.1f}x) | \\: Clear | Shift+Wheel: Zoom | Q: Quit"
            ]
        else:
            controls = [
                f"↑↓: Layer | Shift+Wheel: Zoom | C: Colormap | L: Log | P: Pctl | Samples: {samples}",
                "H: Hebbian | O: Circles | R: Reset/Rebuild | F: FFT | N: Influence | Q: Quit"
            ]
        
        y = self.height - 40
        for line in controls:
            text = self.font.render(line, True, GRAY)
            self.screen.blit(text, (10, y))
            y += 18
    
    def draw_notifications(self):
        """Draw recent notifications."""
        if self.connection is None:
            return
        
        now = time.time()
        y = 100
        
        # Draw notifications (fade out after 3 seconds)
        for ts, msg, color_name in list(self.connection.notifications):
            age = now - ts
            if age > 5.0:
                continue
            
            # Fade out
            alpha = max(0, min(255, int(255 * (1 - age / 5.0))))
            
            if color_name == "green":
                color = (0, 255, 0)
            elif color_name == "yellow":
                color = (255, 255, 0)
            elif color_name == "red":
                color = (255, 100, 100)
            else:
                color = (255, 255, 255)
            
            # Render with alpha (create surface)
            text_surface = self.font.render(msg, True, color)
            text_surface.set_alpha(alpha)
            self.screen.blit(text_surface, (self.width - text_surface.get_width() - 20, y))
            y += 20
    
    def draw_status_bar(self):
        """Draw top status bar."""
        # Connection status
        if self.connection and self.connection.connected:
            status = f"● Connected | Turn {self.connection.current_turn} | Token {self.connection.current_token} | Scans: {self.connection.total_scans}"
            color = GREEN
            
            # Add intervention indicator
            if self.connection.intervention_confirmed:
                status += " | 🔧 STEERING"
        else:
            status = "○ Disconnected"
            color = RED
        
        text = self.font.render(status, True, color)
        self.screen.blit(text, (10, 10))
        
        # Mode indicators
        modes = []
        if self.paused:
            modes.append("PAUSED")
        if self.diff_mode:
            modes.append("DIFF")
        if self.show_grid:
            modes.append("GRID")
        if self.spectral_mode:
            modes.append(f"SPECTRAL ({self.spectral_method})")
        if self.influence_mode:
            modes.append("INFLUENCE")
        if self.hebbian_mode:
            modes.append("HEBBIAN")
        if self.circlepack_mode:
            modes.append("CIRCLES")
        if self.log_scale:
            modes.append("LOG")
        if self.percentile_clip:
            modes.append("P1-99")
        if not self.auto_window:
            modes.append(f"W:{self.window_width:.1f} L:{self.window_level:.1f}")
        
        mode_text = " | ".join(modes) if modes else ""
        if mode_text:
            text = self.font.render(mode_text, True, YELLOW)
            self.screen.blit(text, (self.width - 200, 10))
        
        # Colormap
        cmap_text = f"Colormap: {self.COLORMAPS[self.colormap_idx]}"
        text = self.font.render(cmap_text, True, WHITE)
        self.screen.blit(text, (self.width // 2 - 50, 10))
    
    def draw_wl_overlay(self):
        """Draw window/level overlay when adjusting."""
        if not self.wl_dragging:
            return
        
        # Semi-transparent background
        overlay = pygame.Surface((220, 90), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        
        x = self.width // 2 - 110
        y = self.height // 2 - 45
        self.screen.blit(overlay, (x, y))
        
        # Labels
        title = self.font_large.render("Window / Level", True, CYAN)
        self.screen.blit(title, (x + 35, y + 8))
        
        width_text = self.font.render(f"Width: {self.window_width:.2f}", True, WHITE)
        level_text = self.font.render(f"Level: {self.window_level:.2f}", True, WHITE)
        
        self.screen.blit(width_text, (x + 20, y + 38))
        self.screen.blit(level_text, (x + 120, y + 38))
        
        # Hint
        hint = self.font.render("drag: horiz=width, vert=level", True, GRAY)
        self.screen.blit(hint, (x + 15, y + 65))
    
    def feed_hebbian(self, activations: np.ndarray):
        """Feed activations to Hebbian layout for learning."""
        if self.hebbian_layout is None:
            return
        
        self.hebbian_layout.observe(activations)
        self.hebbian_samples_collected = len(self.hebbian_layout.observations)
    
    def learn_hebbian(self, method: str = "pca"):
        """Learn Hebbian layout from collected samples."""
        if self.hebbian_layout is None:
            print("[Viewer] Hebbian layout not available")
            return
        
        if len(self.hebbian_layout.observations) < 20:
            print(f"[Viewer] Need more samples ({len(self.hebbian_layout.observations)}/20 minimum)")
            return
        
        print(f"[Viewer] Learning Hebbian layout from {len(self.hebbian_layout.observations)} samples...")
        
        if method == "umap":
            self.hebbian_layout.learn_layout_umap()
        elif method == "tsne":
            self.hebbian_layout.learn_layout_tsne()
        else:
            self.hebbian_layout.learn_layout_pca()
        
        self.hebbian_mode = True
        print("[Viewer] Hebbian layout learned! Press H to toggle view.")
    
    def render_hebbian_surface(self, activations: np.ndarray) -> pygame.Surface:
        """Render activations using Hebbian layout."""
        if self.hebbian_layout is None or not self.hebbian_layout.layout_learned:
            # Fallback to grid
            return self.render_layer_surface(activations, self.COLORMAPS[self.colormap_idx])
        
        # Get RGB image from Hebbian layout
        colormap = self.COLORMAPS[self.colormap_idx]
        rgb = self.hebbian_layout.render(activations, colormap)
        
        # Convert to pygame surface
        h, w = rgb.shape[:2]
        surface = pygame.Surface((w, h))
        
        # Copy pixels
        pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
        
        return surface
    
    def feed_circlepack(self, activations: np.ndarray):
        """Feed activations to circle packing hierarchy."""
        if self.circlepack is None:
            return
        
        self.circlepack.observe(activations)
        self.circlepack_samples = len(self.circlepack.observations)
    
    def build_circlepack(self):
        """Build circle packing hierarchy from collected samples."""
        if self.circlepack is None:
            print("[Viewer] Circle packing not available")
            return
        
        if len(self.circlepack.observations) < 20:
            print(f"[Viewer] Need more samples ({len(self.circlepack.observations)}/20 minimum)")
            return
        
        print(f"[Viewer] Building circle hierarchy from {len(self.circlepack.observations)} samples...")
        self.circlepack.build_hierarchy()
        self.circlepack_mode = True
        print("[Viewer] Circle hierarchy built! Press O to toggle view.")
    
    def render_circlepack_surface(self, activations: np.ndarray) -> pygame.Surface:
        """Render activations using hierarchical circle packing."""
        if self.circlepack is None or not self.circlepack.hierarchy_built:
            return self.render_layer_surface(activations, self.COLORMAPS[self.colormap_idx])
        
        # Higher base resolution
        size = 800
        rgb = self.circlepack.render(activations, size=size)
        
        # Convert to pygame surface
        surface = pygame.Surface((size, size))
        pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
        
        return surface
    
    def handle_circlepack_click(self, pos: Tuple[int, int]):
        """Handle click in circlepack mode."""
        if self.circlepack is None or self.circlepack_image_rect is None:
            return
        
        rect = self.circlepack_image_rect
        
        # Check for double-click
        current_time = time.time()
        is_double_click = (
            current_time - self.last_click_time < 0.4 and
            abs(pos[0] - self.last_click_pos[0]) < 10 and
            abs(pos[1] - self.last_click_pos[1]) < 10
        )
        self.last_click_time = current_time
        self.last_click_pos = pos
        
        # Convert screen coords to normalized 0-1 coords
        if not rect.collidepoint(pos):
            # Click outside - deselect
            if self.circlepack.selected_node:
                self.circlepack.selected_node.selected = False
                self.circlepack.selected_node = None
            return
        
        # Normalize to 0-1
        x = (pos[0] - rect.x) / rect.width
        y = (pos[1] - rect.y) / rect.height
        
        # Debug: show click position
        print(f"[Viewer] Click at normalized ({x:.3f}, {y:.3f})")
        
        # Select cluster
        node = self.circlepack.select_at_point(x, y)
        if node:
            print(f"[Viewer] Selected cluster {node.label} ({node.neuron_count} neurons) at ({node.x:.3f}, {node.y:.3f}) r={node.radius:.3f}")
            
            # Double-click: zoom in
            if is_double_click and node.children:
                self.circlepack.zoom_into_selected()
                print(f"[Viewer] Zoomed into {self.circlepack.view_root.label}")
        else:
            print(f"[Viewer] No cluster found at ({x:.3f}, {y:.3f})")
    
    def update_circlepack_hover(self, pos: Tuple[int, int]):
        """Update hover state for circlepack."""
        if self.circlepack is None or self.circlepack_image_rect is None:
            return
        
        rect = self.circlepack_image_rect
        
        if not rect.collidepoint(pos):
            # Outside - clear hover
            if self.circlepack.hovered_node:
                self.circlepack.hovered_node.hovered = False
                self.circlepack.hovered_node = None
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            return
        
        # Normalize to 0-1
        x = (pos[0] - rect.x) / rect.width
        y = (pos[1] - rect.y) / rect.height
        
        # Update hover
        node = self.circlepack.hover_at_point(x, y)
        if node:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
    
    def circlepack_boost(self, amount: float = None):
        """Boost selected cluster."""
        if self.circlepack is None or self.circlepack.selected_node is None:
            print("[Viewer] No cluster selected")
            return
        
        mult = amount if amount else self.circlepack_boost_amount
        node = self.circlepack.selected_node
        self.circlepack.set_intervention(node, mult)
        print(f"[Viewer] Boosting {node.label}: {node.neuron_count} neurons × {mult:.2f}")
        self.apply_circlepack_interventions()
    
    def circlepack_ablate(self, amount: float = 0.0):
        """Ablate selected cluster (0 = full ablation)."""
        if self.circlepack is None or self.circlepack.selected_node is None:
            print("[Viewer] No cluster selected")
            return
        
        node = self.circlepack.selected_node
        self.circlepack.set_intervention(node, amount)
        print(f"[Viewer] ABLATING {node.label}: {node.neuron_count} neurons → {amount}")
        if amount == 0:
            print(f"         WARNING: Full ablation of {node.neuron_count} neurons!")
        self.apply_circlepack_interventions()
    
    def circlepack_clear_intervention(self):
        """Clear intervention on selected cluster or all."""
        if self.circlepack is None:
            return
        
        if self.circlepack.selected_node:
            self.circlepack.clear_intervention(self.circlepack.selected_node)
        else:
            self.circlepack.clear_intervention()
        
        # Send to Anima
        self.apply_circlepack_interventions()
    
    def apply_circlepack_interventions(self):
        """Send circlepack interventions to Anima."""
        if self.circlepack is None:
            return
        
        # Get all neurons with interventions
        neuron_mults = self.circlepack.get_intervention_neurons()
        
        if not neuron_mults:
            # Clear all
            if self.connection and self.connection.connected:
                self.connection.clear_interventions()
                self.connection.intervention_confirmed = False
                print("[Viewer] Cleared all interventions")
            return
        
        # Send to Anima - use layer 22 where samples were collected
        # TODO: track sample layer dynamically
        target_layer = 22
        if self.connection and self.connection.connected:
            if self.connection.send_interventions(neuron_mults, layer=target_layer):
                print(f"[Viewer] Sent interventions for {len(neuron_mults)} neurons to layer {target_layer}")
            else:
                print("[Viewer] Failed to send interventions")
        else:
            print("[Viewer] Not connected - interventions not sent")
    
    def draw_circlepack_info(self):
        """Draw cluster info panel for circlepack mode."""
        if self.circlepack is None:
            return
        
        # Panel background
        panel_x = 10
        panel_y = 70
        panel_w = 280
        panel_h = 220
        
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        self.screen.blit(panel, (panel_x, panel_y))
        
        y = panel_y + 10
        x = panel_x + 10
        
        # View info
        view_label = self.circlepack.view_root.label if self.circlepack.view_root else "All"
        view_neurons = self.circlepack.view_root.neuron_count if self.circlepack.view_root else 0
        text = self.font.render(f"View: {view_label} ({view_neurons} neurons)", True, WHITE)
        self.screen.blit(text, (x, y))
        y += 22
        
        # Selected or hovered cluster info
        node = self.circlepack.selected_node or self.circlepack.hovered_node
        if node:
            label_color = YELLOW if self.circlepack.selected_node else CYAN
            label_text = "Selected" if self.circlepack.selected_node else "Hover"
            text = self.font.render(f"{label_text}: {node.label}", True, label_color)
            self.screen.blit(text, (x, y))
            y += 18
            
            text = self.font.render(f"  Neurons: {node.neuron_count}", True, WHITE)
            self.screen.blit(text, (x, y))
            y += 18
            
            text = self.font.render(f"  Activation: {node.activation:.3f}", True, WHITE)
            self.screen.blit(text, (x, y))
            y += 18
            
            text = self.font.render(f"  Children: {len(node.children)}", True, WHITE)
            self.screen.blit(text, (x, y))
            y += 18
            
            # Intervention status
            if node.label in self.circlepack.cluster_interventions:
                mult = self.circlepack.cluster_interventions[node.label]
                color = GREEN if mult > 1 else RED
                text = self.font.render(f"  Intervention: {mult:.2f}x", True, color)
            else:
                text = self.font.render(f"  Intervention: none", True, GRAY)
            self.screen.blit(text, (x, y))
            y += 22
        else:
            text = self.font.render("Hover over a cluster", True, GRAY)
            self.screen.blit(text, (x, y))
            y += 40
        
        # Controls help
        y = panel_y + panel_h - 55
        text = self.font.render(f"B: Boost ({self.circlepack_boost_amount:.1f}x)  X: Ablate", True, CYAN)
        self.screen.blit(text, (x, y))
        y += 18
        text = self.font.render("Double-click/Enter: Zoom in", True, CYAN)
        self.screen.blit(text, (x, y))
        y += 18
        text = self.font.render("Shift+wheel: Zoom  \\: Clear", True, CYAN)
        self.screen.blit(text, (x, y))
        
        # Active interventions count
        n_interventions = len(self.circlepack.cluster_interventions)
        if n_interventions > 0:
            text = self.font.render(f"Active interventions: {n_interventions}", True, YELLOW)
            self.screen.blit(text, (self.width - 200, 70))
    
    def draw_layer_slider(self, scans: Dict[int, LayerScan]):
        """Draw vertical layer slider on right side."""
        slider_x = self.width - 40
        slider_y = 50
        slider_height = self.height - 150
        
        # Background
        pygame.draw.rect(self.screen, DARK_GRAY, (slider_x - 5, slider_y, 30, slider_height))
        
        # Layer markers
        for layer in range(self.num_layers):
            y = slider_y + int((layer / (self.num_layers - 1)) * slider_height)
            
            # Color based on whether we have data
            if layer in scans:
                color = GREEN if layer != self.current_layer else YELLOW
            else:
                color = GRAY
            
            # Current layer is larger
            if layer == self.current_layer:
                pygame.draw.circle(self.screen, color, (slider_x + 10, y), 8)
                label = self.font.render(f"L{layer}", True, WHITE)
                self.screen.blit(label, (slider_x - 35, y - 7))
            else:
                pygame.draw.circle(self.screen, color, (slider_x + 10, y), 3)
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                
                elif event.key == pygame.K_ESCAPE:
                    # Escape: zoom out in circlepack, otherwise quit
                    if self.circlepack_mode and self.circlepack:
                        if self.circlepack.view_root != self.circlepack.root:
                            self.circlepack.zoom_out()
                            print(f"[Viewer] Zoomed out to {self.circlepack.view_root.label}")
                            self.needs_redraw = True
                        else:
                            # At root, exit circlepack mode
                            self.circlepack_mode = False
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                            print("[Viewer] Exited circle mode")
                    else:
                        return False
                
                elif event.key == pygame.K_UP:
                    self.current_layer = max(0, self.current_layer - 1)
                    self.needs_redraw = True
                
                elif event.key == pygame.K_DOWN:
                    self.current_layer = min(self.num_layers - 1, self.current_layer + 1)
                    self.needs_redraw = True
                
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom = min(20, self.zoom + 1)
                    self.needs_redraw = True
                
                elif event.key == pygame.K_MINUS:
                    self.zoom = max(1, self.zoom - 1)
                    self.needs_redraw = True
                
                elif event.key == pygame.K_c:
                    self.colormap_idx = (self.colormap_idx + 1) % len(self.COLORMAPS)
                    self.needs_redraw = True
                
                elif event.key == pygame.K_d:
                    self.diff_mode = not self.diff_mode
                    self.needs_redraw = True
                
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                    self.needs_redraw = True
                
                elif event.key == pygame.K_h:
                    self.show_histogram = not self.show_histogram
                
                elif event.key == pygame.K_t:
                    self.show_stats = not self.show_stats
                
                elif event.key == pygame.K_b:
                    self.show_hotspots = not self.show_hotspots
                    self.needs_redraw = True
                
                elif event.key == pygame.K_i:
                    self.show_neuron_info = not self.show_neuron_info
                
                elif event.key == pygame.K_l:
                    # Toggle log scale
                    self.log_scale = not self.log_scale
                    self.needs_redraw = True
                    print(f"[Viewer] Log scale: {'ON' if self.log_scale else 'OFF'}")
                
                elif event.key == pygame.K_p:
                    # Toggle percentile clipping
                    self.percentile_clip = not self.percentile_clip
                    self.needs_redraw = True
                    print(f"[Viewer] Percentile clip (1-99%): {'ON' if self.percentile_clip else 'OFF'}")
                
                elif event.key == pygame.K_w:
                    # Reset to auto window
                    self.auto_window = True
                    self.needs_redraw = True
                    print("[Viewer] Window reset to auto (right-drag to adjust)")
                
                elif event.key == pygame.K_f:
                    # Toggle spectral/FFT mode
                    self.spectral_mode = not self.spectral_mode
                    if self.spectral_mode:
                        self.influence_mode = False  # Disable influence
                    self.needs_redraw = True
                    print(f"[Viewer] Spectral mode: {'ON' if self.spectral_mode else 'OFF'}")
                
                elif event.key == pygame.K_m:
                    # Cycle spectral method
                    methods = ["variance", "magnitude", "position"]
                    idx = methods.index(self.spectral_method)
                    self.spectral_method = methods[(idx + 1) % len(methods)]
                    print(f"[Viewer] Spectral method: {self.spectral_method}")
                
                elif event.key == pygame.K_x:
                    # Clear spectral history
                    self.spectral_history.clear()
                    self.cached_influence = None
                    print("[Viewer] Spectral history cleared")
                
                elif event.key == pygame.K_e:
                    # Export analysis data
                    self.export_analysis()
                
                elif event.key == pygame.K_h:
                    # H: Toggle Hebbian mode / learn layout
                    if self.hebbian_layout is None:
                        print("[Viewer] Hebbian not available")
                    elif not self.hebbian_layout.layout_learned:
                        # Learn layout
                        self.learn_hebbian("pca")
                    else:
                        # Toggle mode
                        self.hebbian_mode = not self.hebbian_mode
                        if self.hebbian_mode:
                            self.circlepack_mode = False
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                        print(f"[Viewer] Hebbian mode: {'ON' if self.hebbian_mode else 'OFF'}")
                    self.needs_redraw = True
                
                elif event.key == pygame.K_o:
                    # O: Toggle circle pack mode / build hierarchy
                    if self.circlepack is None:
                        print("[Viewer] Circle packing not available")
                    elif not self.circlepack.hierarchy_built:
                        # Build hierarchy
                        self.build_circlepack()
                    else:
                        # Toggle mode
                        self.circlepack_mode = not self.circlepack_mode
                        if self.circlepack_mode:
                            self.hebbian_mode = False
                        print(f"[Viewer] Circle pack mode: {'ON' if self.circlepack_mode else 'OFF'}")
                    self.needs_redraw = True
                
                elif event.key == pygame.K_r:
                    # R: Rebuild hierarchy (if in circle mode) or reset view
                    if self.circlepack_mode and self.circlepack:
                        print("[Viewer] Rebuilding circle hierarchy...")
                        self.circlepack.hierarchy_built = False
                        self.build_circlepack()
                    else:
                        # Reset view
                        self.pan_x = 0
                        self.pan_y = 0
                        self.zoom = 8
                    self.needs_redraw = True
                
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    # Enter: Zoom into selected cluster
                    if self.circlepack_mode and self.circlepack:
                        if self.circlepack.zoom_into_selected():
                            print(f"[Viewer] Zoomed into {self.circlepack.view_root.label}")
                    self.needs_redraw = True
                
                elif event.key == pygame.K_BACKSPACE:
                    # Backspace: Zoom out in circle mode
                    if self.circlepack_mode and self.circlepack:
                        if self.circlepack.zoom_out():
                            print(f"[Viewer] Zoomed out to {self.circlepack.view_root.label}")
                    self.needs_redraw = True
                
                elif event.key == pygame.K_b:
                    # B: Boost selected cluster
                    if self.circlepack_mode:
                        self.circlepack_boost(1.5)
                    self.needs_redraw = True
                
                elif event.key == pygame.K_x:
                    # X: Ablate selected cluster (zero out)
                    if self.circlepack_mode:
                        self.circlepack_ablate(0.0)
                    self.needs_redraw = True
                
                elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSLASH:
                    # Delete or \: Clear intervention on selected or all
                    if self.circlepack_mode:
                        self.circlepack_clear_intervention()
                    self.needs_redraw = True
                
                elif event.key == pygame.K_LEFTBRACKET:
                    # [: Decrease boost amount
                    if self.circlepack_mode:
                        self.circlepack_boost_amount = max(0.1, self.circlepack_boost_amount - 0.1)
                        print(f"[Viewer] Boost amount: {self.circlepack_boost_amount:.1f}x")
                
                elif event.key == pygame.K_RIGHTBRACKET:
                    # ]: Increase boost amount
                    if self.circlepack_mode:
                        self.circlepack_boost_amount = min(3.0, self.circlepack_boost_amount + 0.1)
                        print(f"[Viewer] Boost amount: {self.circlepack_boost_amount:.1f}x")
                
                elif event.key == pygame.K_n:
                    # Toggle influence/connectivity mode
                    self.influence_mode = not self.influence_mode
                    if self.influence_mode:
                        self.spectral_mode = False  # Disable other modes
                    self.cached_influence = None  # Force recompute
                    print(f"[Viewer] Influence mode: {'ON' if self.influence_mode else 'OFF'}")
                
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                elif event.key == pygame.K_s:
                    self.save_screenshot()
                
                elif event.key in range(pygame.K_1, pygame.K_9 + 1):
                    # 1-9 jump to layer positions
                    idx = event.key - pygame.K_1
                    self.current_layer = min(self.num_layers - 1, int(idx * self.num_layers / 9))
                    self.needs_redraw = True
            
            elif event.type == pygame.MOUSEWHEEL:
                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT or mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META:
                    # Shift/Ctrl/Cmd + wheel: Zoom
                    self.zoom = max(1, min(20, self.zoom + event.y))
                else:
                    # Navigate layers
                    self.current_layer = max(0, min(self.num_layers - 1, self.current_layer - event.y))
                self.needs_redraw = True
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_META or mods & pygame.KMOD_CTRL:
                        # Cmd+click (Mac) or Ctrl+click - window/level
                        self.wl_dragging = True
                        self.wl_drag_start = event.pos
                        # Initialize from current data if auto mode
                        if self.auto_window and self.last_activations is not None:
                            min_val = float(self.last_activations.min())
                            max_val = float(self.last_activations.max())
                            self.window_level = (min_val + max_val) / 2
                            self.window_width = max(0.1, max_val - min_val)
                            self.auto_window = False
                        self.wl_start_level = self.window_level
                        self.wl_start_width = self.window_width
                    else:
                        # Regular left click - pan/select
                        self.dragging = True
                        self.drag_start = event.pos
                        self.drag_moved = False
                elif event.button == 3:  # Right click also works
                    self.wl_dragging = True
                    self.wl_drag_start = event.pos
                    if self.auto_window and self.last_activations is not None:
                        min_val = float(self.last_activations.min())
                        max_val = float(self.last_activations.max())
                        self.window_level = (min_val + max_val) / 2
                        self.window_width = max(0.1, max_val - min_val)
                        self.auto_window = False
                    self.wl_start_level = self.window_level
                    self.wl_start_width = self.window_width
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if self.wl_dragging:
                        self.wl_dragging = False
                    elif not self.drag_moved:
                        # This was a click, not a drag
                        if self.circlepack_mode and self.circlepack and self.circlepack_image_rect:
                            # Circle pack click - select cluster
                            self.handle_circlepack_click(event.pos)
                        else:
                            # Regular click - toggle neuron selection
                            neuron_idx = self.screen_to_neuron(*event.pos)
                            if neuron_idx is not None:
                                if self.selected_neuron == neuron_idx:
                                    self.selected_neuron = None  # Deselect
                                else:
                                    self.selected_neuron = neuron_idx
                                    print(f"[Viewer] Selected neuron {neuron_idx}")
                            else:
                                self.selected_neuron = None
                    self.dragging = False
                elif event.button == 3:
                    self.wl_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                # Update hover neuron (for grid mode)
                neuron_idx = self.screen_to_neuron(*event.pos)
                if neuron_idx != self.hover_neuron_idx:
                    self.hover_neuron_idx = neuron_idx
                    self.hover_pos = event.pos
                
                # Update circlepack hover
                if self.circlepack_mode and self.circlepack and self.circlepack_image_rect:
                    self.update_circlepack_hover(event.pos)
                
                if self.dragging:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    if abs(dx) > 3 or abs(dy) > 3:
                        self.drag_moved = True
                    self.pan_x += dx
                    self.pan_y += dy
                    self.drag_start = event.pos
                    self.needs_redraw = True
                
                elif self.wl_dragging:
                    # Right-drag: horizontal = width, vertical = level
                    dx = event.pos[0] - self.wl_drag_start[0]
                    dy = event.pos[1] - self.wl_drag_start[1]
                    
                    # Sensitivity scales with current values
                    width_sensitivity = self.wl_start_width * 0.005
                    level_sensitivity = self.wl_start_width * 0.005
                    
                    # Horizontal = window width (drag right = wider)
                    self.window_width = max(0.01, self.wl_start_width + dx * width_sensitivity)
                    
                    # Vertical = window level (drag up = higher, note pygame y is inverted)
                    self.window_level = self.wl_start_level - dy * level_sensitivity
                    
                    self.needs_redraw = True
            
            elif event.type == pygame.VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self.needs_redraw = True
        
        return True
    
    def save_screenshot(self):
        """Save current view as screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.screenshot_dir, f"mri_L{self.current_layer}_{timestamp}.png")
        pygame.image.save(self.screen, path)
        print(f"Screenshot saved: {path}")
    
    def run(self):
        """Main loop."""
        running = True
        
        while running:
            running = self.handle_events()
            
            # Clear screen
            self.screen.fill(BLACK)
            
            # Get current scans
            scans = {}
            prev_scans = {}
            if self.connection:
                if not self.paused:
                    scans = self.connection.get_scans()
                    prev_scans = self.connection.get_previous_scans()
                    
                    # Always update spectral history (even when not in spectral mode)
                    for layer_idx, scan in scans.items():
                        self.update_spectral_history(layer_idx, scan.activations)
                        
                        # Feed Hebbian layout (collect from layer 22)
                        if layer_idx == 22 and self.hebbian_layout is not None:
                            self.feed_hebbian(scan.activations)
                        
                        # Feed circle packing (collect from layer 22)
                        if layer_idx == 22 and self.circlepack is not None:
                            self.feed_circlepack(scan.activations)
                else:
                    scans = self.connection.get_scans()  # Use cached
            
            # Main view
            if self.influence_mode:
                # Influence/connectivity heatmap
                # Recompute if layer changed or no cache
                if self.cached_influence is None or self.influence_layer_cache != self.current_layer:
                    self.cached_influence = self.compute_influence_map(self.current_layer)
                    self.influence_layer_cache = self.current_layer
                
                if self.cached_influence is not None:
                    surface = self.render_influence_surface(self.cached_influence)
                    
                    # Scale
                    scaled_w = self.img_width * self.zoom
                    scaled_h = self.img_height * self.zoom
                    scaled = pygame.transform.scale(surface, (scaled_w, scaled_h))
                    
                    # Center with pan offset
                    x = (self.width - scaled_w) // 2 + self.pan_x
                    y = (self.height - scaled_h) // 2 + self.pan_y
                    self.screen.blit(scaled, (x, y))
                    
                    # Update image rect for mouse mapping
                    self.image_rect = pygame.Rect(x, y, scaled_w, scaled_h)
                    
                    # Title overlay
                    n_samples = len(self.spectral_history.get(self.current_layer, []))
                    title = f"INFLUENCE MAP - Layer {self.current_layer} ({n_samples} samples)"
                    title_surf = self.font_large.render(title, True, YELLOW)
                    self.screen.blit(title_surf, (10, 40))
                    
                    # Show top hubs
                    if self.cached_influence is not None:
                        top_indices = np.argsort(self.cached_influence)[-5:][::-1]
                        hub_text = "Top hubs: " + ", ".join(f"{i}" for i in top_indices)
                        hub_surf = self.font.render(hub_text, True, WHITE)
                        self.screen.blit(hub_surf, (10, 70))
                else:
                    msg = f"Need 10+ samples (have {len(self.spectral_history.get(self.current_layer, []))})"
                    msg_surf = self.font_large.render(msg, True, GRAY)
                    self.screen.blit(msg_surf, (self.width//2 - 150, self.height//2))
            
            elif self.spectral_mode:
                # Spectral analysis view
                spectral_surface = self.render_spectral_view(scans)
                
                # Center it
                x = (self.width - spectral_surface.get_width()) // 2
                y = (self.height - spectral_surface.get_height()) // 2
                self.screen.blit(spectral_surface, (x, y))
                
            elif self.show_grid:
                # Grid view of all layers
                grid_surface = self.render_grid_view(scans)
                
                # Scale and center
                scaled_w = int(grid_surface.get_width() * self.zoom / 4)
                scaled_h = int(grid_surface.get_height() * self.zoom / 4)
                scaled = pygame.transform.scale(grid_surface, (scaled_w, scaled_h))
                
                x = (self.width - scaled_w) // 2 + self.pan_x
                y = (self.height - scaled_h) // 2 + self.pan_y
                self.screen.blit(scaled, (x, y))
                
            else:
                # Single layer view
                colormap = self.COLORMAPS[self.colormap_idx]
                
                if self.current_layer in scans:
                    scan = scans[self.current_layer]
                    
                    # Store for window/level initialization
                    self.last_activations = scan.activations
                    
                    if self.hebbian_mode and self.hebbian_layout and self.hebbian_layout.layout_learned:
                        # Hebbian layout view
                        surface = self.render_hebbian_surface(scan.activations)
                        
                        # Scale to fit (Hebbian is 128x128)
                        scaled_w = 128 * self.zoom
                        scaled_h = 128 * self.zoom
                        scaled = pygame.transform.scale(surface, (scaled_w, scaled_h))
                        
                        # Center with pan offset
                        x = (self.width - scaled_w) // 2 + self.pan_x
                        y = (self.height - scaled_h) // 2 + self.pan_y
                        self.screen.blit(scaled, (x, y))
                        
                        # Title
                        title = f"HEBBIAN LAYOUT - Layer {self.current_layer} ({self.hebbian_samples_collected} samples)"
                        title_surf = self.font_large.render(title, True, CYAN)
                        self.screen.blit(title_surf, (10, 40))
                    
                    elif self.circlepack_mode and self.circlepack and self.circlepack.hierarchy_built:
                        # Circle packing hierarchy view
                        surface = self.render_circlepack_surface(scan.activations)
                        
                        # Zoom control - base size 800, zoom scales it
                        base_size = surface.get_width()
                        # Map zoom 1-20 to scale 0.5x to 2x
                        scale = 0.5 + (self.zoom - 1) * 0.08
                        scaled_w = int(base_size * scale)
                        scaled_h = int(base_size * scale)
                        scaled = pygame.transform.smoothscale(surface, (scaled_w, scaled_h))
                        
                        # Center with pan offset
                        x = (self.width - scaled_w) // 2 + self.pan_x
                        y = (self.height - scaled_h) // 2 + self.pan_y
                        self.screen.blit(scaled, (x, y))
                        
                        # Store rect for click mapping
                        self.circlepack_image_rect = pygame.Rect(x, y, scaled_w, scaled_h)
                        
                        # Title with navigation breadcrumb
                        path = self.circlepack.view_root.label if self.circlepack.view_root else "All"
                        title = f"CIRCLE HIERARCHY - {path} - Layer {self.current_layer}"
                        title_surf = self.font_large.render(title, True, CYAN)
                        self.screen.blit(title_surf, (10, 40))
                        
                        # Draw cluster info panel
                        self.draw_circlepack_info()
                        
                    else:
                        # Normal grid view
                        if self.diff_mode and self.current_layer in prev_scans:
                            # Show difference
                            prev = prev_scans[self.current_layer]
                            diff = scan.activations - prev.activations
                            surface = self.render_layer_surface(diff, "signed", mark_hotspots=self.show_hotspots)
                        else:
                            surface = self.render_layer_surface(scan.activations, colormap, mark_hotspots=self.show_hotspots)
                        
                        # Scale
                        scaled_w = self.img_width * self.zoom
                        scaled_h = self.img_height * self.zoom
                        scaled = pygame.transform.scale(surface, (scaled_w, scaled_h))
                        
                        # Center with pan offset
                        x = (self.width - scaled_w) // 2 + self.pan_x
                        y = (self.height - scaled_h) // 2 + self.pan_y
                        self.screen.blit(scaled, (x, y))
                        
                        # Update image rect for mouse mapping
                        self.image_rect = pygame.Rect(x, y, scaled_w, scaled_h)
                        
                        # Highlight selected/hovered neuron
                        highlight_idx = self.selected_neuron if self.selected_neuron else self.hover_neuron_idx
                        if highlight_idx is not None:
                            grid_x, grid_y = self.neuron_to_grid_pos(highlight_idx)
                            # Convert to screen coords
                            screen_x = x + int(grid_x * self.zoom)
                            screen_y = y + int(grid_y * self.zoom)
                            # Draw highlight box
                            color = CYAN if self.selected_neuron else YELLOW
                            pygame.draw.rect(self.screen, color, 
                                            (screen_x, screen_y, self.zoom, self.zoom), 2)
                    
                    # Histogram overlay (both modes)
                    if self.show_histogram:
                        hist_rect = pygame.Rect(50, self.height - 200, 300, 150)
                        self.draw_histogram(scan.activations, hist_rect)
                    
                    # Stats panel (both modes)
                    if self.show_stats:
                        self.draw_stats_panel(scan, self.width - 220, 50)
                    
                    # Neuron info panel
                    self.draw_neuron_info(scans)
                else:
                    # No data for this layer
                    text = self.font_large.render(f"Layer {self.current_layer} - No Data", True, GRAY)
                    self.screen.blit(text, (self.width // 2 - 100, self.height // 2))
                    
                    if self.show_stats:
                        self.draw_stats_panel(None, self.width - 220, 50)
            
            # UI overlays
            self.draw_status_bar()
            self.draw_layer_slider(scans)
            self.draw_controls_help()
            self.draw_notifications()
            self.draw_wl_overlay()  # Window/level indicator when dragging
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        # Cleanup
        if self.connection:
            self.connection.disconnect()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="LLMRI Viewer - Interactive Neural Activation Scanner")
    parser.add_argument("--host", default="localhost", help="MRI server host")
    parser.add_argument("--port", type=int, default=9999, help="MRI server port")
    parser.add_argument("--offline", action="store_true", help="Run without connecting")
    parser.add_argument("--width", type=int, default=1200, help="Window width")
    parser.add_argument("--height", type=int, default=800, help="Window height")
    args = parser.parse_args()
    
    viewer = MRIViewer(args.width, args.height)
    
    if not args.offline:
        print(f"Connecting to {args.host}:{args.port}...")
        if viewer.connect(args.host, args.port):
            print("Connected! Starting viewer...")
        else:
            print("Connection failed. Running in offline mode.")
    else:
        print("Running in offline mode.")
    
    viewer.run()


if __name__ == "__main__":
    main()