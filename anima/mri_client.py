#!/usr/bin/env python3
"""
LLMRI Client - Real-time neural activation visualization.

Connects to Anima's MRI server and renders live brain scans.

Usage:
    python mri_client.py --port 9999 --mode grid
    python mri_client.py --port 9999 --mode live --layer 20
    python mri_client.py --port 9999 --mode diff
"""

import socket
import struct
import json
import threading
import time
import math
import numpy as np
import argparse
import os
from typing import Dict, Optional, List
from datetime import datetime
from collections import deque
from dataclasses import dataclass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import colorsys
except ImportError:
    colorsys = None


@dataclass
class LayerScan:
    """Received scan data."""
    timestamp: float
    layer_idx: int
    turn: int
    token_idx: int
    activations: np.ndarray
    stats: Dict[str, float]


class MRIClient:
    """
    Connects to Anima's MRI server and receives activation streams.
    """
    
    def __init__(self, host: str = "localhost", port: int = 9999):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False
        
        # Received data
        self.layer_scans: Dict[int, LayerScan] = {}  # Latest scan per layer
        self.scan_history: deque = deque(maxlen=1000)
        self.current_turn = 0
        self.current_token = 0
        
        # Image rendering
        self.hidden_dim = 4608  # Gemma-27B default
        self.num_layers = 46
        self.img_width, self.img_height = self._compute_dimensions(self.hidden_dim)
        
        # Output
        self.output_dir = "llmri_live"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Callbacks
        self.on_scan: Optional[callable] = None
        self.on_turn: Optional[callable] = None
        self.on_disconnect: Optional[callable] = None
        
        self._lock = threading.Lock()
    
    def _compute_dimensions(self, hidden_dim: int):
        sqrt = int(math.sqrt(hidden_dim))
        for w in range(sqrt, 0, -1):
            if hidden_dim % w == 0:
                return (w, hidden_dim // w)
        side = int(math.ceil(math.sqrt(hidden_dim)))
        return (side, side)
    
    def connect(self) -> bool:
        """Connect to MRI server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"[MRI Client] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[MRI Client] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server."""
        self.running = False
        self.connected = False
        if self.socket:
            self.socket.close()
        print("[MRI Client] Disconnected")
    
    def start_receiving(self):
        """Start receiving data in background thread."""
        self.running = True
        self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._recv_thread.start()
    
    def _receive_loop(self):
        """Receive and process messages from server."""
        while self.running and self.connected:
            try:
                # Read header (4 bytes)
                header = self._recv_exactly(4)
                if not header:
                    break
                
                msg_len = struct.unpack('!I', header)[0]
                
                # Read body
                body = self._recv_exactly(msg_len)
                if not body:
                    break
                
                # Parse message
                data = json.loads(body.decode('utf-8'))
                self._handle_message(data)
                
            except Exception as e:
                if self.running:
                    print(f"[MRI Client] Receive error: {e}")
                break
        
        self.connected = False
        if self.on_disconnect:
            self.on_disconnect()
    
    def _recv_exactly(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def _handle_message(self, data: Dict):
        """Process received message."""
        msg_type = data.get("type")
        
        if msg_type == "scan":
            # Decode activations
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
                self.layer_scans[scan.layer_idx] = scan
                self.scan_history.append(scan)
                self.current_turn = scan.turn
                self.current_token = scan.token_idx
            
            if self.on_scan:
                self.on_scan(scan)
                
        elif msg_type == "turn_start":
            self.current_turn = data["turn"]
            self.current_token = 0
            
            with self._lock:
                self.layer_scans.clear()
            
            if self.on_turn:
                self.on_turn(data["turn"])
    
    def get_latest_scans(self) -> Dict[int, LayerScan]:
        """Get latest scan for each layer."""
        with self._lock:
            return dict(self.layer_scans)
    
    def render_layer(self, activations: np.ndarray, colormap: str = "viridis") -> 'Image':
        """Render activations as image."""
        if not HAS_PIL:
            raise RuntimeError("PIL not installed")
        
        total_pixels = self.img_width * self.img_height
        if len(activations) < total_pixels:
            activations = np.concatenate([activations, np.zeros(total_pixels - len(activations))])
        elif len(activations) > total_pixels:
            activations = activations[:total_pixels]
        
        acts_2d = activations.reshape(self.img_height, self.img_width)
        
        # Normalize
        min_val, max_val = acts_2d.min(), acts_2d.max()
        if max_val - min_val > 1e-8:
            normalized = (acts_2d - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(acts_2d)
        
        rgb = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        if colormap == "grayscale":
            gray = (normalized * 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif colormap == "hsv" and colorsys:
            for i in range(self.img_height):
                for j in range(self.img_width):
                    h = normalized[i, j] * 0.8
                    s = 0.9
                    v = 0.5 + normalized[i, j] * 0.5
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    rgb[i, j] = [int(r*255), int(g*255), int(b*255)]
        elif colormap == "signed":
            # For diffs: blue=neg, white=zero, red=pos
            abs_max = max(abs(acts_2d.min()), abs(acts_2d.max()))
            if abs_max > 1e-8:
                signed = acts_2d / abs_max
            else:
                signed = np.zeros_like(acts_2d)
            
            for i in range(self.img_height):
                for j in range(self.img_width):
                    v = signed[i, j]
                    if v < 0:
                        rgb[i, j] = [int((1+v)*255), int((1+v)*255), 255]
                    else:
                        rgb[i, j] = [255, int((1-v)*255), int((1-v)*255)]
        else:  # viridis-like
            for i in range(self.img_height):
                for j in range(self.img_width):
                    t = normalized[i, j]
                    r = max(0, min(1, 0.27 + t * 0.7))
                    g = max(0, min(1, 0.0 + t * 0.9))
                    b = max(0, min(1, 0.33 + t * 0.3 - t * t * 0.3))
                    rgb[i, j] = [int(r*255), int(g*255), int(b*255)]
        
        return Image.fromarray(rgb, mode='RGB')
    
    def render_grid(self, colormap: str = "viridis", scale: int = 6) -> Optional[str]:
        """Render all layers as grid."""
        scans = self.get_latest_scans()
        if not scans:
            return None
        
        cols = 8
        rows = math.ceil(self.num_layers / cols)
        pad = 2
        
        grid_w = cols * self.img_width + (cols - 1) * pad
        grid_h = rows * self.img_height + (rows - 1) * pad
        
        grid = Image.new('RGB', (grid_w, grid_h), color=(20, 20, 20))
        
        for layer_idx, scan in sorted(scans.items()):
            img = self.render_layer(scan.activations, colormap)
            
            row, col = divmod(layer_idx, cols)
            x = col * (self.img_width + pad)
            y = row * (self.img_height + pad)
            grid.paste(img, (x, y))
        
        grid = grid.resize((grid_w * scale, grid_h * scale), Image.Resampling.NEAREST)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"live_t{self.current_turn}_{timestamp}.png")
        grid.save(path)
        
        return path


class LiveViewer:
    """
    Real-time terminal visualization of neural activations.
    """
    
    def __init__(self, client: MRIClient, mode: str = "stats", target_layer: int = 20):
        self.client = client
        self.mode = mode
        self.target_layer = target_layer
        self.last_render = 0
        self.render_interval = 0.1  # 10 FPS max
        
        # Hook into client
        client.on_scan = self.on_scan
        client.on_turn = self.on_turn
    
    def on_scan(self, scan: LayerScan):
        """Handle incoming scan."""
        now = time.time()
        if now - self.last_render < self.render_interval:
            return
        self.last_render = now
        
        if self.mode == "stats":
            self._render_stats(scan)
        elif self.mode == "layer":
            if scan.layer_idx == self.target_layer:
                self._render_layer(scan)
        elif self.mode == "all":
            self._render_all()
    
    def on_turn(self, turn: int):
        """Handle new turn."""
        print(f"\n{'='*60}")
        print(f"  TURN {turn}")
        print(f"{'='*60}\n")
    
    def _render_stats(self, scan: LayerScan):
        """Show stats for scan."""
        s = scan.stats
        bar_len = int((s['mean'] + 5) / 10 * 40)  # Rough scale
        bar = '█' * max(0, min(40, bar_len))
        
        print(f"\rL{scan.layer_idx:2d} T{scan.token_idx:3d} | "
              f"μ={s['mean']:+.2f} σ={s['std']:.2f} | {bar}", end='', flush=True)
    
    def _render_layer(self, scan: LayerScan):
        """Render single layer as ASCII."""
        acts = scan.activations
        
        # Downsample to terminal width
        width = 60
        height = 10
        
        # Simple downsampling
        step = len(acts) // (width * height)
        downsampled = acts[::max(1, step)][:width * height]
        
        if len(downsampled) < width * height:
            downsampled = np.concatenate([downsampled, np.zeros(width * height - len(downsampled))])
        
        grid = downsampled.reshape(height, width)
        
        # Normalize
        min_v, max_v = grid.min(), grid.max()
        if max_v - min_v > 1e-8:
            norm = (grid - min_v) / (max_v - min_v)
        else:
            norm = np.zeros_like(grid)
        
        # ASCII render
        chars = ' ░▒▓█'
        print(f"\033[{height+2}A")  # Move cursor up
        print(f"Layer {scan.layer_idx} | Token {scan.token_idx}")
        for row in norm:
            line = ''.join(chars[min(4, int(v * 5))] for v in row)
            print(line)
        print()
    
    def _render_all(self):
        """Render grid summary."""
        scans = self.client.get_latest_scans()
        if not scans:
            return
        
        # One line per layer
        print("\033[48A")  # Move up
        for layer_idx in range(self.client.num_layers):
            scan = scans.get(layer_idx)
            if scan:
                s = scan.stats
                bar_len = int((s['mean'] + 5) / 10 * 30)
                bar = '█' * max(0, min(30, bar_len))
                print(f"L{layer_idx:2d} | {bar:<30} | μ={s['mean']:+.3f}")
            else:
                print(f"L{layer_idx:2d} | {'·'*30} |")


def main():
    parser = argparse.ArgumentParser(description="LLMRI Client - Real-time neural visualization")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--mode", choices=["stats", "layer", "all", "save"], default="stats",
                       help="Visualization mode")
    parser.add_argument("--layer", type=int, default=20, help="Target layer for 'layer' mode")
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument("--save-interval", type=float, default=1.0, 
                       help="Interval between saved images (seconds)")
    args = parser.parse_args()
    
    client = MRIClient(args.host, args.port)
    
    if not client.connect():
        return
    
    client.start_receiving()
    
    if args.mode == "save":
        # Periodically save grid images
        print(f"[Save mode] Saving to {client.output_dir}/ every {args.save_interval}s")
        try:
            while client.connected:
                time.sleep(args.save_interval)
                path = client.render_grid(args.colormap)
                if path:
                    print(f"  Saved: {path}")
        except KeyboardInterrupt:
            pass
    else:
        # Live terminal view
        viewer = LiveViewer(client, args.mode, args.layer)
        
        print(f"[Live mode: {args.mode}] Press Ctrl+C to stop")
        if args.mode == "all":
            # Print empty lines for cursor positioning
            for _ in range(48):
                print()
        
        try:
            while client.connected:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    
    client.disconnect()


if __name__ == "__main__":
    main()