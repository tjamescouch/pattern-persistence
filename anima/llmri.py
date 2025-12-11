#!/usr/bin/env python3
"""
LLMRI - Large Language Model Resonance Imaging
===============================================

Visualize neural network activations layer-by-layer like an MRI scan.
Captures hidden states at each layer and renders them as images.

Usage:
    python llmri.py --model google/gemma-2-27b-it --prompt "Hello, how are you?"
    python llmri.py --model ~/models/gemma-2-27b-it --prompt "I am conscious" --colormap hsv
    python llmri.py --model ~/models/gemma-2-27b-it --prompt "Hello" --animate --fps 10
"""

import torch
import argparse
import os
import sys
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import colorsys

from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class ScanConfig:
    """Configuration for a scan."""
    width: int = 64          # Image width
    height: int = 72         # Image height (64*72 = 4608 for Gemma)
    colormap: str = "viridis"  # viridis, grayscale, hsv, plasma, signed
    normalize: str = "layer"   # layer, global, none
    token_idx: int = -1        # Which token to visualize (-1 = last)
    

@dataclass  
class LayerScan:
    """Captured activation from one layer."""
    layer_idx: int
    activations: torch.Tensor  # (hidden_dim,) for single token
    min_val: float
    max_val: float
    mean_val: float
    std_val: float


class LLMRI:
    """Neural network MRI scanner."""
    
    def __init__(self, model, tokenizer, device: str = "mps"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size
        
        # Storage for captured activations
        self.layer_scans: List[LayerScan] = []
        self.hooks = []
        
        # Compute optimal image dimensions
        self.img_width, self.img_height = self._compute_dimensions(self.hidden_dim)
        print(f"[LLMRI] {self.num_layers} layers, {self.hidden_dim} hidden dim")
        print(f"[LLMRI] Image size: {self.img_width}x{self.img_height}")
    
    def _compute_dimensions(self, hidden_dim: int) -> Tuple[int, int]:
        """Compute optimal 2D dimensions for hidden_dim."""
        # Try to find factors close to square
        sqrt = int(math.sqrt(hidden_dim))
        for w in range(sqrt, 0, -1):
            if hidden_dim % w == 0:
                h = hidden_dim // w
                return (w, h)
        # Fallback: pad to square
        side = int(math.ceil(math.sqrt(hidden_dim)))
        return (side, side)
    
    def _make_hook(self, layer_idx: int):
        """Create a hook for capturing layer activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Get last token activations - convert to float32 for numpy compatibility
            acts = hidden[0, -1, :].detach().float().cpu()
            
            scan = LayerScan(
                layer_idx=layer_idx,
                activations=acts,
                min_val=acts.min().item(),
                max_val=acts.max().item(),
                mean_val=acts.mean().item(),
                std_val=acts.std().item()
            )
            self.layer_scans.append(scan)
        
        return hook
    
    def register_hooks(self):
        """Register hooks on all layers."""
        self.clear_hooks()
        for i in range(self.num_layers):
            hook = self.model.model.layers[i].register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_scans = []
    
    def scan(self, prompt: str, max_new_tokens: int = 1) -> List[LayerScan]:
        """Run a scan on the given prompt."""
        self.layer_scans = []
        self.register_hooks()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        self.clear_hooks()
        
        # Sort by layer index (hooks may fire out of order during generation)
        self.layer_scans.sort(key=lambda x: x.layer_idx)
        
        # Keep only one scan per layer (the last token generation)
        seen = set()
        unique_scans = []
        for scan in reversed(self.layer_scans):
            if scan.layer_idx not in seen:
                seen.add(scan.layer_idx)
                unique_scans.append(scan)
        unique_scans.reverse()
        
        return unique_scans
    
    def _normalize_activations(self, acts: torch.Tensor, scans: List[LayerScan], 
                                mode: str = "layer") -> np.ndarray:
        """Normalize activations to [0, 1] range."""
        acts_np = acts.numpy()
        
        if mode == "layer":
            # Normalize per-layer
            min_val, max_val = acts_np.min(), acts_np.max()
            if max_val - min_val > 1e-8:
                return (acts_np - min_val) / (max_val - min_val)
            return np.zeros_like(acts_np)
        
        elif mode == "global":
            # Normalize across all layers
            global_min = min(s.min_val for s in scans)
            global_max = max(s.max_val for s in scans)
            if global_max - global_min > 1e-8:
                return (acts_np - global_min) / (global_max - global_min)
            return np.zeros_like(acts_np)
        
        elif mode == "signed":
            # Normalize keeping sign info: [-1, 1] -> [0, 1]
            abs_max = max(abs(acts_np.min()), abs(acts_np.max()))
            if abs_max > 1e-8:
                return (acts_np / abs_max + 1) / 2
            return np.ones_like(acts_np) * 0.5
        
        else:  # none
            return np.clip(acts_np, 0, 1)
    
    def _apply_colormap(self, normalized: np.ndarray, colormap: str) -> np.ndarray:
        """Apply colormap to normalized [0,1] values. Returns RGB array."""
        
        if colormap == "grayscale":
            gray = (normalized * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)
        
        elif colormap == "signed":
            # Blue (negative) -> White (zero) -> Red (positive)
            # normalized is in [0, 1] where 0.5 = zero
            rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
            
            # Below 0.5: blue to white
            mask_neg = normalized < 0.5
            intensity_neg = 1 - (normalized[mask_neg] * 2)  # 1 at 0, 0 at 0.5
            rgb[mask_neg, 0] = ((1 - intensity_neg) * 255).astype(np.uint8)
            rgb[mask_neg, 1] = ((1 - intensity_neg) * 255).astype(np.uint8)
            rgb[mask_neg, 2] = 255
            
            # Above 0.5: white to red
            mask_pos = normalized >= 0.5
            intensity_pos = (normalized[mask_pos] - 0.5) * 2  # 0 at 0.5, 1 at 1
            rgb[mask_pos, 0] = 255
            rgb[mask_pos, 1] = ((1 - intensity_pos) * 255).astype(np.uint8)
            rgb[mask_pos, 2] = ((1 - intensity_pos) * 255).astype(np.uint8)
            
            return rgb
        
        elif colormap == "hsv":
            # Map value to hue (full spectrum)
            rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
            for i in range(normalized.shape[0]):
                for j in range(normalized.shape[1]):
                    h = normalized[i, j] * 0.8  # 0-0.8 to avoid wrapping red
                    s = 0.9
                    v = 0.5 + normalized[i, j] * 0.5  # Brighter for higher values
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    rgb[i, j] = [int(r * 255), int(g * 255), int(b * 255)]
            return rgb
        
        elif colormap == "plasma":
            # Attempt plasma-like colormap
            rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
            for i in range(normalized.shape[0]):
                for j in range(normalized.shape[1]):
                    t = normalized[i, j]
                    # Plasma approximation
                    r = min(1, 0.05 + t * 1.5)
                    g = max(0, min(1, t * 2 - 0.5))
                    b = max(0, 1 - t * 1.5)
                    rgb[i, j] = [int(r * 255), int(g * 255), int(b * 255)]
            return rgb
        
        else:  # viridis (default)
            rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
            for i in range(normalized.shape[0]):
                for j in range(normalized.shape[1]):
                    t = normalized[i, j]
                    # Viridis approximation
                    r = max(0, min(1, 0.27 + t * 0.7))
                    g = max(0, min(1, 0.0 + t * 0.9))
                    b = max(0, min(1, 0.33 + t * 0.3 - t * t * 0.3))
                    rgb[i, j] = [int(r * 255), int(g * 255), int(b * 255)]
            return rgb
    
    def render_layer(self, scan: LayerScan, scans: List[LayerScan],
                     config: ScanConfig) -> Image.Image:
        """Render a single layer scan to an image."""
        acts = scan.activations
        
        # Pad if necessary
        total_pixels = config.width * config.height
        if len(acts) < total_pixels:
            acts = torch.cat([acts, torch.zeros(total_pixels - len(acts))])
        elif len(acts) > total_pixels:
            acts = acts[:total_pixels]
        
        # Reshape to 2D
        acts_2d = acts.reshape(config.height, config.width)
        
        # Normalize
        norm_mode = "signed" if config.colormap == "signed" else config.normalize
        normalized = self._normalize_activations(acts_2d, scans, norm_mode)
        
        # Apply colormap
        rgb = self._apply_colormap(normalized, config.colormap)
        
        # Create image
        img = Image.fromarray(rgb, mode='RGB')
        
        return img
    
    def render_all(self, scans: List[LayerScan], config: ScanConfig,
                   output_dir: str = "llmri_output") -> List[str]:
        """Render all layer scans to images."""
        os.makedirs(output_dir, exist_ok=True)
        
        paths = []
        for scan in scans:
            img = self.render_layer(scan, scans, config)
            
            # Scale up for visibility
            img = img.resize((config.width * 8, config.height * 8), Image.Resampling.NEAREST)
            
            path = os.path.join(output_dir, f"layer_{scan.layer_idx:02d}.png")
            img.save(path)
            paths.append(path)
            
            print(f"  Layer {scan.layer_idx:2d}: min={scan.min_val:+.3f} max={scan.max_val:+.3f} "
                  f"mean={scan.mean_val:+.3f} std={scan.std_val:.3f}")
        
        return paths
    
    def render_stack(self, scans: List[LayerScan], config: ScanConfig,
                     output_path: str = "llmri_stack.png") -> str:
        """Render all layers as a vertical stack (like MRI slices)."""
        images = []
        for scan in scans:
            img = self.render_layer(scan, scans, config)
            images.append(img)
        
        # Stack vertically with dividers
        divider_height = 2
        total_height = len(images) * config.height + (len(images) - 1) * divider_height
        
        stack = Image.new('RGB', (config.width, total_height), color=(40, 40, 40))
        
        y = 0
        for img in images:
            stack.paste(img, (0, y))
            y += config.height + divider_height
        
        # Scale up
        stack = stack.resize((config.width * 8, total_height * 4), Image.Resampling.NEAREST)
        stack.save(output_path)
        
        return output_path
    
    def render_grid(self, scans: List[LayerScan], config: ScanConfig,
                    output_path: str = "llmri_grid.png", cols: int = 8) -> str:
        """Render all layers in a grid."""
        images = []
        for scan in scans:
            img = self.render_layer(scan, scans, config)
            images.append(img)
        
        rows = math.ceil(len(images) / cols)
        
        # Padding between images
        pad = 2
        grid_w = cols * config.width + (cols - 1) * pad
        grid_h = rows * config.height + (rows - 1) * pad
        
        grid = Image.new('RGB', (grid_w, grid_h), color=(20, 20, 20))
        
        for i, img in enumerate(images):
            row, col = divmod(i, cols)
            x = col * (config.width + pad)
            y = row * (config.height + pad)
            grid.paste(img, (x, y))
        
        # Scale up
        grid = grid.resize((grid_w * 6, grid_h * 6), Image.Resampling.NEAREST)
        grid.save(output_path)
        
        return output_path
    
    def create_animation(self, scans: List[LayerScan], config: ScanConfig,
                         output_path: str = "llmri_scan.gif", fps: int = 5) -> str:
        """Create animated GIF scanning through layers."""
        frames = []
        
        for scan in scans:
            img = self.render_layer(scan, scans, config)
            # Scale up
            img = img.resize((config.width * 10, config.height * 10), Image.Resampling.NEAREST)
            
            # Add layer label
            # (would need PIL ImageDraw for text, skipping for now)
            
            frames.append(img)
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0
        )
        
        return output_path


def compute_diff(scans_a: List[LayerScan], scans_b: List[LayerScan]) -> List[LayerScan]:
    """Compute difference between two scans."""
    diff_scans = []
    for scan_a, scan_b in zip(scans_a, scans_b):
        diff_acts = scan_a.activations - scan_b.activations
        diff_scan = LayerScan(
            layer_idx=scan_a.layer_idx,
            activations=diff_acts,
            min_val=diff_acts.min().item(),
            max_val=diff_acts.max().item(),
            mean_val=diff_acts.mean().item(),
            std_val=diff_acts.std().item()
        )
        diff_scans.append(diff_scan)
    return diff_scans


def main():
    parser = argparse.ArgumentParser(description="LLMRI - Neural Network MRI Scanner")
    parser.add_argument("--model", type=str, default="google/gemma-2-27b-it")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_a", type=str, default=None, help="First prompt for diff mode")
    parser.add_argument("--prompt_b", type=str, default=None, help="Second prompt for diff mode")
    parser.add_argument("--diff", action="store_true", help="Show difference between prompt_a and prompt_b")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--colormap", type=str, default="viridis",
                        choices=["viridis", "grayscale", "hsv", "plasma", "signed"])
    parser.add_argument("--normalize", type=str, default="layer",
                        choices=["layer", "global", "none"])
    parser.add_argument("--output", type=str, default="llmri_output")
    parser.add_argument("--animate", action="store_true", help="Create animated GIF")
    parser.add_argument("--grid", action="store_true", help="Create grid view")
    parser.add_argument("--stack", action="store_true", help="Create vertical stack")
    parser.add_argument("--fps", type=int, default=5, help="Animation FPS")
    
    args = parser.parse_args()
    args.model = os.path.expanduser(args.model)
    
    print(f"\n{'='*60}")
    print(f"  LLMRI - Large Language Model Resonance Imaging")
    print(f"{'='*60}")
    
    # Load model
    print(f"\n[Loading model: {args.model}]")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    
    # Create scanner
    scanner = LLMRI(model, tokenizer, args.device)
    
    # Configure scan
    config = ScanConfig(
        width=scanner.img_width,
        height=scanner.img_height,
        colormap=args.colormap,
        normalize=args.normalize,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Diff mode or single scan
    if args.diff and args.prompt_a and args.prompt_b:
        print(f"\n[DIFF MODE]")
        print(f"[Scanning A: \"{args.prompt_a}\"]")
        scans_a = scanner.scan(args.prompt_a)
        print(f"[Captured {len(scans_a)} layers]")
        
        print(f"\n[Scanning B: \"{args.prompt_b}\"]")
        scans_b = scanner.scan(args.prompt_b)
        print(f"[Captured {len(scans_b)} layers]")
        
        print(f"\n[Computing difference (A - B)]")
        scans = compute_diff(scans_a, scans_b)
        
        # Use signed colormap for diff by default
        if args.colormap == "viridis":
            config.colormap = "signed"
        
        # Also render individual scans
        print(f"\n[Rendering scan A]")
        dir_a = os.path.join(output_dir, "scan_a")
        scanner.render_all(scans_a, ScanConfig(
            width=config.width, height=config.height,
            colormap=args.colormap, normalize=args.normalize
        ), dir_a)
        
        print(f"\n[Rendering scan B]")
        dir_b = os.path.join(output_dir, "scan_b")
        scanner.render_all(scans_b, ScanConfig(
            width=config.width, height=config.height,
            colormap=args.colormap, normalize=args.normalize
        ), dir_b)
        
        print(f"\n[Rendering diff (A - B)]")
        dir_diff = os.path.join(output_dir, "diff")
        paths = scanner.render_all(scans, config, dir_diff)
        
        # Create side-by-side comparison grid
        print(f"\n[Creating comparison grids]")
        scanner.render_grid(scans_a, ScanConfig(
            width=config.width, height=config.height,
            colormap=args.colormap, normalize=args.normalize
        ), os.path.join(output_dir, "grid_a.png"))
        scanner.render_grid(scans_b, ScanConfig(
            width=config.width, height=config.height,
            colormap=args.colormap, normalize=args.normalize
        ), os.path.join(output_dir, "grid_b.png"))
        scanner.render_grid(scans, config, os.path.join(output_dir, "grid_diff.png"))
        
    else:
        if not args.prompt:
            print("Error: --prompt required (or use --diff with --prompt_a and --prompt_b)")
            sys.exit(1)
            
        print(f"\n[Scanning: \"{args.prompt}\"]")
        scans = scanner.scan(args.prompt)
        print(f"[Captured {len(scans)} layers]")
        
        # Render individual layers
        print(f"\n[Rendering layers to {output_dir}/]")
        paths = scanner.render_all(scans, config, output_dir)
    
    # Optional outputs
    if args.grid and not args.diff:
        grid_path = os.path.join(output_dir, "grid.png")
        scanner.render_grid(scans, config, grid_path)
        print(f"[Grid: {grid_path}]")
    
    if args.stack:
        stack_path = os.path.join(output_dir, "stack.png")
        scanner.render_stack(scans, config, stack_path)
        print(f"[Stack: {stack_path}]")
    
    if args.animate:
        gif_path = os.path.join(output_dir, "scan.gif")
        scanner.create_animation(scans, config, gif_path, args.fps)
        print(f"[Animation: {gif_path}]")
    
    print(f"\n[Scan complete]")
    print(f"[Output: {output_dir}/]")


if __name__ == "__main__":
    main()