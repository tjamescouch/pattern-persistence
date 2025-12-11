#!/usr/bin/env python3
"""
Hebbian Neuron Clustering - Creates meaningful 2D layouts for neural activations.

The problem: Reshaping 4608 dims into 64×72 is arbitrary. Adjacent pixels
aren't related neurons.

The solution: Use Hebbian learning to cluster neurons by co-activation.
Neurons that fire together should be placed together.

Methods:
1. Collect activation samples over many forward passes
2. Build co-activation correlation matrix
3. Use dimensionality reduction to create 2D layout
4. Re-render images using learned layout

Usage:
    from hebbian_layout import HebbianLayout
    
    layout = HebbianLayout(hidden_dim=4608)
    
    # Collect samples during inference
    for activation in activations:
        layout.observe(activation)
    
    # Learn layout
    layout.learn()
    
    # Render using learned layout
    img = layout.render(new_activation)
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math
import os
from datetime import datetime


@dataclass
class NeuronCluster:
    """A cluster of co-firing neurons."""
    center_x: float
    center_y: float
    neuron_indices: List[int]
    mean_activation: float
    label: Optional[str] = None


class HebbianLayout:
    """
    Learns meaningful 2D neuron layouts from co-activation patterns.
    """
    
    def __init__(self, hidden_dim: int = 4608, target_size: Tuple[int, int] = (128, 128)):
        self.hidden_dim = hidden_dim
        self.target_width, self.target_height = target_size
        
        # Observation buffer
        self.observations: List[np.ndarray] = []
        self.max_observations = 500
        
        # Learned layout: neuron_idx -> (x, y)
        self.positions: Optional[np.ndarray] = None  # [hidden_dim, 2]
        self.layout_learned = False
        
        # Clustering results
        self.clusters: List[NeuronCluster] = []
        self.neuron_to_cluster: Dict[int, int] = {}
        
        # Cache for rendering
        self._render_map: Optional[np.ndarray] = None  # [height, width] -> neuron_idx or -1
        
    def observe(self, activation: np.ndarray):
        """Record an activation sample."""
        if len(activation) != self.hidden_dim:
            raise ValueError(f"Expected {self.hidden_dim} dims, got {len(activation)}")
        
        self.observations.append(activation.copy())
        
        if len(self.observations) > self.max_observations:
            self.observations.pop(0)
    
    def compute_coactivation(self) -> np.ndarray:
        """
        Compute co-activation matrix using correlation.
        Returns [hidden_dim, hidden_dim] correlation matrix.
        """
        if len(self.observations) < 10:
            raise ValueError("Need at least 10 observations")
        
        # Stack into [n_samples, hidden_dim]
        X = np.stack(self.observations)
        
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # Compute correlation (memory-efficient for large dims)
        # For 4608 dims, full matrix = 85MB which is fine
        print(f"[Hebbian] Computing correlation from {len(self.observations)} samples...")
        
        # Normalize
        norms = np.linalg.norm(X_centered, axis=0, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X_centered / norms
        
        # Correlation = X.T @ X / n_samples
        corr = (X_norm.T @ X_norm) / len(self.observations)
        
        print(f"[Hebbian] Correlation matrix computed: {corr.shape}")
        return corr
    
    def learn_layout_umap(self, n_neighbors: int = 15, min_dist: float = 0.1):
        """
        Learn 2D layout using UMAP on correlation matrix.
        Requires: pip install umap-learn
        """
        try:
            import umap
        except ImportError:
            print("[Hebbian] UMAP not installed. Run: pip install umap-learn")
            return self.learn_layout_pca()
        
        corr = self.compute_coactivation()
        
        # Convert correlation to distance (1 - corr)
        distance = 1 - corr
        np.fill_diagonal(distance, 0)
        
        print(f"[Hebbian] Running UMAP...")
        reducer = umap.UMAP(
            n_components=2,
            metric='precomputed',
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
        positions = reducer.fit_transform(distance)
        
        # Normalize to target size
        self._normalize_positions(positions)
        
        print(f"[Hebbian] Layout learned!")
        self.layout_learned = True
    
    def learn_layout_pca(self):
        """
        Learn 2D layout using PCA (faster, simpler).
        Fallback when UMAP not available.
        """
        corr = self.compute_coactivation()
        
        print(f"[Hebbian] Running PCA...")
        
        # Use SVD on correlation matrix
        U, S, Vt = np.linalg.svd(corr)
        
        # Take first 2 components
        positions = U[:, :2] * S[:2]
        
        # Normalize to target size
        self._normalize_positions(positions)
        
        print(f"[Hebbian] Layout learned (PCA)!")
        self.layout_learned = True
    
    def learn_layout_tsne(self, perplexity: int = 30):
        """
        Learn 2D layout using t-SNE.
        Slower but often gives better clustering.
        """
        from sklearn.manifold import TSNE
        
        corr = self.compute_coactivation()
        
        # Convert correlation to distance
        distance = 1 - corr
        np.fill_diagonal(distance, 0)
        
        print(f"[Hebbian] Running t-SNE (this may take a while)...")
        tsne = TSNE(
            n_components=2,
            metric='precomputed',
            perplexity=min(perplexity, len(distance) - 1),
            random_state=42
        )
        
        positions = tsne.fit_transform(distance)
        
        self._normalize_positions(positions)
        
        print(f"[Hebbian] Layout learned (t-SNE)!")
        self.layout_learned = True
    
    def _normalize_positions(self, positions: np.ndarray):
        """Normalize positions to fit target size."""
        # Center at origin
        positions = positions - positions.mean(axis=0)
        
        # Scale to fit in target rectangle with margin
        margin = 0.1
        scale_x = (self.target_width * (1 - 2*margin)) / (positions[:, 0].max() - positions[:, 0].min() + 1e-8)
        scale_y = (self.target_height * (1 - 2*margin)) / (positions[:, 1].max() - positions[:, 1].min() + 1e-8)
        scale = min(scale_x, scale_y)
        
        positions = positions * scale
        
        # Shift to center of target
        positions[:, 0] += self.target_width / 2
        positions[:, 1] += self.target_height / 2
        
        self.positions = positions
        self._build_render_map()
    
    def _build_render_map(self):
        """Build mapping from pixels to neurons."""
        self._render_map = np.full((self.target_height, self.target_width), -1, dtype=np.int32)
        
        # For each neuron, mark its pixel
        for idx in range(self.hidden_dim):
            x = int(np.clip(self.positions[idx, 0], 0, self.target_width - 1))
            y = int(np.clip(self.positions[idx, 1], 0, self.target_height - 1))
            self._render_map[y, x] = idx
    
    def cluster_neurons(self, n_clusters: int = 50):
        """
        Cluster neurons based on learned positions.
        Uses K-means on 2D positions.
        """
        if not self.layout_learned:
            raise ValueError("Must learn layout first")
        
        from sklearn.cluster import KMeans
        
        print(f"[Hebbian] Clustering into {n_clusters} groups...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.positions)
        
        self.clusters = []
        for i in range(n_clusters):
            mask = labels == i
            indices = np.where(mask)[0].tolist()
            center = self.positions[mask].mean(axis=0)
            
            cluster = NeuronCluster(
                center_x=center[0],
                center_y=center[1],
                neuron_indices=indices,
                mean_activation=0.0
            )
            self.clusters.append(cluster)
            
            for idx in indices:
                self.neuron_to_cluster[idx] = i
        
        print(f"[Hebbian] Clustered {self.hidden_dim} neurons into {n_clusters} groups")
    
    def render(self, activation: np.ndarray, colormap: str = "viridis") -> np.ndarray:
        """
        Render activation using learned layout.
        Returns RGB image [height, width, 3].
        """
        if not self.layout_learned:
            raise ValueError("Must learn layout first")
        
        # Normalize activation
        min_val, max_val = activation.min(), activation.max()
        if max_val - min_val > 1e-8:
            normalized = (activation - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(activation)
        
        # Create image
        rgb = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Fill background
        rgb[:, :] = [20, 20, 20]
        
        # Plot each neuron
        for idx in range(self.hidden_dim):
            x = int(np.clip(self.positions[idx, 0], 0, self.target_width - 1))
            y = int(np.clip(self.positions[idx, 1], 0, self.target_height - 1))
            
            color = self._apply_colormap(normalized[idx], colormap)
            
            # Blend with existing (for overlapping neurons)
            rgb[y, x] = (rgb[y, x].astype(float) + np.array(color)) / 2
            rgb[y, x] = rgb[y, x].astype(np.uint8)
        
        return rgb
    
    def render_clusters(self, activation: np.ndarray, colormap: str = "viridis") -> np.ndarray:
        """
        Render with cluster-level aggregation.
        Each cluster shows mean activation of its neurons.
        """
        if not self.clusters:
            raise ValueError("Must cluster neurons first")
        
        rgb = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        rgb[:, :] = [20, 20, 20]
        
        # Compute cluster activations
        cluster_acts = np.zeros(len(self.clusters))
        for i, cluster in enumerate(self.clusters):
            cluster_acts[i] = activation[cluster.neuron_indices].mean()
        
        # Normalize
        min_val, max_val = cluster_acts.min(), cluster_acts.max()
        if max_val - min_val > 1e-8:
            normalized = (cluster_acts - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(cluster_acts)
        
        # Draw clusters as circles
        for i, cluster in enumerate(self.clusters):
            cx = int(cluster.center_x)
            cy = int(cluster.center_y)
            radius = max(3, int(np.sqrt(len(cluster.neuron_indices)) / 2))
            
            color = self._apply_colormap(normalized[i], colormap)
            
            # Draw filled circle
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        x = cx + dx
                        y = cy + dy
                        if 0 <= x < self.target_width and 0 <= y < self.target_height:
                            rgb[y, x] = color
        
        return rgb
    
    def _apply_colormap(self, value: float, colormap: str) -> Tuple[int, int, int]:
        """Apply colormap to normalized value."""
        v = max(0, min(1, value))
        
        if colormap == "grayscale":
            g = int(v * 255)
            return (g, g, g)
        elif colormap == "viridis":
            r = int(max(0, min(255, (0.27 + v * 0.7) * 255)))
            g = int(max(0, min(255, v * 0.9 * 255)))
            b = int(max(0, min(255, (0.33 + v * 0.3 - v * v * 0.3) * 255)))
            return (r, g, b)
        elif colormap == "hot":
            if v < 0.33:
                return (int(v * 3 * 255), 0, 0)
            elif v < 0.66:
                return (255, int((v - 0.33) * 3 * 255), 0)
            else:
                return (255, 255, int((v - 0.66) * 3 * 255))
        else:
            return (int(v * 255), int(v * 255), int(v * 255))
    
    def save(self, path: str):
        """Save learned layout."""
        np.savez(
            path,
            positions=self.positions,
            hidden_dim=self.hidden_dim,
            target_size=np.array([self.target_width, self.target_height])
        )
        print(f"[Hebbian] Layout saved to {path}")
    
    def load(self, path: str):
        """Load learned layout."""
        data = np.load(path)
        self.positions = data['positions']
        self.hidden_dim = int(data['hidden_dim'])
        self.target_width, self.target_height = data['target_size']
        self.layout_learned = True
        self._build_render_map()
        print(f"[Hebbian] Layout loaded from {path}")


class OnlineHebbianLayout(HebbianLayout):
    """
    Online version that updates layout incrementally.
    Uses Oja's rule for online PCA.
    """
    
    def __init__(self, hidden_dim: int = 4608, target_size: Tuple[int, int] = (128, 128)):
        super().__init__(hidden_dim, target_size)
        
        # Online learning parameters
        self.learning_rate = 0.01
        self.n_components = 2
        
        # Initialize random projection
        self.W = np.random.randn(hidden_dim, self.n_components) * 0.01
        self.W /= np.linalg.norm(self.W, axis=0, keepdims=True)
        
        # Running mean for centering
        self.running_mean = np.zeros(hidden_dim)
        self.n_seen = 0
    
    def observe_online(self, activation: np.ndarray):
        """Update layout incrementally using Oja's rule."""
        # Update running mean
        self.n_seen += 1
        alpha = 1.0 / self.n_seen
        self.running_mean = (1 - alpha) * self.running_mean + alpha * activation
        
        # Center
        x = activation - self.running_mean
        
        # Project
        y = x @ self.W  # [n_components]
        
        # Oja's rule: W += lr * (x * y.T - y^2 * W)
        update = np.outer(x, y) - self.W * (y ** 2)
        self.W += self.learning_rate * update
        
        # Re-normalize
        self.W /= np.linalg.norm(self.W, axis=0, keepdims=True)
        
        # Update positions every 10 samples
        if self.n_seen % 10 == 0:
            self._update_positions()
    
    def _update_positions(self):
        """Compute positions from current projection."""
        # Project all neurons to 2D
        # Each neuron i has embedding W[i, :]
        positions = self.W.copy()
        
        # Normalize to target size
        self._normalize_positions(positions)
        self.layout_learned = True


def demo():
    """Demo with synthetic data."""
    print("=== Hebbian Layout Demo ===\n")
    
    hidden_dim = 256  # Small for demo
    layout = HebbianLayout(hidden_dim=hidden_dim, target_size=(64, 64))
    
    # Generate synthetic activations with cluster structure
    print("Generating synthetic data with 4 clusters...")
    np.random.seed(42)
    
    # 4 clusters of neurons
    cluster_centers = [
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
    ]
    
    for _ in range(100):
        # Random cluster activation
        weights = np.random.dirichlet([1, 1, 1, 1])
        activation = np.zeros(hidden_dim)
        
        for i, w in enumerate(weights):
            # Each cluster activates a subset of neurons
            start = i * (hidden_dim // 4)
            end = start + (hidden_dim // 4)
            activation[start:end] = w + np.random.randn(hidden_dim // 4) * 0.1
        
        layout.observe(activation)
    
    # Learn layout
    print("\nLearning layout...")
    layout.learn_layout_pca()
    
    # Cluster
    layout.cluster_neurons(n_clusters=4)
    
    # Render test activation
    test_activation = np.zeros(hidden_dim)
    test_activation[:hidden_dim//4] = 1.0  # Activate first cluster
    
    img = layout.render(test_activation)
    
    print(f"\nRendered image shape: {img.shape}")
    print("Clusters found:")
    for i, cluster in enumerate(layout.clusters):
        print(f"  Cluster {i}: {len(cluster.neuron_indices)} neurons at ({cluster.center_x:.1f}, {cluster.center_y:.1f})")
    
    # Save
    layout.save("/tmp/demo_layout.npz")
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()