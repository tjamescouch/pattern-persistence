#!/usr/bin/env python3
"""
Hierarchical Circle Packing for Neural Activations.

Neurons are grouped by co-activation into a hierarchy:
- Root circle contains everything
- Major clusters are large circles inside
- Sub-clusters are circles within those
- Individual neurons are small circles at leaves

Circle size modulated by activation level.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class CircleNode:
    """A node in the circle hierarchy."""
    x: float = 0.0
    y: float = 0.0
    radius: float = 1.0
    
    # Content
    neuron_indices: List[int] = field(default_factory=list)
    children: List['CircleNode'] = field(default_factory=list)
    parent: Optional['CircleNode'] = None
    
    # For rendering
    activation: float = 0.0  # Mean activation of contained neurons
    depth: int = 0
    label: str = ""
    
    # Selection state
    selected: bool = False
    hovered: bool = False
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    @property
    def neuron_count(self) -> int:
        return len(self.neuron_indices)


class HierarchicalCirclePack:
    """
    Builds and renders hierarchical circle packing from neural activations.
    """
    
    def __init__(self, hidden_dim: int = 4608):
        self.hidden_dim = hidden_dim
        self.root: Optional[CircleNode] = None
        self.hierarchy_built = False
        
        # Observation buffer for building hierarchy
        self.observations: List[np.ndarray] = []
        self.max_observations = 200
        
        # Hierarchy parameters
        self.n_top_clusters = 12  # Major groupings
        self.n_sub_clusters = 8   # Sub-groupings within each
        
        # Selection and navigation state
        self.selected_node: Optional[CircleNode] = None
        self.hovered_node: Optional[CircleNode] = None
        self.view_root: Optional[CircleNode] = None  # Current view (can zoom into sub-tree)
        
        # Intervention state
        self.cluster_interventions: Dict[str, float] = {}  # label -> multiplier
        
    def observe(self, activation: np.ndarray):
        """Record activation sample."""
        self.observations.append(activation.copy())
        if len(self.observations) > self.max_observations:
            self.observations.pop(0)
    
    def build_hierarchy(self):
        """Build hierarchical clustering from observations."""
        if len(self.observations) < 20:
            print(f"[CirclePack] Need 20+ samples (have {len(self.observations)})")
            return
        
        print(f"[CirclePack] Building hierarchy from {len(self.observations)} samples...")
        
        # Compute correlation matrix
        X = np.stack(self.observations)
        X_centered = X - X.mean(axis=0)
        norms = np.linalg.norm(X_centered, axis=0, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X_centered / norms
        corr = (X_norm.T @ X_norm) / len(self.observations)
        
        # Convert to distance
        distance = 1 - np.abs(corr)
        np.fill_diagonal(distance, 0)
        
        # Hierarchical clustering using simple agglomerative approach
        # For speed, we'll use sklearn if available, else a simple method
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # First level: major clusters
            clustering = AgglomerativeClustering(
                n_clusters=self.n_top_clusters,
                metric='precomputed',
                linkage='average'
            )
            top_labels = clustering.fit_predict(distance)
            
        except ImportError:
            # Fallback: random assignment (not ideal but works)
            print("[CirclePack] sklearn not available, using simple clustering")
            top_labels = np.arange(self.hidden_dim) % self.n_top_clusters
        
        # Build tree structure
        self.root = CircleNode(
            x=0.5, y=0.5, radius=0.45,
            neuron_indices=list(range(self.hidden_dim)),
            depth=0, label="All",
            parent=None
        )
        self.view_root = self.root
        
        # Create top-level clusters
        for cluster_id in range(self.n_top_clusters):
            indices = np.where(top_labels == cluster_id)[0].tolist()
            if len(indices) == 0:
                continue
            
            child = CircleNode(
                neuron_indices=indices,
                depth=1,
                label=f"C{cluster_id}",
                parent=self.root
            )
            
            # Sub-cluster if large enough
            if len(indices) > self.n_sub_clusters * 2:
                self._add_subclusters(child, distance, indices)
            
            self.root.children.append(child)
        
        # Pack circles
        self._pack_circles(self.root)
        
        # Debug: print top-level cluster positions
        print(f"[CirclePack] Built {len(self.root.children)} top clusters")
        for i, child in enumerate(self.root.children[:5]):  # Show first 5
            print(f"  {child.label}: pos=({child.x:.3f}, {child.y:.3f}) r={child.radius:.3f} neurons={child.neuron_count}")
        
        self.hierarchy_built = True
        print(f"[CirclePack] Hierarchy built: {self.n_top_clusters} clusters")
    
    def _add_subclusters(self, parent: CircleNode, distance: np.ndarray, indices: List[int]):
        """Add sub-clusters to a parent node."""
        if len(indices) < self.n_sub_clusters:
            return
        
        # Extract sub-distance matrix
        idx_array = np.array(indices)
        sub_distance = distance[np.ix_(idx_array, idx_array)]
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            n_sub = min(self.n_sub_clusters, len(indices) // 3)
            if n_sub < 2:
                return
                
            clustering = AgglomerativeClustering(
                n_clusters=n_sub,
                metric='precomputed',
                linkage='average'
            )
            sub_labels = clustering.fit_predict(sub_distance)
            
            for sub_id in range(n_sub):
                sub_indices = [indices[i] for i in range(len(indices)) if sub_labels[i] == sub_id]
                if sub_indices:
                    child = CircleNode(
                        neuron_indices=sub_indices,
                        depth=parent.depth + 1,
                        label=f"{parent.label}.{sub_id}",
                        parent=parent
                    )
                    parent.children.append(child)
                    
                    # Add third level for large sub-clusters
                    if len(sub_indices) > 50 and parent.depth < 2:
                        self._add_subclusters(child, distance, sub_indices)
                    
        except ImportError:
            # No sklearn - just split evenly
            chunk_size = max(1, len(indices) // self.n_sub_clusters)
            for i in range(self.n_sub_clusters):
                start = i * chunk_size
                end = start + chunk_size if i < self.n_sub_clusters - 1 else len(indices)
                sub_indices = indices[start:end]
                if sub_indices:
                    child = CircleNode(
                        neuron_indices=sub_indices,
                        depth=parent.depth + 1,
                        label=f"{parent.label}.{i}",
                        parent=parent
                    )
                    parent.children.append(child)
    
    def _pack_circles(self, node: CircleNode, depth: int = 0):
        """
        Pack child circles inside parent circle.
        Uses a simple radial layout algorithm.
        """
        if not node.children:
            return
        
        n = len(node.children)
        if n == 0:
            return
        
        # Calculate child radii based on neuron count
        total_neurons = len(node.neuron_indices)
        
        # Reserve space for children - use more of parent
        available_radius = node.radius * 0.75
        
        # Assign radii proportional to sqrt of neuron count
        child_weights = [math.sqrt(len(c.neuron_indices)) for c in node.children]
        total_weight = sum(child_weights)
        
        if total_weight == 0:
            return
        
        # Position children in a circle around center
        if n == 1:
            # Single child in center
            child = node.children[0]
            child.x = node.x
            child.y = node.y
            child.radius = available_radius * 0.8
        elif n <= 4:
            # Very few children - bigger circles, tighter ring
            angle_step = 2 * math.pi / n
            
            max_child_radius = available_radius * 0.6
            for i, child in enumerate(node.children):
                weight_fraction = child_weights[i] / total_weight
                child.radius = max(
                    available_radius * 0.25,
                    min(max_child_radius, available_radius * math.sqrt(weight_fraction) * 1.5)
                )
            
            ring_radius = node.radius * 0.32
            
            for i, child in enumerate(node.children):
                angle = i * angle_step - math.pi / 2
                child.x = node.x + ring_radius * math.cos(angle)
                child.y = node.y + ring_radius * math.sin(angle)
        elif n <= 8:
            # Few children - arrange in ring with larger circles
            angle_step = 2 * math.pi / n
            
            max_child_radius = available_radius * 0.5
            for i, child in enumerate(node.children):
                weight_fraction = child_weights[i] / total_weight
                child.radius = max(
                    available_radius * 0.2,
                    min(max_child_radius, available_radius * math.sqrt(weight_fraction) * 1.4)
                )
            
            ring_radius = node.radius * 0.35
            
            for i, child in enumerate(node.children):
                angle = i * angle_step - math.pi / 2
                child.x = node.x + ring_radius * math.cos(angle)
                child.y = node.y + ring_radius * math.sin(angle)
        else:
            # Many children - use tighter packing with two rings
            angle_step = 2 * math.pi / n
            
            max_child_radius = available_radius * 0.4
            for i, child in enumerate(node.children):
                weight_fraction = child_weights[i] / total_weight
                child.radius = max(
                    available_radius * 0.12,
                    min(max_child_radius, available_radius * math.sqrt(weight_fraction) * 1.3)
                )
            
            # Two rings for many children
            inner_ring = node.radius * 0.22
            outer_ring = node.radius * 0.40
            
            for i, child in enumerate(node.children):
                ring = outer_ring if i % 2 == 0 else inner_ring
                angle = i * angle_step - math.pi / 2
                child.x = node.x + ring * math.cos(angle)
                child.y = node.y + ring * math.sin(angle)
        
        # Recursively pack children
        for child in node.children:
            self._pack_circles(child, depth + 1)
    
    def update_activations(self, activations: np.ndarray):
        """Update activation values throughout hierarchy."""
        if self.root is None:
            return
        
        self._update_node_activation(self.root, activations)
    
    def _update_node_activation(self, node: CircleNode, activations: np.ndarray):
        """Recursively update activation for a node."""
        if node.neuron_indices:
            node.activation = float(np.mean(activations[node.neuron_indices]))
        
        for child in node.children:
            self._update_node_activation(child, activations)
    
    def render(self, activations: np.ndarray, size: int = 800) -> np.ndarray:
        """
        Render hierarchy as RGB image.
        Returns [size, size, 3] uint8 array.
        """
        if self.view_root is None:
            # Return empty
            return np.zeros((size, size, 3), dtype=np.uint8)
        
        # Update activations
        self.update_activations(activations)
        
        # Normalize activations for coloring
        act_min = activations.min()
        act_max = activations.max()
        act_range = act_max - act_min if act_max > act_min else 1
        
        # Create image
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :] = [15, 15, 20]  # Dark background
        
        # Determine which nodes to render on top (use labels since nodes aren't hashable)
        skip_labels = set()
        if self.selected_node:
            skip_labels.add(self.selected_node.label)
        if self.hovered_node and self.hovered_node != self.selected_node:
            skip_labels.add(self.hovered_node.label)
        
        # Render from current view root, skip top nodes first
        self._render_node(self.view_root, img, size, act_min, act_range, skip_labels=skip_labels)
        
        # Render hovered on top (if not selected)
        if self.hovered_node and self.hovered_node != self.selected_node:
            self._render_node(self.hovered_node, img, size, act_min, act_range, skip_labels=set())
        
        # Render selected subtree on very top
        if self.selected_node:
            self._render_node(self.selected_node, img, size, act_min, act_range, skip_labels=set())
        
        return img
    
    def _render_node(self, node: CircleNode, img: np.ndarray, size: int, 
                     act_min: float, act_range: float, skip_labels: set = None):
        """Render a node and its children."""
        if skip_labels is None:
            skip_labels = set()
        
        # Skip this node if we're deferring its rendering
        if node.label in skip_labels:
            return
        
        # Convert normalized coords to pixels
        cx = int(node.x * size)
        cy = int(node.y * size)
        
        # Base radius from layout
        base_r = int(node.radius * size)
        
        # Modulate radius by activation (±30%)
        norm_act = (node.activation - act_min) / act_range if act_range > 0 else 0.5
        norm_act = max(0, min(1, norm_act))
        
        # Size modulation: low activation = smaller, high = larger
        size_mod = 0.7 + 0.6 * norm_act  # Range: 0.7x to 1.3x
        r = max(2, int(base_r * size_mod))
        
        if r < 1:
            return
        
        # Check for intervention
        has_intervention = node.label in self.cluster_interventions
        intervention_mult = self.cluster_interventions.get(node.label, 1.0)
        
        # Color based on depth and cluster ID
        if node.depth == 0:
            # Root: very subtle, almost invisible
            self._draw_filled_circle(img, cx, cy, r, (25, 25, 35))
        else:
            # Get a consistent random color based on label
            color = self._get_cluster_color(node.label, norm_act)
            
            # Tint based on intervention
            if has_intervention:
                if intervention_mult > 1.0:
                    # Boost: tint toward green
                    color = (color[0] // 2, min(255, color[1] + 80), color[2] // 2)
                elif intervention_mult < 1.0:
                    # Ablate: tint toward red
                    color = (min(255, color[0] + 80), color[1] // 2, color[2] // 2)
                elif intervention_mult == 0:
                    # Full ablation: dark red
                    color = (100, 20, 20)
            
            self._draw_filled_circle(img, cx, cy, r, color)
        
        # Hover highlight (bright cyan ring with glow)
        if node.hovered and not node.selected:
            # Outer glow
            self._draw_circle_ring(img, cx, cy, r + 6, (0, 180, 180), thickness=4)
            # Inner bright ring
            self._draw_circle_ring(img, cx, cy, r + 2, (0, 255, 255), thickness=3)
        
        # Selection highlight (yellow/white rings)
        if node.selected:
            self._draw_circle_ring(img, cx, cy, r + 4, (255, 255, 0), thickness=4)
            self._draw_circle_ring(img, cx, cy, r + 8, (255, 255, 255), thickness=2)
        
        # Render children on top
        for child in node.children:
            self._render_node(child, img, size, act_min, act_range, skip_labels=skip_labels)
    
    def _draw_circle_ring(self, img: np.ndarray, cx: int, cy: int, r: int,
                          color: Tuple[int, int, int], thickness: int = 2):
        """Draw a ring (circle outline)."""
        h, w = img.shape[:2]
        
        y_min = max(0, cy - r - thickness)
        y_max = min(h, cy + r + thickness + 1)
        x_min = max(0, cx - r - thickness)
        x_max = min(w, cx + r + thickness + 1)
        
        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        
        inner = (r - thickness) ** 2
        outer = (r + thickness) ** 2
        mask = (dist_sq >= inner) & (dist_sq <= outer)
        
        img[y_min:y_max, x_min:x_max][mask] = color
    
    def _get_cluster_color(self, label: str, activation: float) -> Tuple[int, int, int]:
        """Get a distinct color for a cluster, modulated by activation."""
        # Use label hash for consistent random color
        import hashlib
        hash_val = int(hashlib.md5(label.encode()).hexdigest()[:8], 16)
        
        # Generate hue from hash (0-360)
        hue = (hash_val % 360) / 360.0
        
        # Saturation based on depth (deeper = less saturated)
        depth = label.count('.')
        saturation = max(0.4, 0.9 - depth * 0.15)
        
        # Value/brightness based on activation
        value = 0.3 + 0.7 * activation
        
        # HSV to RGB
        r, g, b = self._hsv_to_rgb(hue, saturation, value)
        
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB (all values 0-1)."""
        if s == 0:
            return (v, v, v)
        
        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        i = i % 6
        if i == 0:
            return (v, t, p)
        elif i == 1:
            return (q, v, p)
        elif i == 2:
            return (p, v, t)
        elif i == 3:
            return (p, q, v)
        elif i == 4:
            return (t, p, v)
        else:
            return (v, p, q)
    
    def _draw_filled_circle(self, img: np.ndarray, cx: int, cy: int, r: int, 
                            color: Tuple[int, int, int]):
        """Draw a filled circle on the image."""
        h, w = img.shape[:2]
        
        # Clip to bounds for efficiency
        y_min = max(0, cy - r)
        y_max = min(h, cy + r + 1)
        x_min = max(0, cx - r)
        x_max = min(w, cx + r + 1)
        
        # Create local mask
        y, x = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        
        # Apply color
        img[y_min:y_max, x_min:x_max][mask] = color
    
    def get_cluster_at_point(self, x: float, y: float) -> Optional[CircleNode]:
        """Find which cluster contains a point (normalized 0-1 coords)."""
        if self.view_root is None:
            return None
        
        # Collect all nodes that contain this point
        candidates = []
        self._collect_containing_nodes(self.view_root, x, y, candidates)
        
        if not candidates:
            return None
        
        # Sort by: 1) smaller radius first (more specific), 2) deeper in tree
        # This ensures clicking on a small circle inside a big one selects the small one
        candidates.sort(key=lambda n: (n.radius, -n.depth))
        
        return candidates[0]
    
    def _collect_containing_nodes(self, node: CircleNode, x: float, y: float, 
                                   candidates: List[CircleNode]):
        """Collect all nodes containing the point."""
        dist = math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2)
        
        # Very generous hit area - minimum 5% of view, or actual radius * 1.3
        hit_radius = max(node.radius * 1.3, 0.05)
        
        if dist <= hit_radius:
            if node.depth > 0:  # Don't include root
                candidates.append(node)
        
        # Always check children (they might be outside parent due to layout)
        for child in node.children:
            self._collect_containing_nodes(child, x, y, candidates)
    
    def _find_deepest_containing(self, node: CircleNode, x: float, y: float) -> Optional[CircleNode]:
        """Find deepest node containing point (legacy, use get_cluster_at_point)."""
        dist = math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2)
        if dist > node.radius:
            return None
        
        # Check children (deepest first)
        for child in node.children:
            result = self._find_deepest_containing(child, x, y)
            if result is not None:
                return result
        
        return node
    
    def select_at_point(self, x: float, y: float) -> Optional[CircleNode]:
        """Select cluster at point. Returns selected node or None."""
        # Clear previous selection
        if self.selected_node:
            self.selected_node.selected = False
        
        node = self.get_cluster_at_point(x, y)
        if node and node != self.view_root:  # Don't select root
            node.selected = True
            self.selected_node = node
            return node
        
        self.selected_node = None
        return None
    
    def hover_at_point(self, x: float, y: float) -> Optional[CircleNode]:
        """Update hover state. Returns hovered node or None."""
        # Clear previous hover
        if self.hovered_node:
            self.hovered_node.hovered = False
        
        node = self.get_cluster_at_point(x, y)
        if node and node != self.view_root:  # Don't hover root
            node.hovered = True
            self.hovered_node = node
            return node
        
        self.hovered_node = None
        return None
    
    def zoom_into_selected(self) -> bool:
        """Zoom view into selected cluster. Returns True if zoomed."""
        if self.selected_node and self.selected_node.children:
            self.view_root = self.selected_node
            self.selected_node.selected = False
            self.selected_node = None
            # Repack for new view
            self._repack_for_view()
            return True
        return False
    
    def zoom_out(self) -> bool:
        """Zoom view out to parent. Returns True if zoomed."""
        if self.view_root and self.view_root.parent:
            self.view_root = self.view_root.parent
            if self.selected_node:
                self.selected_node.selected = False
                self.selected_node = None
            self._repack_for_view()
            return True
        elif self.view_root != self.root:
            self.view_root = self.root
            self._repack_for_view()
            return True
        return False
    
    def zoom_to_root(self):
        """Reset view to root."""
        self.view_root = self.root
        if self.selected_node:
            self.selected_node.selected = False
            self.selected_node = None
        self._repack_for_view()
    
    def _repack_for_view(self):
        """Repack circles for current view root."""
        if self.view_root is None:
            return
        
        # Reset position of view root to center
        self.view_root.x = 0.5
        self.view_root.y = 0.5
        self.view_root.radius = 0.45
        
        # Repack children
        self._pack_circles(self.view_root)
    
    def set_intervention(self, node: CircleNode, multiplier: float):
        """Set intervention multiplier for a cluster."""
        if node is None:
            return
        self.cluster_interventions[node.label] = multiplier
        print(f"[CirclePack] Intervention on {node.label}: {multiplier:.2f}x ({node.neuron_count} neurons)")
    
    def clear_intervention(self, node: CircleNode = None):
        """Clear intervention for a cluster or all clusters."""
        if node is None:
            self.cluster_interventions.clear()
            print("[CirclePack] Cleared all interventions")
        elif node.label in self.cluster_interventions:
            del self.cluster_interventions[node.label]
            print(f"[CirclePack] Cleared intervention on {node.label}")
    
    def get_intervention_neurons(self) -> Dict[int, float]:
        """Get all neurons with interventions and their multipliers."""
        neuron_multipliers = {}
        
        for label, multiplier in self.cluster_interventions.items():
            node = self._find_node_by_label(self.root, label)
            if node:
                print(f"[CirclePack] Intervention {label}: {len(node.neuron_indices)} neurons × {multiplier:.2f}")
                for idx in node.neuron_indices:
                    # If neuron is in multiple intervened clusters, multiply
                    if idx in neuron_multipliers:
                        neuron_multipliers[idx] *= multiplier
                    else:
                        neuron_multipliers[idx] = multiplier
            else:
                print(f"[CirclePack] WARNING: Node {label} not found!")
        
        if neuron_multipliers:
            print(f"[CirclePack] Total: {len(neuron_multipliers)} unique neurons")
        
        return neuron_multipliers
    
    def _find_node_by_label(self, node: CircleNode, label: str) -> Optional[CircleNode]:
        """Find node by label."""
        if node.label == label:
            return node
        for child in node.children:
            result = self._find_node_by_label(child, label)
            if result:
                return result
        return None
    
    def save(self, path: str):
        """Save hierarchy structure."""
        if self.root is None:
            print("[CirclePack] No hierarchy to save")
            return
        
        # Serialize tree
        def serialize_node(node):
            return {
                'x': node.x, 'y': node.y, 'radius': node.radius,
                'neuron_indices': node.neuron_indices,
                'depth': node.depth, 'label': node.label,
                'children': [serialize_node(c) for c in node.children]
            }
        
        import json
        with open(path, 'w') as f:
            json.dump(serialize_node(self.root), f)
        
        print(f"[CirclePack] Saved to {path}")
    
    def load(self, path: str):
        """Load hierarchy structure."""
        import json
        
        def deserialize_node(data):
            node = CircleNode(
                x=data['x'], y=data['y'], radius=data['radius'],
                neuron_indices=data['neuron_indices'],
                depth=data['depth'], label=data['label']
            )
            node.children = [deserialize_node(c) for c in data['children']]
            return node
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.root = deserialize_node(data)
        self.hierarchy_built = True
        print(f"[CirclePack] Loaded from {path}")


def demo():
    """Demo with synthetic data."""
    print("=== Circle Packing Demo ===\n")
    
    hidden_dim = 256
    packer = HierarchicalCirclePack(hidden_dim=hidden_dim)
    packer.n_top_clusters = 4
    packer.n_sub_clusters = 3
    
    # Generate observations with cluster structure
    print("Generating synthetic data...")
    np.random.seed(42)
    
    for _ in range(50):
        activation = np.random.randn(hidden_dim)
        # Add cluster structure
        cluster = np.random.randint(4)
        start = cluster * (hidden_dim // 4)
        end = start + (hidden_dim // 4)
        activation[start:end] += 2
        
        packer.observe(activation)
    
    # Build hierarchy
    packer.build_hierarchy()
    
    # Render
    test_act = np.random.randn(hidden_dim)
    test_act[64:128] += 3  # Light up second cluster
    
    img = packer.render(test_act, size=400)
    
    print(f"Rendered image shape: {img.shape}")
    
    # Save image
    try:
        from PIL import Image
        Image.fromarray(img).save("/tmp/circle_pack_demo.png")
        print("Saved to /tmp/circle_pack_demo.png")
    except ImportError:
        print("PIL not available for saving")


if __name__ == "__main__":
    demo()