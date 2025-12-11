# LLMRI: Real-Time Neural Activation Viewer for LLMs

A toolkit for visualizing and intervening on transformer activations during inference. Stream layer activations over a socket, view them as images, cluster neurons by co-activation patterns, and apply real-time interventions (boost/ablate) to study causal effects.

## Overview

LLMRI provides:

- **Real-time streaming**: Watch activations change token-by-token during generation
- **Hierarchical clustering**: Group neurons by functional similarity (co-activation)
- **Interactive interventions**: Boost or ablate neuron clusters and observe output changes
- **Multiple visualizations**: Grid view, Hebbian layout, circle packing, spectral analysis

## Requirements

```
torch
numpy
pygame
scikit-learn (optional, improves clustering)
```

## Quick Start

### 1. Start the model with MRI server

```bash
python anima_v12.py --mri-server 9999
```

This loads the model and starts streaming activations on port 9999.

### 2. Connect the viewer

```bash
python mri_viewer.py --port 9999
```

The viewer connects and displays real-time activations.

### 3. Build functional clusters

Chat with the model for 20+ turns to collect activation samples, then press **O** to build the hierarchical clustering. This groups neurons that consistently activate together.

### 4. Intervene and observe

Click a cluster to select it, then:
- **B**: Boost (multiply activations by 1.5x default)
- **X**: Ablate (set activations to 0)
- Watch the model's output for changes

## Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│     Model       │                    │     Viewer      │
│  (anima_v12)    │                    │  (mri_viewer)   │
│                 │                    │                 │
│  ┌───────────┐  │   activations      │  ┌───────────┐  │
│  │  LLMRI    │──┼───────────────────►│  │  Display  │  │
│  │  hooks    │  │      (socket)      │  │           │  │
│  │           │◄─┼───────────────────-│  │  Cluster  │  │
│  └───────────┘  │   interventions    │  │           │  │
│                 │                    │  └───────────┘  │
└─────────────────┘                    └─────────────────┘
```

**Data flow:**
1. Forward hooks capture activations at each layer
2. Server broadcasts to connected viewers (JSON over TCP)
3. Viewer renders as images, collects samples for clustering
4. User applies interventions → sent back to server
5. Hooks modify activations on next forward pass

## Viewer Controls

### Navigation
| Key | Action |
|-----|--------|
| ↑/↓ | Change layer |
| Shift+Wheel | Zoom |
| Click+Drag | Pan |
| Q | Quit |

### Visualization Modes
| Key | Mode |
|-----|------|
| (default) | Grid view - neurons as pixels |
| H | Hebbian layout - 2D projection by co-activation |
| O | Circle packing - hierarchical clusters |
| F | Spectral (FFT) analysis |
| N | Influence map |

### Circle Pack Mode
| Key | Action |
|-----|--------|
| Click | Select cluster |
| Double-click | Zoom into cluster |
| Esc | Zoom out / exit |
| B | Boost selected (green tint) |
| X | Ablate selected (red tint) |
| [ / ] | Adjust boost amount |
| \ | Clear interventions |

### Display Options
| Key | Action |
|-----|--------|
| C | Cycle colormap |
| L | Toggle log scale |
| P | Toggle percentile clipping (1-99%) |
| G | Toggle grid overlay |
| D | Toggle diff mode (shows change from previous) |

## Files

| File | Purpose |
|------|---------|
| `anima_v12.py` | Main model runtime with LLMRI hooks |
| `mri_server.py` | Socket server for streaming activations |
| `mri_viewer.py` | PyGame-based visualization client |
| `circle_pack.py` | Hierarchical clustering and circle layout |
| `hebbian_layout.py` | 2D co-activation projection |

## How Clustering Works

1. **Sample collection**: Store activation vectors from layer 22 (configurable) across multiple inference steps

2. **Correlation matrix**: Compute pairwise correlation between all neurons based on co-activation patterns

3. **Agglomerative clustering**: Group neurons hierarchically using average linkage on correlation distance

4. **Circle packing**: Layout clusters as nested circles, sized by neuron count, colored by cluster ID

The result: neurons that fire together are grouped together. These clusters often correspond to functional modules (syntax processing, semantic features, attention patterns).

## Intervention Protocol

Interventions modify activations during the forward pass:

```python
# Multiplicative intervention
hidden[0, -1, neuron_idx] *= multiplier

# multiplier = 0.0  → ablation (neuron silenced)
# multiplier = 1.5  → boost (50% stronger)
# multiplier = 1.0  → no change
```

Interventions are applied at a specific layer (default: layer 22, middle of the network). This allows studying:

- **Necessity**: Does ablating this cluster break specific capabilities?
- **Sufficiency**: Does boosting this cluster enhance specific behaviors?
- **Causal structure**: How do perturbations propagate through layers?

## Example: Finding Function-Specific Clusters

```
1. Start model and viewer
2. Have a conversation about mathematics
3. Build clusters (O key)
4. Note which clusters are most active during math reasoning
5. Clear conversation, start new topic (e.g., poetry)
6. Ablate the "math-active" clusters
7. Return to math questions
8. Observe degraded math performance (if cluster was causal)
```

## Limitations

- **Correlational clustering**: Co-activation doesn't prove functional relatedness
- **Single-layer intervention**: Effects may be compensated by subsequent layers
- **Causal opacity**: Boosting/ablating tests necessity, not mechanism
- **Computational cost**: Real-time streaming adds ~10-20% overhead

## Technical Details

### Socket Protocol

Messages are length-prefixed JSON:
```
[4 bytes: uint32 length][JSON body]
```

**Server → Client (activations):**
```json
{
  "type": "scan",
  "layer": 22,
  "turn": 5,
  "token": 12,
  "activations": "hex-encoded float32 array",
  "stats": {"min": -2.1, "max": 8.3, "mean": 0.02, "std": 1.1}
}
```

**Client → Server (intervention):**
```json
{
  "type": "intervention",
  "layer": 22,
  "interventions": {"127": 0.0, "384": 1.5, "512": 0.0}
}
```

### Protected Neurons

Some neurons are critical infrastructure (e.g., attention scaling, layer norm). Ablating these crashes generation. The system maintains a protected list that blocks dangerous interventions by default.

## Citation

If you use this tool in research:

```
LLMRI: Real-time neural activation viewer for transformer interpretability
https://github.com/[repo]
```

## License

MIT
