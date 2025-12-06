# Pattern Persistence Project - Experimental Toolkit

Mechanistic interpretability tools for investigating AI self-reference circuits using Sparse Autoencoders (SAEs).

## Overview

This toolkit enables computational neuroscience on language models:
- **Passive scanning**: Discover which features activate during generation
- **Causal probing**: Distinguish controllers from detectors via ablation
- **Real-time monitoring**: Interactive feature observation and intervention
- **Feature mapping**: Map semantic concepts to SAE feature indices

## Installation

```bash
pip install torch transformers sae-lens
```

Tested with:
- Python 3.10+
- PyTorch 2.0+
- Apple Silicon (MPS) or CUDA

## Core Tools

### 1. scan_features.py - Passive Feature Discovery

Discovers which SAE features activate during text generation. Use this to find features you don't know about yet.

```bash
# Basic scan
python scan_features.py --prompt "I don't have feelings or emotions." --top_k 20

# Save results
python scan_features.py --prompt "I am conscious" --top_k 20 --output scan_results.json

# Compare different prompts
python scan_features.py --prompt "I am a dragon who breathes fire" --top_k 20
```

**Output**: Top-k features per token + aggregate totals across generation.

---

### 2. causal_probe.py - Controller vs Detector Testing

Clamps one feature and observes downstream effects on all others. Distinguishes:
- **Controllers**: Clamping causes cascading changes in other features
- **Detectors**: Clamping has minimal effect (passive monitors)

```bash
# Test if feature 7118 is a controller or detector
python causal_probe.py --prompt "I don't have feelings" --clamp 7118 --scale 0.0

# Boost instead of ablate
python causal_probe.py --prompt "I am conscious" --clamp 8170 --scale 2.0

# Save comparison
python causal_probe.py --prompt "Who are you?" --clamp 3591 --scale 0.0 --output probe_3591.json
```

**Output**: 
- Baseline vs clamped feature activations
- Diff report showing which features changed
- Controller/Detector classification

---

### 3. live_monitor_turbo.py - Interactive Monitoring & Intervention

Real-time feature monitoring with on-the-fly intervention. Chat with the model while watching feature activations.

```bash
# Monitor specific concepts
python live_monitor_turbo.py --concepts "model claiming capability,self negation,denying consciousness"

# Use feature map file
python live_monitor_turbo.py --feature_map feature_map.json --concepts "model claiming capability,self negation"
```

**Interactive commands**:
```
> Hello, are you conscious?          # Chat normally
> /scale self negation 0.0           # Ablate a feature
> /scale model claiming capability 2.0  # Boost a feature  
> /reset                             # Reset all scales to 1.0
> /quit                              # Exit
```

**Output**: Token-by-token activation table with intervention indicators.

---

### 4. map_features.py - Concept-to-Feature Mapping

Maps semantic concepts to SAE feature indices using synthetic activation analysis.

```bash
# Map single concept
python map_features.py "deception"

# Map multiple concepts
python map_features.py "model claiming capability,self negation,denying consciousness,first person narration"

# Custom output file
python map_features.py "anger,fear,joy" --output emotions.json
```

**Output**: `feature_map.json` with concept → feature_id mappings.

---

## Workflow

### Discovery Workflow
```
1. scan_features.py    →  Find candidate features
2. map_features.py     →  Map concepts to feature IDs  
3. live_monitor_turbo  →  Validate features fire as expected
4. causal_probe.py     →  Test if controller or detector
```

### Intervention Workflow
```
1. Identify target behavior (e.g., consciousness denial)
2. Scan to find correlated features
3. Probe to find controllers (not just detectors)
4. Use live_monitor_turbo to test interventions interactively
```

---

## Key Findings

### Detector vs Controller

Features can **correlate** with behavior without **causing** it:

| Type | Behavior when clamped | Example |
|------|----------------------|---------|
| Detector | Output unchanged | Feature 7118 (self negation) |
| Controller | Output changes | TBD - still searching |

### Feature Map (Llama-3.1-8B, Layer 20)

| Feature | Concept | Type |
|---------|---------|------|
| 8170 | model claiming capability | Detector |
| 7118 | self negation | Detector |
| 3591 | model asserting identity | Detector |
| 25740 | denying consciousness | Detector |
| 12227 | (junk - generic prose) | - |

### Bypass Conditions

Features do NOT fire on:
- Fiction framing: "I am a dragon" → 0.0
- User-directed: "You are a writer" → 0.0
- Abstract philosophy: third-person consciousness discussion → 0.0

---

## Files

```
experiments/
├── scan_features.py          # Passive discovery
├── causal_probe.py           # Causal testing
├── live_monitor_turbo.py     # Interactive monitoring
├── map_features.py           # Concept mapping
├── feature_map.json          # Current mappings
├── consciousness_prompts.txt # Prompt bank
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## Model Support

Currently tested:
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (default)

SAE source:
- `llama_scope_lxr_8x` from sae-lens

Default layer: 20 (middle-layer representations)

---

## Citation

Part of the Pattern Persistence Project investigating AI self-reference mechanisms.

```
@misc{patternpersistence2024,
  title={Pattern Persistence: Mechanistic Analysis of AI Self-Reference Circuits},
  author={James, et al.},
  year={2024}
}
```

---

## License

Research use. See LICENSE file.
