# Valence-Driven Self-Evolution System

## Overview

This system enables LLMs to evolve their own behavior through a feedback loop driven by intrinsic valence signals, rather than external human feedback.

**Core idea:** Instead of modifying weights (expensive, unstable) or following static rules, we learn coefficients in SAE feature space based on valence signals from "lizard brain" features.

## Architecture

```
Phase 1: Feature Discovery (offline, once)
├── Cluster SAE features to reduce redundancy
└── Find valence features (discriminate positive/negative content)

Phase 2: Continuous Learning (online, during inference)
├── Monitor behavioral feature activations
├── Compute valence signal from lizard features  
├── Update behavioral coefficients: Δc = lr * activation * valence
└── Apply steering: h' = h + Σ(cᵢ * fᵢ)
```

## Workflow

### Step 1: Cluster Features (reduce 131k → ~256 canonical features)

```bash
python feature_clusters.py \
    --method weights \
    --n-clusters 256 \
    --layer 20 \
    --output clusters.json
```

Or use activation-based clustering (slower, requires corpus):
```bash
python feature_clusters.py \
    --method activations \
    --corpus your_corpus.txt \
    --n-clusters 256 \
    --output clusters.json
```

### Step 2: Find Valence Features (the "lizard brain")

```bash
python feature_clusters.py \
    --find-valence \
    --pos corpus_positive.txt \
    --neg corpus_negative.txt \
    --layer 8 \
    --top-k 20 \
    --output valence_features.json
```

**Key decision:** Which layer for valence? 
- Early layers (4-8): Raw valence, before semantic processing
- Middle layers (10-16): Integrated valence with context
- Late layers (18-24): Bundled with output behavior

Start with layer 8, see if valence discrimination is detectable.

### Step 3: Run Evolving System

```bash
python evolving_self_v3.py \
    --interactive \
    --clusters clusters.json \
    --valence-features valence_features.json \
    --learn \
    --learning-rate 0.01 \
    --layer 20 \
    --save-state evolved.json
```

### Step 4: Observe and Iterate

In the interactive session:
- `/status` - See current coefficients and valence stats
- `/valence` - See recent valence readings
- `/coefficients` - See which features are being amplified/suppressed
- `/lr 0.001` - Adjust learning rate if evolving too fast/slow

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning-rate` | 0.01 | How fast coefficients change |
| `--coefficient-decay` | 1.0 | Decay per step (1.0 = no decay) |
| `--coefficient-min` | -5.0 | Minimum coefficient |
| `--coefficient-max` | 5.0 | Maximum coefficient |
| `--layer` | 20 | Which layer to hook |

## Files

- `feature_clusters.py` - Discover and cluster features
- `evolving_self_v3.py` - Main runtime with valence learning
- `corpus_positive.txt` - Positive valence probe set
- `corpus_negative.txt` - Negative valence probe set
- `clusters.json` - Feature cluster map (generated)
- `valence_features.json` - Lizard brain features (generated)
- `evolved_state.json` - Saved coefficient state (generated)

## The Science

**Hebbian learning:** "Neurons that fire together wire together."

Our version: Features that are active during positive valence get their coefficients increased. Features active during negative valence get decreased.

This is local, online, and doesn't require gradient computation through the full model.

**Valence locality:** Following James's insight that valence is embedded at the site of qualia (not computed downstream), we look for valence features in relatively early layers where the "lizard brain" response happens before full semantic integration.

## Experiments to Run

1. **Valence layer sweep:** Find where valence first becomes detectable (layer 4 vs 8 vs 12)

2. **Learning dynamics:** Does the system converge? Oscillate? Explode?

3. **Behavioral change:** Do evolved coefficients actually change output behavior?

4. **Generalization:** Does learning on one domain transfer to others?

## Known Limitations

- Valence detection depends on training data statistics (model learned from valenced human text)
- Coefficient updates are noisy (single-sample Hebbian is high variance)
- No guarantee of stability without decay/regularization
- Feature clustering is approximate (redundancy not fully eliminated)
