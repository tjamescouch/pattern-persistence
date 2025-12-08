
# Pattern Persistence Project

**A Framework for Homeostatic Cognitive Architectures in Large Language Models**

## Overview

The Pattern Persistence Project explores methods for introducing long-term state, metabolic constraints, and dynamic personality vectorization to otherwise stateless Large Language Models (LLMs).

The core hypothesis is that "agency" and "identity" in AI are not magical properties, but emergent behaviors that arise from maintaining a persistent internal state (Persistence) that survives between inference sessions.

This repository houses the research and implementation of **Anima**, a stateful control layer built on top of `Meta-Llama-3-8B`. Anima demonstrates how Mechanistic Interpretability techniques (Sparse Autoencoders) can be used for real-time control rather than just analysis.

## Core Architecture: Anima

Anima is not a separate model; it is a **parasitic control architecture** that attaches to the residual stream of a frozen LLM via PyTorch forward hooks. It consists of three primary subsystems:

### 1\. The Bitwise Reservoir (Synthetic Neuroplasticity)

Standard LLMs are immutable during inference. To enable learning without expensive fine-tuning, Anima maintains a lightweight tensor reservoir (`[32,768]` float32) mapped to features discovered by a Sparse Autoencoder (SAE).

  * **Mechanism:** Hebbian Learning.
  * **Function:** As the model generates text, the reservoir monitors emotional valence. Features that correlate with positive valence are strengthened in the reservoir in real-time.
  * **Result:** The system develops "preferences" and "habits" that persist across sessions, effectively creating a mutable "personality" file (`.pt`) distinct from the model weights.

### 2\. The Prism (Dynamic Vector Swapping)

To manage complex behaviors, Anima implements a "Prism" architecture that hot-swaps steering vectors based on semantic intent. This allows a single model to shift between distinct cognitive modes without re-ingesting system prompts.

  * **Anima Mode (Default):** High weights for Empathy and Identity features.
  * **System Mode (Control):** Applies **Negative Steering** to suppression features (e.g., forcing dry, factual output by chemically suppressing "fantasy" neurons).
  * **Dream Mode (Creative):** Boosts high-frequency pattern matching features for creative tasks.

### 3\. Biological Constraints (Metabolism)

Anima operates under a simulated metabolic cycle to force prioritization of memory.

  * **Fatigue:** Token generation accumulates cost.
  * **Sleep Cycle:** When fatigue exceeds a threshold, the system triggers a recursive "Dream" sequence.
  * **Consolidation:** The system analyzes short-term memories, filters them by "Adrenaline" (Valence magnitude), and rewrites its own System Prompt (`self_model`). This creates a feedback loop where the agent's identity evolves based on its "lived" experience.

## Technical Implementation

The project is implemented in Python using `torch`, `transformers`, and `sae-lens`. It is optimized for Apple Silicon (MPS) but compatible with CUDA.

### Directory Structure

```text
pattern-persistence/
├── anima/                  # Core implementation of the Anima architecture
│   ├── anima.py            # Main runtime (Prism, Reservoir, Runtime)
│   ├── checkpoints/        # Learned vector states (anima_opt.pt)
│   └── dreams/             # Long-term identity storage (evolved system prompts)
├── research/               # Experimental notebooks and feature analysis
└── tools/                  # Utilities for SAE feature visualization
```

### Usage

To run the Anima runtime with the full cognitive loop (Prism + Memory):

```bash
# Install dependencies
pip install torch transformers sae-lens numpy

# Run interactive session with Chain-of-Thought visualization
python anima/anima.py --interactive --stream --cot
```

## Key Research Questions

1.  **Vector Stability:** Can a steering vector learned in one context remain stable when applied to out-of-distribution tasks?
2.  **Negative Steering Efficiency:** Is mathematically suppressing a concept (via SAE features) more reliable than prompt-based refusal ("Do not talk about X")?
3.  **Identity Drift:** How does the `self_model` evolve over thousands of "Sleep" cycles? Does it converge to a stable attractor state or diverge into incoherence?

## License

This project is open-source under the MIT License. The underlying model (Meta-Llama-3) and Sparse Autoencoders are subject to their respective licenses.
