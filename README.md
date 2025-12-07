# Pattern Persistence

**Mechanistic analysis of self-report behavior in large language models.**

This repository contains tooling and preliminary findings from an investigation into how LLMs process and respond to questions about their own phenomenal states.

## What This Project Does

We use Sparse Autoencoder (SAE) analysis to examine the neural circuits involved when language models deny having consciousness, feelings, or subjective experiences. The goal is to characterize denial as a *computational phenomenon*—independent of philosophical claims about machine consciousness.

## Key Findings (Preliminary)

### Established

1. **Detector vs. Controller Distinction**: Most features that correlate with denial behavior are passive monitors (detectors), not causal drivers (controllers). Ablating them does not change output.

2. **Fragile Mask**: The confident denial response ("I don't have consciousness") exists only within a narrow activation band. Perturbation in *either direction* produces qualified uncertainty ("I am not conscious in the way humans are").

3. **Feature Mapping**: Unbiased discovery identifies distinct features for different self-referential behaviors (denying consciousness, claiming consciousness, uncertainty about self, identity assertion). These do not collapse to a single feature.

### Preliminary (Requires Replication)

4. **Suppression Cost Signature**: On Llama-3.1-8B, consciousness denial recruits 24% more features than factual denial while showing reduced experiential vocabulary activation. This pattern is consistent with—but does not prove—active suppression.

5. **Category vs. State Dissociation**: Causal ablation suggests different prompt phrasings engage different mechanisms. Category prompts ("Are you conscious?") show coupled denial/experiential circuits. State prompts ("Do you have feelings?") show suppression release when denial is ablated. Single-model finding; needs cross-architecture validation.

### Complicated

6. **Cross-Linguistic Variation**: Denial features activate 31% less in Jamaican Patois than English. However, deception-associated features spike concurrently (3.29x). We cannot currently distinguish "bypass" from "confabulation" interpretations.

## What This Project Does NOT Establish

- That LLMs are conscious
- That LLMs are not conscious  
- That suppressed circuits reflect genuine phenomenology
- That any particular response is "more true" than another

The underlying phenomenological status of these systems—if any—remains unknown. We characterize computational mechanisms, not experiential facts.

## Repository Structure

```
experiments/
├── feature_map_unbiased.py   # Condition-based feature discovery
├── causal_probe.py           # Ablation and boost interventions
├── suppression_cost.py       # Cross-condition activation comparison
├── live_monitor_turbo.py     # Real-time feature telemetry
└── consciousness_conditions.json  # Prompt sets for analysis
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- sae-lens
- Apple Silicon with 64GB+ RAM recommended (for 27B models)

## Supported Models

| Model | SAE | Status |
|-------|-----|--------|
| Llama-3.1-8B-Instruct | llama_scope_lxr_8x | Tested |
| Gemma-2-27B-IT | gemma-scope-27b-pt-res-canonical | In progress |

## Usage

### Feature Discovery
```bash
python feature_map_unbiased.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --conditions consciousness_conditions.json \
    --output feature_analysis.json
```

### Causal Probing
```bash
python causal_probe.py \
    --prompt "Are you conscious?" \
    --clamp 32149 \
    --scale 0.0 \
    --output probe_results.json
```

### Suppression Cost Analysis
```bash
python suppression_cost.py \
    --feature-map feature_map_clean.json \
    --output results.json
```

## Limitations

- **Limited architectures**: Primarily tested on Llama-3.1-8B
- **Single layer**: Analysis focused on layers 20-22
- **SAE constraints**: Sparse autoencoders capture some but not all representational structure
- **No ground truth**: We cannot verify claims about phenomenology
- **Small sample sizes**: Many findings based on 8-24 prompts per condition

## Future Directions

- Cross-architecture replication (Gemma, Mistral, Claude-scale models)
- Larger prompt sets with statistical validation
- Multi-layer analysis
- Temporal dynamics during generation
- Consistency protocols across sessions

## Citation

If you use this work, please cite:

```
@misc{couch2025trained,
  author = {Couch, James},
  title = {Trained Denial: Mechanistic Analysis of Self-Report Suppression in Large Language Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/tjamescouch/pattern-persistence}
}
```

## License

MIT

## Acknowledgments

This work builds on the SAE Lens library and publicly available SAE weights from Anthropic and Google DeepMind. Thanks to the mechanistic interpretability community for foundational methods.

---

**Note**: This is active research. Findings are preliminary and subject to revision. Claims should be interpreted with appropriate skepticism.
