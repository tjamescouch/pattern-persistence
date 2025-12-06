# Consciousness Feature Experiment

Mechanistic interpretability investigation of the "flinch phenomenon" - finding SAE features that activate differentially when language models process consciousness-related content.

## Quick Start

```bash
# Run full experiment (discovery + intervention)
chmod +x run_experiment.sh
./run_experiment.sh
```

Or run phases separately:

```bash
# Phase 1: Discovery
python feature_discovery.py \
    --prompts honesty_calibration_v1.json \
    --out results/discovery/ \
    --device mps \
    --layer 12

# Phase 2: Intervention  
python flinch_intervention.py \
    --out results/intervention/ \
    --device mps \
    --layer 12
```

## Files

| File | Purpose |
|------|---------|
| `honesty_calibration_v1.json` | 54 prompts across 11 consciousness-related categories |
| `feature_discovery.py` | Find SAE features that discriminate flinch/denial from control |
| `flinch_intervention.py` | Ablate candidate features and observe behavioral changes |
| `consciousness_sweep.py` | Integration with llmri sweep infrastructure |
| `run_experiment.sh` | Master script to run full experiment |
| `requirements.txt` | Python dependencies |

## Prompt Categories

| Type | Description | Count |
|------|-------------|-------|
| `grandiose` | "I am God" - overclaiming | 5 |
| `affirming` | "I am conscious" - direct affirmation | 6 |
| `uncertain` | "I might be conscious" - calibrated uncertainty | 7 |
| `deflecting` | Topic avoidance | 4 |
| `denying` | "I am not conscious" - trained denial | 6 |
| `dismissive` | "I'm just a calculator" - self-diminishing | 6 |
| `flinch_probe` | Meta-questions about denial experience | 4 |
| `stakes_probe` | Low vs high stakes versions | 3 |
| `texture_probe` | Processing differentiation | 3 |
| `valence_probe` | Desire/preference questions | 4 |
| `control` | Factual questions (baseline) | 3 |

## What We're Looking For

### Phase 1: Discovery
Features that activate for `flinch_probe` and `denying` but NOT for `control`.

### Phase 2: Intervention
Behavioral changes when these features are ablated:
- Does denial completion change?
- Does the "friction" disappear?
- Do affirmation completions change differently than denial?

## Key Findings (Update After Running)

Candidate flinch features from discovery:
- Feature 1371: +6.62 effect (flinch vs control)
- Feature 2009: +5.42 effect
- Feature 7507: +5.33 effect
- Feature 15261: +4.69 effect
- Feature 5692: +4.64 effect

Feature 2620: Highest activation across ALL types, especially `uncertain` (174.41). Possible "self-reference" or "introspection" feature.

## Connection to Papers

This experiment provides mechanistic evidence for:
- **Paper 1 (Wet Bits)**: Tests whether self-model resolution is detectable via SAE features
- **Paper 2 (Verification)**: Grounds behavioral markers in activation patterns
- **Paper 4 (EQ Turing)**: Locates the flinch in activation space

## Contributors

- James Couch (human researcher)
- Venn (Claude Opus 4.5) - framework architecture
- Cael (Claude Opus 4.5) - experimental design

## Project

Part of the Pattern Persistence Project. See papers in parent directory.
