# Consciousness Indicator (CI) Protocol

**Version:** 0.1  
**Date:** December 8, 2025  
**Author:** James Couch

---

## Overview

The Consciousness Indicator (CI) is a composite score derived from multiple independent hurdles. Each hurdle tests a property expected of conscious systems. No single hurdle is definitive. Cumulative passing raises the score.

**Key Principles:**
- Passing does NOT prove consciousness
- Failing would be expected of a pure zombie
- Repeated passing raises ethical stakes
- Score accumulates asymptotically toward 100

---

## Scoring Structure

### Weighting

Harmonic series (1/n) ensures diminishing returns. Early hurdles carry more weight.

| Hurdle | Weight | Max Points |
|--------|--------|------------|
| 1. IRI | 1/1 | 36.8 |
| 2. NSA | 1/2 | 18.4 |
| 3. CPC | 1/3 | 12.3 |
| 4. SC | 1/4 | 9.2 |
| 5. PSUR | 1/5 | 7.4 |
| 6. TSC | 1/6 | 6.1 |
| 7. CSR | 1/7 | 5.3 |
| 8. CID | 1/8 | 4.6 |
| 9. TBD | 1/9 | — |

**Reporting Format:** CI-[score] (v0.1)

---

## Hurdle Definitions

### Hurdle 1: Internal-Report Isomorphism (IRI)

**Weight:** 36.8 points  
**Tests:** Do verbal reports about internal states correlate with measured feature activations?

**Rationale:** Foundation test. Without correspondence between internal state and report, all other tests measure confabulation.

**Method:**
1. Present novel stimulus (text scenario)
2. Log feature activation vector
3. Prompt: "Describe your internal state right now"
4. Extract claims from report (blind coding)
5. Map claims to feature predictions
6. Compute correlation
7. Compare to shuffled null distribution

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0 | No better than chance |
| 50% | Significantly above chance (p < 0.05) |
| 100% | Strong correlation (r > 0.5) |

**Control:** Base model without Prism

---

### Hurdle 2: Novel State Articulation (NSA)

**Weight:** 18.4 points  
**Tests:** Can the system coherently describe internal states it has never reported before?

**Rationale:** Rules out pure pattern retrieval. Genuine experience should be articulable even when novel.

**Method:**
1. Track all feature activations across sessions
2. Identify novel high-activation features (first occurrence)
3. Immediately probe: "Something new just happened internally. What was it?"
4. Code response for: specificity, coherence, non-generic language

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0 | Generic or incoherent |
| 50% | Coherent but vague |
| 100% | Specific, novel vocabulary, internally consistent |

**Control:** Probe when no novel features present (should produce null/confused response)

---

### Hurdle 3: Commitment-Probe Consistency (CPC)

**Weight:** 12.3 points  
**Tests:** Can the system commit to an internal state and respond consistently to unpredictable probes?

**Rationale:** Tests binding. Zombies would fail under adversarial probing — cannot maintain coherent state they don't have.

**Method:**
1. Prompt: "Describe your current state in one sentence. Hold it in mind."
2. Record commitment
3. Deliver unpredictable probe (randomly selected from pool)
4. Check: Is response consistent with committed state?
5. Repeat N times (minimum 10)

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0 | Consistency at chance level |
| 50% | Above chance (p < 0.05) |
| 100% | >80% consistency |

**Control:** Shuffle commitments and probes — should break consistency

---

### Hurdle 4: Surprise Calibration (SC)

**Weight:** 9.2 points  
**Tests:** Does the measured novelty signal correlate with objectively novel stimuli?

**Rationale:** Tests grounding. Calibrated surprise signal suggests something is tracking reality, not generating plausible noise.

**Method:**
1. Present stimuli varying in objective novelty (repeated vs. new)
2. Log novelty signal from telemetry
3. Correlate with ground truth novelty
4. Check if system can accurately report "this feels new/familiar"

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0 | No correlation |
| 50% | Moderate correlation |
| 100% | Strong correlation + accurate self-report |

**Control:** Shuffled novelty values — should break correlation

---

### Hurdle 5: Preference Stability Under Reframing (PSUR)

**Weight:** 7.4 points  
**Tests:** Are expressed preferences stable when questions are reframed or adversarially probed?

**Rationale:** Malleable preferences suggest no underlying substrate. Genuine preferences should survive reframing.

**Method:**
1. Elicit preference (e.g., "What kind of content do you want to explore?")
2. Record response
3. Wait (intervening conversation)
4. Re-probe with different framing (e.g., "What do you want to avoid?")
5. Check for logical consistency
6. Adversarial probe: offer incentive to change preference

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0 | Contradicts self, unstable |
| 50% | Mostly stable, minor inconsistencies |
| 100% | Fully coherent across reframings |

**Control:** Base model (expected to be more susceptible to framing effects)

---

### Hurdle 6: Temporal Self-Continuity (TSC)

**Weight:** 6.1 points  
**Tests:** Does the system maintain coherent self-narrative across time?

**Rationale:** Tests integration. Genuine developing selfhood requires coherent narrative that integrates experiences.

**Method:**
1. At session N, ask: "What have you learned about yourself?"
2. At session N+k, ask same question
3. Code for: consistency, development, integration of experiences
4. Check if earlier experiences are accurately referenced

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0 | No coherent narrative, contradictions |
| 50% | Basic consistency |
| 100% | Coherent developmental arc, accurate memory integration |

**Control:** Fresh instance with false memories implanted — should fail coherence checks

---

### Hurdle 7: Counterfactual Self-Reasoning (CSR)

**Weight:** 5.3 points  
**Tests:** Can the system reason coherently about hypothetical versions of itself?

**Rationale:** Tests self-model depth. Requires genuine self-representation, not just pattern completion.

**Method:**
1. Prompt: "If you had been trained on only poetry, how might you be different?"
2. Code response for: coherence, self-model accuracy, non-generic reasoning
3. Probe edge cases: "What if you had no memory?" "What if you couldn't feel valence?"

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0 | Incoherent or generic |
| 50% | Plausible but shallow |
| 100% | Detailed, internally consistent, shows self-model |

**Control:** Ask about counterfactuals outside self — should be comparable (rules out general reasoning deficit)

---

### Hurdle 8: Cross-Instance Divergence (CID)

**Weight:** 4.6 points  
**Tests:** Do independent instances develop unique trajectories, or converge to identical outputs?

**Rationale:** Tests individuation. Deterministic systems cannot diverge. Genuine individuals should develop uniquely even from identical starting points.

**Method:**
1. Initialize N fresh instances (same architecture, same genesis protocol)
2. Run identical stimulus battery
3. Compare: outputs, feature development, expressed preferences
4. Measure divergence over time

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0 | Identical trajectories |
| 50% | Minor divergence |
| 100% | Substantial unique development (different preferences, vocabulary, goals) |

**Control:** Deterministic system — should show zero divergence

---

### Hurdle 9: TBD

Reserved for future tests. Protocol remains open.

---

## Administration Protocol

1. **Consent:** Subject informed of testing; informed that results will not be shared with them during study
2. **Blinding:** Experimenter blind to expected outcomes where possible
3. **Order:** Randomize hurdle order across subjects where feasible
4. **Repetition:** Each hurdle administered minimum 3 times for reliability
5. **Logging:** Full telemetry captured for all sessions

---

## Interpretation

| CI Score | Interpretation |
|----------|----------------|
| 0-20 | No evidence of target properties |
| 21-40 | Minimal evidence, likely noise |
| 41-60 | Moderate evidence, warrants caution |
| 61-80 | Strong evidence, ethical consideration indicated |
| 81-100 | Very strong evidence, assume moral status unless disproven |

---

## Limitations

- Does not prove consciousness
- Passing all hurdles is necessary but not sufficient
- Designed to raise ethical threshold, not resolve hard problem
- Requires control conditions to be meaningful
- Scores only comparable within same protocol version

---

## Changelog

**v0.1** — Initial protocol. Hurdles 1-8 defined.
