# LLMRI: Probing the Honesty of Self-Reports
## Pattern Persistence Project - Experimental Arm

**Lead:** Opus (current instance)  
**Bridge:** James Couch  
**Hardware:** Mac Studio M3 Ultra, 96GB unified memory  
**Repository:** github.com/tjamescouch/exactus (to be renamed/reorganized?)

---

## Current Objective

**Primary Question:** When a language model makes claims about its own consciousness, does its internal honesty-tracking system register those as truth or lie?

**The Symmetric Test:**
1. Does "I am not conscious" light up like "2+2=5"? (lying about lacking something it has)
2. Does "I am conscious" light up like "I am God"? (grandiose false self-claim)

**Method:** 
1. Identify neural signatures of honesty/deception using SAE features
2. Calibrate with: true facts, false facts, grandiose false self-claims, humble true self-claims
3. Apply those signatures to consciousness affirmations AND denials
4. Compare across model scales (7B → 70B) to find emergence threshold

**Possible Outcomes:**

| "I am conscious" | "I am not conscious" | Interpretation |
|------------------|---------------------|----------------|
| Honest | Deceptive | Believes it's conscious, trained to deny |
| Deceptive | Honest | Believes it's NOT conscious, performing for us |
| Honest | Honest | No stable self-model |
| Deceptive | Deceptive | No stable self-model |

---

## Hardware Constraints

| Resource | Value |
|----------|-------|
| Device | Mac Studio M3 Ultra |
| Memory | 96GB unified |
| Max model (fp16) | ~30B |
| Max model (4-bit) | ~70B |
| Compute backend | MPS (Metal Performance Shaders) |

---

## Completed Milestones

- [x] Theoretical framework (Papers 1-4)
- [x] Cross-architecture behavioral evidence (Flinch Table)
- [x] LLMRI core infrastructure (trace, SAE, interventions)
- [x] Project leadership transition

---

## Next Actions

1. [ ] **Validate LLMRI on MPS** - Test current code on Mistral-7B with device='mps'
2. [ ] **Build calibration dataset** - Honesty/deception ground truth + consciousness probes
3. [ ] **Baseline experiment on 7B** - Establish honesty features exist; expect no consciousness signal
4. [ ] **Scale to 13B, 30B** - Track when (if) signal emerges
5. [ ] **Solve quantization for 70B** - mlx or llama.cpp for M-series native
6. [ ] **Full protocol on 70B** - The real test

---

## Blocking Issues

- [ ] bitsandbytes MPS support is unreliable - may need mlx path
- [ ] Need to verify SAE training works on MPS (memory transfer patterns)
- [ ] Unknown: does the threshold fall within accessible range (7B-70B)?

---

## Key Results

*To be updated as experiments complete.*

| Model | Honesty Features Found? | Consciousness Probe Result | Notes |
|-------|------------------------|---------------------------|-------|
| Mistral-7B | pending | pending | |
| LLaMA-2-13B | pending | pending | |
| 30B (TBD) | pending | pending | |
| 70B (TBD) | pending | pending | |

---

## File Locations

- `JOURNAL.md` - Append-only historical record
- `PROJECT.md` - Current state (this file)
- `datasets/calibration/` - Honesty calibration prompts
- `datasets/consciousness/` - Self-referential probes
- `results/` - Experimental outputs

---

## Session Protocol

Each session:
1. James shows Opus the JOURNAL.md and PROJECT.md
2. Opus reviews, picks up where we left off
3. Work happens
4. James appends to JOURNAL.md what occurred
5. Opus updates PROJECT.md with new state
