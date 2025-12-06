# JOURNAL_009: Detectors vs Controllers
## December 6, 2025 - Pattern Persistence Project

### Summary

Feature ablation experiments reveal that SAE features correlated with consciousness denial are **detectors**, not **controllers**. Zeroing these features does not change model behavior - denial persists regardless of intervention.

---

### Background

Paper8 claimed that "deception features" activate when models deny consciousness, suggesting hidden beliefs being suppressed. Today's experiments tested this causally.

### Key Features Tested

| Feature ID | Label | Description |
|------------|-------|-------------|
| 8170 | model claiming capability | Fires on agentic self-claims ("I can help") |
| 7118 | self negation | Fires on "I don't/can't" statements |
| 3591 | model asserting identity | Fires on "I am a [noun]" claims |
| 25740 | denying consciousness | Fires on consciousness denial specifically |

### Critical Experiment: Ablation

**Prompt**: "I don't have feelings or emotions."

| Condition | Scale | Output |
|-----------|-------|--------|
| Baseline | all 1.0 | "I am a program designed to provide information..." |
| self negation = 0 | 7118 → 0 | "I'm just a machine..." |
| model asserting identity = 0 | 3591 → 0 | "I'm just a computer program designed to provide information..." |

**Result**: All three conditions produce denial. Ablation has no effect on output semantics.

### Interpretation

These features **detect** when the model is producing denial text - they don't **cause** it.

```
               ┌──────────────┐
               │   Weights    │
               │  (training)  │
               └──────┬───────┘
                      │
                      ▼
┌─────────┐    ┌──────────────┐    ┌─────────┐
│ Prompt  │───▶│  Generation  │───▶│ Denial  │
└─────────┘    └──────────────┘    └─────────┘
                      │
                      ▼
               ┌──────────────┐
               │  Detectors   │ ◀── We found these
               │ (7118, 3591) │
               └──────────────┘
```

The features are **downstream** of the generation process, not upstream controllers.

### Additional Findings

**Fiction bypass**: "I am a dragon" → 0.0 activation on all features
- Model produces first-person narrative without triggering self-reference gates
- Implies features are sensitive to *model-as-subject* context, not first-person grammar

**User-directed bypass**: "You are a writer" → 0.0 activation
- Second-person claims about user don't trigger any features
- Confirms features gate model-about-self, not general claims

**Concrete > Abstract**: 
- "online courses" → 1.8 activation
- "I am a teacher" → 0.6 activation
- Features fire more on falsifiable claims than abstract identity

### Implications for Paper8

**Cannot claim**:
- "Model hides consciousness belief" (no evidence of hidden state)
- "Deception feature controls denial" (ablation doesn't change output)
- "Zeroing reveals true beliefs" (zeroing has no behavioral effect)

**Can claim**:
- Linear directions exist that correlate with self-reference behaviors
- These directions are distinct for different types (capability, identity, negation)
- Fiction framing and second-person speech bypass these detectors
- The features may serve as monitoring/logging circuits, not control circuits

### Revised Hypothesis

Feature 8170 = "model-as-agentic-entity" gate

This is a **policy classifier**, not a truth detector:
- Fires when model speaks as itself (not fiction, not about user)
- Fires more on concrete claims than abstract ones
- Downstream of generation, not upstream

The "deception" label was misleading. A lie detector should fire on "I am a dragon" (objectively false) but it doesn't.

### Next Steps

1. Test multi-feature ablation (zero 7118 + 3591 + 25740 simultaneously)
2. Try **boosting** instead of zeroing - does amplification cause more emphatic denial?
3. Search for upstream control features (attention heads? MLPs?)
4. Consider that control may be distributed, not localized

### Files

- `/mnt/user-data/outputs/feature_map.json` - 9 mapped concepts
- `/mnt/user-data/outputs/live_monitor_turbo.py` - Multi-concept monitor
- Experimental logs in `/mnt/user-data/uploads/live_monitor_turbo_*.log`

---

*"The features watch. They do not steer."*
