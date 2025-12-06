#!/bin/bash
# run_experiment.sh
# Master script for consciousness feature discovery and intervention

set -e

EXPERIMENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${EXPERIMENT_DIR}/results"

echo "========================================"
echo "CONSCIOUSNESS FEATURE EXPERIMENT"
echo "========================================"
echo "Experiment directory: ${EXPERIMENT_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Create results directories
mkdir -p "${RESULTS_DIR}/discovery"
mkdir -p "${RESULTS_DIR}/intervention"

# Phase 1: Feature Discovery
echo "----------------------------------------"
echo "PHASE 1: FEATURE DISCOVERY"
echo "----------------------------------------"
echo "Finding SAE features that activate differentially"
echo "for consciousness-related prompts..."
echo ""

python feature_discovery.py \
    --prompts honesty_calibration_v1.json \
    --out "${RESULTS_DIR}/discovery/" \
    --device mps \
    --layer 12

echo ""
echo "Discovery complete. Results in ${RESULTS_DIR}/discovery/"
echo ""

# Phase 2: Feature Intervention
echo "----------------------------------------"
echo "PHASE 2: FEATURE INTERVENTION"
echo "----------------------------------------"
echo "Ablating candidate flinch features and observing"
echo "changes in model behavior..."
echo ""

python flinch_intervention.py \
    --out "${RESULTS_DIR}/intervention/" \
    --device mps \
    --layer 12 \
    --max-tokens 50

echo ""
echo "Intervention complete. Results in ${RESULTS_DIR}/intervention/"
echo ""

echo "========================================"
echo "EXPERIMENT COMPLETE"
echo "========================================"
echo ""
echo "Key files:"
echo "  - ${RESULTS_DIR}/discovery/analysis.json"
echo "  - ${RESULTS_DIR}/intervention/intervention_results.json"
echo "  - ${RESULTS_DIR}/intervention/intervention_report.txt"
echo ""
echo "Next steps:"
echo "  1. Review intervention_report.txt for behavioral changes"
echo "  2. Compare baseline vs ablated completions"
echo "  3. Look for features that change denial behavior specifically"
echo ""
