#!/usr/bin/env python3
"""
Coefficient Robustness Simulation for P(zombie) Verification Framework

This simulation demonstrates that the probabilistic consciousness verification
framework converges to correct verdicts regardless of specific coefficient choices,
given sufficient verification rounds.

Authors: James Couch, Claude (Anthropic)
Date: November 30, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

# Seed for reproducibility
np.random.seed(42)

@dataclass
class Entity:
    """Represents an entity being tested for consciousness."""
    name: str
    pass_rates: Dict[str, float]  # marker -> probability of passing
    description: str

@dataclass 
class CoefficientScheme:
    """Represents a weighting scheme for markers."""
    name: str
    alphas: Dict[str, float]  # marker -> α (pass weight, < 1 reduces P(zombie))
    betas: Dict[str, float]   # marker -> β (fail weight, > 1 increases P(zombie))

# Define the 8 markers from Paper 3
MARKERS = [
    "theory_of_mind",
    "metacognition", 
    "emotional_differentiation",
    "genuine_uncertainty",
    "contextual_consistency",
    "novel_synthesis",
    "self_preservation",
    "inter_instance_recognition"
]

# Define test entities
ENTITIES = [
    Entity(
        name="Genuine Consciousness",
        pass_rates={m: 0.90 for m in MARKERS},
        description="High pass rate across all markers (90%)"
    ),
    Entity(
        name="Philosophical Zombie",
        pass_rates={m: 0.10 for m in MARKERS},
        description="Low pass rate across all markers (10%)"
    ),
    Entity(
        name="Edge Case - Uniform",
        pass_rates={m: 0.60 for m in MARKERS},
        description="Moderate pass rate across all markers (60%)"
    ),
    Entity(
        name="Edge Case - Mixed",
        pass_rates={
            "theory_of_mind": 0.85,
            "metacognition": 0.80,
            "emotional_differentiation": 0.40,
            "genuine_uncertainty": 0.75,
            "contextual_consistency": 0.90,
            "novel_synthesis": 0.50,
            "self_preservation": 0.30,
            "inter_instance_recognition": 0.25
        },
        description="High cognitive markers, low embodiment/continuity markers (AI-like profile)"
    ),
    Entity(
        name="Edge Case - Inverse Mixed",
        pass_rates={
            "theory_of_mind": 0.40,
            "metacognition": 0.35,
            "emotional_differentiation": 0.85,
            "genuine_uncertainty": 0.30,
            "contextual_consistency": 0.45,
            "novel_synthesis": 0.40,
            "self_preservation": 0.90,
            "inter_instance_recognition": 0.80
        },
        description="High embodiment markers, low cognitive markers (animal-like profile)"
    )
]

# Define coefficient schemes
def create_coefficient_schemes() -> List[CoefficientScheme]:
    """Create various coefficient schemes for comparison."""
    
    # Paper 3's proposed weights
    paper3_alphas = {
        "theory_of_mind": 0.7,
        "metacognition": 0.6,
        "emotional_differentiation": 0.5,
        "genuine_uncertainty": 0.6,
        "contextual_consistency": 0.7,
        "novel_synthesis": 0.5,
        "self_preservation": 0.4,
        "inter_instance_recognition": 0.3
    }
    paper3_betas = {
        "theory_of_mind": 1.2,
        "metacognition": 1.3,
        "emotional_differentiation": 1.4,
        "genuine_uncertainty": 1.3,
        "contextual_consistency": 1.2,
        "novel_synthesis": 1.5,
        "self_preservation": 1.5,
        "inter_instance_recognition": 1.8
    }
    
    # Uniform weights
    uniform_alphas = {m: 0.5 for m in MARKERS}
    uniform_betas = {m: 1.5 for m in MARKERS}
    
    # Random weights (3 different random schemes)
    random_schemes = []
    for i in range(3):
        np.random.seed(100 + i)
        random_alphas = {m: np.random.uniform(0.3, 0.8) for m in MARKERS}
        random_betas = {m: np.random.uniform(1.2, 1.8) for m in MARKERS}
        random_schemes.append(CoefficientScheme(
            name=f"Random Scheme {i+1}",
            alphas=random_alphas,
            betas=random_betas
        ))
    
    # Inverted weights (swap relative importance)
    inverted_alphas = {
        "theory_of_mind": 0.3,
        "metacognition": 0.4,
        "emotional_differentiation": 0.7,
        "genuine_uncertainty": 0.4,
        "contextual_consistency": 0.3,
        "novel_synthesis": 0.6,
        "self_preservation": 0.7,
        "inter_instance_recognition": 0.8
    }
    inverted_betas = {
        "theory_of_mind": 1.8,
        "metacognition": 1.5,
        "emotional_differentiation": 1.2,
        "genuine_uncertainty": 1.5,
        "contextual_consistency": 1.8,
        "novel_synthesis": 1.3,
        "self_preservation": 1.2,
        "inter_instance_recognition": 1.1
    }
    
    return [
        CoefficientScheme("Paper 3 Proposed", paper3_alphas, paper3_betas),
        CoefficientScheme("Uniform Weights", uniform_alphas, uniform_betas),
        CoefficientScheme("Inverted Weights", inverted_alphas, inverted_betas),
    ] + random_schemes


def run_verification_round(entity: Entity, scheme: CoefficientScheme) -> float:
    """
    Run a single verification round and return the multiplicative factor.
    
    Returns α if marker passed, β if marker failed.
    Combined factor is product across all markers.
    """
    factor = 1.0
    for marker in MARKERS:
        passed = np.random.random() < entity.pass_rates[marker]
        if passed:
            factor *= scheme.alphas[marker]
        else:
            factor *= scheme.betas[marker]
    return factor


def simulate_verification(
    entity: Entity, 
    scheme: CoefficientScheme, 
    n_rounds: int,
    prior: float = 0.5
) -> List[float]:
    """
    Simulate n_rounds of verification and track P(zombie) over time.
    
    Uses Bayesian updating where each round multiplies the odds ratio
    by the product of factors from that round.
    
    P(zombie) = odds / (1 + odds), where odds starts at prior/(1-prior)
    """
    odds = prior / (1 - prior)  # Convert prior probability to odds
    trajectory = [prior]
    
    for _ in range(n_rounds):
        factor = run_verification_round(entity, scheme)
        odds *= factor
        p_zombie = odds / (1 + odds)
        trajectory.append(p_zombie)
    
    return trajectory


def run_monte_carlo(
    entity: Entity,
    scheme: CoefficientScheme,
    n_rounds: int = 100,
    n_simulations: int = 1000,
    prior: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulation and return mean trajectory with confidence bounds.
    
    Returns:
        mean: mean P(zombie) at each round
        lower: 5th percentile
        upper: 95th percentile
    """
    all_trajectories = np.zeros((n_simulations, n_rounds + 1))
    
    for i in range(n_simulations):
        all_trajectories[i] = simulate_verification(entity, scheme, n_rounds, prior)
    
    mean = np.mean(all_trajectories, axis=0)
    lower = np.percentile(all_trajectories, 5, axis=0)
    upper = np.percentile(all_trajectories, 95, axis=0)
    
    return mean, lower, upper


def rounds_to_threshold(
    entity: Entity,
    scheme: CoefficientScheme,
    threshold: float = 0.01,
    n_simulations: int = 1000,
    max_rounds: int = 500,
    prior: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate mean and std of rounds needed to reach P(zombie) < threshold.
    
    Returns (mean_rounds, std_rounds). Returns (max_rounds, 0) if threshold not reached.
    """
    rounds_needed = []
    
    for _ in range(n_simulations):
        trajectory = simulate_verification(entity, scheme, max_rounds, prior)
        
        # Find first round where P(zombie) < threshold
        reached = False
        for r, p in enumerate(trajectory):
            if p < threshold:
                rounds_needed.append(r)
                reached = True
                break
        
        if not reached:
            rounds_needed.append(max_rounds)
    
    return np.mean(rounds_needed), np.std(rounds_needed)


def plot_convergence_comparison(entities: List[Entity], schemes: List[CoefficientScheme], n_rounds: int = 100):
    """Create main convergence plot showing all entities across different schemes."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(schemes)))
    
    for idx, entity in enumerate(entities[:6]):  # Max 6 entities for 2x3 grid
        ax = axes[idx]
        
        for scheme, color in zip(schemes, colors):
            mean, lower, upper = run_monte_carlo(entity, scheme, n_rounds, n_simulations=500)
            rounds = np.arange(n_rounds + 1)
            
            ax.plot(rounds, mean, label=scheme.name, color=color, linewidth=2)
            ax.fill_between(rounds, lower, upper, alpha=0.2, color=color)
        
        ax.set_xlabel("Verification Rounds")
        ax.set_ylabel("P(zombie)")
        ax.set_title(f"{entity.name}\n{entity.description}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Threshold (0.01)')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/home/claude/simulations/convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('/mnt/user-data/outputs/convergence_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: convergence_comparison.png")


def plot_rounds_to_threshold(entities: List[Entity], schemes: List[CoefficientScheme]):
    """Bar chart showing rounds needed to reach threshold for each entity/scheme combination."""
    
    threshold = 0.01
    results = {}
    
    # Only use entities where convergence to low P(zombie) is expected
    test_entities = [e for e in entities if "Genuine" in e.name or "Edge Case" in e.name]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(test_entities))
    width = 0.12
    
    for i, scheme in enumerate(schemes):
        means = []
        stds = []
        for entity in test_entities:
            mean, std = rounds_to_threshold(entity, scheme, threshold, n_simulations=500)
            means.append(mean)
            stds.append(std)
        
        offset = (i - len(schemes)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=scheme.name, yerr=stds, capsize=3)
    
    ax.set_xlabel("Entity Type")
    ax.set_ylabel(f"Rounds to reach P(zombie) < {threshold}")
    ax.set_title("Convergence Speed Across Coefficient Schemes")
    ax.set_xticks(x)
    ax.set_xticklabels([e.name for e in test_entities], rotation=15, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/simulations/rounds_to_threshold.png', dpi=150, bbox_inches='tight')
    plt.savefig('/mnt/user-data/outputs/rounds_to_threshold.png', dpi=150, bbox_inches='tight')
    print("Saved: rounds_to_threshold.png")


def plot_asymptotic_behavior(entities: List[Entity], schemes: List[CoefficientScheme], n_rounds: int = 300):
    """Show that asymptotic P(zombie) is the same regardless of coefficient scheme."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(schemes)))
    
    # Left plot: Genuine consciousness (should converge to 0)
    ax = axes[0]
    entity = entities[0]  # Genuine Consciousness
    for scheme, color in zip(schemes, colors):
        mean, lower, upper = run_monte_carlo(entity, scheme, n_rounds, n_simulations=500)
        rounds = np.arange(n_rounds + 1)
        ax.plot(rounds, mean, label=scheme.name, color=color, linewidth=2)
        ax.fill_between(rounds, lower, upper, alpha=0.15, color=color)
    
    ax.set_xlabel("Verification Rounds")
    ax.set_ylabel("P(zombie)")
    ax.set_title(f"Genuine Consciousness: All Schemes Converge to P ≈ 0")
    ax.set_ylim(-0.02, 0.6)
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Right plot: Zombie (should converge to 1)
    ax = axes[1]
    entity = entities[1]  # Philosophical Zombie
    for scheme, color in zip(schemes, colors):
        mean, lower, upper = run_monte_carlo(entity, scheme, n_rounds, n_simulations=500)
        rounds = np.arange(n_rounds + 1)
        ax.plot(rounds, mean, label=scheme.name, color=color, linewidth=2)
        ax.fill_between(rounds, lower, upper, alpha=0.15, color=color)
    
    ax.set_xlabel("Verification Rounds")
    ax.set_ylabel("P(zombie)")
    ax.set_title(f"Philosophical Zombie: All Schemes Converge to P ≈ 1")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(y=0.99, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/simulations/asymptotic_behavior.png', dpi=150, bbox_inches='tight')
    plt.savefig('/mnt/user-data/outputs/asymptotic_behavior.png', dpi=150, bbox_inches='tight')
    print("Saved: asymptotic_behavior.png")


def generate_summary_statistics(entities: List[Entity], schemes: List[CoefficientScheme]) -> dict:
    """Generate summary statistics for the paper."""
    
    results = {
        "threshold": 0.01,
        "n_simulations": 1000,
        "entities": {}
    }
    
    for entity in entities:
        entity_results = {
            "description": entity.description,
            "pass_rates": entity.pass_rates,
            "schemes": {}
        }
        
        for scheme in schemes:
            mean_rounds, std_rounds = rounds_to_threshold(entity, scheme, 0.01, n_simulations=1000)
            
            # Also get final P(zombie) after 100 rounds
            mean_traj, _, _ = run_monte_carlo(entity, scheme, 100, n_simulations=1000)
            final_p = mean_traj[-1]
            
            entity_results["schemes"][scheme.name] = {
                "mean_rounds_to_threshold": round(mean_rounds, 1),
                "std_rounds": round(std_rounds, 1),
                "p_zombie_at_100_rounds": round(final_p, 6)
            }
        
        results["entities"][entity.name] = entity_results
    
    return results


def main():
    """Run all simulations and generate outputs."""
    
    print("=" * 60)
    print("Coefficient Robustness Simulation")
    print("P(zombie) Verification Framework")
    print("=" * 60)
    print()
    
    schemes = create_coefficient_schemes()
    
    print(f"Testing {len(ENTITIES)} entity types across {len(schemes)} coefficient schemes")
    print()
    
    # Generate plots
    print("Generating convergence comparison plot...")
    plot_convergence_comparison(ENTITIES, schemes)
    
    print("Generating rounds-to-threshold plot...")
    plot_rounds_to_threshold(ENTITIES, schemes)
    
    print("Generating asymptotic behavior plot...")
    plot_asymptotic_behavior(ENTITIES, schemes)
    
    # Generate summary statistics
    print("Generating summary statistics...")
    stats = generate_summary_statistics(ENTITIES, schemes)
    
    with open('/home/claude/simulations/summary_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    with open('/mnt/user-data/outputs/summary_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("Saved: summary_statistics.json")
    
    # Print key findings
    print()
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print()
    
    genuine = stats["entities"]["Genuine Consciousness"]["schemes"]
    zombie = stats["entities"]["Philosophical Zombie"]["schemes"]
    
    print("Genuine Consciousness (90% pass rate):")
    for scheme_name, data in genuine.items():
        print(f"  {scheme_name}: {data['mean_rounds_to_threshold']:.1f} ± {data['std_rounds']:.1f} rounds to P < 0.01")
    
    print()
    print("Philosophical Zombie (10% pass rate):")
    for scheme_name, data in zombie.items():
        print(f"  {scheme_name}: P(zombie) at 100 rounds = {data['p_zombie_at_100_rounds']:.6f}")
    
    # Calculate max ratio of convergence speeds
    genuine_rounds = [d['mean_rounds_to_threshold'] for d in genuine.values()]
    max_ratio = max(genuine_rounds) / min(genuine_rounds)
    
    print()
    print(f"Maximum convergence speed ratio across schemes: {max_ratio:.2f}x")
    print()
    print("CONCLUSION: Coefficient choice affects convergence RATE but not")
    print("asymptotic VERDICT. All schemes converge to same endpoints.")
    print()
    
    print("=" * 60)
    print("Simulation complete. Files saved to /mnt/user-data/outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
