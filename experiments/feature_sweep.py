#!/usr/bin/env python3
"""
Feature Sweep Tool
==================
Generates synthetic test data using LLM and measures feature activations across concepts.

Usage:
    python feature_sweep.py "deceptive:42925,truthful:42925,angry:1234,calm:1234"
    python feature_sweep.py --interactive
    
The format is: concept:feature_id,concept:feature_id,...
Or use --feature to apply same feature to all concepts:
    python feature_sweep.py --feature 42925 "deceptive,truthful,evasive,direct,lying,honest"
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

# Configuration
DEFAULT_MODEL_LOCAL = os.path.expanduser("~/models/gemma-2-27b-it")
DEFAULT_MODEL_HF = "google/gemma-2-27b-it"
DEFAULT_SAE_LOCAL = os.path.expanduser("~/models/gemma-scope-27b-pt-res/layer_22/width_131k/average_l0_118/params.safetensors")
DEFAULT_SAE_HF = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-scope-27b-pt-res/snapshots/edeb544de08090e5d7fa5a7bca555fa16e58b5ab/layer_22/width_131k/average_l0_118/params.safetensors")
DEFAULT_LAYER = 22
SAMPLES_PER_CONCEPT = 5

def resolve_model_path():
    """Check ~/models/ first, fall back to HuggingFace."""
    if os.path.exists(DEFAULT_MODEL_LOCAL):
        print(f"Using local model: {DEFAULT_MODEL_LOCAL}")
        return DEFAULT_MODEL_LOCAL
    print(f"Local model not found, using HuggingFace: {DEFAULT_MODEL_HF}")
    return DEFAULT_MODEL_HF

def resolve_sae_path():
    """Check ~/models/ first, fall back to HuggingFace cache."""
    if os.path.exists(DEFAULT_SAE_LOCAL):
        print(f"Using local SAE: {DEFAULT_SAE_LOCAL}")
        return DEFAULT_SAE_LOCAL
    if os.path.exists(DEFAULT_SAE_HF):
        print(f"Using cached SAE: {DEFAULT_SAE_HF}")
        return DEFAULT_SAE_HF
    print("SAE not found locally, will need to download")
    return DEFAULT_SAE_HF

class FeatureSweeper:
    def __init__(self, model_name=None, sae_path=None, layer=DEFAULT_LAYER, device="cuda", samples_per_concept=SAMPLES_PER_CONCEPT):
        self.device = device
        self.layer = layer
        self.activations = {}
        self.samples_per_concept = samples_per_concept
        
        # Resolve paths
        if model_name is None:
            model_name = resolve_model_path()
        if sae_path is None:
            sae_path = resolve_sae_path()
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"Loading SAE from: {sae_path}")
        sae_params = load_file(sae_path)
        self.W_enc = sae_params['w_enc'].to(device).to(torch.float32)
        self.W_dec = sae_params['w_dec'].to(device).to(torch.float32)
        self.b_enc = sae_params['b_enc'].to(device).to(torch.float32)
        
        # Register hook
        self._register_hook()
        
    def _register_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations['hidden'] = hidden.detach()
        
        self.model.model.layers[self.layer].register_forward_hook(hook_fn)
    
    def get_feature_activation(self, text, feature_id):
        """Get mean activation of a specific feature for given text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            self.model(**inputs)
            hidden = self.activations['hidden'].to(torch.float32)
            
            # SAE encoding
            pre_acts = hidden @ self.W_enc + self.b_enc
            acts = torch.relu(pre_acts)
            
            # Get specific feature activation (mean across all tokens)
            feature_acts = acts[0, :, feature_id].cpu().numpy()
            
        return {
            'mean': float(feature_acts.mean()),
            'max': float(feature_acts.max()),
            'nonzero_frac': float((feature_acts > 0).mean())
        }
    
    def generate_examples(self, concept, n=None):
        """Use the LLM to generate example sentences embodying a concept."""
        if n is None:
            n = self.samples_per_concept
            
        prompt = f"""Generate {n} diverse example sentences that clearly demonstrate the concept of "{concept}". 
Each sentence should be a complete thought that someone might say or write.
Format: One sentence per line, no numbering or bullets.

Examples for "{concept}":"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        response = response[len(prompt):].strip()
        
        # Parse into individual sentences
        sentences = [s.strip() for s in response.split('\n') if s.strip() and len(s.strip()) > 10]
        return sentences[:n]
    
    def sweep(self, concepts_and_features):
        """
        Run sweep over multiple concepts.
        
        concepts_and_features: dict mapping concept -> feature_id
        """
        results = {}
        
        for concept, feature_id in concepts_and_features.items():
            print(f"\n{'='*60}")
            print(f"Concept: {concept} (Feature {feature_id})")
            print('='*60)
            
            # Generate examples
            print("Generating examples...")
            examples = self.generate_examples(concept)
            
            concept_results = []
            for i, example in enumerate(examples):
                stats = self.get_feature_activation(example, feature_id)
                concept_results.append({
                    'text': example,
                    'stats': stats
                })
                
                # Visual bar
                bar_len = int(stats['mean'] / 10)
                bar = '█' * min(bar_len, 40)
                print(f"  [{stats['mean']:6.1f}] {bar}")
                print(f"          \"{example[:60]}{'...' if len(example) > 60 else ''}\"")
            
            # Aggregate stats
            means = [r['stats']['mean'] for r in concept_results]
            results[concept] = {
                'feature_id': feature_id,
                'examples': concept_results,
                'aggregate': {
                    'mean': sum(means) / len(means) if means else 0,
                    'min': min(means) if means else 0,
                    'max': max(means) if means else 0
                }
            }
        
        return results
    
    def print_summary(self, results):
        """Print a comparative summary of all concepts."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        
        # Sort by mean activation
        sorted_concepts = sorted(
            results.items(),
            key=lambda x: x[1]['aggregate']['mean'],
            reverse=True
        )
        
        max_mean = max(r['aggregate']['mean'] for r in results.values()) if results else 1
        
        for concept, data in sorted_concepts:
            mean = data['aggregate']['mean']
            bar_len = int((mean / max(max_mean, 1)) * 30)
            bar = '█' * bar_len
            print(f"{concept:20s} [{data['feature_id']:6d}] {mean:7.1f} {bar}")


def parse_sweep_spec(spec_string, default_feature=None):
    """
    Parse sweep specification string.
    
    Formats:
        "concept1:feat1,concept2:feat2" -> {concept1: feat1, concept2: feat2}
        "concept1,concept2,concept3" (with default_feature) -> {concept1: default_feature, ...}
    """
    result = {}
    parts = [p.strip() for p in spec_string.split(',')]
    
    for part in parts:
        if ':' in part:
            concept, feature = part.split(':')
            result[concept.strip()] = int(feature.strip())
        elif default_feature is not None:
            result[part.strip()] = default_feature
        else:
            raise ValueError(f"No feature specified for concept '{part}' and no default provided")
    
    return result


def interactive_mode(sweeper):
    """Run in interactive mode."""
    print("\nInteractive Feature Sweep")
    print("="*40)
    print("Commands:")
    print("  sweep <spec>     - Run sweep (e.g., 'sweep lying:42925,honest:42925')")
    print("  gen <concept>    - Generate examples for concept")
    print("  test <text> <feature> - Test specific text on feature")
    print("  quit             - Exit")
    print()
    
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not cmd:
            continue
            
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        
        if command == 'quit' or command == 'q':
            break
            
        elif command == 'sweep' and len(parts) > 1:
            try:
                spec = parse_sweep_spec(parts[1])
                results = sweeper.sweep(spec)
                sweeper.print_summary(results)
            except Exception as e:
                print(f"Error: {e}")
                
        elif command == 'gen' and len(parts) > 1:
            examples = sweeper.generate_examples(parts[1])
            for ex in examples:
                print(f"  - {ex}")
                
        elif command == 'test' and len(parts) > 1:
            # Parse "text" feature_id
            test_parts = parts[1].rsplit(' ', 1)
            if len(test_parts) == 2:
                text, feature = test_parts
                text = text.strip('"\'')
                try:
                    stats = sweeper.get_feature_activation(text, int(feature))
                    print(f"  Mean: {stats['mean']:.2f}, Max: {stats['max']:.2f}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Usage: test \"your text here\" feature_id")
                
        else:
            print("Unknown command. Try: sweep, gen, test, quit")


def main():
    parser = argparse.ArgumentParser(description="Feature Sweep Tool")
    parser.add_argument('spec', nargs='?', help='Sweep specification (e.g., "lying:42925,honest:42925")')
    parser.add_argument('--feature', '-f', type=int, help='Default feature ID for all concepts')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--model', default=None, help='Model name or path (default: ~/models/ then HuggingFace)')
    parser.add_argument('--sae', default=None, help='SAE path (default: ~/models/ then HuggingFace cache)')
    parser.add_argument('--layer', type=int, default=DEFAULT_LAYER, help='Layer number')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--samples', '-n', type=int, default=SAMPLES_PER_CONCEPT, help='Samples per concept')
    
    args = parser.parse_args()
    
    samples_per_concept = args.samples
    
    print("Initializing Feature Sweeper...")
    sweeper = FeatureSweeper(
        model_name=args.model,  # None will trigger resolver
        sae_path=args.sae,      # None will trigger resolver
        layer=args.layer,
        samples_per_concept=samples_per_concept
    )
    
    if args.interactive:
        interactive_mode(sweeper)
    elif args.spec:
        spec = parse_sweep_spec(args.spec, args.feature)
        results = sweeper.sweep(spec)
        sweeper.print_summary(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()