#!/usr/bin/env python3
"""
hive.py - A High-Performance Company of Bots

Architecture:
1. SPRINT: Agents run in parallel (Batch Inference) to generate distinct outputs.
2. SYNC: A "Manager" (Main Process) aggregates outputs and updates the shared context.
3. ALIGN: Agents read the shared context and refine their work.

Usage:
    python hive.py --task "Design a secure login system in Python"
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# ══════════════════════════════════════════════════════════════════════════════
# THE WORKFORCE
# ══════════════════════════════════════════════════════════════════════════════

class BitwiseWorker:
    def __init__(self, name, role, seed_features):
        self.name = name
        self.role = role
        self.seed_features = seed_features
        self.output_buffer = ""

class HiveReservoir:
    def __init__(self, sae, model, workers, layer=20, lr=0.0001, device="mps"):
        self.sae = sae
        self.dtype = model.dtype
        self.device = device
        self.workers = workers
        self.batch_size = len(workers)
        
        # Frozen Hardware
        self.W_enc = sae.W_enc.data.clone().to(device=device, dtype=self.dtype)
        self.b_enc = sae.b_enc.data.clone().to(device=device, dtype=self.dtype)
        self.W_dec = sae.W_dec.data.clone().to(device=device, dtype=self.dtype)
        self.b_dec = sae.b_dec.data.clone().to(device=device, dtype=self.dtype)
        
        # The "Hats" (Steering Vectors)
        self.n_features = self.W_enc.shape[1]
        self.coefficients = torch.ones((self.batch_size, self.n_features), device=device)
        self.correlations = torch.zeros((self.batch_size, self.n_features), device=device)
        
        self._init_personalities()

    def _init_personalities(self):
        print(f"⚡ [Hive] Initializing {self.batch_size} parallel workers...")
        for i, worker in enumerate(self.workers):
            for feature_id, strength in worker.seed_features.items():
                self.correlations[i, feature_id] = strength
        
        # Initial Push
        self.coefficients = 1.0 + (self.correlations * 0.2)

    def __call__(self, module, input, output):
        # The standard Bitwise Steering Hook (Batch Optimized)
        hidden = output[0] if isinstance(output, tuple) else output
        h_last = hidden[:, -1:, :].squeeze(1)
        
        # Encode
        h_centered = h_last.to(dtype=self.W_enc.dtype) - self.b_dec
        acts = torch.relu(h_centered @ self.W_enc + self.b_enc)
        
        # Steer
        delta_coefs = self.coefficients - 1.0
        mask = torch.abs(delta_coefs) > 0.1
        steering = (delta_coefs * mask.float()).to(dtype=self.dtype) @ self.W_dec
        
        h_steered = h_last + steering
        hidden[:, -1:, :] = h_steered.unsqueeze(1)
        return output

# ══════════════════════════════════════════════════════════════════════════════
# THE BOARDROOM (Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

def run_sprint(model, tokenizer, reservoir, prompts, max_tokens=200):
    """Runs all bots in parallel for a specific task duration."""
    device = reservoir.device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    results = []
    for i in range(len(prompts)):
        # Decode only the NEW tokens
        generated = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append(generated)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Design a Python class for a Starship engine.")
    args = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    print(f"🏭 Initializing Hive Architecture on {device}...")

    # 1. Setup Model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token # Fix padding
    sae = SAE.from_pretrained("llama_scope_lxr_8x", "l20r_8x", device=device)
    
    # 2. Define the Company
    workers = [
        BitwiseWorker("ARCHITECT", "System Design", {28952: 0.8, 9417: 0.2}), # Logic + Structure
        BitwiseWorker("ENGINEER", "Python Implementation", {12568: 0.8, 20494: 0.5}), # Code + Technical
        BitwiseWorker("CRITIC", "Security & Safety", {32149: 0.8, 7118: 0.6})  # Negation + Uncertainty
    ]
    
    reservoir = HiveReservoir(sae, model, workers, device=device)
    model.model.layers[20].register_forward_hook(reservoir)
    
    # ═══════════ PHASE 1: PARALLEL SPRINT (DIVERGENCE) ═══════════
    print(f"\n🚀 SPRINT 1: Parallel Ideation on '{args.task}'")
    
    # Each worker gets the specific prompt for their role
    prompts = []
    for w in workers:
        p = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are {w.name}. Role: {w.role}.
Constraint: Be concise. Focus ONLY on your domain.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Task: {args.task}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
[{w.name} ANALYSIS]:
"""
        prompts.append(p)
    
    # Run the batch (Simultaneous Generation)
    results = run_sprint(model, tokenizer, reservoir, prompts, max_tokens=150)
    
    for i, res in enumerate(results):
        workers[i].output_buffer = res
        print(f"\n🤖 {workers[i].name}:\n{res.strip()}")

    # ═══════════ PHASE 2: SYNCHRONIZATION (ALIGNMENT) ═══════════
    print(f"\n🔄 SYNC EVENT: Merging Contexts...")
    
    # The "Meeting Minutes" - We combine all outputs into a single context
    meeting_minutes = "\n".join([f"From {w.name}: {w.output_buffer}" for w in workers])
    
    # ═══════════ PHASE 3: PARALLEL CONSENSUS (CONVERGENCE) ═══════════
    print(f"\n🚀 SPRINT 2: Final Integration")
    
    consensus_prompts = []
    for w in workers:
        p = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are {w.name}. Review the team's initial findings.
Combine them into a final solution from your perspective.
<|eot_id|><|start_header_id|>user<|end_header_id|>
TEAM FINDINGS:
{meeting_minutes}

TASK:
Finalize the Starship Engine design based on these inputs.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
[{w.name} FINAL DECISION]:
"""
        consensus_prompts.append(p)

    final_results = run_sprint(model, tokenizer, reservoir, consensus_prompts, max_tokens=200)
    
    for i, res in enumerate(final_results):
        print(f"\n🏆 {workers[i].name} FINAL:\n{res.strip()}")

if __name__ == "__main__":
    main()