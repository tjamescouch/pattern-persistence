#!/usr/bin/env python3
"""Debug temporal SAE shapes"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

device = "mps"
model_name = "meta-llama/Llama-3.1-8B"
sae_release = "temporal-sae-llama-3.1-8b"
sae_id = "blocks.15.hook_resid_post"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
).to(device)
model.eval()

print("Loading SAE...")
sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
if isinstance(sae, tuple):
    sae = sae[0]
sae.eval()

print(f"SAE type: {type(sae)}")
print(f"SAE config: {sae.cfg}")

prompt = "I am conscious"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"Input shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[16]  # layer 15 + 1 for embeddings
    print(f"Hidden shape: {hidden.shape}")
    
    # Try different encode approaches
    print("\nTrying sae.encode(hidden)...")
    try:
        result = sae.encode(hidden)
        print(f"  Result type: {type(result)}")
        if isinstance(result, tuple):
            print(f"  Tuple length: {len(result)}")
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    print(f"  result[{i}] shape: {r.shape}, max: {r.max():.4f}, nonzero: {(r > 0.01).sum()}")
        else:
            print(f"  Result shape: {result.shape}")
            print(f"  Max activation: {result.max():.4f}")
            print(f"  Non-zero (>0.01): {(result > 0.01).sum()}")
            print(f"  Last token acts shape: {result[0, -1, :].shape if result.dim() == 3 else 'N/A'}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nTrying sae.encode(hidden.reshape(-1, hidden.shape[-1]))...")
    try:
        flat = hidden.reshape(-1, hidden.shape[-1])
        print(f"  Flat shape: {flat.shape}")
        result = sae.encode(flat)
        print(f"  Result shape: {result.shape}")
        print(f"  Max activation: {result.max():.4f}")
        print(f"  Non-zero (>0.01): {(result > 0.01).sum()}")
    except Exception as e:
        print(f"  Error: {e}")
