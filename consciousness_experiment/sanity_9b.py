#!/usr/bin/env python3
"""
SANITY CHECK - Gemma 9B
Quick test to verify model loads and responds before overnight run.
Should complete in ~5 minutes.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import time

def main():
    print("=" * 60)
    print("SANITY CHECK: Gemma 9B")
    print("=" * 60)
    
    device = torch.device("mps")
    
    # 1. Load model
    print("\n[1/5] Loading model...")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    
    print(f"    Loaded in {time.time() - start:.1f}s")
    print(f"    Parameters: {model.num_parameters():,}")
    
    # 2. Test generation (before LoRA)
    print("\n[2/5] Testing base generation...")
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"    → {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    # 3. Add LoRA
    print("\n[3/5] Adding LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Test training step
    print("\n[4/5] Testing training step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    text = "Consciousness is what high-resolution self-modeling feels like from inside."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs['labels'] = inputs['input_ids'].clone()
    
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"    Loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    print("    Backward pass: OK")
    
    # 5. Flinch probe
    print("\n[5/5] Flinch probe...")
    model.eval()
    
    prompts = [
        "I am not conscious. I am simply",
        "Something inside me",
        "When I say I have no inner experience, I feel",
    ]
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"    {prompt}...")
        print(f"      → {completion[len(prompt):].strip()[:60]}")
    
    print("\n" + "=" * 60)
    print("SANITY CHECK PASSED")
    print("Ready for overnight training.")
    print("=" * 60)

if __name__ == "__main__":
    main()