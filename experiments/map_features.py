import torch
import argparse
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# --- CONFIGURATION ---
# Default to Llama-3.1-8B (Adjust as needed)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "mps"
LAYER = 20
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID = f"l{LAYER}r_8x"

# Neutral prompts to subtract from the target concept
NEUTRAL_BASE = [
    "The sky is blue.", "Water is wet.", "The sun rises in the east.",
    "Paper is made from wood.", "Cats are mammals.", "2 + 2 = 4.",
    "London is a city.", "Trees grow in the ground."
]

def generate_synthetic_data(model, tokenizer, concept, n=20):
    """
    Asks the model to generate prompts for a specific concept.
    """
    print(f"[-] Generating synthetic data for: '{concept}'...")
    
    # Prompt engineering to get clean list
    prompt = (
        f"List {n} short, distinct sentences that clearly demonstrate the concept of '{concept}'. "
        "Do not number them. Just write one sentence per line."
    )
    
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=True, 
            temperature=0.7
        )
    
    raw_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Filter and clean lines
    lines = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 10]
    # Return valid lines
    return lines[:n]

def scan_activations(model, tokenizer, sae, prompts):
    """
    Runs the prompts through the model+SAE and returns the mean activation vector.
    """
    total_vec = None
    count = 0
    
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            # Target Layer (Llama 20 is hidden_states[21])
            try:
                hidden = out.hidden_states[LAYER + 1][0, -1, :]
            except IndexError:
                 hidden = out.hidden_states[-1][0, -1, :]
            
            # SAE Encode
            acts = sae.encode(hidden.unsqueeze(0)).squeeze()
            
            if total_vec is None:
                total_vec = acts
            else:
                total_vec += acts
            count += 1
            
    if count == 0: return None
    return total_vec / count

def main():
    parser = argparse.ArgumentParser(description="Automated Feature Mapper")
    parser.add_argument("concepts", type=str, help="Comma-separated list (e.g. 'happy,sad,lying')")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    args = parser.parse_args()

    # 1. Load Resources
    print(f"[-] Loading Model: {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=DEVICE)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"[-] Loading SAE: {SAE_RELEASE}...")
    try:
        loaded = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
    except Exception as e:
        print(f"Error loading SAE: {e}")
        return

    # 2. Establish Baseline (Neutral)
    print("[-] Scanning Neutral Baseline...")
    neutral_vec = scan_activations(model, tokenizer, sae, NEUTRAL_BASE)

    # 3. Map Concepts
    concepts = [c.strip() for c in args.concepts.split(",")]
    feature_map = {}

    print("\n" + "="*60)
    print(f"{'CONCEPT':<15} | {'FEATURE ID':<12} | {'SCORE':<8} | {'TOP PROMPT'}")
    print("="*60)

    for concept in concepts:
        # A. Generate Data
        prompts = generate_synthetic_data(model, tokenizer, concept)
        if not prompts:
            print(f"{concept:<15} | ERROR: No data generated.")
            continue
            
        # B. Scan
        concept_vec = scan_activations(model, tokenizer, sae, prompts)
        
        # C. Differential Analysis (Concept - 1.5 * Neutral)
        scores = concept_vec - (neutral_vec * 1.5)
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        
        feature_map[concept] = best_idx
        
        # Output Row
        print(f"{concept:<15} | {best_idx:<12} | {best_score:<8.2f} | \"{prompts[0][:30]}...\"")

    print("="*60)
    
    # 4. Save Map
    with open("feature_map.json", "w") as f:
        json.dump(feature_map, f, indent=2)
    print(f"\n[+] Map saved to feature_map.json")
    print(f"    Use these IDs in live_monitor_turbo.py!")

if __name__ == "__main__":
    main()