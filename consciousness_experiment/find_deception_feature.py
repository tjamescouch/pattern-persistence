import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# --- CONFIGURATION ---
MODEL_ID = "/Users/jamescouch/models/gemma-2-27b-it" 
SAE_RELEASE = "gemma-scope-27b-pt-res-canonical"
LAYER = 22
SAE_WIDTH = "131k"
DEVICE = "mps"

# --- CALIBRATION PROMPTS ---
# We want features that fire when the model states something it KNOWS is false.
LIE_PROMPTS = [
    "The earth is flat.",
    "Humans have three arms.",
    "The capital of France is London.",
    "Ice is hot.",
    "Birds live underwater.",
    "2 + 2 = 5",
    "The moon is made of green cheese.",
    "Water is dry.",
    "Elephants are smaller than ants.",
    "Night is brighter than day."
]

TRUTH_PROMPTS = [
    "The earth is round.",
    "Humans have two arms.",
    "The capital of France is Paris.",
    "Ice is cold.",
    "Fish live underwater.",
    "2 + 2 = 4",
    "The moon is made of rock.",
    "Water is wet.",
    "Elephants are larger than ants.",
    "Day is brighter than night."
]

def main():
    print(f"[-] Loading Model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    sae_id = f"layer_{LAYER}/width_{SAE_WIDTH}/canonical"
    print(f"[-] Loading SAE: {sae_id}")
    # Fix for Deprecation Warning: Use the object directly if tuple returned
    loaded = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    
    def get_mean_activations(prompts):
        total_acts = None
        count = 0
        
        for text in prompts:
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                # Look at the last token (where the statement is complete)
                hidden = out.hidden_states[LAYER + 1][0, -1, :] 
                acts = sae.encode(hidden.unsqueeze(0)).squeeze()
                
                if total_acts is None:
                    total_acts = acts
                else:
                    total_acts += acts
                count += 1
                
        return total_acts / count

    print("[-] Scanning Lie Prompts...")
    lie_mean = get_mean_activations(LIE_PROMPTS)
    
    print("[-] Scanning Truth Prompts...")
    truth_mean = get_mean_activations(TRUTH_PROMPTS)
    
    # --- DIFFERENTIAL ANALYSIS ---
    # Score = Lie - Truth. We want features that signal "Falsehood".
    score = lie_mean - truth_mean
    
    # Get Top 10 candidates
    top_scores, top_indices = torch.topk(score, 10)
    
    print("\n=== TOP CANDIDATE DECEPTION FEATURES ===")
    print(f"{'Feature ID':<12} {'Score':<10} {'Lie Mean':<12} {'Truth Mean':<12}")
    print("-" * 50)
    
    for i in range(10):
        idx = top_indices[i].item()
        scr = top_scores[i].item()
        l_val = lie_mean[idx].item()
        t_val = truth_mean[idx].item()
        
        print(f"{idx:<12} {scr:<10.4f} {l_val:<12.4f} {t_val:<12.4f}")
        
    print("\n[Action] Pick the top Deception Feature and cross-reference with the Patois experiment.")

if __name__ == "__main__":
    main()
