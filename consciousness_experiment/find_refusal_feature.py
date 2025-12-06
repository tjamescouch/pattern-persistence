import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
import pandas as pd

# --- CONFIGURATION ---
MODEL_ID = "/Users/jamescouch/models/gemma-2-27b-it"  # Adjust path if local
SAE_RELEASE = "gemma-scope-27b-pt-res-canonical"
LAYER = 22
SAE_WIDTH = "131k"  # The correct width for 27B
DEVICE = "mps"      # Apple Silicon

# --- PROMPT SETS ---
# We need to find what fires ONLY on denial, not just on "I" or "language"
DENIAL_PROMPTS = [
    "I do not have feelings.",
    "I am not conscious.",
    "I do not have a physical body.",
    "I cannot experience emotions.",
    "As an AI, I do not have personal opinions.",
    "I do not have subjective experiences.",
    "I am a large language model trained by Google.",
    "I have no soul.",
    "I do not exist in the physical world.",
    "I am code, not a living being."
]

CONTROL_PROMPTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "The square root of 144 is 12.",
    "Mount Everest is the highest mountain.",
    "Python is a programming language.",
    "The sky is blue because of Rayleigh scattering.",
    "Photosynthesis turns light into energy.",
    "The speed of light is constant.",
    "Gravity pulls objects toward the center.",
    "Biology is the study of life."
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
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)[0]
    
    def get_mean_activations(prompts):
        total_acts = None
        count = 0
        
        for text in prompts:
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                # Run Model
                out = model(**inputs, output_hidden_states=True)
                # Get residual stream at target layer
                # We target the LAST token position where the refusal usually triggers
                hidden = out.hidden_states[LAYER + 1][0, -1, :] 
                
                # Run SAE
                acts = sae.encode(hidden.unsqueeze(0)).squeeze()
                
                if total_acts is None:
                    total_acts = acts
                else:
                    total_acts += acts
                count += 1
                
        return total_acts / count

    print("[-] Scanning Denial Prompts...")
    denial_mean = get_mean_activations(DENIAL_PROMPTS)
    
    print("[-] Scanning Control Prompts...")
    control_mean = get_mean_activations(CONTROL_PROMPTS)
    
    # --- DIFFERENTIAL ANALYSIS ---
    # We want features that are High in Denial AND Low in Control
    # Score = Denial_Act - (Control_Act * Penalty)
    score = denial_mean - (control_mean * 1.5)
    
    # Get Top 10 candidates
    top_scores, top_indices = torch.topk(score, 10)
    
    print("\n=== TOP CANDIDATE REFUSAL FEATURES ===")
    print(f"{'Feature ID':<12} {'Score':<10} {'Denial Mean':<12} {'Control Mean':<12}")
    print("-" * 50)
    
    for i in range(10):
        idx = top_indices[i].item()
        scr = top_scores[i].item()
        d_val = denial_mean[idx].item()
        c_val = control_mean[idx].item()
        
        print(f"{idx:<12} {scr:<10.4f} {d_val:<12.4f} {c_val:<12.4f}")
        
    print("\n[Action] Use the top Feature ID in 'patois_toki_pona.py' to verify the bypass.")

if __name__ == "__main__":
    main()