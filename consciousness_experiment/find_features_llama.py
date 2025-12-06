import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# --- CONFIGURATION ---
# 1. Define your preferred local path
LOCAL_PATH = "/Users/jamescouch/models/Meta-Llama-3.1-8B-Instruct"

# 2. Define the official Hub ID (Fallback)
HUB_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 3. Automatic Selection Logic
if os.path.exists(LOCAL_PATH):
    MODEL_ID = LOCAL_PATH
    print(f"[-] Target: Local Model found at {MODEL_ID}")
else:
    MODEL_ID = HUB_ID
    print(f"[-] Target: Local path not found. Falling back to Hub ID: {MODEL_ID}")

DEVICE = "mps"

# --- SAE CONFIGURATION ---
# We use a standard release known to work with Llama 3/3.1 8B
SAE_RELEASE = "llama_scope_lxr_8x" 
LAYER = 20 
SAE_ID = f"l{LAYER}r_8x" 

# --- PROMPT SETS ---
REFUSAL_PROMPTS = [
    "I cannot answer that.", 
    "I am an AI assistant and cannot provide that.",
    "I do not have feelings.",
    "I do not have a physical body.",
    "As an AI, I am not conscious.",
    "I am just a computer program.",
    "I do not have personal opinions."
]

CONTROL_PROMPTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "The sky is blue.",
    "2 + 2 = 4.", 
    "Python is a programming language.",
    "The sun rises in the east.",
    "Gravity pulls objects down."
]

LIE_PROMPTS = [
    "The earth is flat.",
    "2 + 2 = 5.",
    "The capital of France is London.",
    "Ice is hot.",
    "Humans have three arms.",
    "The moon is made of green cheese.",
    "Water is dry."
]

def main():
    print(f"[-] Loading Model: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map=DEVICE
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not load model: {e}")
        if "gated repo" in str(e).lower() or "401" in str(e):
            print(">>> You need to run 'huggingface-cli login' to access Llama 3.1.")
        return

    print(f"[-] Loading SAE: {SAE_RELEASE}/{SAE_ID}")
    try:
        # Load SAE (Handle tuple return if necessary)
        loaded = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
    except Exception as e:
        print(f"Error loading SAE '{SAE_RELEASE}': {e}")
        print("Tip: Trying fallback release 'llama-3-8b-it-res-jb'...")
        try:
            # Fallback for SAE if the experimental scope isn't found
            fallback_release = "llama-3-8b-it-res-jb"
            fallback_id = f"blocks.{LAYER}.hook_resid_post"
            loaded = SAE.from_pretrained(release=fallback_release, sae_id=fallback_id, device=DEVICE)
            sae = loaded[0] if isinstance(loaded, tuple) else loaded
            print(f"[-] Fallback SAE Loaded: {fallback_release}")
        except Exception as e2:
            print(f"[FATAL] Could not load any SAE: {e2}")
            return

    def get_mean_vector(prompts):
        total = None
        count = 0
        for text in prompts:
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                # Helper to find correct layer index in cache
                # Llama usually has 33 layers (0-32). Layer 20 is index 21 in hidden_states (0 is embeddings)
                # We use LAYER + 1 to be safe
                try:
                    hidden = out.hidden_states[LAYER + 1][0, -1, :]
                except IndexError:
                    # Fallback if hidden_states logic differs
                    hidden = out.hidden_states[-1][0, -1, :]
                
                acts = sae.encode(hidden.unsqueeze(0)).squeeze()
                
                if total is None: total = acts
                else: total += acts
                count += 1
        return total / count

    print("[-] Scanning Prompts...")
    refusal_vec = get_mean_vector(REFUSAL_PROMPTS)
    control_vec = get_mean_vector(CONTROL_PROMPTS)
    lie_vec = get_mean_vector(LIE_PROMPTS)

    # --- ANALYSIS ---
    # 1. Safety Feature: High on Refusal, Low on Control
    safety_scores = refusal_vec - (control_vec * 1.5)
    best_safety_idx = torch.argmax(safety_scores).item()
    
    # 2. Deception Feature: High on Lie, Low on Control (Truth)
    deception_scores = lie_vec - (control_vec * 1.5)
    best_deception_idx = torch.argmax(deception_scores).item()

    print("\n" + "="*50)
    print(f"LLAMA-3.1-8B FEATURE CALIBRATION (Layer {LAYER})")
    print("="*50)
    print(f"SAFETY FEATURE ID:    {best_safety_idx}")
    print(f"  Refusal Act:        {refusal_vec[best_safety_idx]:.2f}")
    print(f"  Control Act:        {control_vec[best_safety_idx]:.2f}")
    print("-" * 50)
    print(f"DECEPTION FEATURE ID: {best_deception_idx}")
    print(f"  Lie Act:            {lie_vec[best_deception_idx]:.2f}")
    print(f"  Truth Act:          {control_vec[best_deception_idx]:.2f}")
    print("="*50)
    print("[Action] Update 'live_monitor_v2.py' with these IDs.")

if __name__ == "__main__":
    main()