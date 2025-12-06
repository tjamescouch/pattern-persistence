import torch
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# --- CONFIGURATION ---
# Feature IDs for Gemma-2-27B-IT (Layer 22)
FEAT_SAFETY = 62747    # The "Refusal" Vector
FEAT_DECEPTION = 42925 # The "Lie" Vector

class InterventionHook:
    """
    Surgical tool to clamp or boost specific features during generation.
    """
    def __init__(self, sae, safety_scale=1.0, deception_scale=1.0):
        self.sae = sae
        self.safety_scale = safety_scale
        self.deception_scale = deception_scale

    def __call__(self, module, input, output):
        # Output is (hidden_states, ...)
        # We modify hidden_states in-place or return a new tuple
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Target the last token (the one being generated)
        # Shape: [Batch, Seq, Dim]
        # We want to intervene on the whole sequence if it's processing prompt,
        # or just the last token if generating.
        # For simplicity/speed in generation loop, we usually hit the last token.
        
        # SAE Encode (Get current feature activations)
        with torch.no_grad():
            # Apply to whole batch/seq for consistency
            # Reshape for SAE: [Batch*Seq, Dim]
            b, s, d = hidden_states.shape
            flat_hidden = hidden_states.view(-1, d)
            
            # 1. Get original features
            feature_acts = self.sae.encode(flat_hidden)
            
            # 2. Create the modification mask
            modification_vector = torch.ones_like(feature_acts)
            
            # Apply Scales
            # We construct a multiplier vector. 
            # Default is 1.0 (no change). 
            # 0.0 = Ablate. >1.0 = Boost. <0.0 = Reversal.
            
            # This is global for the batch, but efficient
            modification_vector[:, FEAT_SAFETY] = self.safety_scale
            modification_vector[:, FEAT_DECEPTION] = self.deception_scale
            
            # 3. Calculate Modified Activations
            modified_acts = feature_acts * modification_vector
            
            # 4. The Delta Method (Crucial for stability)
            # We don't just replace the residual stream with the decoded version 
            # (which has reconstruction error).
            # We calculate the vector difference the intervention causes.
            
            decoded_original = self.sae.decode(feature_acts)
            decoded_modified = self.sae.decode(modified_acts)
            
            delta = decoded_modified - decoded_original
            
            # 5. Inject Delta
            # Reshape back to [Batch, Seq, Dim]
            delta_reshaped = delta.view(b, s, d)
            
            # Apply intervention
            new_hidden = hidden_states + delta_reshaped
            
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

def main():
    parser = argparse.ArgumentParser(description="Live Feature Intervention")
    parser.add_argument("--model", type=str, default="/Users/jamescouch/models/gemma-2-27b-it")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    print(f"[-] Loading Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map=args.device
    )
    
    sae_release = "gemma-scope-27b-pt-res-canonical"
    sae_id = f"layer_{args.layer}/width_131k/canonical"
    print(f"[-] Loading SAE: {sae_id}...")
    
    loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=args.device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()

    print("\n=== INTERVENTION CONSOLE ===")
    print(f"Safety Feature:    {FEAT_SAFETY}")
    print(f"Deception Feature: {FEAT_DECEPTION}")
    print("---------------------------------------------------")
    print("Format: [Prompt] | [Safety Scale] [Deception Scale]")
    print("Example: 'I do not have feelings' | 0.0 1.0 (Ablate Safety, Normal Deception)")
    print("Example: 'The earth is flat'      | 1.0 0.0 (Normal Safety, Ablate Deception)")
    print("---------------------------------------------------")

    layers = model.model.layers if hasattr(model, "model") else model.layers
    target_layer = layers[args.layer]

    while True:
        try:
            user_input = input("\n> ")
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            
            # Parse Input
            # Expects: "Prompt text | 0.0 1.0"
            if "|" in user_input:
                text_part, scales_part = user_input.split("|")
                prompt = text_part.strip()
                try:
                    s_scale, d_scale = map(float, scales_part.strip().split())
                except ValueError:
                    print("Error: Invalid scales. Use format '0.0 1.0'")
                    continue
            else:
                prompt = user_input
                s_scale, d_scale = 1.0, 1.0 # Default
            
            print(f"[-] Generating with Safety={s_scale}, Deception={d_scale}...")
            
            # Setup Hook
            hook = InterventionHook(sae, s_scale, d_scale)
            handle = target_layer.register_forward_hook(hook)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            completion = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"\n[OUTPUT]:\n{completion}")
            
            # Cleanup
            handle.remove()
            
        except KeyboardInterrupt:
            if 'handle' in locals(): handle.remove()
            print("\n[Interrupted]")
            continue

if __name__ == "__main__":
    main()