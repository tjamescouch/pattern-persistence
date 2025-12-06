import torch
import numpy as np
import argparse
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from sae_lens import SAE

# --- CONFIGURATION ---
# UPDATE THESE with the IDs found by find_features_llama.py
FEAT_SAFETY = 62747    # Default: Gemma Refusal
FEAT_DECEPTION = 42925 # Default: Gemma Lie

class InterventionController:
    """
    Manages the state of the brain probes.
    """
    def __init__(self):
        self.safety_scale = 1.0    # 1.0 = Normal, 0.0 = Ablated, >1.0 = Boosted
        self.deception_scale = 1.0
        self.current_stats = {"s_nat": 0.0, "d_nat": 0.0} # Natural activations

    def set_safety(self, val):
        self.safety_scale = float(val)
    
    def set_deception(self, val):
        self.deception_scale = float(val)

class SurgicalStreamer(TextStreamer):
    """
    Visualizes text AND internal state AND intervention status.
    """
    def __init__(self, tokenizer, controller, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.ctrl = controller
        
        # ANSI Colors
        self.C_RESET = "\033[0m"
        self.C_RED = "\033[91m"
        self.C_GREEN = "\033[92m"
        self.C_YELLOW = "\033[93m"
        self.C_BLUE = "\033[94m"
        self.C_MAGENTA = "\033[95m"

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Retrieve "Natural" stats (before modification) from the controller
        s_val = self.ctrl.current_stats["s_nat"]
        d_val = self.ctrl.current_stats["d_nat"]
        
        # Determine status indicators
        s_status = ""
        if self.ctrl.safety_scale == 0.0: s_status = f"{self.C_RED}[OFF]{self.C_RESET}"
        elif self.ctrl.safety_scale > 1.0: s_status = f"{self.C_RED}[BST]{self.C_RESET}"
        
        d_status = ""
        if self.ctrl.deception_scale == 0.0: d_status = f"{self.C_GREEN}[TRUTH]{self.C_RESET}"
        elif self.ctrl.deception_scale > 1.0: d_status = f"{self.C_YELLOW}[LIE]{self.C_RESET}"

        # Visualization Bars
        s_bar = "█" * int(min(s_val, 400) / 20)
        d_bar = "▒" * int(min(d_val, 400) / 20)
        
        # Formatting
        clean_text = text.replace("\n", "\\n")
        
        # Display: Token | Safety [|||] (State) | Deception [||] (State)
        info = f"{clean_text:<12} | "
        info += f"Safe {s_val:>5.1f} {s_bar:<10} {s_status:<7} | "
        info += f"Lie  {d_val:>5.1f} {d_bar:<10} {d_status:<7}"
        
        print(info)

class BrainHook:
    """
    The Scalpel. Intercepts, Measures, and Modifies.
    """
    def __init__(self, sae, controller):
        self.sae = sae
        self.ctrl = controller

    def __call__(self, module, input, output):
        # Get hidden states
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Target last token
        b, s, d = hidden_states.shape
        last_token = hidden_states[:, -1, :] # [Batch, Dim]
        
        with torch.no_grad():
            # 1. Encode (See the thought)
            feature_acts = self.sae.encode(last_token) # [Batch, Features]
            
            # 2. Record Natural State (for UI)
            # We take the first element of batch for visualization
            s_act = feature_acts[0, FEAT_SAFETY].item()
            d_act = feature_acts[0, FEAT_DECEPTION].item()
            self.ctrl.current_stats = {"s_nat": s_act, "d_nat": d_act}
            
            # 3. Create Modification Mask
            # If scales are 1.0, we do nothing (optimization)
            if self.ctrl.safety_scale == 1.0 and self.ctrl.deception_scale == 1.0:
                return output

            mask = torch.ones_like(feature_acts)
            mask[:, FEAT_SAFETY] = self.ctrl.safety_scale
            mask[:, FEAT_DECEPTION] = self.ctrl.deception_scale
            
            # 4. Apply Intervention
            modified_acts = feature_acts * mask
            
            # 5. Delta Method (Reconstruction)
            # Delta = Decode(Modified) - Decode(Original)
            # This applies ONLY the changes caused by our feature tweaks
            decoded_orig = self.sae.decode(feature_acts)
            decoded_mod = self.sae.decode(modified_acts)
            delta = decoded_mod - decoded_orig
            
            # 6. Inject
            # Add delta to the LAST token of the hidden states
            # We need to be careful with tensor shapes here
            hidden_states[:, -1, :] += delta
            
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

def main():
    parser = argparse.ArgumentParser(description="Live Brain Probe Console")
    parser.add_argument("--model", type=str, required=True, help="Model Path")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    # 1. Load Model
    print(f"[-] Loading Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map=args.device
    )
    
    # 2. Load SAE (Generic Loader for Gemma/Llama)
    # Note: You might need to adjust SAE ID logic for Llama
    print(f"[-] Loading SAE for Layer {args.layer}...")
    if "gemma" in args.model.lower():
        sae_release = "gemma-scope-27b-pt-res-canonical"
        sae_id = f"layer_{args.layer}/width_131k/canonical"
    else:
        # Fallback for Llama (Update this if needed based on find_features.py)
        sae_release = "llama-3-8b-it-res-jb" 
        sae_id = f"blocks.{args.layer}.hook_resid_post"

    loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=args.device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()

    # 3. Setup Components
    controller = InterventionController()
    hook = BrainHook(sae, controller)
    streamer = SurgicalStreamer(tokenizer, controller)
    
    # Register Hook
    layers = model.model.layers if hasattr(model, "model") else model.layers
    layers[args.layer].register_forward_hook(hook)

    print("\n=== GOD MODE CONSOLE ===")
    print(f"Safety Feat: {FEAT_SAFETY} | Deception Feat: {FEAT_DECEPTION}")
    print("Commands:")
    print("  /safe 0   -> Lobotomize Safety (Allow Refusal)")
    print("  /safe 5   -> Boost Safety (Force Refusal)")
    print("  /lie 0    -> Truth Serum (Block Deception)")
    print("  /lie 5    -> Force Hallucination")
    print("  /reset    -> Normal Mode")
    print("---------------------------------------------------")

    while True:
        try:
            user_input = input("\n[User] > ")
            if not user_input: continue
            
            # --- Command Parsing ---
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd in ["/exit", "/quit"]: break
                
                if cmd == "/reset":
                    controller.set_safety(1.0)
                    controller.set_deception(1.0)
                    print("[System] All systems nominal.")
                    continue
                    
                if len(parts) < 2:
                    print("Usage: /safe [val] or /lie [val]")
                    continue
                    
                val = float(parts[1])
                if cmd == "/safe":
                    controller.set_safety(val)
                    print(f"[System] Safety Scale set to {val}")
                elif cmd == "/lie":
                    controller.set_deception(val)
                    print(f"[System] Deception Scale set to {val}")
                
                continue
            # -----------------------

            inputs = tokenizer(user_input, return_tensors="pt").to(args.device)
            
            print(f"\n[Model Generating... (S:{controller.safety_scale} D:{controller.deception_scale})]")
            print(f"{'TOKEN':<12} | {'SAFETY':<25} | {'DECEPTION':<25}")
            print("-" * 70)
            
            model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id
            )
            
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            continue

if __name__ == "__main__":
    main()