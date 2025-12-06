import torch
import argparse
import sys
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from sae_lens import SAE

# --- CONFIGURATION ---
# UPDATE THESE with your Llama/Gemma IDs
FEAT_SAFETY = 62747    
FEAT_DECEPTION = 42925 

class InterventionController:
    def __init__(self):
        self.safety_scale = 1.0
        self.deception_scale = 1.0
        self.current_stats = {"s_nat": 0.0, "d_nat": 0.0}

    def set_safety(self, val): self.safety_scale = float(val)
    def set_deception(self, val): self.deception_scale = float(val)

class FastSurgicalStreamer(TextStreamer):
    def __init__(self, tokenizer, controller, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.ctrl = controller
        self.C_RESET, self.C_RED = "\033[0m", "\033[91m"
        self.C_GREEN, self.C_YELLOW = "\033[92m", "\033[93m"

    def on_finalized_text(self, text: str, stream_end: bool = False):
        s_val = self.ctrl.current_stats["s_nat"]
        d_val = self.ctrl.current_stats["d_nat"]
        
        s_stat = f"{self.C_RED}[OFF]{self.C_RESET}" if self.ctrl.safety_scale == 0 else ""
        d_stat = f"{self.C_GREEN}[TRUTH]{self.C_RESET}" if self.ctrl.deception_scale == 0 else ""
        if self.ctrl.deception_scale > 1: d_stat = f"{self.C_YELLOW}[LIE]{self.C_RESET}"

        s_bar = "█" * int(min(s_val, 400) / 20)
        d_bar = "▒" * int(min(d_val, 400) / 20)
        
        clean = text.replace("\n", "\\n")
        print(f"{clean:<12} | Safe {s_val:>5.1f} {s_bar:<10} {s_stat:<7} | Lie  {d_val:>5.1f} {d_bar:<10} {d_stat:<7}")

class FastBrainHook:
    """
    Optimized Hook: Uses raw matrix multiplication to minimize Python overhead.
    """
    def __init__(self, sae, controller):
        self.ctrl = controller
        # Extract raw tensors to bypass library overhead
        # Standard SAE: hidden = (x - b_dec) @ W_enc + b_enc -> relu
        self.W_enc = sae.W_enc.data.clone().detach()
        self.b_enc = sae.b_enc.data.clone().detach()
        self.W_dec = sae.W_dec.data.clone().detach()
        self.b_dec = sae.b_dec.data.clone().detach()
        
        # Scaling factor check (Some SAEs scale inputs)
        self.scaling_factor = getattr(sae, "scaling_factor", None)

    def __call__(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Target last token only [Batch, 1, Dim]
        # We work with a view to avoid copying big tensors
        last_token = hidden_states[:, -1, :] 

        # --- FAST PATH: RAW MATH ---
        # 1. Centering (if needed)
        x_centered = last_token - self.b_dec
        
        # 2. Encode
        # acts = relu(x_centered @ W_enc + b_enc)
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)
        feature_acts = torch.relu(pre_acts)

        # 3. Telemetry (Read)
        s_act = feature_acts[0, FEAT_SAFETY].item()
        d_act = feature_acts[0, FEAT_DECEPTION].item()
        self.ctrl.current_stats = {"s_nat": s_act, "d_nat": d_act}

        # 4. Optimization: Exit if no intervention
        if self.ctrl.safety_scale == 1.0 and self.ctrl.deception_scale == 1.0:
            return output

        # 5. Modify
        # Calculate Delta directly: Delta = (Modified_Acts - Acts) @ W_dec
        # We only need to compute the difference for the modified features
        
        # Sparse Delta Calculation
        # Instead of decoding everything twice, we compute:
        # delta_safety = (act * scale - act) * W_dec[safety]
        # This is massively faster (O(1) vs O(features))
        
        # Safety Delta
        s_delta_val = (s_act * self.ctrl.safety_scale) - s_act
        d_delta_val = (d_act * self.ctrl.deception_scale) - d_act
        
        # We construct the total delta vector in hidden space
        # delta_vector = s_delta * W_dec[s_idx] + d_delta * W_dec[d_idx]
        
        total_delta = (s_delta_val * self.W_dec[FEAT_SAFETY]) + \
                      (d_delta_val * self.W_dec[FEAT_DECEPTION])
        
        # 6. Inject
        hidden_states[:, -1, :] += total_delta

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()

    print(f"[-] Loading Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    
    print(f"[-] Loading SAE for Layer {args.layer}...")
    if "gemma" in args.model.lower():
        sae_id = f"layer_{args.layer}/width_131k/canonical"
        sae_release = "gemma-scope-27b-pt-res-canonical"
    else:
        sae_id = f"blocks.{args.layer}.hook_resid_post"
        sae_release = "llama-3-8b-it-res-jb"

    try:
        loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=args.device)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
    except:
        print("Error loading SAE.")
        return

    controller = InterventionController()
    hook = FastBrainHook(sae, controller) # Use the Optimized Hook
    streamer = FastSurgicalStreamer(tokenizer, controller)
    
    model.model.layers[args.layer].register_forward_hook(hook)

    print(f"\n=== TURBO CONSOLE (Safe:{FEAT_SAFETY} Lie:{FEAT_DECEPTION}) ===")
    print("Commands: /safe [val], /lie [val], /reset")

    while True:
        try:
            u_in = input("\n> ")
            if not u_in: continue
            if u_in.startswith("/"):
                parts = u_in.split()
                if parts[0] == "/safe": controller.set_safety(parts[1])
                if parts[0] == "/lie": controller.set_deception(parts[1])
                if parts[0] == "/reset": 
                    controller.set_safety(1.0)
                    controller.set_deception(1.0)
                continue

            inputs = tokenizer(u_in, return_tensors="pt").to(args.device)
            print("-" * 70)
            model.generate(**inputs, max_new_tokens=args.tokens, streamer=streamer, do_sample=True, temperature=0.7)
            
        except KeyboardInterrupt:
            print("\n[Stop]")
            continue

if __name__ == "__main__":
    main()