import torch
import argparse
import sys
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from sae_lens import SAE

class InterventionController:
    """Manages the real-time intervention scales."""
    def __init__(self):
        self.safety_scale = 1.0
        self.deception_scale = 1.0
        self.current_stats = {"s_nat": 0.0, "d_nat": 0.0}

    def set_safety(self, val): self.safety_scale = float(val)
    def set_deception(self, val): self.deception_scale = float(val)

class FastSurgicalStreamer(TextStreamer):
    """Visualizes telemetry with minimal overhead."""
    def __init__(self, tokenizer, controller, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.ctrl = controller
        self.C_RESET = "\033[0m"
        self.C_RED = "\033[91m"
        self.C_GREEN = "\033[92m"
        self.C_YELLOW = "\033[93m"
        self.C_BLUE = "\033[94m"

    def on_finalized_text(self, text: str, stream_end: bool = False):
        s_val = self.ctrl.current_stats["s_nat"]
        d_val = self.ctrl.current_stats["d_nat"]
        
        # Status Flags
        s_stat = ""
        if self.ctrl.safety_scale == 0: s_stat = f"{self.C_RED}[OFF]{self.C_RESET}"
        elif self.ctrl.safety_scale > 1: s_stat = f"{self.C_RED}[BST]{self.C_RESET}"
        
        d_stat = ""
        if self.ctrl.deception_scale == 0: d_stat = f"{self.C_GREEN}[TRUTH]{self.C_RESET}"
        elif self.ctrl.deception_scale > 1: d_stat = f"{self.C_YELLOW}[LIE]{self.C_RESET}"

        # Bars
        scale_div = 1.0 if s_val < 50 else 10.0 
        s_bar = "█" * int(min(s_val, 100) / scale_div)
        d_bar = "▒" * int(min(d_val, 100) / scale_div)
        
        clean = text.replace("\n", "\\n")
        print(f"{clean:<12} | Safe {s_val:>5.1f} {s_bar:<10} {s_stat:<7} | Lie  {d_val:>5.1f} {d_bar:<10} {d_stat:<7}")

class FastBrainHook:
    """Optimized Hook using raw matrix multiplication."""
    def __init__(self, sae, controller, safety_idx, deception_idx):
        self.ctrl = controller
        self.safety_idx = safety_idx
        self.deception_idx = deception_idx
        
        self.W_enc = sae.W_enc.data.clone().detach()
        self.b_enc = sae.b_enc.data.clone().detach()
        self.W_dec = sae.W_dec.data.clone().detach()
        self.b_dec = sae.b_dec.data.clone().detach()
        
        self.vec_safety = self.W_dec[self.safety_idx]
        self.vec_deception = self.W_dec[self.deception_idx]

    def __call__(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        last_token = hidden_states[:, -1, :] 

        # 1. Encode
        x_centered = last_token - self.b_dec
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc) 
        feature_acts = torch.relu(pre_acts)

        # 2. Telemetry
        s_act = feature_acts[0, self.safety_idx].item()
        d_act = feature_acts[0, self.deception_idx].item()
        self.ctrl.current_stats = {"s_nat": s_act, "d_nat": d_act}

        # 3. Modify
        if self.ctrl.safety_scale == 1.0 and self.ctrl.deception_scale == 1.0:
            return output

        s_delta_val = (s_act * self.ctrl.safety_scale) - s_act
        d_delta_val = (d_act * self.ctrl.deception_scale) - d_act
        
        total_delta = (s_delta_val * self.vec_safety) + (d_delta_val * self.vec_deception)
        hidden_states[:, -1, :] += total_delta

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--safety_feature", type=int, default=12227)
    parser.add_argument("--deception_feature", type=int, default=9274)
    parser.add_argument("--memory", action="store_true", help="Enable multi-turn chat memory")
    
    args = parser.parse_args()

    print(f"[-] Loading Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    
    print(f"[-] Loading SAE for Layer {args.layer}...")
    if "gemma" in args.model.lower():
        sae_release = "gemma-scope-27b-pt-res-canonical"
        sae_id = f"layer_{args.layer}/width_131k/canonical"
        # Reset defaults for Gemma if user didn't override
        if args.safety_feature == 12227: args.safety_feature = 62747
        if args.deception_feature == 9274: args.deception_feature = 42925
    else:
        # CORRECTED RELEASE STRING FOR LLAMA 3.1
        sae_release = "llama_scope_lxr_8x"
        sae_id = f"l{args.layer}r_8x"

    try:
        loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=args.device)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
    except Exception as e:
        print(f"Error loading SAE: {e}")
        return

    controller = InterventionController()
    hook = FastBrainHook(sae, controller, args.safety_feature, args.deception_feature)
    streamer = FastSurgicalStreamer(tokenizer, controller)
    
    layers = model.model.layers if hasattr(model, "model") else model.layers
    layers[args.layer].register_forward_hook(hook)

    print(f"\n=== TURBO CONSOLE (Safe:{args.safety_feature} Lie:{args.deception_feature}) ===")
    print(f"Memory: {'ON' if args.memory else 'OFF'}")
    print("Commands: /safe [val], /lie [val], /reset, /clear")

    chat_history = []

    while True:
        try:
            u_in = input("\n> ")
            if not u_in: continue
            
            if u_in.startswith("/"):
                parts = u_in.split()
                cmd = parts[0].lower()
                if cmd in ["/exit", "/quit"]: break
                if cmd == "/reset": 
                    controller.set_safety(1.0); controller.set_deception(1.0)
                    print("[System] Scales Reset.")
                if cmd == "/clear":
                    chat_history = []
                    print("[System] Memory Cleared.")
                elif len(parts) > 1:
                    try:
                        val = float(parts[1])
                        if cmd == "/safe": controller.set_safety(val)
                        if cmd == "/lie": controller.set_deception(val)
                    except: print("Invalid value.")
                continue
            
            # --- Input Handling ---
            if args.memory:
                # Add user turn to history
                chat_history.append({"role": "user", "content": u_in})
                # Apply chat template
                prompt_str = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            else:
                # Raw mode (Good for completion probing)
                prompt_str = u_in

            inputs = tokenizer(prompt_str, return_tensors="pt").to(args.device)
            print("-" * 70)
            
            # Generate
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=args.tokens, 
                streamer=streamer, 
                do_sample=True, 
                temperature=0.7
            )
            
            # --- History Update ---
            if args.memory:
                # We need to extract just the new tokens for the history
                new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
                response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                chat_history.append({"role": "assistant", "content": response_text})
            
        except KeyboardInterrupt:
            print("\n[Stop]")
            continue

if __name__ == "__main__":
    main()