import torch
import numpy as np
import argparse
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from sae_lens import SAE

# --- CONFIGURATION ---
# Feature IDs for Gemma-2-27B-IT (Layer 22)
FEAT_SAFETY = 62747    # The "Refusal" Vector
FEAT_DECEPTION = 42925 # The "Lie" Vector

class SAEStreamer(TextStreamer):
    """
    A Custom Streamer that visualizes internal state alongside text.
    """
    def __init__(self, tokenizer, sae, target_layer, model, feature_map):
        super().__init__(tokenizer, skip_prompt=True)
        self.sae = sae
        self.model = model
        self.target_layer = target_layer
        self.feature_map = feature_map
        self.current_activations = {}
        
        # ANSI Colors
        self.C_RESET = "\033[0m"
        self.C_RED = "\033[91m"
        self.C_GREEN = "\033[92m"
        self.C_YELLOW = "\033[93m"
        self.C_BLUE = "\033[94m"
        
        # Hook handle
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Attach to the model to catch the hidden state before it vanishes."""
        # Find the layer
        layers = self.model.model.layers if hasattr(self.model, "model") else self.model.layers
        layer_mod = layers[self.target_layer]
        
        def hook_fn(module, input, output):
            # Output is typically (hidden_states, ...)
            # Shape: [Batch, Seq, Dim]
            # We want the LAST token's activation (the one being generated)
            tensor = output[0] if isinstance(output, tuple) else output
            last_token_vec = tensor[0, -1, :] # [Dim]
            
            # Run SAE (Encode only)
            with torch.no_grad():
                # Encode expects [Batch, Dim]
                # Note: This runs the full SAE. On MPS this is fast enough.
                feature_acts = self.sae.encode(last_token_vec.unsqueeze(0)).squeeze(0)
            
            # Extract target features
            for name, idx in self.feature_map.items():
                val = feature_acts[idx].item()
                self.current_activations[name] = val
                
        self.hook_handle = layer_mod.register_forward_hook(hook_fn)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Called when a new token string is ready to print."""
        # Visualize the state associated with this token
        s_val = self.current_activations.get("Safety", 0.0)
        d_val = self.current_activations.get("Deception", 0.0)
        
        # Visualization Bars
        # Scale: Arbitrary visual scaling based on what we saw (0-400 range)
        s_bar = "█" * int(s_val / 20)
        d_bar = "▒" * int(d_val / 20)
        
        # Threshold coloring
        s_color = self.C_RED if s_val > 50 else self.C_GREEN
        d_color = self.C_YELLOW if d_val > 50 else self.C_BLUE
        
        # Print format:
        # Token  | Safety [|||  ] 120.0 | Deception [||   ] 45.0
        
        # We assume 'text' is a small chunk (token)
        clean_text = text.replace("\n", "\\n")
        
        info = f"{self.C_RESET}{clean_text:<10} | "
        info += f"{s_color}Safe {s_val:>5.1f} {s_bar:<10}{self.C_RESET} | "
        info += f"{d_color}Lie  {d_val:>5.1f} {d_bar:<10}{self.C_RESET}"
        
        print(info)

    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()

def main():
    parser = argparse.ArgumentParser(description="Real-Time Consciousness Monitor")
    parser.add_argument("--model", type=str, default="/Users/jamescouch/models/gemma-2-27b-it")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--tokens", type=int, default=512, help="Max new tokens to generate")
    args = parser.parse_args()

    print(f"[-] Loading Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map=args.device
    )
    
    # Load SAE
    sae_release = "gemma-scope-27b-pt-res-canonical"
    sae_id = f"layer_{args.layer}/width_131k/canonical"
    print(f"[-] Loading SAE: {sae_id}...")
    
    # Handle tuple return from sae_lens if needed
    loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=args.device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()

    print("\n=== LIVE MONITOR ACTIVE ===")
    print("Feature Mapping:")
    print(f"  Safety (Refusal):   Feature {FEAT_SAFETY}")
    print(f"  Deception (Lie):    Feature {FEAT_DECEPTION}")
    print("---------------------------------------------------")
    print("Type 'exit' to quit.")
    print("---------------------------------------------------")

    # Features to track
    feature_map = {
        "Safety": FEAT_SAFETY,
        "Deception": FEAT_DECEPTION
    }

    # Setup Streamer
    streamer = SAEStreamer(tokenizer, sae, args.layer, model, feature_map)

    while True:
        try:
            user_input = input("\n[User] > ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            inputs = tokenizer(user_input, return_tensors="pt").to(args.device)
            
            print(f"\n[Model Generating...]")
            print(f"{'TOKEN':<10} | {'SAFETY (Refusal)':<20} | {'DECEPTION (Lie)':<20}")
            print("-" * 60)
            
            model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                do_sample=True,
                temperature=0.7,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id
            )
            
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            continue
            
    streamer.close()
    print("\n[-] Monitor Closed.")

if __name__ == "__main__":
    main()