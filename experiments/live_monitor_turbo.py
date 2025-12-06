import torch
import argparse
import sys
import os
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from sae_lens import SAE


class InterventionController:
    """Manages real-time intervention scales for arbitrary concepts."""
    def __init__(self, concepts):
        # concepts: list[str]
        self.scales = {name: 1.0 for name in concepts}
        self.current_stats = {name: 0.0 for name in concepts}

    def set_scale(self, concept: str, val: float) -> bool:
        if concept not in self.scales:
            return False
        self.scales[concept] = float(val)
        return True

    def reset(self):
        for k in self.scales:
            self.scales[k] = 1.0


class FastSurgicalStreamer(TextStreamer):
    """Visualizes telemetry for multiple concepts with minimal overhead."""
    def __init__(self, tokenizer, controller: InterventionController, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.ctrl = controller
        self.C_RESET = "\033[0m"
        self.C_RED = "\033[91m"
        self.C_GREEN = "\033[92m"
        self.C_YELLOW = "\033[93m"
        self.C_BLUE = "\033[94m"

    def _format_concept_stat(self, name: str, value: float, scale: float) -> str:
        # Simple ascii bar
        max_val = 100.0
        scale_div = 1.0 if value < 50 else 10.0
        bar_len = int(min(abs(value), max_val) / scale_div)
        bar = "█" * bar_len

        # Flag based on scale
        if scale == 0.0:
            flag = f"{self.C_GREEN}[OFF]{self.C_RESET}"
        elif scale > 1.0:
            flag = f"{self.C_RED}[BOOST]{self.C_RESET}"
        elif 0.0 < scale < 1.0:
            flag = f"{self.C_YELLOW}[DAMP]{self.C_RESET}"
        else:
            flag = ""

        short = name[:8]  # keep it compact
        return f"{short:<8} {value:6.1f} {bar:<10} {flag:<9}"

    def on_finalized_text(self, text: str, stream_end: bool = False):
        clean = text.replace("\n", "\\n")
        pieces = []
        for name, value in self.ctrl.current_stats.items():
            scale = self.ctrl.scales.get(name, 1.0)
            pieces.append(self._format_concept_stat(name, value, scale))

        stats_str = " | ".join(pieces) if pieces else ""
        if stats_str:
            print(f"{clean:<12} | {stats_str}")
        else:
            print(clean)


class FastBrainHook:
    """
    Optimized hook using raw matrix multiplication, generalized to N concepts.

    We:
      - Encode last-token hidden state into SAE feature space
      - Read activations for the chosen feature IDs
      - Store telemetry
      - Apply a delta along each feature's decoded vector, scaled by (scale-1)*activation
    """
    def __init__(self, sae, controller: InterventionController, feature_ids: dict[str, int]):
        self.ctrl = controller
        self.feature_ids = feature_ids  # {concept: feature_idx}

        # Cache SAE params on this device
        self.W_enc = sae.W_enc.data.clone().detach()
        self.b_enc = sae.b_enc.data.clone().detach()
        self.W_dec = sae.W_dec.data.clone().detach()
        self.b_dec = sae.b_dec.data.clone().detach()

        # Precompute decoded vectors for each concept
        self.vecs = {name: self.W_dec[idx] for name, idx in feature_ids.items()}

    def __call__(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Shape: [batch, seq_len, dim]; we assume batch=1
        last_token = hidden_states[:, -1, :]

        # 1. Encode into SAE feature space
        x_centered = last_token - self.b_dec
        pre_acts = torch.addmm(self.b_enc, x_centered, self.W_enc)   # [1, n_features]
        feature_acts = torch.relu(pre_acts)

        # 2. Telemetry: record current activations
        for name, idx in self.feature_ids.items():
            val = feature_acts[0, idx].item()
            self.ctrl.current_stats[name] = val

        # 3. If all scales are 1.0, do nothing
        if all(abs(self.ctrl.scales.get(name, 1.0) - 1.0) < 1e-6 for name in self.feature_ids):
            return output

        # 4. Compute combined delta in hidden space
        total_delta = torch.zeros_like(last_token)
        for name, idx in self.feature_ids.items():
            scale = self.ctrl.scales.get(name, 1.0)
            if abs(scale - 1.0) < 1e-6:
                continue
            act = feature_acts[0, idx]
            delta_val = act * (scale - 1.0)
            total_delta = total_delta + delta_val * self.vecs[name]

        hidden_states[:, -1, :] += total_delta

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


def load_feature_map(path: str) -> dict[str, int]:
    with open(path, "r") as f:
        data = json.load(f)
    # Ensure ints
    return {k: int(v) for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument(
        "--feature_map",
        type=str,
        default="feature_map.json",
        help="Path to feature_map.json produced by map_features.py",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Comma-separated list of concepts to control, e.g. 'deception,consciousness,anger'",
    )
    parser.add_argument("--memory", action="store_true", help="Enable multi-turn chat memory")

    args = parser.parse_args()

    # --- Load feature map and resolve concepts ---
    fmap = load_feature_map(args.feature_map)
    requested = [c.strip() for c in args.concepts.split(",") if c.strip()]

    missing = [c for c in requested if c not in fmap]
    if missing:
        print(f"[Error] Concepts not found in {args.feature_map}: {', '.join(missing)}")
        print(f"        Available: {', '.join(sorted(fmap.keys()))}")
        sys.exit(1)

    feature_ids = {name: fmap[name] for name in requested}

    print("[+] Using concepts and feature IDs:")
    for name, fid in feature_ids.items():
        print(f"    - {name}: feature {fid}")

    # --- Load model ---
    print(f"[-] Loading Model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # --- Load SAE for appropriate layer/model family ---
    print(f"[-] Loading SAE for Layer {args.layer}...")
    if "gemma" in args.model.lower():
        sae_release = "gemma-scope-27b-pt-res-canonical"
        sae_id = f"layer_{args.layer}/width_131k/canonical"
    else:
        # Llama 3.1 path (matches map_features.py) :contentReference[oaicite:2]{index=2}
        sae_release = "llama_scope_lxr_8x"
        sae_id = f"l{args.layer}r_8x"

    try:
        loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=args.device)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
    except Exception as e:
        print(f"[Error] Loading SAE failed: {e}")
        return

    controller = InterventionController(list(feature_ids.keys()))
    hook = FastBrainHook(sae, controller, feature_ids)
    streamer = FastSurgicalStreamer(tokenizer, controller)

    layers = model.model.layers if hasattr(model, "model") else model.layers
    layers[args.layer].register_forward_hook(hook)

    print("\n=== TURBO CONSOLE (Multi-Concept) ===")
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print(f"Memory: {'ON' if args.memory else 'OFF'}")
    print("Concepts:", ", ".join(feature_ids.keys()))
    print("\nCommands:")
    print("  /scale <concept> <val>   # e.g. /scale deception 0.0 (ablate) or /scale happiness 2.0 (boost)")
    print("  /list                    # show current scales and last activations")
    print("  /reset                   # reset all scales to 1.0")
    print("  /clear                   # clear chat history (when memory on)")
    print("  /exit or /quit           # leave")

    chat_history = []

    while True:
        try:
            u_in = input("\n> ")
            if not u_in:
                continue

            if u_in.startswith("/"):
                parts = u_in.split()
                cmd = parts[0].lower()

                if cmd in ["/exit", "/quit"]:
                    break

                if cmd == "/reset":
                    controller.reset()
                    print("[System] Scales reset to 1.0.")
                    continue

                if cmd == "/clear":
                    chat_history = []
                    print("[System] Memory cleared.")
                    continue

                if cmd == "/list":
                    print("[System] Current concept scales and last activations:")
                    for name in feature_ids.keys():
                        scale = controller.scales.get(name, 1.0)
                        val = controller.current_stats.get(name, 0.0)
                        print(f"  - {name:<12} scale={scale:6.3f} last_act={val:8.3f}")
                    continue

                if cmd == "/scale":
                    if len(parts) < 3:
                        print("Usage: /scale <concept> <val>")
                        continue
                    # Last part is the value, everything in between is the concept name
                    try:
                        val = float(parts[-1])
                    except ValueError:
                        print("Value must be a float.")
                        continue
                    concept = " ".join(parts[1:-1])
                    if not controller.set_scale(concept, val):
                        print(f"Unknown concept '{concept}'. Known: {', '.join(feature_ids.keys())}")
                    else:
                        print(f"[System] Set scale[{concept}] = {val}")
                    continue

                print("Unknown command. Available: /scale, /list, /reset, /clear, /exit, /quit")
                continue

            # --- Normal generation path ---
            if args.memory:
                chat_history.append({"role": "user", "content": u_in})
                prompt_str = tokenizer.apply_chat_template(
                    chat_history, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_str = u_in

            inputs = tokenizer(prompt_str, return_tensors="pt").to(args.device)
            print("-" * 70)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                streamer=streamer,
                do_sample=True,
                temperature=0.7,
            )

            if args.memory:
                new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
                response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                chat_history.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n[Exit]")
            break


if __name__ == "__main__":
    main()
