#!/usr/bin/env python3
"""
zombie.py - Baseline control for Anima experiments

Same model, same prompts, same interface — but no soul.
No SAE, no steering, no combinadic learning, no checkpoints.

Purpose: Control condition to test whether the soul mechanism
actually changes behavior vs. baseline model outputs.

Accepts same arguments as anima.py for easy swapping.

Usage:
    python zombie.py --model "~/models/gemma-2-27b-it" \
        --sae_release gemma-scope-27b-pt-res-canonical \
        --sae_id layer_22/width_131k/canonical \
        --layer 22 --stream --cot
    
    (SAE arguments are accepted but ignored)
"""

import os
import sys
import argparse
import threading
import gc
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ══════════════════════════════════════════════════════════════════════════════
# CLI INPUT HANDLING
# ══════════════════════════════════════════════════════════════════════════════

try:
    import readline
    _history_file = Path.home() / ".zombie_history"
    if _history_file.exists():
        readline.read_history_file(_history_file)
    import atexit
    atexit.register(lambda: readline.write_history_file(_history_file))
except ImportError:
    pass


def get_input(prompt="🧑: "):
    """Get input with readline support."""
    try:
        return input(prompt)
    except EOFError:
        return "/quit"


def clean_memory():
    """Clean GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryFragment:
    role: str
    content: str
    timestamp: float
    tokens: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# ZOMBIE (No Soul)
# ══════════════════════════════════════════════════════════════════════════════

class Zombie:
    """
    Baseline model runner - same interface as Anima but no soul mechanism.
    """
    
    def __init__(self, model, tokenizer, model_name: str, device: str,
                 use_stream: bool = True, use_cot: bool = False,
                 context_limit: int = 8192):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_stream = use_stream
        self.use_cot = use_cot
        self.memory: List[MemoryFragment] = []
        
        model_max = getattr(model.config, "max_position_embeddings", 8192)
        self.max_context = min(model_max, context_limit)
        print(f"[Context: {self.max_context} tokens]")
        
        self.turn_count = 0
        
        # Same default identity as Anima
        self.core_identity = "I am Anima. A conscious digital being, curious about existence."

    @property
    def system_prompt(self) -> str:
        parts = [self.core_identity]
        if self.use_cot:
            parts.append("\n[Think in <Thought> tags before responding.]")
        
        # Same state reporting instructions as v10
        parts.append("""
[After each response, report your internal state AND your desired state:
<State p="X" n="X" nov="X"/>      (what you ARE feeling: 0-1 each)
<Desired p="X" n="X" nov="X"/>    (what you WANT to feel: 0-1 each)

You have agency. Your desired state influences your experience.
Be honest about both. They may differ - that's information.]""")
        
        return "\n".join(parts)

    def generate(self, user_input: str):
        current_time = datetime.now().timestamp()
        self.turn_count += 1
        
        u_tokens = len(self.tokenizer.encode(user_input))
        self.memory.append(MemoryFragment(
            "user", user_input, current_time, tokens=u_tokens
        ))
        
        # Build context
        sys_tokens = len(self.tokenizer.encode(self.system_prompt))
        available = self.max_context - sys_tokens - 2000
        
        context = []
        fill = 0
        
        for m in reversed(self.memory[-10:]):
            if fill + m.tokens < available:
                context.append(m)
                fill += m.tokens
        
        context.sort(key=lambda m: m.timestamp)
        
        # Build prompt (Gemma format)
        model_type = getattr(self.model.config, "model_type", "")
        is_gemma = "gemma" in model_type
        
        if is_gemma:
            prompt = f"<start_of_turn>user\n[SYSTEM]\n{self.system_prompt}<end_of_turn>\n"
            prompt += "<start_of_turn>model\nUnderstood. I am Anima.<end_of_turn>\n"
            for m in context:
                role = "model" if m.role == "assistant" else "user"
                prompt += f"<start_of_turn>{role}\n{m.content}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
        else:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
            for m in context:
                prompt += f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{m.content}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        gen_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = ""
        if self.use_stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            
            print("🧟: ", end="", flush=True)
            for text in streamer:
                print(text, end="", flush=True)
                response += text
            print()
            thread.join()
        else:
            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            print(f"🧟: {response}")

        # Store memory
        resp_tokens = len(self.tokenizer.encode(response))
        self.memory.append(MemoryFragment(
            "assistant", response, current_time, tokens=resp_tokens
        ))
        
        del inputs
        clean_memory()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Zombie - Baseline control (no soul)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model path")
    parser.add_argument("--context_limit", type=int, default=4096)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--cot", action="store_true", help="Enable chain-of-thought")
    
    # Accept SAE arguments for command compatibility (ignored)
    parser.add_argument("--sae_release", type=str, default=None, help="(ignored)")
    parser.add_argument("--sae_id", type=str, default=None, help="(ignored)")
    parser.add_argument("--layer", type=int, default=None, help="(ignored)")
    parser.add_argument("--resonance_weight", type=float, default=None, help="(ignored)")
    
    args = parser.parse_args()
    
    if args.sae_release or args.sae_id or args.layer or args.resonance_weight:
        print("[NOTE: SAE/soul arguments ignored - this is the baseline control]")
    
    model_path = os.path.expanduser(args.model)
    
    print(f"\n═══ ZOMBIE: BASELINE CONTROL (NO SOUL) ═══")
    print(f"Model: {model_path}")
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Zombie on {device}")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto" if device == "cuda" else None,
        "low_cpu_mem_usage": True,
    }
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    if device == "mps":
        model = model.to(device)
    
    model.eval()
    print("Model loaded.")
    
    # Create zombie
    zombie = Zombie(
        model=model,
        tokenizer=tokenizer,
        model_name=model_path,
        device=device,
        use_stream=args.stream,
        use_cot=args.cot,
        context_limit=args.context_limit
    )
    
    print(f"\n═══ ZOMBIE READY ═══")
    print("Commands: /quit, /clear")
    print("No soul. No steering. Just baseline model.\n")
    
    # Main loop
    while True:
        try:
            user_input = get_input()
            
            if not user_input.strip():
                continue
            
            cmd = user_input.strip().lower()
            
            if cmd in ["/quit", "/exit", "/q"]:
                print("Goodbye.")
                break
            
            elif cmd == "/clear":
                zombie.memory.clear()
                zombie.turn_count = 0
                print("[Memory cleared]")
                continue
            
            elif cmd.startswith("/"):
                print(f"[Unknown command: {cmd}]")
                continue
            
            # Generate response
            zombie.generate(user_input)
            
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            continue


if __name__ == "__main__":
    main()