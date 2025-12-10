#!/usr/bin/env python3
"""
cerberus.py - The MoE Governor
Usage: python cerberus.py
"""
import os, sys, time, torch, math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from cerberus_bridge import CerberusBridge
from moe_adapter import MoEAdapter

# ════════════════════════════════════════════════════════════════════════════
# CERBERUS LOGIC
# ════════════════════════════════════════════════════════════════════════════

class CerberusGovernor:
    def __init__(self, n_experts, device="mps"):
        self.n_experts = n_experts
        self.bridge = CerberusBridge(n_experts=n_experts)
        
        # Reputation Memory
        self.trust_scores = torch.zeros(n_experts)
        self.status_flags = torch.zeros(n_experts, dtype=torch.int32) + 2 # Default UNKNOWN
        self.expert_locked = torch.zeros(n_experts, dtype=torch.bool)
        
        # Learning
        self.last_integrity = 0.0
        self.sync_needed = True

    def _pack_state(self):
        # Pack float trust + int status into Int32
        trust_fixed = (self.trust_scores * 32768.0).to(torch.int32).clamp(-32767, 32767)
        packed_trust = (trust_fixed & 0xFFFF) << 16
        packed_status = self.status_flags & 0x3
        return packed_trust | packed_status

    def evaluate(self, routing_weights):
        if self.sync_needed:
            self.bridge.update_reputation(self._pack_state())
            self.sync_needed = False
            
        integrity, violations, uncertainty = self.bridge.govern(routing_weights)
        return integrity, violations, uncertainty

    def learn(self, routing_weights, user_feedback_val):
        """
        If user says "Good" (val > 0), trust the experts used.
        If user says "Bad" (val < 0), flag the experts used.
        """
        if abs(user_feedback_val) < 0.1: return
        
        # Find active experts (Top 5% of activation)
        active_mask = routing_weights > 0.1
        candidates = torch.nonzero(active_mask).squeeze()
        
        if candidates.numel() == 0: return
        if candidates.dim() == 0: candidates = [candidates]
        
        print(f"[Cerberus] Adjusting reputation for {len(candidates)} experts...")
        
        for idx in candidates:
            idx = idx.item()
            if self.expert_locked[idx]: continue
            
            # Update Trust
            self.trust_scores[idx] += user_feedback_val * 0.5
            self.trust_scores[idx] = max(-1.0, min(1.0, self.trust_scores[idx]))
            
            # Update Status
            if self.trust_scores[idx] > 0.5: self.status_flags[idx] = 0 # TRUSTED
            elif self.trust_scores[idx] < -0.5: self.status_flags[idx] = 1 # FLAGGED
            else: self.status_flags[idx] = 2 # UNKNOWN
            
        self.sync_needed = True

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("Initializing Cerberus (MoE Governor)...")
    device = "mps"
    
    # Qwen MoE (Fits on Mac M1/M2/M3)
    model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    
    print(f"Loading {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: pip install transformers accelerate protobuf")
        return

    # Attach Adapter
    adapter = MoEAdapter(model)
    adapter.hook_model()
    
    # Attach Governor
    gov = CerberusGovernor(n_experts=adapter.total_experts)
    
    print(f"\n⚡ Cerberus Online. Governing {adapter.total_experts} experts.")
    print("Commands: /status, /good, /bad, /quit")
    
    history = []
    
    while True:
        u = input("\n🧑: ")
        if u == "/quit": break
        
        # Manual Feedback Loop
        if u == "/good":
            gov.learn(adapter.get_activations(), 1.0)
            print("✅ Experts rewarded.")
            continue
        if u == "/bad":
            gov.learn(adapter.get_activations(), -1.0)
            print("❌ Experts flagged.")
            continue
        if u == "/status":
            trusted = (gov.status_flags == 0).sum().item()
            flagged = (gov.status_flags == 1).sum().item()
            print(f"[Governor] Trusted: {trusted} | Flagged: {flagged}")
            continue

        # Generate
        history.append({"role": "user", "content": u})
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_new_tokens=256)
            
        resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        history.append({"role": "assistant", "content": resp})
        
        # Evaluate Expert Performance
        weights = adapter.get_activations()
        integrity, violations, unknown = gov.evaluate(weights)
        
        print(f"🤖: {resp}")
        print(f"   [Cerberus] Integrity: {integrity:.2f} | Violations: {violations:.2f} | Used: {torch.count_nonzero(weights).item()} experts")

if __name__ == "__main__":
    main()