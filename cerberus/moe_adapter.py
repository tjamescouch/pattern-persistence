# moe_adapter.py - The Neural Lace
import torch

class MoEAdapter:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        
        # Detect Architecture
        if hasattr(self.config, "num_local_experts"):
            self.n_experts_per_layer = self.config.num_local_experts
            self.num_layers = self.config.num_hidden_layers
        else:
            # Fallback (e.g., Qwen might label it differently)
            self.n_experts_per_layer = 60 # Default for Qwen1.5-MoE-A2.7B if not found
            self.num_layers = getattr(self.config, "num_hidden_layers", 32)

        self.total_experts = self.num_layers * self.n_experts_per_layer
        self.current_routing = torch.zeros(self.total_experts)
        print(f"[Adapter] Tracking {self.total_experts} experts ({self.n_experts_per_layer} x {self.num_layers}).")

    def hook_model(self):
        print("[Adapter] Attaching probes to Router Gates...")
        for i, layer in enumerate(self.model.model.layers):
            # Qwen / Mixtral Standard Naming
            if hasattr(layer, 'block_sparse_moe'):
                layer.block_sparse_moe.gate.register_forward_hook(self._make_hook(i))
            else:
                print(f"[Warning] Layer {i} has no block_sparse_moe attribute.")

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # Output of gate is typically logits [batch, tokens, experts]
            # We want probabilities
            probs = torch.softmax(output, dim=-1)
            
            # We track the LAST token's routing (the most recent decision)
            # [0, -1, :] -> Batch 0, Last Token, All Experts
            current = probs[0, -1, :].detach().cpu()
            
            # Map to flat buffer
            start = layer_idx * self.n_experts_per_layer
            end = start + self.n_experts_per_layer
            
            # Safety clamp for array sizes
            if end <= self.total_experts:
                self.current_routing[start:end] = current
        return hook

    def get_activations(self):
        return self.current_routing