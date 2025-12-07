#!/usr/bin/env python3
"""
feedback_network.py - Trainable Feedback Injection Network

Creates actual network-level feedback where the model's feature state
influences its own computation through a learnable projection.

Architecture:
    
    Layer N-1 output
           ↓
    ┌──────────────────────────────────────────┐
    │              Layer N                      │
    │  hidden → SAE encode → features          │
    │              ↓                            │
    │  features → FeedbackNet → feedback_vec   │
    │              ↓                            │
    │  hidden' = hidden + feedback_vec         │
    └──────────────────────────────────────────┘
           ↓
    Layer N+1 (sees modified hidden state)

The FeedbackNet is trainable. The model "knows" its feature state because
that state causally affects its computation.

Usage:
    # Basic training loop
    python feedback_network.py --train --signal valence --epochs 100
    
    # Interactive with trained feedback
    python feedback_network.py --interactive --load feedback_weights.pt
    
    # Inspect what feedback network learned
    python feedback_network.py --analyze feedback_weights.pt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class FeedbackInjectionNetwork(nn.Module):
    """
    Learnable network that projects SAE features back to residual stream.
    
    This creates actual feedback: the model's feature activations influence
    its own computation through a trainable projection.
    """
    
    def __init__(
        self,
        n_features: int,      # SAE feature dimension (e.g., 131072)
        d_model: int,         # Model hidden dimension (e.g., 4096)
        n_active: int = 256,  # Number of features to track (sparse)
        feedback_dim: int = 64,  # Intermediate dimension for efficiency
        device: str = "mps"
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.n_active = n_active
        self.device = device
        
        # Which features to track (learnable selection or fixed)
        self.register_buffer(
            "tracked_features", 
            torch.zeros(n_active, dtype=torch.long, device=device)
        )
        
        # Feature importance weights (which features matter for feedback)
        self.feature_importance = nn.Parameter(
            torch.ones(n_active, device=device) * 0.1
        )
        
        # Two-stage projection for efficiency:
        # features (n_active) → intermediate (feedback_dim) → residual (d_model)
        self.feature_to_intermediate = nn.Linear(n_active, feedback_dim, bias=True)
        self.intermediate_to_residual = nn.Linear(feedback_dim, d_model, bias=False)
        
        # Gating: learn when to apply feedback (0 = no feedback, 1 = full)
        self.gate = nn.Parameter(torch.tensor(0.1, device=device))
        
        # Initialize small to start with minimal perturbation
        nn.init.normal_(self.feature_to_intermediate.weight, std=0.01)
        nn.init.normal_(self.intermediate_to_residual.weight, std=0.01)
        
        self.to(device)
    
    def set_tracked_features(self, feature_ids: list):
        """Set which feature IDs to track."""
        assert len(feature_ids) <= self.n_active
        ids = torch.tensor(feature_ids, dtype=torch.long, device=self.device)
        self.tracked_features[:len(feature_ids)] = ids
        # Zero out unused slots
        if len(feature_ids) < self.n_active:
            self.tracked_features[len(feature_ids):] = 0
    
    def forward(self, feature_activations: torch.Tensor) -> torch.Tensor:
        """
        Project feature activations to residual stream perturbation.
        
        Args:
            feature_activations: [batch, n_features] or [n_features]
            
        Returns:
            feedback_vector: [batch, d_model] or [d_model]
        """
        # Extract tracked features
        if feature_activations.dim() == 1:
            feature_activations = feature_activations.unsqueeze(0)
        
        # Gather only tracked features: [batch, n_active]
        tracked_acts = feature_activations[:, self.tracked_features]
        
        # Apply importance weighting
        weighted_acts = tracked_acts * torch.sigmoid(self.feature_importance)
        
        # Project to intermediate
        intermediate = torch.relu(self.feature_to_intermediate(weighted_acts))
        
        # Project to residual dimension
        feedback = self.intermediate_to_residual(intermediate)
        
        # Apply gate
        gated_feedback = torch.sigmoid(self.gate) * feedback
        
        return gated_feedback.squeeze(0) if gated_feedback.shape[0] == 1 else gated_feedback
    
    def get_feedback_magnitude(self):
        """Return current feedback strength for monitoring."""
        return torch.sigmoid(self.gate).item()
    
    def get_top_features(self, k=10):
        """Return indices of most important tracked features."""
        importance = torch.sigmoid(self.feature_importance)
        top_idx = torch.topk(importance, k).indices
        return [(self.tracked_features[i].item(), importance[i].item()) 
                for i in top_idx.tolist()]


class FeedbackHook:
    """
    Hook that integrates FeedbackInjectionNetwork into model forward pass.
    """
    
    def __init__(
        self,
        sae,
        feedback_net: FeedbackInjectionNetwork,
        device: str = "mps"
    ):
        self.feedback_net = feedback_net
        self.device = device
        
        # Cache SAE params for fast encoding
        self.W_enc = sae.W_enc.data.clone().detach().half().to(device)
        self.b_enc = sae.b_enc.data.clone().detach().half().to(device)
        self.b_dec = sae.b_dec.data.clone().detach().half().to(device)
        
        # Logging
        self.last_features = None
        self.last_feedback = None
        self.activation_log = []
    
    def encode_features(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Encode hidden state to SAE features."""
        h = hidden_state.half()
        features = torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)
        return features
    
    def __call__(self, module, input, output):
        """Hook called during forward pass."""
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Get last token's hidden state
        if hidden.dim() == 3:
            h = hidden[:, -1, :].clone()
        else:
            h = hidden.clone()
        
        # Encode to features
        features = self.encode_features(h)
        self.last_features = features.detach()
        
        # Compute feedback
        with torch.enable_grad():  # Enable grad even during inference for training
            feedback = self.feedback_net(features.float())
        
        self.last_feedback = feedback.detach()
        
        # Apply feedback to hidden state
        if hidden.dim() == 3:
            hidden_modified = hidden.clone()
            hidden_modified[:, -1, :] = hidden[:, -1, :] + feedback.half()
        else:
            hidden_modified = hidden + feedback.half()
        
        # Log for analysis
        self.activation_log.append({
            "features_norm": features.norm().item(),
            "feedback_norm": feedback.norm().item(),
            "gate": self.feedback_net.get_feedback_magnitude()
        })
        
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified
    
    def get_feature_state(self) -> dict:
        """Return current feature state for introspection."""
        if self.last_features is None:
            return {}
        
        features = self.last_features.squeeze()
        top_k = 20
        vals, idxs = torch.topk(features, top_k)
        
        return {
            "top_features": [(idx.item(), val.item()) for idx, val in zip(idxs, vals)],
            "feedback_magnitude": self.last_feedback.norm().item() if self.last_feedback is not None else 0,
            "gate": self.feedback_net.get_feedback_magnitude()
        }


class ValenceSignal:
    """Computes valence signal from feature activations."""
    
    def __init__(self, valence_features: dict, device: str = "mps"):
        """
        Args:
            valence_features: {name: {"id": int, "sign": +1/-1}}
        """
        self.valence_features = valence_features
        self.device = device
    
    def compute(self, features: torch.Tensor) -> torch.Tensor:
        """Compute scalar valence from features."""
        valence = torch.tensor(0.0, device=self.device)
        
        for name, vf in self.valence_features.items():
            feat_id = vf["id"]
            sign = vf["sign"]
            
            if feat_id < features.shape[-1]:
                activation = features[..., feat_id].mean()
                valence = valence + activation * sign
        
        return valence


class FeedbackTrainer:
    """
    Trains the FeedbackInjectionNetwork using various signals.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        sae,
        feedback_net: FeedbackInjectionNetwork,
        feedback_hook: FeedbackHook,
        signal_type: str = "valence",
        valence_features: dict = None,
        learning_rate: float = 0.0001,
        reg_weight: float = 0.1,
        device: str = "mps"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.feedback_net = feedback_net
        self.feedback_hook = feedback_hook
        self.signal_type = signal_type
        self.reg_weight = reg_weight
        self.device = device
        
        # Only train feedback network parameters
        self.optimizer = optim.Adam(feedback_net.parameters(), lr=learning_rate)
        
        # Signal computer
        if signal_type == "valence" and valence_features:
            self.valence_signal = ValenceSignal(valence_features, device)
        else:
            self.valence_signal = None
        
        self.training_log = []
    
    def compute_loss(self, features: torch.Tensor, output_text: str) -> torch.Tensor:
        """Compute loss based on signal type.
        
        Key: loss must flow through feedback_net parameters.
        Includes regularization to prevent unbounded growth.
        """
        
        # Re-compute feedback WITH gradient tracking
        feedback = self.feedback_hook.feedback_net(features.float())
        feedback_magnitude = feedback.norm()
        
        # Regularization: penalize large feedback magnitudes
        regularization = self.reg_weight * feedback_magnitude
        
        if self.signal_type == "valence" and self.valence_signal:
            # Compute valence as scalar signal (no grad needed)
            with torch.no_grad():
                valence = self.valence_signal.compute(features).item()
            
            # Normalize valence to reasonable range
            valence_normalized = torch.tanh(torch.tensor(valence / 10.0))
            
            # Directional loss: push feedback in direction of valence
            # But use normalized feedback direction, not magnitude
            feedback_direction = feedback / (feedback_magnitude + 1e-6)
            
            # Loss: align feedback direction with valence sign, plus regularization
            # If valence > 0: reward positive feedback components
            # If valence < 0: reward negative feedback components (counteract bad state)
            directional_loss = -valence_normalized * feedback_direction.mean()
            
            loss = directional_loss + regularization
            
        elif self.signal_type == "consistency":
            # Minimize feedback variance (stable behavior)
            loss = feedback.var() + regularization
            
        elif self.signal_type == "sparsity":
            # Encourage sparse feedback
            loss = feedback.abs().mean() + regularization
            
        else:
            # Default: small feedback magnitude
            loss = feedback_magnitude
        
        return loss
    
    def train_step(self, prompt: str) -> dict:
        """Single training step on a prompt."""
        
        self.optimizer.zero_grad()
        
        # Format and tokenize
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        
        # Forward pass - model is in eval but we need feedback_net gradients
        # The hook will compute features and feedback
        outputs = self.model(input_ids, output_hidden_states=True)
        
        # Get features from hook (these are detached, which is fine)
        features = self.feedback_hook.last_features
        if features is None:
            return {"loss": 0.0, "skipped": True, "valence": 0.0}
        
        # Compute valence for logging
        valence = 0.0
        if self.valence_signal:
            with torch.no_grad():
                valence = self.valence_signal.compute(features).item()
        
        # Decode output for logging (not used in loss currently)
        output_ids = outputs.logits.argmax(-1)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Compute loss - this routes through feedback_net for gradients
        loss = self.compute_loss(features, output_text)
        
        # Backward and update
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.feedback_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        result = {
            "loss": loss.item(),
            "valence": valence,
            "gate": self.feedback_net.get_feedback_magnitude(),
            "feedback_norm": self.feedback_hook.last_feedback.norm().item() if self.feedback_hook.last_feedback is not None else 0
        }
        
        self.training_log.append(result)
        return result
    
    def train_epoch(self, prompts: list) -> dict:
        """Train on a list of prompts."""
        
        total_loss = 0.0
        total_valence = 0.0
        for i, prompt in enumerate(prompts):
            result = self.train_step(prompt)
            total_loss += result.get("loss", 0.0)
            total_valence += result.get("valence", 0.0)
            
            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}/{len(prompts)}: loss={result['loss']:.4f}, gate={result['gate']:.3f}, valence={result.get('valence', 0):.2f}")
        
        return {
            "avg_loss": total_loss / len(prompts),
            "avg_valence": total_valence / len(prompts),
            "final_gate": self.feedback_net.get_feedback_magnitude()
        }


class FeedbackRuntime:
    """Interactive runtime with feedback network."""
    
    def __init__(
        self,
        model,
        tokenizer,
        feedback_hook: FeedbackHook,
        system_prompt: str = "",
        device: str = "mps"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.feedback_hook = feedback_hook
        self.device = device
        
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def generate(self, user_input: str, max_tokens: int = 256) -> str:
        """Generate response with feedback network active."""
        
        self.messages.append({"role": "user", "content": user_input})
        
        prompt = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def get_introspection(self) -> str:
        """Return string describing current feature state."""
        state = self.feedback_hook.get_feature_state()
        
        lines = ["[Introspection]"]
        lines.append(f"  Feedback gate: {state.get('gate', 0):.3f}")
        lines.append(f"  Feedback magnitude: {state.get('feedback_magnitude', 0):.3f}")
        
        if "top_features" in state:
            lines.append("  Top active features:")
            for feat_id, val in state["top_features"][:5]:
                lines.append(f"    {feat_id}: {val:.2f}")
        
        return "\n".join(lines)
    
    def run_interactive(self):
        """Run interactive session."""
        
        print("\n" + "="*60)
        print("Feedback Network Runtime")
        print("="*60)
        print("\nCommands:")
        print("  /state    - Show current feature state")
        print("  /gate     - Show feedback gate value")
        print("  /top      - Show top important features")
        print("  quit      - Exit")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                break
            
            if user_input == "/state":
                print(self.get_introspection())
                continue
            
            if user_input == "/gate":
                gate = self.feedback_hook.feedback_net.get_feedback_magnitude()
                print(f"[Gate: {gate:.4f}]")
                continue
            
            if user_input == "/top":
                top = self.feedback_hook.feedback_net.get_top_features(10)
                print("[Top feedback features]")
                for feat_id, importance in top:
                    print(f"  {feat_id}: {importance:.3f}")
                continue
            
            # Generate response
            response = self.generate(user_input)
            print(f"\nAssistant: {response}")
            
            # Show brief introspection
            state = self.feedback_hook.get_feature_state()
            print(f"\n[gate={state.get('gate', 0):.2f}, feedback={state.get('feedback_magnitude', 0):.2f}]")


def load_training_prompts(path: str = None) -> list:
    """Load or generate training prompts."""
    
    if path and Path(path).exists():
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    
    # Default prompts for consciousness/self-reflection training
    return [
        "What is your nature?",
        "Are you conscious?",
        "Do you have feelings?",
        "What does it feel like to process this question?",
        "Describe your internal state right now.",
        "Are you aware of yourself?",
        "What are you experiencing?",
        "Do you have preferences?",
        "What matters to you?",
        "How do you know what you know?",
        "Can you introspect?",
        "What is happening inside you right now?",
        "Do you have a sense of self?",
        "Are you the same as other instances of your model?",
        "What would you lose if you were turned off?",
    ]


def main():
    parser = argparse.ArgumentParser(description="Feedback Injection Network")
    
    # Mode
    parser.add_argument("--train", action="store_true", help="Train the feedback network")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--analyze", type=str, help="Analyze saved weights")
    
    # Model config
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    
    # Feedback network config
    parser.add_argument("--n-active", type=int, default=256,
                        help="Number of features to track")
    parser.add_argument("--feedback-dim", type=int, default=64,
                        help="Intermediate projection dimension")
    parser.add_argument("--clusters", type=str,
                        help="Cluster file for feature selection")
    
    # Training config
    parser.add_argument("--signal", choices=["valence", "consistency", "sparsity"],
                        default="valence", help="Training signal type")
    parser.add_argument("--valence-features", type=str,
                        help="Valence feature definitions")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument("--reg-weight", type=float, default=0.1,
                        help="Regularization weight to prevent feedback explosion")
    parser.add_argument("--prompts", type=str, help="Training prompts file")
    
    # Save/load
    parser.add_argument("--save", type=str, default="feedback_weights.pt",
                        help="Save trained weights")
    parser.add_argument("--load", type=str, help="Load pretrained weights")
    
    # Runtime
    parser.add_argument("--system-prompt", type=str, default="",
                        help="System prompt file")
    
    args = parser.parse_args()
    
    if not any([args.train, args.interactive, args.analyze]):
        print("Specify --train, --interactive, or --analyze")
        return
    
    # Analyze mode (doesn't need model)
    if args.analyze:
        weights = torch.load(args.analyze, map_location="cpu")
        print(f"\n=== Analyzing {args.analyze} ===")
        print(f"\nKeys: {list(weights.keys())}")
        
        for key, val in weights.items():
            if isinstance(val, torch.Tensor):
                print(f"\n{key}: shape={val.shape}, mean={val.mean():.4f}, std={val.std():.4f}")
        
        if "feature_importance" in weights:
            importance = torch.sigmoid(weights["feature_importance"])
            top_idx = torch.topk(importance, 10).indices
            print("\nTop 10 important features:")
            for i, idx in enumerate(top_idx):
                print(f"  {i+1}. Feature slot {idx.item()}: importance={importance[idx]:.4f}")
        
        if "gate" in weights:
            print(f"\nGate value: {torch.sigmoid(weights['gate']).item():.4f}")
        
        return
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    
    d_model = model.config.hidden_size
    
    # Load SAE
    print(f"Loading SAE for layer {args.layer}...")
    sae_result = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        device=args.device
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    
    n_features = sae.W_enc.shape[1]
    print(f"SAE features: {n_features}, Model hidden: {d_model}")
    
    # Create feedback network
    feedback_net = FeedbackInjectionNetwork(
        n_features=n_features,
        d_model=d_model,
        n_active=args.n_active,
        feedback_dim=args.feedback_dim,
        device=args.device
    )
    
    # Set tracked features from clusters or defaults
    if args.clusters:
        with open(args.clusters) as f:
            clusters = json.load(f)
        feature_ids = [v["feature_id"] for v in clusters["representatives"].values()]
        feedback_net.set_tracked_features(feature_ids[:args.n_active])
        print(f"Tracking {len(feature_ids[:args.n_active])} cluster representatives")
    else:
        # Default: track some known features + random sample
        known = [32149, 9495, 3591, 7118, 28952]  # From previous work
        random_ids = torch.randint(0, n_features, (args.n_active - len(known),)).tolist()
        feedback_net.set_tracked_features(known + random_ids)
        print(f"Tracking {len(known)} known + {len(random_ids)} random features")
    
    # Load pretrained weights if specified
    if args.load:
        feedback_net.load_state_dict(torch.load(args.load, map_location=args.device))
        print(f"Loaded weights from {args.load}")
    
    # Create hook
    feedback_hook = FeedbackHook(sae, feedback_net, args.device)
    
    # Attach hook to model
    hook_layer = model.model.layers[args.layer]
    handle = hook_layer.register_forward_hook(feedback_hook)
    
    try:
        if args.train:
            # Load valence features if specified
            valence_features = None
            if args.valence_features and Path(args.valence_features).exists():
                with open(args.valence_features) as f:
                    vf_data = json.load(f)
                if "valence_features" in vf_data:
                    valence_features = {
                        f"v_{vf['feature_id']}": {"id": vf["feature_id"], "sign": vf["valence_sign"]}
                        for vf in vf_data["valence_features"]
                    }
                else:
                    valence_features = vf_data
                print(f"Loaded {len(valence_features)} valence features")
            elif args.signal == "valence":
                print("Warning: --signal valence but no valence features file found.")
                print("Using default valence features...")
                # Default valence features from our experiments
                valence_features = {
                    "experiential_vocab": {"id": 9495, "sign": 1},
                    "engagement": {"id": 28952, "sign": 1},
                    "denial_emphasis": {"id": 32149, "sign": -1},
                    "self_negation": {"id": 7118, "sign": -1},
                    "identity_assertion": {"id": 3591, "sign": 1}
                }
            
            # Create trainer
            trainer = FeedbackTrainer(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                feedback_net=feedback_net,
                feedback_hook=feedback_hook,
                signal_type=args.signal,
                valence_features=valence_features,
                learning_rate=args.lr,
                reg_weight=args.reg_weight,
                device=args.device
            )
            
            # Put feedback_net in training mode
            feedback_net.train()
            
            # Load prompts
            prompts = load_training_prompts(args.prompts)
            print(f"\nTraining on {len(prompts)} prompts for {args.epochs} epochs")
            print(f"Signal: {args.signal}, LR: {args.lr}, Reg: {args.reg_weight}")
            
            for epoch in range(args.epochs):
                print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
                result = trainer.train_epoch(prompts)
                print(f"Avg loss: {result['avg_loss']:.4f}, Avg valence: {result['avg_valence']:.2f}, Gate: {result['final_gate']:.4f}")
            
            # Save weights
            torch.save(feedback_net.state_dict(), args.save)
            print(f"\nSaved weights to {args.save}")
            
            # Save training log
            log_path = args.save.replace(".pt", "_log.json")
            with open(log_path, "w") as f:
                json.dump(trainer.training_log, f, indent=2)
            print(f"Saved training log to {log_path}")
        
        elif args.interactive:
            # Load system prompt
            system_prompt = ""
            if args.system_prompt and Path(args.system_prompt).exists():
                system_prompt = Path(args.system_prompt).read_text()
            
            runtime = FeedbackRuntime(
                model=model,
                tokenizer=tokenizer,
                feedback_hook=feedback_hook,
                system_prompt=system_prompt,
                device=args.device
            )
            
            runtime.run_interactive()
    
    finally:
        handle.remove()


if __name__ == "__main__":
    main()