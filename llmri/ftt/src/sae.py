# -----------------------------------------------------------------------------
# Project: FTT (Fast Tensor Transform)
# Module: SAE Examiner (Diagnostics)
# Path: llmri/ftt/src/examine.py
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import argparse
import json
import matplotlib.pyplot as plt # Optional, for future plotting
import numpy as np

# Robust imports
try:
    from .engine import FTT_Loader
    from .sae import SparseAutoencoder
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from engine import FTT_Loader
    from sae import SparseAutoencoder

def examine_sae(sae_path: str, ftt_path: str, batch_size: int = 4096):
    print(f"[-] Examining SAE at: {sae_path}")
    
    # 1. Load Config
    config_path = os.path.join(sae_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    input_dim = config["input_dim"]
    expansion = config["expansion"]
    hidden_dim = input_dim * expansion
    
    print(f"    Arch: {input_dim} -> {hidden_dim} (x{expansion})")
    
    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SparseAutoencoder(input_dim, expansion).to(device)
    
    weights_path = os.path.join(sae_path, "sae_weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. Load Data
    dataset = FTT_Loader(ftt_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    # 4. Run Diagnostics
    print(f"\n[>] Running Inference on {len(dataset)} vectors...")
    
    total_l0 = 0
    total_loss = 0
    feature_activity = torch.zeros(hidden_dim, device=device)
    
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            x = x.to(device)
            
            recon, latents = model(x)
            
            # Metrics
            l0 = (latents > 0).float().sum(dim=1).mean()
            loss = nn.functional.mse_loss(recon, x)
            
            # Track activity (for dead neuron detection)
            # We count how many times each feature fired in this batch
            batch_activity = (latents > 0).float().sum(dim=0)
            feature_activity += batch_activity
            
            total_l0 += l0.item()
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"    Batch {i}/{len(dataloader)} | L0: {l0:.1f} | Loss: {loss:.5f}")

    # 5. Summary Stats
    avg_l0 = total_l0 / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    
    # Dead Neurons: Features that never fired once
    dead_neurons = (feature_activity == 0).sum().item()
    dead_pct = (dead_neurons / hidden_dim) * 100
    
    print(f"\n[Diagnostic Report]")
    print(f"-------------------")
    print(f"MSE Loss:      {avg_loss:.5f}")
    print(f"Average L0:    {avg_l0:.1f} (Target: 20-100)")
    print(f"Sparsity:      {(1 - avg_l0/hidden_dim)*100:.2f}%")
    print(f"Dead Neurons:  {dead_neurons} ({dead_pct:.2f}%)")
    
    # Interpretation
    if dead_pct > 50:
        print("\n⚠️  CRITICAL: >50% Dead Neurons. Training likely collapsed or L1 too high.")
    elif avg_l0 > 500:
        print("\n⚠️  WARNING: L0 is very high. L1 coefficient might be too low.")
    else:
        print("\n✅ SAE Health: GOOD. Ready for feature interpretation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae", type=str, required=True, help="Path to SAE folder")
    parser.add_argument("--ftt", type=str, required=True, help="Path to FTT data")
    args = parser.parse_args()
    
    examine_sae(args.sae, args.ftt)