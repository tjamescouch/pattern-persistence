# -----------------------------------------------------------------------------
# Project: FTT (Fast Tensor Transform)
# Module: Distribution Diagnostic (The "Bin Checker")
# Path: llmri/ftt/src/diagnose.py
# -----------------------------------------------------------------------------

import torch
import sys
import os
import argparse
import numpy as np

# Robust import for 'engine'
try:
    from .engine import FTT_Loader
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from engine import FTT_Loader

def print_ascii_histogram(data, bins=20, range_min=-128, range_max=127):
    counts, bin_edges = np.histogram(data.numpy(), bins=bins, range=(range_min, range_max))
    max_count = counts.max()
    
    print(f"\n    Distribution (Int8 values):")
    for count, edge in zip(counts, bin_edges):
        bar_len = int(40 * count / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"    [{int(edge):4d}]: {bar} ({count})")

def diagnose_distribution(ftt_path: str, samples: int = 1000):
    print(f"[-] Diagnosing Distribution in: {ftt_path}")
    
    loader = FTT_Loader(ftt_path)
    print(f"    Total Rows: {len(loader)}")
    
    # 1. Random Sampling
    indices = torch.randint(0, len(loader), (samples,))
    
    # We want to look at the RAW Int8 values before scaling to see bin usage
    # We access the mmap directly to bypass the engine's float conversion
    raw_mmap = loader.mmap
    
    # Collect sample data
    # Note: numpy memmap indexing returns new arrays, so this is safe
    batch_raw = raw_mmap[indices.numpy()] 
    batch_tensor = torch.from_numpy(batch_raw).float() # Convert to float just for stats calculation
    
    # 2. Key Metrics
    
    # A. Utilization (How many distinct int8 values are used?)
    unique_vals = torch.unique(batch_tensor).numel()
    utilization_pct = (unique_vals / 256) * 100
    
    # B. Centering (Is the mean near 0?)
    mean_val = batch_tensor.mean().item()
    
    # C. Shift (Is min > 0?)
    min_val = batch_tensor.min().item()
    max_val = batch_tensor.max().item()
    
    # D. Saturation (Are we hitting the rails -127 or 127?)
    clipped_low = (batch_tensor == -127).sum().item()
    clipped_high = (batch_tensor == 127).sum().item()
    clipped_pct = ((clipped_low + clipped_high) / batch_tensor.numel()) * 100
    
    print(f"\n[Analysis Report]")
    print(f"-------------------")
    print(f"Range Used:     [{min_val:.0f} to {max_val:.0f}]")
    print(f"Mean Value:     {mean_val:.2f} (Target: ~0)")
    print(f"Unique Values:  {unique_vals}/256 ({utilization_pct:.1f}%)")
    print(f"Saturation:     {clipped_pct:.2f}% (Values at -127/127)")
    
    # 3. Visual Check
    print_ascii_histogram(batch_tensor.view(-1))
    
    # 4. Verdict
    print(f"\n[Verdict]")
    if min_val > 0:
        print("⚠️  FAIL: POSITIVE SHIFT DETECTED.")
        print("    Data is strictly positive. Symmetric quantization is wasting 50% of bits.")
        print("    Recommendation: Switch to Asymmetric Quantization.")
    elif utilization_pct < 10:
        print("⚠️  FAIL: LOW BIT DEPTH.")
        print("    Less than 10% of int8 range is used. Resolution loss likely.")
        print("    Recommendation: Check scaling logic or data variance.")
    elif mean_val > 30 or mean_val < -30:
        print("⚠️  WARNING: POOR CENTERING.")
        print("    Distribution is heavily skewed.")
    else:
        print("✅ PASS: SYMMETRIC QUANTIZATION SUITABLE.")
        print("    Data is roughly centered and uses the int8 range well.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to FTT archive")
    args = parser.parse_args()
    
    diagnose_distribution(args.path)