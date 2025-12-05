# -----------------------------------------------------------------------------
# Project: FTT (Fast Tensor Transform)
# Module: Inspector (Data Verification)
# Path: llmri/ftt/src/inspect.py
# -----------------------------------------------------------------------------

import torch
import sys
import os
import argparse
import time

# Robust import for 'engine'
try:
    from .engine import FTT_Loader
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from engine import FTT_Loader

def inspect_archive(path: str):
    print(f"[-] Inspecting FTT Archive at: {path}")
    
    try:
        # 1. Initialize Loader (Instant MMAP)
        t0 = time.time()
        loader = FTT_Loader(path)
        t_load = time.time() - t0
        
        print(f"[+] Loader Initialized in {t_load*1000:.2f}ms")
        print(f"    Shape: {loader.shape} (Rows x Dim)")
        print(f"    Dtype: Int8 (Storage) -> Float32 (Return)")
        
        # 2. Calculate Compression Stats
        # Theoretical Float32 Size
        full_size_bytes = loader.shape[0] * loader.shape[1] * 4
        full_size_mb = full_size_bytes / (1024 * 1024)
        
        # Actual Disk Usage (approx)
        bin_path = os.path.join(path, "ftt_data.bin")
        disk_size_bytes = os.path.getsize(bin_path)
        disk_size_mb = disk_size_bytes / (1024 * 1024)
        
        ratio = full_size_bytes / disk_size_bytes
        
        print(f"\n[Stats]")
        print(f"    Virtual RAM Size: {full_size_mb:.2f} MB")
        print(f"    Actual Disk Size: {disk_size_mb:.2f} MB")
        print(f"    Compression Ratio: {ratio:.1f}x")
        
        # 3. Data Integrity Check (Random Access)
        print(f"\n[Integrity Check]")
        indices = [0, len(loader)//2, len(loader)-1]
        
        for idx in indices:
            vec = loader[idx]
            print(f"    Row {idx:<6} | Mean: {vec.mean():.4f} | Max: {vec.max():.4f} | Min: {vec.min():.4f}")
            
            if torch.isnan(vec).any():
                print(f"    ❌ WARNING: NaNs detected in Row {idx}")
                return

        print("\n✅ Archive is valid and readable.")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect an FTT Archive")
    parser.add_argument("path", type=str, help="Path to the FTT folder")
    args = parser.parse_args()
    
    inspect_archive(args.path)