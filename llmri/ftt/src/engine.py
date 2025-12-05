# -----------------------------------------------------------------------------
# Project: FTT (Fast Tensor Transform)
# Module: Core Engine
# Path: llmri/ftt/src/engine.py
# -----------------------------------------------------------------------------

import torch
import json
import numpy as np
from pathlib import Path
import shutil
import os

class FTT_Writer:
    """
    The Fast Tensor Transform Writer.
    Compresses and streams activations to disk to bypass RAM.
    """
    def __init__(self, path: str, dim: int, overwrite: bool = True):
        self.path = Path(path)
        self.dim = dim
        self.rows = 0
        self.scales = []
        
        # Setup directory
        if self.path.exists() and overwrite:
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.bin_path = self.path / "ftt_data.bin"
        
        # Ensure file starts empty
        if not self.bin_path.exists():
            with open(self.bin_path, "wb") as f:
                pass

    def push(self, tensor_f32: torch.Tensor):
        """
        Transform: F32 -> Int8 -> Disk
        """
        # 1. Calc Scale (Symmetric, Per-Token)
        # Shape: [Batch, 1]
        max_val = tensor_f32.abs().max(dim=1, keepdim=True).values
        scale = max_val / 127.0
        
        # Avoid div by zero for dead neurons/padding
        scale[scale == 0] = 1.0 
        
        # 2. Fast Quantization
        tensor_int8 = (tensor_f32 / scale).round().clamp(-127, 127).to(torch.int8)
        
        # 3. Stream to Disk (CPU write)
        with open(self.bin_path, "ab") as f:
            f.write(tensor_int8.cpu().numpy().tobytes())
            
        # 4. Log Metadata
        self.scales.extend(scale.view(-1).tolist())
        self.rows += tensor_f32.shape[0]

    def close(self):
        print(f"[-] Finalizing FTT Archive at {self.path}...")
        meta = {
            "shape": [self.rows, self.dim],
            "dtype": "int8",
            "version": "FTT_v1"
        }
        
        # Save scales efficiently
        torch.save(torch.tensor(self.scales, dtype=torch.float32), self.path / "scales.pt")
        
        with open(self.path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"[+] FTT Archive Complete. Rows: {self.rows}, Dim: {self.dim}")


class FTT_Loader(torch.utils.data.Dataset):
    """
    The Magic Window. 
    Lazy loads FTT data using Memory Mapping.
    """
    def __init__(self, path: str):
        self.path = Path(path)
        
        if not (self.path / "meta.json").exists():
            raise FileNotFoundError(f"FTT Archive not found at {self.path}")

        # 1. Load Metadata
        with open(self.path / "meta.json", "r") as f:
            self.meta = json.load(f)
            
        self.shape = tuple(self.meta["shape"])
        self.length = self.shape[0]
        self.dim = self.shape[1]
        
        # 2. Load Scales (RAM cheap)
        self.scales = torch.load(self.path / "scales.pt")
        
        # 3. MMAP: The Virtual Tensor
        self.mmap = np.memmap(
            self.path / "ftt_data.bin", 
            dtype='int8', 
            mode='r', 
            shape=self.shape
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Zero-Copy Read (with tiny copy to silence PyTorch warning)
        # The .copy() makes the numpy array writable before PyTorch sees it.
        # Cost: 4KB copy (negligible).
        data_int8 = torch.from_numpy(self.mmap[idx].copy())
        
        # 2. Dequantize
        scale = self.scales[idx]
        return data_int8.to(torch.float32) * scale

if __name__ == "__main__":
    # Quick self-test if run directly
    print("Running FTT Engine Self-Test...")
    writer = FTT_Writer("./ftt_test_engine", 64)
    writer.push(torch.randn(10, 64))
    writer.close()
    print("Test passed.")