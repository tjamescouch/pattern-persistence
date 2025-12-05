# -----------------------------------------------------------------------------
# Project: FTT (Fast Tensor Transform)
# Module: Converter (The Surgical Mock Edition)
# Path: llmri/ftt/src/convert.py
# -----------------------------------------------------------------------------

import torch
from pathlib import Path
import sys
import os
import glob
import argparse
from types import ModuleType

# Robust import for 'engine'
try:
    from .engine import FTT_Writer
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from engine import FTT_Writer

def mock_missing_dependencies():
    """
    Surgically injects a fake 'llmri.trace' module to satisfy pickle.
    """
    print("[-] Activating Skeleton Key for 'llmri.trace'...")

    # 1. Ensure 'llmri' exists in sys.modules
    if 'llmri' not in sys.modules:
        sys.modules['llmri'] = ModuleType('llmri')
    
    parent_module = sys.modules['llmri']

    # 2. Create the fake 'trace' module
    if 'llmri.trace' not in sys.modules:
        fake_trace = ModuleType('llmri.trace')
        sys.modules['llmri.trace'] = fake_trace
    else:
        fake_trace = sys.modules['llmri.trace']

    # 3. CRITICAL: Graft the child onto the parent
    # This ensures "import llmri; llmri.trace" works
    if not hasattr(parent_module, 'trace'):
        setattr(parent_module, 'trace', fake_trace)

    # 4. Create the Stub Class
    class GenericStub:
        def __init__(self, *args, **kwargs):
            # Pickle typically bypasses __init__, but we define it just in case
            pass
        def __repr__(self):
            return f"<GenericStub keys={list(self.__dict__.keys())}>"
            
    # 5. Inject the classes into the fake module
    targets = ['ActivationTrace', 'ActivationSlice']
    
    for target in targets:
        if not hasattr(fake_trace, target):
            setattr(fake_trace, target, GenericStub)
        
    print(f"[+] Skeleton Key Active. Mocked: {', '.join(targets)}")

def _smart_extract_tensor(obj, depth=0):
    """
    Recursively searches for the LARGEST tensor in an object/dict/list.
    Returns (Tensor, info_string) or (None, info_string)
    """
    if depth > 2: # Prevent infinite recursion
        return None
        
    candidates = []
    
    # 1. Is it a tensor?
    if isinstance(obj, torch.Tensor):
        return obj

    # 2. Is it a list/tuple?
    if isinstance(obj, (list, tuple)):
        # Check if it's a list of tensors (stack them)
        if len(obj) > 0 and isinstance(obj[0], torch.Tensor):
            try:
                return torch.stack(list(obj))
            except:
                pass
        # Otherwise recurse items
        for item in obj:
            t = _smart_extract_tensor(item, depth + 1)
            if t is not None:
                candidates.append(t)

    # 3. Is it a dictionary?
    elif isinstance(obj, dict):
        for v in obj.values():
            t = _smart_extract_tensor(v, depth + 1)
            if t is not None:
                candidates.append(t)

    # 4. Is it an Object (Stub)?
    elif hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            t = _smart_extract_tensor(v, depth + 1)
            if t is not None:
                candidates.append(t)

    # Selection Heuristic: Return the largest tensor found
    if candidates:
        # Filter out Nones just in case
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return None
        # Sort by element count (numel) descending
        candidates.sort(key=lambda x: x.numel(), reverse=True)
        return candidates[0]
        
    return None

def ingest_existing_data(source_pattern: str, output_dir: str):
    # Activate the Skeleton Key immediately
    mock_missing_dependencies()
    
    source_pattern = os.path.expanduser(source_pattern)
    files = sorted(glob.glob(source_pattern))
    
    if not files:
        print(f"[!] No files found matching: {source_pattern}")
        return

    print(f"[-] Found {len(files)} files. Converting...")
    
    writer = None
    total_vectors = 0
    
    for fpath in files:
        print(f"    Processing {os.path.basename(fpath)}... ", end="")
        try:
            # Load with security checks disabled to allow our Mock class
            loaded_obj = torch.load(fpath, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"FAILED (Load Error): {e}")
            continue
            
        tensor = None
        
        # --- Strategy A: Check 'slices' attribute specifically ---
        slices = None
        if hasattr(loaded_obj, 'slices'):
            slices = loaded_obj.slices
        elif isinstance(loaded_obj, dict) and 'slices' in loaded_obj:
            slices = loaded_obj['slices']
            
        if slices is not None:
            # Normalize to list
            if isinstance(slices, dict): 
                slices = list(slices.values())
            
            if isinstance(slices, (list, tuple)) and len(slices) > 0:
                # print(f"[Scanning {len(slices)} slices] ", end="")
                collected = []
                for i, s in enumerate(slices):
                    t = _smart_extract_tensor(s)
                    if t is not None:
                        collected.append(t)
                    else:
                        # Debug info for the first failed slice
                        if i == 0:
                            print(f"[Slice 0 Content: {s}] ", end="")
                            
                if collected:
                    try:
                        tensor = torch.cat(collected, dim=0)
                    except Exception as e:
                        print(f"[Merge Error: {e}] ", end="")

        # --- Strategy B: Fallback to scanning the whole object ---
        if tensor is None:
            tensor = _smart_extract_tensor(loaded_obj)

        # --- Final Check ---
        if tensor is None:
            print(f"SKIPPED (Could not find tensor)")
            if hasattr(loaded_obj, '__dict__'):
                print(f"    Object keys: {list(loaded_obj.__dict__.keys())}")
            continue

        # --- Flattening & Writing ---
        if tensor.dim() == 3:
            tensor = tensor.view(-1, tensor.shape[-1])
            
        if writer is None:
            dim = tensor.shape[-1]
            writer = FTT_Writer(output_dir, dim)
            print(f"[Init dim={dim}] ", end="")

        writer.push(tensor)
        rows = tensor.shape[0]
        total_vectors += rows
        print(f"Done (+{rows} rows)")
        
        del loaded_obj
        del tensor

    if writer:
        writer.close()
        print(f"Total Vectors Processed: {total_vectors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt files to FTT")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    args = parser.parse_args()
    
    ingest_existing_data(args.source, args.dest)