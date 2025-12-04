# llmri/utils/io.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def save_activation_trace(trace, path: str | Path) -> None:
    """
    Save an ActivationTrace to disk using torch.save.

    For Phase 1 we serialize the whole dataclass directly.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, p)


def load_activation_trace(path: str | Path):
    """
    Load an ActivationTrace from disk.

    On PyTorch >= 2.6, torch.load defaults to weights_only=True and uses a
    restricted unpickler. We explicitly allowlist llmri's classes using
    torch.serialization.safe_globals.

    Assumes the file was created by llmri itself (trusted).
    """
    p = Path(path)

    # Prefer the new safe_globals mechanism when available.
    try:
        from torch.serialization import safe_globals  # type: ignore[attr-defined]
        # Local import to avoid circulars at module import time.
        from ..trace import ActivationTrace, ActivationSlice  # type: ignore[import]

        with safe_globals([ActivationTrace, ActivationSlice]):
            trace = torch.load(p)
        return trace

    except ImportError:
        # Older PyTorch: no safe_globals. Use weights_only=False explicitly.
        trace = torch.load(p, weights_only=False)  # type: ignore[call-arg]
        return trace


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
