# llmri/utils/hf_adapter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch.nn as nn


@dataclass
class HFBlockDesc:
    layer_index: int
    attn_q_name: Optional[str]
    attn_k_name: Optional[str]
    attn_v_name: Optional[str]
    attn_o_name: Optional[str]
    mlp_in_name: Optional[str]
    mlp_out_name: Optional[str]


@dataclass
class HFModelStructure:
    num_layers: int
    layers: List[HFBlockDesc]


def infer_model_structure(model: nn.Module) -> HFModelStructure:
    """
    Infer a simple description of transformer blocks for a GPT/LLaMA-like HF causal LM.

    We currently support two broad families:

    - LLaMA-style:   model.layers (inside a LlamaForCausalLM with attribute 'model')
    - GPT-style:     model.transformer.h

    If we can't match either, we fail fast.
    """

    # LLaMA-style (LlamaForCausalLM from HF):
    #
    #   model: LlamaForCausalLM
    #   model.model: LlamaModel
    #   model.model.layers: ModuleList of blocks
    #
    # BUT the parameter names use "model.layers.*", not "model.model.layers.*".
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = list(model.model.layers)
        # This prefix must match the actual parameter names as seen in
        # dict(model.named_parameters()).keys()
        prefix = "model.layers"
        return _describe_llama_like(model, blocks, prefix)

    # GPT-style (GPT-2 / GPT-J etc.):
    #
    #   model: GPT2LMHeadModel / similar
    #   model.transformer.h: ModuleList of blocks
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = list(model.transformer.h)
        prefix = "transformer.h"
        return _describe_gpt_like(model, blocks, prefix)

    raise RuntimeError(
        "Unsupported HF model architecture for llmri.hf_adapter: "
        "expected LLaMA-style (model.model.layers) or GPT-style (transformer.h)."
    )


def _describe_llama_like(model: nn.Module, blocks: List[nn.Module], prefix: str) -> HFModelStructure:
    """
    LLaMA/Mistral style:
      - each block has .self_attn.(q_proj,k_proj,v_proj,o_proj)
      - MLP is typically .mlp.(gate_proj, up_proj, down_proj)

    We expose:
      - attn_q: q_proj.weight
      - attn_k: k_proj.weight
      - attn_v: v_proj.weight
      - attn_o: o_proj.weight
      - mlp_in: up_proj.weight (or gate_proj.weight as fallback)
      - mlp_out: down_proj.weight
    """
    layers_desc: List[HFBlockDesc] = []

    for i, _block in enumerate(blocks):
        block_prefix = f"{prefix}.{i}"

        attn_q_name = _maybe_param_name(model, f"{block_prefix}.self_attn.q_proj.weight")
        attn_k_name = _maybe_param_name(model, f"{block_prefix}.self_attn.k_proj.weight")
        attn_v_name = _maybe_param_name(model, f"{block_prefix}.self_attn.v_proj.weight")
        attn_o_name = _maybe_param_name(model, f"{block_prefix}.self_attn.o_proj.weight")

        # For MLP we choose a representative "in" and "out" weight.
        mlp_in_name = (
            _maybe_param_name(model, f"{block_prefix}.mlp.up_proj.weight")
            or _maybe_param_name(model, f"{block_prefix}.mlp.gate_proj.weight")
        )
        mlp_out_name = _maybe_param_name(model, f"{block_prefix}.mlp.down_proj.weight")

        layers_desc.append(
            HFBlockDesc(
                layer_index=i,
                attn_q_name=attn_q_name,
                attn_k_name=attn_k_name,
                attn_v_name=attn_v_name,
                attn_o_name=attn_o_name,
                mlp_in_name=mlp_in_name,
                mlp_out_name=mlp_out_name,
            )
        )

    return HFModelStructure(num_layers=len(layers_desc), layers=layers_desc)


def _describe_gpt_like(model: nn.Module, blocks: List[nn.Module], prefix: str) -> HFModelStructure:
    """
    GPT-2 / GPT-J style:
      - attn.c_attn.weight (combined qkv) and attn.c_proj.weight
      - mlp.c_fc.weight, mlp.c_proj.weight

    We map:
      - q,k,v all to c_attn.weight for now (Phase 1 doesn't need them separated)
      - attn_o to c_proj.weight
      - mlp_in to c_fc.weight
      - mlp_out to c_proj.weight
    """
    layers_desc: List[HFBlockDesc] = []

    for i, _block in enumerate(blocks):
        block_prefix = f"{prefix}.{i}"

        c_attn = _maybe_param_name(model, f"{block_prefix}.attn.c_attn.weight")
        c_proj = _maybe_param_name(model, f"{block_prefix}.attn.c_proj.weight")
        c_fc = _maybe_param_name(model, f"{block_prefix}.mlp.c_fc.weight")
        c_mlp_proj = _maybe_param_name(model, f"{block_prefix}.mlp.c_proj.weight")

        layers_desc.append(
            HFBlockDesc(
                layer_index=i,
                attn_q_name=c_attn,
                attn_k_name=c_attn,
                attn_v_name=c_attn,
                attn_o_name=c_proj,
                mlp_in_name=c_fc,
                mlp_out_name=c_mlp_proj,
            )
        )

    return HFModelStructure(num_layers=len(layers_desc), layers=layers_desc)


def _maybe_param_name(model: nn.Module, name: str) -> Optional[str]:
    """
    Return name if it exists as an attribute chain, otherwise None.

    We walk the dotted path except for the final part, which is assumed
    to be a parameter/buffer name.
    """
    module: nn.Module = model
    parts = name.split(".")

    try:
        for p in parts[:-1]:
            module = getattr(module, p)
        getattr(module, parts[-1])
        return name
    except AttributeError:
        return None
