# llmri/core.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils.hf_adapter import (
    infer_model_structure,
    HFModelStructure,
)

ComponentType = Literal[
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "mlp_in",
    "mlp_out",
]


@dataclass
class LayerComponentInfo:
    layer_index: int
    component: ComponentType
    weight_shape: Tuple[int, ...]
    name: str  # module path in the HF model


@dataclass
class LayerInfo:
    index: int
    components: List[LayerComponentInfo]


@dataclass
class ModelInfo:
    model_id: str
    n_params: int
    n_layers: int
    d_model: int
    vocab_size: int
    layer_infos: List[LayerInfo]


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    info: ModelInfo


def load_model(model_id: str, device: str = "cuda") -> LoadedModel:
    """
    Load a causal LM and tokenizer from Hugging Face and derive a ModelInfo.

    device:
      - "cpu"
      - "cuda" (single-GPU; we use device_map='auto')
      - "mps"  (Apple Silicon GPU)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dev = torch.device(device)
    is_gpu = dev.type in ("cuda", "mps")
    dtype = torch.float16 if is_gpu else torch.float32

    if dev.type == "cuda":
        # Let HF/accelerate shard across available CUDA devices.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        # CPU or MPS: load then move.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        model.to(dev)

    struct: HFModelStructure = infer_model_structure(model)

    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size
    d_model = model.config.hidden_size
    n_layers = struct.num_layers

    layer_infos: List[LayerInfo] = []
    for layer_idx, layer_desc in enumerate(struct.layers):
        components: List[LayerComponentInfo] = []

        if layer_desc.attn_q_name is not None:
            q_weight = _get_param_shape(model, layer_desc.attn_q_name)
            components.append(
                LayerComponentInfo(
                    layer_index=layer_idx,
                    component="attention_q",
                    weight_shape=q_weight,
                    name=layer_desc.attn_q_name,
                )
            )

        if layer_desc.attn_k_name is not None:
            k_weight = _get_param_shape(model, layer_desc.attn_k_name)
            components.append(
                LayerComponentInfo(
                    layer_index=layer_idx,
                    component="attention_k",
                    weight_shape=k_weight,
                    name=layer_desc.attn_k_name,
                )
            )

        if layer_desc.attn_v_name is not None:
            v_weight = _get_param_shape(model, layer_desc.attn_v_name)
            components.append(
                LayerComponentInfo(
                    layer_index=layer_idx,
                    component="attention_v",
                    weight_shape=v_weight,
                    name=layer_desc.attn_v_name,
                )
            )

        if layer_desc.attn_o_name is not None:
            o_weight = _get_param_shape(model, layer_desc.attn_o_name)
            components.append(
                LayerComponentInfo(
                    layer_index=layer_idx,
                    component="attention_o",
                    weight_shape=o_weight,
                    name=layer_desc.attn_o_name,
                )
            )

        if layer_desc.mlp_in_name is not None:
            m_in_weight = _get_param_shape(model, layer_desc.mlp_in_name)
            components.append(
                LayerComponentInfo(
                    layer_index=layer_idx,
                    component="mlp_in",
                    weight_shape=m_in_weight,
                    name=layer_desc.mlp_in_name,
                )
            )

        if layer_desc.mlp_out_name is not None:
            m_out_weight = _get_param_shape(model, layer_desc.mlp_out_name)
            components.append(
                LayerComponentInfo(
                    layer_index=layer_idx,
                    component="mlp_out",
                    weight_shape=m_out_weight,
                    name=layer_desc.mlp_out_name,
                )
            )

        layer_infos.append(LayerInfo(index=layer_idx, components=components))

    info = ModelInfo(
        model_id=model_id,
        n_params=n_params,
        n_layers=n_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        layer_infos=layer_infos,
    )

    return LoadedModel(model=model, tokenizer=tokenizer, info=info)


def _get_param_shape(model: torch.nn.Module, param_name: str) -> Tuple[int, ...]:
    """
    Get the shape of a parameter given its dotted name
    (e.g. 'model.layers.0.mlp.up_proj.weight').
    """
    module = model
    parts = param_name.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    param = getattr(module, parts[-1])
    return tuple(param.shape)