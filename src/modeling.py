"""
modeling.py — Backbone Family Adapters for Attention-Only Replacement
=====================================================================

This module contains adapter/wrapper code to replace attention modules
inside existing Hugging Face backbones while keeping the rest intact.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.longattention_layer import LongAttentionCore


class LongAttentionBartSelfAttentionAdapter(nn.Module):
    """Replace BART/MBART encoder self-attention with LongAttentionCore."""

    def __init__(
        self,
        original_attn: nn.Module,
        core: LongAttentionCore,
        *,
        layer_index: int,
        total_layers: int,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.scaling = original_attn.scaling
        self.dropout = getattr(original_attn, "dropout", 0.0)
        self.is_decoder = getattr(original_attn, "is_decoder", False)

        # Reuse original projection layers for fair comparison.
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.out_proj = original_attn.out_proj

        self.core = core
        self.layer_index = layer_index
        self.total_layers = max(total_layers, 1)
        self.last_info: Optional[Dict[str, torch.Tensor]] = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        # Keep unsupported paths unchanged to preserve compatibility.
        if key_value_states is not None or past_key_value is not None:
            return self.original_attn(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

        bsz, tgt_len, _ = hidden_states.size()
        q = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        k = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        v = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        ratio = self.layer_index / max(self.total_layers - 1, 1)
        attn_output, info = self.core(q, k, v, attention_mask=attention_mask, layer_ratio=ratio)
        self.last_info = info

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        attn_weights_reshaped = None
        if output_attentions:
            # Keep interface without materializing dense LxL matrix.
            attn_weights_reshaped = info["topk_w"].mean(dim=1)

        return attn_output, attn_weights_reshaped, None


class LongAttentionQASelfAttentionAdapter(nn.Module):
    """Replace BERT-like encoder self-attention with LongAttentionCore."""

    def __init__(
        self,
        original_self_attn: nn.Module,
        core: LongAttentionCore,
        *,
        layer_index: int,
        total_layers: int,
    ):
        super().__init__()
        self.original = original_self_attn

        self.query = original_self_attn.query
        self.key = original_self_attn.key
        self.value = original_self_attn.value

        # Common HF pattern for BERT-like modules.
        self.num_attention_heads = getattr(original_self_attn, "num_attention_heads")
        self.attention_head_size = getattr(original_self_attn, "attention_head_size")
        self.all_head_size = getattr(original_self_attn, "all_head_size")

        self.core = core
        self.layer_index = layer_index
        self.total_layers = max(total_layers, 1)
        self.last_info: Optional[Dict[str, torch.Tensor]] = None

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        # Fallback for cross-attn or cached decoding if present.
        if encoder_hidden_states is not None or past_key_value is not None:
            return self.original(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                **kwargs,
            )

        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        ratio = self.layer_index / max(self.total_layers - 1, 1)
        context, info = self.core(q, k, v, attention_mask=attention_mask, layer_ratio=ratio)
        self.last_info = info

        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(new_context_shape)

        if output_attentions:
            attn_probs = info["topk_w"].mean(dim=1)
            return (context, attn_probs)
        return (context,)


class LongAttentionLongformerSelfAttentionAdapter(nn.Module):
    """Replace Longformer self-attention while keeping Longformer layer interface."""

    def __init__(
        self,
        original_self_attn: nn.Module,
        core: LongAttentionCore,
        *,
        layer_index: int,
        total_layers: int,
    ):
        super().__init__()
        self.original = original_self_attn
        self.query = original_self_attn.query
        self.key = original_self_attn.key
        self.value = original_self_attn.value

        self.num_heads = getattr(original_self_attn, "num_heads")
        self.head_dim = getattr(original_self_attn, "head_dim")
        self.embed_dim = getattr(original_self_attn, "embed_dim", self.num_heads * self.head_dim)

        self.core = core
        self.layer_index = layer_index
        self.total_layers = max(total_layers, 1)
        self.last_info: Optional[Dict[str, torch.Tensor]] = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        is_index_masked: Optional[torch.Tensor] = None,
        is_index_global_attn: Optional[torch.Tensor] = None,
        is_global_attn: Optional[bool] = None,
        output_attentions: bool = False,
    ):
        q = self._shape(self.query(hidden_states))
        k = self._shape(self.key(hidden_states))
        v = self._shape(self.value(hidden_states))

        ratio = self.layer_index / max(self.total_layers - 1, 1)
        context, info = self.core(q, k, v, attention_mask=attention_mask, layer_ratio=ratio)
        self.last_info = info

        context = context.permute(0, 2, 1, 3).contiguous().view(hidden_states.size(0), hidden_states.size(1), self.embed_dim)

        if output_attentions:
            attn_probs = info["topk_w"].mean(dim=1)
            return (context, attn_probs)
        return (context,)


def replace_bart_encoder_attention_with_longattention(
    model: nn.Module,
    *,
    num_types: int,
    window_size: int,
    segment_size: int,
    top_k: int,
    alpha_init: float,
    gate_bias_init: float,
) -> int:
    if not hasattr(model, "model") or not hasattr(model.model, "encoder"):
        raise ValueError("Expected a Seq2Seq model with model.encoder.")

    layers = getattr(model.model.encoder, "layers", None)
    if layers is None:
        raise ValueError("Unsupported encoder structure: expected encoder.layers")

    replaced = 0
    total_layers = len(layers)
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        core = LongAttentionCore(
            d_head=attn.head_dim,
            num_types=num_types,
            window_size=window_size,
            segment_size=segment_size,
            top_k=top_k,
            alpha_init=alpha_init,
            gate_bias_init=gate_bias_init,
        )
        layer.self_attn = LongAttentionBartSelfAttentionAdapter(
            attn,
            core,
            layer_index=idx,
            total_layers=total_layers,
        )
        replaced += 1
    return replaced


def replace_qa_encoder_attention_with_longattention(
    encoder: nn.Module,
    *,
    num_types: int,
    window_size: int,
    segment_size: int,
    top_k: int,
    alpha_init: float,
    gate_bias_init: float,
) -> int:
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
    elif hasattr(encoder, "layer"):
        layers = encoder.layer
    else:
        raise ValueError("Unsupported QA encoder structure: expected encoder.layer")

    replaced = 0
    total_layers = len(layers)
    for idx, layer in enumerate(layers):
        attn_container = getattr(layer, "attention", None)
        if attn_container is None or not hasattr(attn_container, "self"):
            continue

        original_self = attn_container.self
        cls_name = original_self.__class__.__name__.lower()
        if "longformer" in cls_name:
            heads = getattr(original_self, "num_heads", None)
            head_dim = getattr(original_self, "head_dim", None)
        else:
            heads = getattr(original_self, "num_attention_heads", None)
            head_dim = getattr(original_self, "attention_head_size", None)
        if heads is None or head_dim is None:
            continue

        core = LongAttentionCore(
            d_head=head_dim,
            num_types=num_types,
            window_size=window_size,
            segment_size=segment_size,
            top_k=top_k,
            alpha_init=alpha_init,
            gate_bias_init=gate_bias_init,
        )
        if "longformer" in cls_name:
            attn_container.self = LongAttentionLongformerSelfAttentionAdapter(
                original_self,
                core,
                layer_index=idx,
                total_layers=total_layers,
            )
        else:
            attn_container.self = LongAttentionQASelfAttentionAdapter(
                original_self,
                core,
                layer_index=idx,
                total_layers=total_layers,
            )
        replaced += 1

    if replaced == 0:
        raise ValueError("No QA self-attention module was replaced; unsupported backbone family.")
    return replaced


def collect_replaced_seq2seq_layer_infos(model: nn.Module) -> List[Dict[str, torch.Tensor]]:
    infos: List[Dict[str, torch.Tensor]] = []
    if not hasattr(model, "model") or not hasattr(model.model, "encoder"):
        return infos
    layers = getattr(model.model.encoder, "layers", None)
    if layers is None:
        return infos
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        info = getattr(attn, "last_info", None)
        if info is not None:
            infos.append(info)
    return infos


def collect_replaced_qa_layer_infos(encoder: nn.Module) -> List[Dict[str, torch.Tensor]]:
    infos: List[Dict[str, torch.Tensor]] = []
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer
    elif hasattr(encoder, "layer"):
        layers = encoder.layer
    else:
        return infos

    for layer in layers:
        attn_container = getattr(layer, "attention", None)
        if attn_container is None:
            continue
        attn = getattr(attn_container, "self", None)
        info = getattr(attn, "last_info", None)
        if info is not None:
            infos.append(info)
    return infos
