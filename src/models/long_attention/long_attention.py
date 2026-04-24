"""
LongAttention v2 Model Builder: Load BART and inject LongAttention with proper weight inheritance.

This module handles:
  1. Loading the pre-trained BartForConditionalGeneration backbone.
  2. Replacing all encoder self-attention layers with LongAttention v2.
  3. Copying BART's pre-trained Q/K/V/Out weights into local branch, typed retrieval,
     and dependency gist projections (with symmetry-breaking noise for typed modules).
  4. Re-establishing weight tying (embed_tokens + lm_head) to prevent missing-key errors.

Following Google LRA convention, the attention algorithm is defined separately
in long_attention_attention.py, and this file handles model construction only.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BartForConditionalGeneration, PreTrainedModel

from .attention import LongAttention

logger = logging.getLogger(__name__)


def build_long_attention_model(
    backbone: str = "facebook/bart-base",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    config: Optional[Dict[str, Any]] = None,
    freeze_backbone: bool = False,
) -> PreTrainedModel:
    """
    Load BART backbone and inject LongAttention v2 into all encoder layers.

    Args:
        backbone:         HuggingFace model ID (e.g. 'facebook/bart-base').
        device_map:       Device placement strategy ('auto', 'cpu', 'cuda').
        torch_dtype:      Floating point precision for model weights.
        config:           Dict with optional keys:
                          - local_window_size (int, default 512)
                          - top_k (int, default 64)
                          - num_types (int, default 3)
                          - bottleneck_ratio (float, default 0.25)
                          - dropout_prob (float, default 0.0)
        freeze_backbone:  If True, freeze all params except the injected attention layers.

    Returns:
        BartForConditionalGeneration with LongAttention v2 in all encoder layers.
    """
    config = config or {}
    local_window_size = config.get("local_window_size", 512)
    top_k = config.get("top_k", 64)
    num_types = config.get("num_types", 3)
    bottleneck_ratio = config.get("bottleneck_ratio", 0.25)
    dropout_prob = config.get("dropout_prob", 0.0)

    logger.info("Loading backbone model: %s (dtype=%s)", backbone, torch_dtype)
    model = BartForConditionalGeneration.from_pretrained(
        backbone,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    # Inject LongAttention v2 into all encoder layers
    layers = model.model.encoder.layers
    replaced_count = 0

    for idx, layer in enumerate(layers):
        original_attn = layer.self_attn
        hidden_size = original_attn.embed_dim
        num_heads = original_attn.num_heads

        try:
            device = next(original_attn.parameters()).device
            dtype = next(original_attn.parameters()).dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        new_attn = LongAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            local_window_size=local_window_size,
            top_k=top_k,
            num_types=num_types,
            bottleneck_ratio=bottleneck_ratio,
            dropout_prob=dropout_prob,
            layer_idx=idx,
            bias=True,
        ).to(device=device, dtype=dtype)

        # ── Weight Inheritance ──────────────────────────────────────────
        with torch.no_grad():
            # Copy for Local Branch
            new_attn.local_attention.q_proj.weight.copy_(original_attn.q_proj.weight)
            new_attn.local_attention.k_proj.weight.copy_(original_attn.k_proj.weight)
            new_attn.local_attention.v_proj.weight.copy_(original_attn.v_proj.weight)
            if hasattr(original_attn.q_proj, 'bias') and original_attn.q_proj.bias is not None:
                new_attn.local_attention.q_proj.bias.copy_(original_attn.q_proj.bias)
                new_attn.local_attention.k_proj.bias.copy_(original_attn.k_proj.bias)
                new_attn.local_attention.v_proj.bias.copy_(original_attn.v_proj.bias)

            # Copy for Long-range Retrieval Queries (with symmetry-breaking noise)
            # Increased noise to 5e-3 to kick optimizer out of symmetry trap
            q_w = new_attn.typed_retrieval.q_proj.weight
            q_w.copy_(original_attn.q_proj.weight.repeat(num_types, 1))
            q_w.data += torch.randn_like(q_w.data) * 5e-3
            if original_attn.q_proj.bias is not None:
                q_b = new_attn.typed_retrieval.q_proj.bias
                q_b.copy_(original_attn.q_proj.bias.repeat(num_types))

            # Copy for Dependency Gist (K, V projections)
            # multi_type_proj weight is (num_types * 2 * D, D)
            kv_w = new_attn.typed_gist.multi_type_proj.weight
            kv_template_w = torch.cat([original_attn.k_proj.weight, original_attn.v_proj.weight], dim=0)
            kv_w.copy_(kv_template_w.repeat(num_types, 1))
            kv_w.data += torch.randn_like(kv_w.data) * 5e-3
            
            if original_attn.k_proj.bias is not None:
                kv_b = new_attn.typed_gist.multi_type_proj.bias
                kv_template_b = torch.cat([original_attn.k_proj.bias, original_attn.v_proj.bias], dim=0)
                kv_b.copy_(kv_template_b.repeat(num_types))

            # Copy Final Output Projection
            new_attn.out_proj.weight.copy_(original_attn.out_proj.weight)
            if hasattr(original_attn.out_proj, 'bias') and original_attn.out_proj.bias is not None:
                new_attn.out_proj.bias.copy_(original_attn.out_proj.bias)

        # Hot-swap
        layer.self_attn = new_attn
        replaced_count += 1

    logger.info("LongAttention injection complete: %d encoder layer(s) replaced.", replaced_count)

    # ── CRITICAL FIX: Re-establish weight tying ─────────────────────────
    # BART shares weights between encoder/decoder embeddings and lm_head.
    # After modifying encoder structure, this link can break during checkpoint
    # save/load, causing 'missing keys' errors and random embedding init.
    model.tie_weights()
    logger.info("Weight tying re-established (embed_tokens + lm_head).")

    # ── Optional backbone freeze ────────────────────────────────────────
    if freeze_backbone:
        logger.info("Freezing base model parameters. Only training LongAttention modules.")
        for name, param in model.named_parameters():
            if "encoder.layers" in name and "self_attn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("LongAttention model ready | Total params: {:,} | Trainable: {:,}".format(
        total_params, trainable_params))

    return model
