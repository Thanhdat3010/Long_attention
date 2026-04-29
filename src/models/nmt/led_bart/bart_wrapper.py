"""
LED Model Builder: Load BART and inject LED attention with proper weight inheritance.

This module handles:
  1. Loading the pre-trained BartForConditionalGeneration backbone.
  2. Replacing all encoder self-attention layers with LEDSelfAttention.
  3. Copying BART's pre-trained Q/K/V/Out weights into both local and global projections.
  4. Re-establishing weight tying (embed_tokens + lm_head) to prevent missing-key errors.

Following Google LRA convention, the attention algorithm is defined separately
in led_attention.py, and this file handles model construction only.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BartForConditionalGeneration, PreTrainedModel

from .led_attention import LEDSelfAttention

logger = logging.getLogger(__name__)


def _extend_learned_position_embeddings(
    model: nn.Module, 
    new_max_length: int, 
    offset: int = 2
) -> None:
    """
    Extend the absolute learned position embeddings of a BART/RoBERTa model 
    using the Copy/Repeat strategy (standardized by Longformer/LED).
    """
    embed_positions = model.embed_positions
    current_max_pos = embed_positions.num_embeddings - offset
    
    if new_max_length <= current_max_pos:
        return

    logger.info(
        f"Extending LED position embeddings: {current_max_pos} -> {new_max_length} "
        f"(Strategy: Copy/Repeat)"
    )

    old_weights = embed_positions.weight.data
    hidden_size = old_weights.size(-1)
    
    special_weights = old_weights[:offset]
    pos_weights = old_weights[offset:]
    
    n_repeats = (new_max_length + current_max_pos - 1) // current_max_pos
    extended_pos_weights = pos_weights.repeat(n_repeats, 1)
    extended_pos_weights = extended_pos_weights[:new_max_length]
    
    new_weights = torch.cat([special_weights, extended_pos_weights], dim=0)
    
    new_embed_positions = type(embed_positions)(new_max_length, hidden_size)
    new_embed_positions.weight.data = new_weights
    
    model.embed_positions = new_embed_positions


def build_led_model(
    backbone: str = "facebook/bart-base",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    config: Optional[Dict[str, Any]] = None,
    freeze_backbone: bool = False,
) -> PreTrainedModel:
    """
    Load BART backbone and inject LED (Longformer-style) attention into all encoder layers.

    Args:
        backbone:         HuggingFace model ID (e.g. 'facebook/bart-base').
        device_map:       Device placement strategy ('auto', 'cpu', 'cuda').
        torch_dtype:      Floating point precision for model weights.
        config:           Dict with optional keys:
                          - local_window_size (int, default 512)
                          - dropout_prob (float, default 0.0)
        freeze_backbone:  If True, freeze all params except the injected attention layers.

    Returns:
        BartForConditionalGeneration with LED attention in all encoder layers.
    """
    config = config or {}
    window_size = config.get("local_window_size", 512)
    dropout_prob = config.get("dropout_prob", 0.0)

    logger.info("Loading backbone model: %s (dtype=%s)", backbone, torch_dtype)
    model = BartForConditionalGeneration.from_pretrained(
        backbone,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    # ── Position Embedding Extension ────────────────────────────────────
    max_length = config.get("max_length", 1024)
    _extend_learned_position_embeddings(model.model.encoder, max_length)
    _extend_learned_position_embeddings(model.model.decoder, max_length)
    model.config.max_position_embeddings = max_length

    # Inject LED attention into all encoder layers
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

        new_attn = LEDSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            dropout_prob=dropout_prob,
            bias=True,
        ).to(device=device, dtype=dtype)

        # ── Weight Inheritance ──────────────────────────────────────────
        # Copy pre-trained BART weights to preserve learned attention patterns.
        with torch.no_grad():
            # Local projections ← BART originals
            new_attn.q_proj.weight.copy_(original_attn.q_proj.weight)
            new_attn.k_proj.weight.copy_(original_attn.k_proj.weight)
            new_attn.v_proj.weight.copy_(original_attn.v_proj.weight)

            # Global projections ← copies of local (per LED/Longformer paper §3.1)
            new_attn.q_proj_global.weight.copy_(original_attn.q_proj.weight)
            new_attn.k_proj_global.weight.copy_(original_attn.k_proj.weight)
            new_attn.v_proj_global.weight.copy_(original_attn.v_proj.weight)

            # Output projection
            new_attn.out_proj.weight.copy_(original_attn.out_proj.weight)

            # Biases
            if hasattr(original_attn.q_proj, 'bias') and original_attn.q_proj.bias is not None:
                new_attn.q_proj.bias.copy_(original_attn.q_proj.bias)
                new_attn.k_proj.bias.copy_(original_attn.k_proj.bias)
                new_attn.v_proj.bias.copy_(original_attn.v_proj.bias)
                new_attn.q_proj_global.bias.copy_(original_attn.q_proj.bias)
                new_attn.k_proj_global.bias.copy_(original_attn.k_proj.bias)
                new_attn.v_proj_global.bias.copy_(original_attn.v_proj.bias)
            if hasattr(original_attn.out_proj, 'bias') and original_attn.out_proj.bias is not None:
                new_attn.out_proj.bias.copy_(original_attn.out_proj.bias)

        # Hot-swap
        layer.self_attn = new_attn
        replaced_count += 1

    logger.info("LED injection complete: %d encoder layer(s) replaced.", replaced_count)

    # ── CRITICAL FIX: Re-establish weight tying ─────────────────────────
    # BART shares weights between encoder/decoder embeddings and lm_head.
    # After modifying encoder structure, this link can break during checkpoint
    # save/load, causing 'missing keys' errors and random embedding init.
    model.tie_weights()
    logger.info("Weight tying re-established (embed_tokens + lm_head).")

    # ── Optional backbone freeze ────────────────────────────────────────
    if freeze_backbone:
        logger.info("Freezing base model parameters. Only training LED attention modules.")
        for name, param in model.named_parameters():
            if "encoder.layers" in name and "self_attn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("LED model ready | Total params: {:,} | Trainable: {:,}".format(
        total_params, trainable_params))

    return model
