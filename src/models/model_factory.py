"""
Model Factory: Backbone Loading and Attention Hot-Swapping for BART.

This module provides:
1. ``build_tokenizer(backbone)``   — Loads BART tokenizer.
2. ``build_model(...)``            — Loads BartForConditionalGeneration and hot-swaps Attention.
3. ``inject_attention(...)``       — Replaces Encoder self-attention layers dynamically.
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel, BartForConditionalGeneration

logger = logging.getLogger(__name__)


def build_tokenizer(backbone: str = "facebook/bart-base") -> Any:
    """Load the HuggingFace tokenizer for BART."""
    logger.info("Loading tokenizer: %s", backbone)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    return tokenizer


def inject_attention(
    model: nn.Module,
    attention_type: str = "long_attention",
    local_window_size: int = 512,
    top_k: int = 64,
    num_types: int = 3,
    bottleneck_ratio: float = 0.25,
    dropout_prob: float = 0.0,
) -> None:
    """
    Dynamically replace all BART ENCODER self-attention layers.

    Args:
        model:             BartForConditionalGeneration instance.
        attention_type:    "vanilla", "led", or "long_attention".
        local_window_size: Window size for Sliding Window / LongAttention.
    """
    if attention_type == "vanilla":
        logger.info("Using standard BART attention — no injection performed.")
        return

    # Lazy imports to break circular dependency
    from .long_attention import LongAttention
    from .local_attention import LocalSlidingWindowAttention
    from .led_attention import LEDSelfAttention

    try:
        layers = model.model.encoder.layers
    except AttributeError:
        raise ValueError("Model does not have model.model.encoder.layers. Ensure it is BART.")

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

        if attention_type == "long_attention":
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
            
        elif attention_type == "led":
            new_attn = LEDSelfAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                window_size=local_window_size,
                dropout_prob=dropout_prob,
                bias=True,
            ).to(device=device, dtype=dtype)
            
        elif attention_type == "sparse":
            # Keep for backward compatibility or remove if not needed
            new_attn = LocalSlidingWindowAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                window_size=local_window_size,
                dropout_prob=dropout_prob,
                bias=True,
            ).to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Hot-swap
        layer.self_attn = new_attn
        replaced_count += 1

    logger.info("Injection complete: %d Encoder layer(s) replaced with '%s'.", replaced_count, attention_type)


def build_model(
    backbone: str = "facebook/bart-base",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    attention_type: str = "standard",
    long_attention_config: Optional[Dict[str, Any]] = None,
    freeze_backbone: bool = False,
) -> PreTrainedModel:
    """
    Load BART backbone and orchestrate the injection of custom attention blocks.
    """
    if attention_type == "standard":
        attention_type = "vanilla"

    logger.info("Loading backbone model: %s (dtype=%s)", backbone, torch_dtype)
    model: PreTrainedModel = BartForConditionalGeneration.from_pretrained(
        backbone,
        device_map=device_map,
    )

    long_attention_config = long_attention_config or {}
    
    inject_attention(
        model=model,
        attention_type=attention_type,
        local_window_size=long_attention_config.get("local_window_size", 512),
        top_k=long_attention_config.get("top_k", 64),
        num_types=long_attention_config.get("num_types", 3),
        bottleneck_ratio=long_attention_config.get("bottleneck_ratio", 0.25),
        dropout_prob=long_attention_config.get("dropout_prob", 0.0),
    )

    # Freeze standard backbone layers if requested, EXCEPT the newly injected attention layers
    if freeze_backbone and attention_type != "vanilla":
        logger.info("Freezing base model parameters. Only training new attention modules.")
        for name, param in model.named_parameters():
            if "encoder.layers" in name and "self_attn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model ready | Total params: {:,} | Trainable: {:,}".format(total_params, trainable_params))

    return model
