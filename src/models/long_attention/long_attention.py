"""
LongAttention v2 Model Builder: Load BART and inject LongAttention with proper weight inheritance.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, PreTrainedModel

# Import the main attention container from attention.py
from .attention import LongAttention

logger = logging.getLogger(__name__)

def _extend_learned_position_embeddings(model: nn.Module, new_max_length: int, offset: int = 2) -> None:
    """
    Extend BART position embeddings using Copy/Repeat strategy (standard Longformer/LED practice).
    """
    embed_positions = model.embed_positions
    current_max_pos = embed_positions.num_embeddings - offset
    
    if new_max_length <= current_max_pos:
        return

    logger.info(f"Extending BART position embeddings: {current_max_pos} -> {new_max_length} (Copy/Repeat)")
    old_weights = embed_positions.weight.data
    hidden_size = old_weights.size(-1)
    
    special_weights = old_weights[:offset]
    pos_weights = old_weights[offset:]
    
    n_repeats = (new_max_length + current_max_pos - 1) // current_max_pos
    extended_pos_weights = pos_weights.repeat(n_repeats, 1)
    extended_pos_weights = extended_pos_weights[:new_max_length]
    
    new_weights = torch.cat([special_weights, extended_pos_weights], dim=0)
    
    # Preserve the same class (BartLearnedPositionalEmbedding) to avoid signature mismatch
    new_embed_positions = type(embed_positions)(new_max_length, hidden_size)
    new_embed_positions.weight.data = new_weights
    model.embed_positions = new_embed_positions

def build_long_attention_model(
    backbone: str = "facebook/bart-base",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    config: Optional[Dict[str, Any]] = None,
    freeze_backbone: bool = False,
) -> PreTrainedModel:
    """
    Build LongAttention model by injecting necessitated, dependency-typed attention into BART.
    """
    config = config or {}
    local_window_size = config.get("local_window_size", 512)
    top_k = config.get("top_k", 64)
    num_types = config.get("num_types", 3)
    bottleneck_ratio = config.get("bottleneck_ratio", 0.25)
    dropout_prob = config.get("dropout_prob", 0.1)

    logger.info("Loading backbone model: %s", backbone)
    model = BartForConditionalGeneration.from_pretrained(
        backbone, device_map=device_map, torch_dtype=torch_dtype
    )

    # ── 1. Position Extension ──────────────────────────────────────────
    max_length = config.get("max_length", 1024)
    _extend_learned_position_embeddings(model.model.encoder, max_length)
    _extend_learned_position_embeddings(model.model.decoder, max_length)
    model.config.max_position_embeddings = max_length

    # ── 2. Attention Injection ─────────────────────────────────────────
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

        # ── 3. Weight Inheritance (Crucial for Fine-tuning) ────────────────
        with torch.no_grad():
            # (a) Local Branch Inheritance
            new_attn.local_attention.q_proj.weight.copy_(original_attn.q_proj.weight)
            new_attn.local_attention.k_proj.weight.copy_(original_attn.k_proj.weight)
            new_attn.local_attention.v_proj.weight.copy_(original_attn.v_proj.weight)
            if original_attn.q_proj.bias is not None:
                new_attn.local_attention.q_proj.bias.copy_(original_attn.q_proj.bias)
                new_attn.local_attention.k_proj.bias.copy_(original_attn.k_proj.bias)
                new_attn.local_attention.v_proj.bias.copy_(original_attn.v_proj.bias)

            # (b) Long-range Retrieval Inheritance
            # Use 5e-3 noise to break symmetry between dependency types
            q_w = new_attn.typed_retrieval.q_proj.weight
            q_w.copy_(original_attn.q_proj.weight.repeat(num_types, 1))
            q_w.data += torch.randn_like(q_w.data) * 2e-2
            if original_attn.q_proj.bias is not None:
                new_attn.typed_retrieval.q_proj.bias.copy_(original_attn.q_proj.bias.repeat(num_types))

            # (c) Gist/Dependency Projection Inheritance
            kv_w = new_attn.typed_gist.multi_type_proj.weight
            kv_template_w = torch.cat([original_attn.k_proj.weight, original_attn.v_proj.weight], dim=0)
            kv_w.copy_(kv_template_w.repeat(num_types, 1))
            kv_w.data += torch.randn_like(kv_w.data) * 2e-2
            
            if original_attn.k_proj.bias is not None:
                kv_b = new_attn.typed_gist.multi_type_proj.bias
                kv_template_b = torch.cat([original_attn.k_proj.bias, original_attn.v_proj.bias], dim=0)
                kv_b.copy_(kv_template_b.repeat(num_types))

            # (d) Final Out Projection Inheritance
            new_attn.out_proj.weight.copy_(original_attn.out_proj.weight)
            if hasattr(original_attn.out_proj, 'bias') and original_attn.out_proj.bias is not None:
                new_attn.out_proj.bias.copy_(original_attn.out_proj.bias)

        # Hot-swap the Attention Layer
        layer.self_attn = new_attn
        replaced_count += 1

    # Re-establish weight tying (embed_tokens + lm_head) to prevent missing keys
    model.tie_weights()
    
    if freeze_backbone:
        logger.info("Freezing backbone — only Injected layers will be trained.")
        for name, param in model.named_parameters():
            if "encoder.layers" in name and "self_attn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info(f"Successfully injected LongAttention into {replaced_count} layers.")
    return model
