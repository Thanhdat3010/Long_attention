"""
Models package — Dispatcher for attention-type-specific model builders.

This replaces the old model_factory.py. Each attention type is a self-contained
subpackage (following Google LRA convention):
  - src/models/led/          → LED (Longformer-style) attention
  - src/models/long_attention/ → LongAttention v2
"""

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration, PreTrainedModel

logger = logging.getLogger(__name__)


def build_tokenizer(backbone: str = "facebook/bart-base"):
    """Load the HuggingFace tokenizer for BART."""
    logger.info("Loading tokenizer: %s", backbone)
    return AutoTokenizer.from_pretrained(backbone)


def build_model(
    task: str = "nmt",
    backbone: str = "facebook/bart-base",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    attention_type: str = "vanilla",
    long_attention_config: Optional[Dict[str, Any]] = None,
    freeze_backbone: bool = False,
) -> PreTrainedModel:
    """
    Build a BART model with the specified attention type.

    Dispatches to the appropriate self-contained model builder:
      - 'vanilla': Unmodified BART (no injection).
      - 'led':     LED (Longformer-style) sliding window + global tokens.
      - 'long_attention': LongAttention v2 (necessity-aware, dependency-typed).

    Args:
        backbone:              HuggingFace model ID.
        device_map:            Device placement strategy.
        torch_dtype:           Floating point precision.
        attention_type:        One of 'vanilla', 'led', 'long_attention'.
        long_attention_config: Dict of attention-specific hyperparameters.
        freeze_backbone:       If True, freeze all params except injected attention.

    Returns:
        BartForConditionalGeneration with the specified attention mechanism.
    """
    if task == "nmt":
        if attention_type in ("standard", "vanilla"):
            logger.info("Loading vanilla BART: %s (dtype=%s)", backbone, torch_dtype)
            model = BartForConditionalGeneration.from_pretrained(
                backbone, device_map=device_map, torch_dtype=torch_dtype
            )
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
        elif attention_type == "led":
            from .nmt.led_bart.bart_wrapper import build_led_model
            model = build_led_model(
                backbone=backbone,
                device_map=device_map,
                torch_dtype=torch_dtype,
                config=long_attention_config,
                freeze_backbone=freeze_backbone,
            )
        elif attention_type == "long_attention":
            from .nmt.long_attention_bart.bart_wrapper import build_long_attention_model
            model = build_long_attention_model(
                backbone=backbone,
                device_map=device_map,
                torch_dtype=torch_dtype,
                config=long_attention_config,
                freeze_backbone=freeze_backbone,
            )
        else:
            raise ValueError(f"Unknown NMT attention type: '{attention_type}'.")

    elif task == "qa":
        # For QA, we expect a RoBERTa backbone
        # We pass num_labels=2 for start/end, and our wrappers add the Yes/No head
        if long_attention_config is None:
            long_attention_config = {}
        long_attention_config["num_labels"] = 2

        if attention_type in ("standard", "vanilla", "longformer"):
            from .qa.longformer_roberta.roberta_wrapper import build_qa_longformer_model
            model = build_qa_longformer_model(backbone, long_attention_config)
            model.inject_longformer_attention(long_attention_config)
            model.to(torch_dtype)
            
        elif attention_type == "long_attention":
            from .qa.long_attention_roberta.roberta_wrapper import build_qa_long_attention_model
            model = build_qa_long_attention_model(backbone, long_attention_config)
            model.inject_long_attention(long_attention_config)
            model.to(torch_dtype)
        else:
            raise ValueError(f"Unknown QA attention type: '{attention_type}'.")
    else:
        raise ValueError(f"Unknown task: '{task}'")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model ready | Total params: {:,} | Trainable: {:,}".format(
        total_params, trainable_params))

    return model
