"""
backbone.py — Backbone-Agnostic Model Builder
===============================================

Supports ANY HuggingFace encoder model as backbone:
  --backbone allenai/longformer-base-4096
  --backbone roberta-base
  --backbone google/bigbird-roberta-base
  --backbone bert-base-uncased
  ...any AutoModel-compatible model

Two model variants:
  1. BaselineModel       — backbone + task head (no modification)
  2. LongAttentionModel  — backbone embeddings + LongAttention v2 layers + task head
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import math

import torch
import torch.nn as nn

from src.modeling import LongAttentionLayer


# ===================================================================== #
#  BaselineModel                                                         #
# ===================================================================== #

class BaselineModel(nn.Module):
    """Vanilla pretrained encoder + task head. No modification."""

    def __init__(self, backbone_name: str, task: str = "qa"):
        super().__init__()
        from transformers import AutoModel
        self.task = task
        self.encoder = AutoModel.from_pretrained(backbone_name)
        h = self.encoder.config.hidden_size

        if task == "qa":
            self.qa_head = nn.Linear(h, 2)
        elif task == "docmt":
            self.lm_head = nn.Linear(h, self.encoder.config.vocab_size)

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, **kw):
        kwargs = {}
        if global_attention_mask is not None:
            kwargs["global_attention_mask"] = global_attention_mask

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        h = out.last_hidden_state
        if self.task == "qa":
            logits = self.qa_head(h)
            return {"start_logits": logits[:, :, 0], "end_logits": logits[:, :, 1]}
        else:
            return {"logits": self.lm_head(h)}


class Seq2SeqBaseline(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM
        self.model = AutoModelForSeq2SeqLM.from_pretrained(backbone_name)

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return {"loss": out.loss, "logits": out.logits}

    def generate(self, **kw):
        return self.model.generate(**kw)


# ===================================================================== #
#  LongAttentionModel                                                    #
# ===================================================================== #

class LongAttentionModel(nn.Module):
    """
    Pretrained embeddings + LongAttention v2 encoder layers + task head.

    Supports gradient checkpointing (--gradient_checkpoint) and
    top-k segment routing (--top_k).
    """

    def __init__(
        self,
        backbone_name: str,
        task: str = "qa",
        num_types: int = 4,
        window_size: int = 256,
        segment_size: int = 64,
        top_k: int = 4,
        gradient_checkpoint: bool = False,
        alpha_init: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        from transformers import AutoModel
        self.task = task

        pretrained = AutoModel.from_pretrained(backbone_name)
        cfg = pretrained.config
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        dropout = getattr(cfg, "hidden_dropout_prob", 0.1)

        # Keep pretrained embeddings
        if hasattr(pretrained, "embeddings"):
            self.embeddings = pretrained.embeddings
        elif hasattr(pretrained, "encoder") and hasattr(pretrained.encoder, "embeddings"):
            self.embeddings = pretrained.encoder.embeddings
        else:
            self.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        # Replace encoder layers with LongAttention v2
        self.layers = nn.ModuleList([
            LongAttentionLayer(
                d_model=self.hidden_size,
                num_heads=num_heads,
                num_types=num_types,
                window_size=window_size,
                segment_size=segment_size,
                top_k=top_k,
                ff_mult=4,
                dropout=dropout,
                alpha_init=alpha_init,
                gate_bias_init=gate_bias_init,
            )
            for _ in range(self.num_layers)
        ])

        # Attempt to copy weights from original pretrained encoder to avoid starting from scratch
        if hasattr(pretrained, "encoder") and hasattr(pretrained.encoder, "layer"):
            orig_layers = pretrained.encoder.layer
        elif hasattr(pretrained, "layer"):
            orig_layers = pretrained.layer
        else:
            orig_layers = None

        if orig_layers is not None:
            for i, la_layer in enumerate(self.layers):
                if i >= len(orig_layers): break
                orig = orig_layers[i]
                
                try:
                    if hasattr(orig, "attention"):
                        attn = orig.attention
                        if hasattr(attn, "self"):
                            q_w = attn.self.query.weight
                            k_w = attn.self.key.weight
                            v_w = attn.self.value.weight
                            q_b = getattr(attn.self.query, "bias", None)
                            k_b = getattr(attn.self.key, "bias", None)
                            v_b = getattr(attn.self.value, "bias", None)

                            la_layer.qkv_proj.weight.data = torch.cat([q_w, k_w, v_w], dim=0)
                            if q_b is not None and k_b is not None and v_b is not None:
                                la_layer.qkv_proj.bias.data = torch.cat([q_b, k_b, v_b], dim=0)

                        if hasattr(attn, "output"):
                            la_layer.out_proj.weight.data = attn.output.dense.weight.data
                            if hasattr(attn.output.dense, "bias"):
                                la_layer.out_proj.bias.data = attn.output.dense.bias.data
                            la_layer.norm1.weight.data = attn.output.LayerNorm.weight.data
                            la_layer.norm1.bias.data = attn.output.LayerNorm.bias.data

                    if hasattr(orig, "intermediate") and hasattr(orig, "output"):
                        la_layer.ff[0].weight.data = orig.intermediate.dense.weight.data
                        if hasattr(orig.intermediate.dense, "bias"):
                            la_layer.ff[0].bias.data = orig.intermediate.dense.bias.data

                        la_layer.ff[3].weight.data = orig.output.dense.weight.data
                        if hasattr(orig.output.dense, "bias"):
                            la_layer.ff[3].bias.data = orig.output.dense.bias.data

                        la_layer.norm2.weight.data = orig.output.LayerNorm.weight.data
                        la_layer.norm2.bias.data = orig.output.LayerNorm.bias.data
                except Exception:
                    pass

        # Enable gradient checkpointing if requested
        if gradient_checkpoint:
            for layer in self.layers:
                layer.use_checkpoint = True

        self.final_norm = nn.LayerNorm(self.hidden_size)

        if task == "qa":
            self.qa_head = nn.Linear(self.hidden_size, 2)
        elif task == "docmt":
            self.lm_head = nn.Linear(self.hidden_size, cfg.vocab_size)

        del pretrained  # free original encoder

    def forward(self, input_ids, attention_mask=None, **kw):
        x = self.embeddings(input_ids)

        layer_infos = []
        for idx, layer in enumerate(self.layers):
            ratio = idx / max(self.num_layers - 1, 1)
            x, info = layer(x, layer_ratio=ratio, attention_mask=attention_mask)
            layer_infos.append(info)

        x = self.final_norm(x)

        if self.task == "qa":
            logits = self.qa_head(x)
            out = {"start_logits": logits[:, :, 0], "end_logits": logits[:, :, 1]}
        else:
            out = {"logits": self.lm_head(x)}

        return out, layer_infos


class LongAttentionBartEncoder(nn.Module):
    def __init__(
        self,
        bart_encoder,
        num_types: int = 4,
        window_size: int = 256,
        segment_size: int = 64,
        top_k: int = 4,
        gradient_checkpoint: bool = False,
        alpha_init: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        cfg = bart_encoder.config
        self.config = cfg
        self.embed_tokens = bart_encoder.embed_tokens
        self.embed_positions = getattr(bart_encoder, "embed_positions", None)
        self.layernorm_embedding = getattr(bart_encoder, "layernorm_embedding", None)
        self.dropout = getattr(cfg, "dropout", 0.1)
        self.embed_scale = math.sqrt(cfg.d_model) if getattr(cfg, "scale_embedding", False) else 1.0
        self.hidden_size = cfg.d_model
        self.num_layers = cfg.encoder_layers
        num_heads = cfg.encoder_attention_heads

        self.layers = nn.ModuleList([
            LongAttentionLayer(
                d_model=self.hidden_size,
                num_heads=num_heads,
                num_types=num_types,
                window_size=window_size,
                segment_size=segment_size,
                top_k=top_k,
                ff_mult=4,
                dropout=cfg.dropout,
                alpha_init=alpha_init,
                gate_bias_init=gate_bias_init,
            )
            for _ in range(self.num_layers)
        ])

        if gradient_checkpoint:
            for layer in self.layers:
                layer.use_checkpoint = True

        self.final_norm = nn.LayerNorm(self.hidden_size)
        self._last_infos = None

    def forward(self, input_ids=None, attention_mask=None, **kw):
        if input_ids is None:
            raise ValueError("LongAttentionBartEncoder requires input_ids")

        x = self.embed_tokens(input_ids) * self.embed_scale
        if self.embed_positions is not None:
            x = x + self.embed_positions(input_ids)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        layer_infos = []
        for idx, layer in enumerate(self.layers):
            ratio = idx / max(self.num_layers - 1, 1)
            x, info = layer(x, layer_ratio=ratio, attention_mask=attention_mask)
            layer_infos.append(info)

        x = self.final_norm(x)
        self._last_infos = layer_infos

        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=x)


class LongAttentionSeq2SeqModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_types: int = 4,
        window_size: int = 256,
        segment_size: int = 64,
        top_k: int = 4,
        gradient_checkpoint: bool = False,
        alpha_init: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM
        self.model = AutoModelForSeq2SeqLM.from_pretrained(backbone_name)
        if getattr(self.model.config, "model_type", "") != "bart":
            raise ValueError("LongAttentionSeq2SeqModel currently supports BART backbones only.")

        encoder = self.model.model.encoder
        self.model.model.encoder = LongAttentionBartEncoder(
            encoder,
            num_types=num_types,
            window_size=window_size,
            segment_size=segment_size,
            top_k=top_k,
            gradient_checkpoint=gradient_checkpoint,
            alpha_init=alpha_init,
            gate_bias_init=gate_bias_init,
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        layer_infos = getattr(self.model.model.encoder, "_last_infos", None)
        return {"loss": out.loss, "logits": out.logits, "layer_infos": layer_infos}

    def generate(self, **kw):
        return self.model.generate(**kw)


# ===================================================================== #
#  Factory                                                               #
# ===================================================================== #

def build_model(args) -> nn.Module:
    if args.task == "docmt":
        if args.model == "baseline":
            model = Seq2SeqBaseline(args.backbone)
            print(f"[model] Seq2SeqBaseline | backbone={args.backbone} | task={args.task}")
        elif args.model == "longattention":
            top_k = getattr(args, "top_k", 4)
            grad_ckpt = getattr(args, "gradient_checkpoint", False)
            alpha_init = getattr(args, "alpha_init", 0.02)
            gate_bias_init = getattr(args, "gate_bias_init", 0.0)
            model = LongAttentionSeq2SeqModel(
                backbone_name=args.backbone,
                num_types=args.num_types,
                window_size=args.window_size,
                segment_size=args.segment_size,
                top_k=top_k,
                gradient_checkpoint=grad_ckpt,
                alpha_init=alpha_init,
                gate_bias_init=gate_bias_init,
            )
            print(f"[model] LongAttentionSeq2SeqModel | backbone={args.backbone} | task={args.task}")
            print(f"        window={args.window_size} segment={args.segment_size} types={args.num_types} top_k={top_k} grad_ckpt={grad_ckpt}")
        else:
            raise ValueError(f"Unknown: {args.model}")

    elif args.model == "baseline":
        model = BaselineModel(args.backbone, args.task)
        print(f"[model] BaselineModel | backbone={args.backbone} | task={args.task}")

    elif args.model == "longattention":
        top_k = getattr(args, "top_k", 4)
        grad_ckpt = getattr(args, "gradient_checkpoint", False)
        alpha_init = getattr(args, "alpha_init", 0.02)
        gate_bias_init = getattr(args, "gate_bias_init", 0.0)
        model = LongAttentionModel(
            backbone_name=args.backbone,
            task=args.task,
            num_types=args.num_types,
            window_size=args.window_size,
            segment_size=args.segment_size,
            top_k=top_k,
            gradient_checkpoint=grad_ckpt,
            alpha_init=alpha_init,
            gate_bias_init=gate_bias_init,
        )
        print(f"[model] LongAttentionModel | backbone={args.backbone} | task={args.task}")
        print(f"        window={args.window_size} segment={args.segment_size} types={args.num_types} top_k={top_k} grad_ckpt={grad_ckpt}")
    else:
        raise ValueError(f"Unknown: {args.model}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        params: {total:,} total, {trainable:,} trainable")
    return model
