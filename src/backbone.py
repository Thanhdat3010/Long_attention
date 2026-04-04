"""
backbone.py — Model Builder (Baseline vs LongAttention attention-only)
======================================================================

Design goal:
- Baseline: keep backbone untouched
- LongAttention: replace ONLY attention modules, keep all other backbone blocks unchanged
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.modeling import (
    collect_replaced_qa_layer_infos,
    collect_replaced_seq2seq_layer_infos,
    replace_bart_encoder_attention_with_longattention,
    replace_qa_encoder_attention_with_longattention,
)


class BaselineModel(nn.Module):
    """Vanilla pretrained encoder + task head."""

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


class LongAttentionQAModel(nn.Module):
    """QA model where only encoder attention modules are replaced by LongAttention."""

    def __init__(
        self,
        backbone_name: str,
        num_types: int = 4,
        window_size: int = 256,
        segment_size: int = 64,
        top_k: int = 4,
        alpha_init: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(backbone_name)
        h = self.encoder.config.hidden_size
        self.qa_head = nn.Linear(h, 2)

        replaced = replace_qa_encoder_attention_with_longattention(
            self.encoder,
            num_types=num_types,
            window_size=window_size,
            segment_size=segment_size,
            top_k=top_k,
            alpha_init=alpha_init,
            gate_bias_init=gate_bias_init,
        )
        print(f"[model] attention-only replacement completed: {replaced} QA self-attn modules")

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, **kw):
        kwargs = {}
        if global_attention_mask is not None:
            kwargs["global_attention_mask"] = global_attention_mask

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        h = out.last_hidden_state
        logits = self.qa_head(h)

        layer_infos = collect_replaced_qa_layer_infos(self.encoder)
        return {"start_logits": logits[:, :, 0], "end_logits": logits[:, :, 1]}, layer_infos


class LongAttentionSeq2SeqModel(nn.Module):
    """Seq2Seq model where only encoder self-attention modules are replaced."""

    def __init__(
        self,
        backbone_name: str,
        num_types: int = 4,
        window_size: int = 256,
        segment_size: int = 64,
        top_k: int = 4,
        alpha_init: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM

        self.model = AutoModelForSeq2SeqLM.from_pretrained(backbone_name)
        model_type = getattr(self.model.config, "model_type", "")
        if model_type not in {"bart", "mbart"}:
            raise ValueError(
                "LongAttentionSeq2SeqModel currently supports BART/MBART families only. "
                f"Got model_type='{model_type}'."
            )

        replaced = replace_bart_encoder_attention_with_longattention(
            self.model,
            num_types=num_types,
            window_size=window_size,
            segment_size=segment_size,
            top_k=top_k,
            alpha_init=alpha_init,
            gate_bias_init=gate_bias_init,
        )
        print(f"[model] attention-only replacement completed: {replaced} encoder self-attn layers")

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        layer_infos = collect_replaced_seq2seq_layer_infos(self.model)
        return {"loss": out.loss, "logits": out.logits, "layer_infos": layer_infos}

    def generate(self, **kw):
        return self.model.generate(**kw)


def build_model(args) -> nn.Module:
    top_k = getattr(args, "top_k", 4)
    alpha_init = getattr(args, "alpha_init", 0.02)
    gate_bias_init = getattr(args, "gate_bias_init", 0.0)

    if args.task == "docmt":
        if args.model == "baseline":
            model = Seq2SeqBaseline(args.backbone)
            print(f"[model] Seq2SeqBaseline | backbone={args.backbone} | task={args.task}")
        elif args.model == "longattention":
            model = LongAttentionSeq2SeqModel(
                backbone_name=args.backbone,
                num_types=args.num_types,
                window_size=args.window_size,
                segment_size=args.segment_size,
                top_k=top_k,
                alpha_init=alpha_init,
                gate_bias_init=gate_bias_init,
            )
            print(f"[model] LongAttentionSeq2SeqModel | backbone={args.backbone} | task={args.task}")
            print(
                f"        window={args.window_size} segment={args.segment_size} "
                f"types={args.num_types} top_k={top_k}"
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

    elif args.task == "qa":
        if args.model == "baseline":
            model = BaselineModel(args.backbone, args.task)
            print(f"[model] BaselineModel | backbone={args.backbone} | task={args.task}")
        elif args.model == "longattention":
            model = LongAttentionQAModel(
                backbone_name=args.backbone,
                num_types=args.num_types,
                window_size=args.window_size,
                segment_size=args.segment_size,
                top_k=top_k,
                alpha_init=alpha_init,
                gate_bias_init=gate_bias_init,
            )
            print(f"[model] LongAttentionQAModel | backbone={args.backbone} | task={args.task}")
            print(
                f"        window={args.window_size} segment={args.segment_size} "
                f"types={args.num_types} top_k={top_k}"
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")
    else:
        raise ValueError(f"Unknown task: {args.task}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        params: {total:,} total, {trainable:,} trainable")
    return model
