import logging
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaForQuestionAnswering, RobertaForMaskedLM
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from .longformer_attention import LongformerSelfAttention

logger = logging.getLogger(__name__)


def _extend_roberta_position_embeddings(model: nn.Module, new_max_length: int) -> None:
    embeddings = model.embeddings
    old_pos_embed = embeddings.position_embeddings
    old_max_pos, hidden_size = old_pos_embed.weight.shape
    
    target_max_pos = new_max_length + 2
    if target_max_pos <= old_max_pos:
        return

    logger.info(f"Extending RoBERTa position embeddings: {old_max_pos} -> {target_max_pos}")
    new_pos_embed = nn.Embedding(target_max_pos, hidden_size, padding_idx=old_pos_embed.padding_idx)
    new_pos_embed.weight.data.normal_(mean=0.0, std=0.02) # Standard initializer_range
    
    # Copy old weights
    with torch.no_grad():
        new_pos_embed.weight.data[:old_max_pos] = old_pos_embed.weight.data
        
        # Copy/repeat strategy for the rest
        n_repeats = (target_max_pos - 2) // (old_max_pos - 2) + 1
        extended = old_pos_embed.weight.data[2:].repeat(n_repeats, 1)
        new_pos_embed.weight.data[old_max_pos:] = extended[: target_max_pos - old_max_pos]
    
    embeddings.position_embeddings = new_pos_embed
    embeddings.register_buffer(
        "position_ids", torch.arange(target_max_pos).expand((1, -1)), persistent=False
    )
    # CRITICAL: Also extend token_type_ids buffer to match new size
    if hasattr(embeddings, "token_type_ids"):
        embeddings.register_buffer(
            "token_type_ids", torch.zeros((1, target_max_pos), dtype=torch.long), persistent=False
        )


@dataclass
class MultiTaskQAOutput(QuestionAnsweringModelOutput):
    yes_no_logits: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None
    gate_val: Optional[torch.FloatTensor] = None


class RobertaLongformerForQA(RobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.yes_no_classifier = nn.Linear(config.hidden_size, 3)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def inject_longformer_attention(self, config: Dict[str, Any]):
        window_size = config.get("local_window_size", 512)
        dropout_prob = config.get("dropout_prob", 0.1)
        max_length = config.get("max_length", 4096)
        
        _extend_roberta_position_embeddings(self.roberta, max_length)
        self.config.max_position_embeddings = max_length + 2
        
        replaced = 0
        for layer in self.roberta.encoder.layer:
            orig_attn = layer.attention.self
            hidden_size = self.config.hidden_size
            num_heads = self.config.num_attention_heads
            
            new_attn = LongformerSelfAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                window_size=window_size,
                dropout_prob=dropout_prob,
            )
            
            with torch.no_grad():
                new_attn.q_proj.weight.copy_(orig_attn.query.weight)
                new_attn.k_proj.weight.copy_(orig_attn.key.weight)
                new_attn.v_proj.weight.copy_(orig_attn.value.weight)
                new_attn.q_proj.bias.copy_(orig_attn.query.bias)
                new_attn.k_proj.bias.copy_(orig_attn.key.bias)
                new_attn.v_proj.bias.copy_(orig_attn.value.bias)
                
                new_attn.k_proj_global.weight.copy_(orig_attn.key.weight)
                new_attn.k_proj_global.weight.data += torch.randn_like(orig_attn.key.weight) * 1e-2
                new_attn.v_proj_global.weight.copy_(orig_attn.value.weight)
                new_attn.v_proj_global.weight.data += torch.randn_like(orig_attn.value.weight) * 1e-2
                new_attn.k_proj_global.bias.copy_(orig_attn.key.bias)
                new_attn.v_proj_global.bias.copy_(orig_attn.value.bias)

            class AttnWrapper(nn.Module):
                def __init__(self, attn):
                    super().__init__()
                    self.attn = attn
                def forward(self, hidden_states, attention_mask=None, **kwargs):
                    custom_mask = None
                    if attention_mask is not None:
                        # HF mask can be (B,1,1,T) or (B,1,T,T) — extract (B,T)
                        if attention_mask.dim() == 4:
                            mask_2d = attention_mask[:, 0, 0, :]  # (B, T)
                        elif attention_mask.dim() == 3:
                            mask_2d = attention_mask[:, 0, :]
                        else:
                            mask_2d = attention_mask
                        is_pad = (mask_2d < -100)
                        B, T = hidden_states.shape[:2]
                        custom_mask = torch.zeros((B, T), device=hidden_states.device)
                        custom_mask.masked_fill_(is_pad, -10000.0)
                        custom_mask[:, 0] = 1.0  # Global <s>
                    
                    out, attn_weights = self.attn(hidden_states, attention_mask=custom_mask)
                    return (out, attn_weights) if kwargs.get("output_attentions") else (out,)
            
            layer.attention.self = AttnWrapper(new_attn)
            replaced += 1
        logger.info(f"Injected Longformer Attention into {replaced} RoBERTa layers.")

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, start_positions=None, end_positions=None, answer_types=None, labels=None, **kwargs):
        if global_attention_mask is not None and attention_mask is not None:
            custom_mask = attention_mask.clone().float()
            custom_mask[attention_mask == 0] = -10000.0
            custom_mask[global_attention_mask == 1] = 1.0
            attention_mask = custom_mask

        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        
        total_loss = None
        if labels is not None:
            prediction_scores = self.lm_head(sequence_output)
            loss_fct = nn.CrossEntropyLoss()
            total_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        cls_output = sequence_output[:, 0, :]
        yes_no_logits = self.yes_no_classifier(cls_output)

        if start_positions is not None and end_positions is not None and labels is None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            span_loss = (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions)) / 2
            # When entire batch is Yes/No (all targets=-1), span_loss is NaN (0/0). Use 0 instead.
            if torch.isnan(span_loss):
                span_loss = torch.tensor(0.0, dtype=start_logits.dtype, device=start_logits.device, requires_grad=True)
            total_loss = span_loss
            if answer_types is not None:
                total_loss = total_loss + 0.5 * nn.CrossEntropyLoss()(yes_no_logits, answer_types)

        return MultiTaskQAOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, yes_no_logits=yes_no_logits)


def build_qa_longformer_model(backbone: str = "roberta-base", config: Dict[str, Any] = None):
    # Load standard model to get weights safely
    base_model = RobertaForQuestionAnswering.from_pretrained(backbone)
    mlm_model = RobertaForMaskedLM.from_pretrained(backbone)
    
    # Create our model with same config
    model = RobertaLongformerForQA(base_model.config)
    # Copy weights
    model.roberta.load_state_dict(base_model.roberta.state_dict())
    model.qa_outputs.load_state_dict(base_model.qa_outputs.state_dict())
    model.lm_head.load_state_dict(mlm_model.lm_head.state_dict(), strict=False)
    
    # Tie LM head weights to word embeddings (Standard in RoBERTa)
    model.lm_head.decoder.weight = model.roberta.embeddings.word_embeddings.weight
    
    return model
