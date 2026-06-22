import logging
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaForQuestionAnswering, RobertaForMaskedLM
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from ...core.attention_core import LongAttention

logger = logging.getLogger(__name__)

def _extend_roberta_position_embeddings(model: nn.Module, new_max_length: int) -> None:
    embeddings = model.embeddings
    old_pos_embed = embeddings.position_embeddings
    old_max_pos, hidden_size = old_pos_embed.weight.shape
    target_max_pos = new_max_length + 2
    if target_max_pos <= old_max_pos: return

    logger.info(f"Extending RoBERTa position embeddings: {old_max_pos} -> {target_max_pos}")
    new_pos_embed = nn.Embedding(target_max_pos, hidden_size, padding_idx=old_pos_embed.padding_idx)
    new_pos_embed.weight.data.normal_(mean=0.0, std=0.02)
    with torch.no_grad():
        new_pos_embed.weight.data[:old_max_pos] = old_pos_embed.weight.data
        n_repeats = (target_max_pos - 2) // (old_max_pos - 2) + 1
        extended = old_pos_embed.weight.data[2:].repeat(n_repeats, 1)
        new_pos_embed.weight.data[old_max_pos:] = extended[: target_max_pos - old_max_pos]
    embeddings.position_embeddings = new_pos_embed
    embeddings.register_buffer("position_ids", torch.arange(target_max_pos).expand((1, -1)), persistent=False)
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


class RobertaLongAttentionForQA(RobertaPreTrainedModel):
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

    def inject_long_attention(self, config: Dict[str, Any]):
        local_window_size = config.get("local_window_size", 512)
        top_k = config.get("top_k", 64)
        num_types = config.get("num_types", 3)
        bottleneck_ratio = config.get("bottleneck_ratio", 0.25)
        dropout_prob = config.get("dropout_prob", 0.1)
        max_length = config.get("max_length", 4096)
        
        _extend_roberta_position_embeddings(self.roberta, max_length)
        self.config.max_position_embeddings = max_length + 2
        
        hidden_size = self.config.hidden_size
        layers = self.roberta.encoder.layer
        num_layers = len(layers)
        
        # Pre-cache các lớp attention gốc để tránh lỗi AttributeError khi thay thế in-place
        orig_attns = [layer.attention.self for layer in layers]
        
        replaced = 0
        for idx, layer in enumerate(layers):
            orig_attn = orig_attns[idx]
            
            try:
                device = next(orig_attn.parameters()).device
                dtype = next(orig_attn.parameters()).dtype
            except StopIteration:
                device = torch.device("cpu")
                dtype = torch.float32

            new_attn = LongAttention(
                hidden_size=hidden_size,
                num_heads=self.config.num_attention_heads,
                local_window_size=local_window_size,
                top_k=top_k,
                num_types=num_types,
                bottleneck_ratio=bottleneck_ratio,
                dropout_prob=dropout_prob,
                layer_idx=idx,
                bias=True,
            ).to(device=device, dtype=dtype)
            
            with torch.no_grad():
                new_attn.local_attention.q_proj.weight.copy_(orig_attn.query.weight)
                new_attn.local_attention.k_proj.weight.copy_(orig_attn.key.weight)
                new_attn.local_attention.v_proj.weight.copy_(orig_attn.value.weight)
                new_attn.local_attention.q_proj.bias.copy_(orig_attn.query.bias)
                new_attn.local_attention.k_proj.bias.copy_(orig_attn.key.bias)
                new_attn.local_attention.v_proj.bias.copy_(orig_attn.value.bias)

                # Cross-Layer Weight Inheritance (CLWI) cho RoBERTa 12 tầng
                orig_attn_0 = orig_attns[idx]
                orig_attn_1 = orig_attns[(idx + num_layers // 3) % num_layers]
                orig_attn_2 = orig_attns[(idx + 2 * (num_layers // 3)) % num_layers]

                q_w = new_attn.typed_retrieval.q_proj.weight
                q_w.data[0:hidden_size].copy_(orig_attn_0.query.weight)
                q_w.data[hidden_size:2*hidden_size].copy_(orig_attn_1.query.weight)
                q_w.data[2*hidden_size:3*hidden_size].copy_(orig_attn_2.query.weight)
                
                if orig_attn_0.query.bias is not None:
                    q_b = new_attn.typed_retrieval.q_proj.bias
                    q_b.data[0:hidden_size].copy_(orig_attn_0.query.bias)
                    q_b.data[hidden_size:2*hidden_size].copy_(orig_attn_1.query.bias)
                    q_b.data[2*hidden_size:3*hidden_size].copy_(orig_attn_2.query.bias)
                    
                kv_w = new_attn.typed_gist.multi_type_proj.weight
                kv_chunk_size = 2 * hidden_size
                
                kv_temp_0 = torch.cat([orig_attn_0.key.weight, orig_attn_0.value.weight], dim=0)
                kv_temp_1 = torch.cat([orig_attn_1.key.weight, orig_attn_1.value.weight], dim=0)
                kv_temp_2 = torch.cat([orig_attn_2.key.weight, orig_attn_2.value.weight], dim=0)
                
                kv_w.data[0:kv_chunk_size].copy_(kv_temp_0)
                kv_w.data[kv_chunk_size:2*kv_chunk_size].copy_(kv_temp_1)
                kv_w.data[2*kv_chunk_size:3*kv_chunk_size].copy_(kv_temp_2)
                
                if orig_attn_0.key.bias is not None:
                    kv_b = new_attn.typed_gist.multi_type_proj.bias
                    kv_b_temp_0 = torch.cat([orig_attn_0.key.bias, orig_attn_0.value.bias], dim=0)
                    kv_b_temp_1 = torch.cat([orig_attn_1.key.bias, orig_attn_1.value.bias], dim=0)
                    kv_b_temp_2 = torch.cat([orig_attn_2.key.bias, orig_attn_2.value.bias], dim=0)
                    
                    kv_b.data[0:kv_chunk_size].copy_(kv_b_temp_0)
                    kv_b.data[kv_chunk_size:2*kv_chunk_size].copy_(kv_b_temp_1)
                    kv_b.data[2*kv_chunk_size:3*kv_chunk_size].copy_(kv_b_temp_2)
                
                new_attn.out_proj.weight.copy_(layer.attention.output.dense.weight)
                new_attn.out_proj.bias.copy_(layer.attention.output.dense.bias)

            class LongAttnWrapper(nn.Module):
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
                        custom_mask.masked_fill_(is_pad, float("-inf"))
                        # Expand to (B, 1, 1, T) for broadcasting with (1, 1, T, T) local mask
                        custom_mask = custom_mask.unsqueeze(1).unsqueeze(2)
                    outputs = self.attn(hidden_states, attention_mask=custom_mask)
                    return (outputs[0],) + outputs[1:]

            layer.attention.self = LongAttnWrapper(new_attn)
            layer.attention.output.dense = nn.Identity()
            replaced += 1
        logger.info(f"Injected LongAttention into {replaced} RoBERTa layers.")

    def forward(self, input_ids=None, attention_mask=None, start_positions=None, end_positions=None, answer_types=None, labels=None, **kwargs):
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
        yes_no_logits = self.yes_no_classifier(sequence_output[:, 0, :])

        if start_positions is not None and end_positions is not None and labels is None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            span_loss = (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions)) / 2
            # When entire batch is Yes/No (all targets=-1), span_loss is NaN (0/0). Use 0 instead.
            if torch.isnan(span_loss):
                span_loss = torch.tensor(0.0, dtype=start_logits.dtype, device=start_logits.device, requires_grad=True)
            total_loss = span_loss
            if answer_types is not None:
                total_loss = total_loss + 1.0 * nn.CrossEntropyLoss()(yes_no_logits, answer_types)

        div_loss, g_val, cnt = 0.0, 0.0, 0
        for layer in self.roberta.encoder.layer:
            attn = layer.attention.self.attn
            if hasattr(attn, 'compute_orthogonality_loss'):
                div_loss += attn.compute_orthogonality_loss()
                g_val += attn.last_gate_val
                cnt += 1
            elif hasattr(attn, 'last_diversity_loss'):
                div_loss += attn.last_diversity_loss
                g_val += attn.last_gate_val
                cnt += 1
        return MultiTaskQAOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, yes_no_logits=yes_no_logits, diversity_loss=div_loss/max(1,cnt), gate_val=g_val/max(1,cnt))


def build_qa_long_attention_model(backbone: str = "roberta-base", config: Dict[str, Any] = None):
    base_model = RobertaForQuestionAnswering.from_pretrained(backbone)
    mlm_model = RobertaForMaskedLM.from_pretrained(backbone)
    model = RobertaLongAttentionForQA(base_model.config)
    model.roberta.load_state_dict(base_model.roberta.state_dict())
    model.qa_outputs.load_state_dict(base_model.qa_outputs.state_dict())
    model.lm_head.load_state_dict(mlm_model.lm_head.state_dict(), strict=False)
    
    # Tie LM head weights to word embeddings (Standard in RoBERTa)
    model.lm_head.decoder.weight = model.roberta.embeddings.word_embeddings.weight
    
    return model
