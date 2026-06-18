import logging
import weakref
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
    """Extend RoBERTa's position embeddings from 514 to new_max_length + 2 using copy/repeat."""
    embeddings = model.embeddings
    old_pos_embed = embeddings.position_embeddings
    old_max_pos, hidden_size = old_pos_embed.weight.shape
    
    target_max_pos = new_max_length + 2
    if target_max_pos <= old_max_pos:
        return

    logger.info(f"Extending RoBERTa position embeddings: {old_max_pos} -> {target_max_pos}")
    new_pos_embed = nn.Embedding(target_max_pos, hidden_size, padding_idx=old_pos_embed.padding_idx)
    new_pos_embed.weight.data.normal_(mean=0.0, std=0.02)
    
    with torch.no_grad():
        new_pos_embed.weight.data[:old_max_pos] = old_pos_embed.weight.data
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
    _tied_weights_keys = {
        "lm_head.decoder.weight": "roberta.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.yes_no_classifier = nn.Linear(config.hidden_size, 3)
        # Temporary storage for global_attention_mask (accessed by AttnWrapper via weakref)
        self._current_global_attention_mask = None
        self._log_mask_once = True
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
        
        # Store weak reference to self for AttnWrapper to access global_attention_mask
        # weakref avoids circular reference issues with nn.Module tree and pickling
        model_ref = weakref.ref(self)
        
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
                # Copy Q, K, V weights from pretrained RoBERTa
                new_attn.q_proj.weight.copy_(orig_attn.query.weight)
                new_attn.k_proj.weight.copy_(orig_attn.key.weight)
                new_attn.v_proj.weight.copy_(orig_attn.value.weight)
                new_attn.q_proj.bias.copy_(orig_attn.query.bias)
                new_attn.k_proj.bias.copy_(orig_attn.key.bias)
                new_attn.v_proj.bias.copy_(orig_attn.value.bias)
                
                # Global K, V initialized from pretrained + small noise
                new_attn.k_proj_global.weight.copy_(orig_attn.key.weight)
                new_attn.k_proj_global.weight.data += torch.randn_like(orig_attn.key.weight) * 1e-2
                new_attn.v_proj_global.weight.copy_(orig_attn.value.weight)
                new_attn.v_proj_global.weight.data += torch.randn_like(orig_attn.value.weight) * 1e-2
                new_attn.k_proj_global.bias.copy_(orig_attn.key.bias)
                new_attn.v_proj_global.bias.copy_(orig_attn.value.bias)
                
                # ---------------------------------------------------------------
                # FIX BUG 2: Copy output.dense pretrained weights → out_proj
                # ---------------------------------------------------------------
                # Previously out_proj was RANDOMLY initialized while output.dense
                # still had pretrained weights. Signal went through:
                #   attention → out_proj (RANDOM) → output.dense (pretrained)
                # Two linear layers back-to-back, with the random one in the
                # middle corrupting all pretrained representations.
                # Fix: Move pretrained weights to out_proj, replace dense with Identity.
                # ---------------------------------------------------------------
                new_attn.out_proj.weight.copy_(layer.attention.output.dense.weight)
                new_attn.out_proj.bias.copy_(layer.attention.output.dense.bias)

            # -------------------------------------------------------------------
            # FIX BUG 1: Global Attention Mask Recovery via Weakref
            # -------------------------------------------------------------------
            # Problem: RoBERTa's get_extended_attention_mask() converts ALL positive
            # mask values to 0.0 and negative to -inf. This destroys the distinction
            # between global tokens (question) and local tokens (context) — both
            # become 0.0. The old AttnWrapper could only hardcode token[0] as global,
            # losing ALL question tokens' global status.
            #
            # With window_size=512, context tokens beyond position ~256 could NOT
            # see the question at all. The model had no idea what to answer.
            #
            # Solution: Store global_attention_mask on the model BEFORE calling
            # self.roberta(), and have AttnWrapper retrieve it via weakref.
            # This completely bypasses get_extended_attention_mask for global info.
            # -------------------------------------------------------------------
            class AttnWrapper(nn.Module):
                def __init__(self, attn, get_model):
                    super().__init__()
                    self.attn = attn
                    # Use object.__setattr__ to avoid nn.Module registering
                    # the weakref callable as a submodule
                    object.__setattr__(self, '_get_model', get_model)
                    object.__setattr__(self, '_logged', False)
                    
                def forward(self, hidden_states, attention_mask=None, **kwargs):
                    model = self._get_model()
                    global_mask = getattr(model, "_current_global_attention_mask", None) if model else None
                    
                    B, T = hidden_states.shape[:2]
                    custom_mask = None
                    
                    # Even if attention_mask is None (e.g. optimized away by HF because no padding),
                    # we MUST build a custom_mask if we have a global_mask.
                    if attention_mask is not None or global_mask is not None:
                        custom_mask = torch.zeros((B, T), device=hidden_states.device)
                        
                        if attention_mask is not None:
                            # HF extended mask: (B,1,1,T) or (B,1,T,T) — extract (B,T)
                            if attention_mask.dim() == 4:
                                mask_2d = attention_mask[:, 0, 0, :]
                            elif attention_mask.dim() == 3:
                                mask_2d = attention_mask[:, 0, :]
                            else:
                                mask_2d = attention_mask
                            
                            is_pad = (mask_2d < -100)
                            custom_mask.masked_fill_(is_pad, -10000.0)
                        
                        # FIX: Restore global token info from saved mask on model
                        if global_mask is not None:
                            custom_mask[global_mask == 1] = 1.0
                        else:
                            # Fallback (e.g. during SPT when no global_attention_mask):
                            # only <s> token is global
                            custom_mask[:, 0] = 1.0
                        
                        # Diagnostic logging (first call only to avoid log spam)
                        if not self._logged:
                            n_global = (custom_mask > 0.5).sum(dim=-1).float().mean().item()
                            n_local = ((custom_mask > -1) & (custom_mask < 0.5)).sum(dim=-1).float().mean().item()
                            n_pad = is_pad.sum(dim=-1).float().mean().item()
                            logger.info(
                                f"[AttnWrapper] Mask stats (first call): "
                                f"Global={n_global:.0f}, Local={n_local:.0f}, Pad={n_pad:.0f}, "
                                f"SeqLen={T}, has_global_mask={global_mask is not None}"
                            )
                            object.__setattr__(self, '_logged', True)
                    
                    out, attn_weights = self.attn(hidden_states, attention_mask=custom_mask)
                    # RoBERTa 5.x always unpacks: output, attn_weights = self.self(...)
                    return out, (attn_weights if kwargs.get("output_attentions") else None)
            
            layer.attention.self = AttnWrapper(new_attn, model_ref)
            
            # FIX BUG 2: Replace output.dense with Identity
            # Pretrained weights already moved to out_proj above.
            # Keeping output.dense would apply TWO linear projections back-to-back.
            layer.attention.output.dense = nn.Identity()
            replaced += 1
            
        logger.info(f"Injected Longformer Attention into {replaced} RoBERTa layers.")
        logger.info(f"  Window size: {window_size}, Max length: {max_length}")
        logger.info(f"  output.dense → nn.Identity() (pretrained weights moved to out_proj)")

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None,
                start_positions=None, end_positions=None, answer_types=None,
                labels=None, **kwargs):
        # -------------------------------------------------------------------
        # FIX BUG 1: Pass standard binary attention_mask (0/1) to RoBERTa.
        # Store global_attention_mask separately for AttnWrappers to access.
        #
        # OLD (broken): Mixed global info INTO attention_mask values (1.0 for
        # both global and attend), which get_extended_attention_mask() converted
        # ALL to 0.0 — destroying the global/local distinction entirely.
        #
        # NEW (fixed): Keep attention_mask as pure 0/1 binary. Store
        # global_attention_mask as a temporary attribute that AttnWrapper reads
        # via weakref, completely bypassing get_extended_attention_mask.
        # -------------------------------------------------------------------
        self._current_global_attention_mask = global_attention_mask
        try:
            outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        finally:
            self._current_global_attention_mask = None  # Clean up to prevent memory leak
        
        sequence_output = outputs[0]
        
        # Diagnostic logging (once per training run)
        if self._log_mask_once:
            if global_attention_mask is not None:
                n_global = (global_attention_mask == 1).sum(dim=-1).float().mean().item()
                n_attend = (attention_mask == 1).sum(dim=-1).float().mean().item() if attention_mask is not None else 0
                logger.info(
                    f"[QA Forward] input_shape={list(input_ids.shape)}, "
                    f"avg_attend_tokens={n_attend:.0f}, avg_global_tokens={n_global:.1f}"
                )
            else:
                logger.info(
                    f"[QA Forward] input_shape={list(input_ids.shape)}, "
                    f"global_attention_mask=None (SPT mode)"
                )
            self._log_mask_once = False
        
        total_loss = None
        if labels is not None:
            # MLM (SPT) mode
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
