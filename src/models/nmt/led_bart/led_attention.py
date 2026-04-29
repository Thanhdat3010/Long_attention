"""
LED Attention: Sliding Window with Global Tokens.

A BART-compatible implementation of Longformer-style attention for LED experiments.
Algorithm faithfully ported from Google Research's Long Range Arena (LRA) repo:
https://github.com/google-research/long-range-arena/blob/master/lra_benchmarks/models/longformer/longformer_attention.py

Key design (matching LRA):
  1. Two separate sets of Q/K/V projections: one for sliding-window, one for global.
  2. Two independent attention computations (local path + global path).
  3. Final output merged via torch.where based on global_attention_mask.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import transformers
from packaging import version

# Transformers >= 4.36 changed BartEncoderLayer signature from 3-tuple to 2-tuple return expectation
TRANSFORMERS_NEW_SIG = version.parse(transformers.__version__) >= version.parse("4.36.0")

class LEDSelfAttention(nn.Module):
    """
    LED (Longformer Encoder-Decoder) Style Attention.
    Combines local sliding window attention with a few global tokens.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 512,
        dropout_prob: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.head_dim)

        # Local Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Global Projections (Official LED Component)
        self.q_proj_global = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj_global = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj_global = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.attn_dropout = nn.Dropout(p=dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * dk)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        global_attention_mask: (B, T) where 1 indicates global attention.
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        
        # Default global mask if not provided (BART sets None by default)
        if global_attention_mask is None:
            global_attention_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
            global_attention_mask[:, 0] = True # <s> is global token
        else:
            global_attention_mask = global_attention_mask.bool()

        # 1. Compute Local and Global Projections
        Q_local = self._split_heads(self.q_proj(hidden_states))
        K_local = self._split_heads(self.k_proj(hidden_states))
        V_local = self._split_heads(self.v_proj(hidden_states))
        
        Q_global = self._split_heads(self.q_proj_global(hidden_states))
        K_global = self._split_heads(self.k_proj_global(hidden_states))
        V_global = self._split_heads(self.v_proj_global(hidden_states))
        
        # 2. Local Path (Sliding Window + Global Tokens access)
        # Note: In LRA, non-global tokens see global tokens via Local Projections (K_local)
        scores_local = torch.matmul(Q_local, K_local.transpose(-2, -1)) / self.scale
        
        idx = torch.arange(T, device=device)
        mask_window = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= (self.window_size // 2)
        is_global = global_attention_mask # (B, T)
        
        # Final Local Mask: Window OR anyone-sees-global
        final_mask_local = mask_window.unsqueeze(0).unsqueeze(0) | is_global.unsqueeze(1).unsqueeze(-2)
        scores_local = scores_local.masked_fill(~final_mask_local, float("-inf"))
        
        if attention_mask is not None:
            scores_local = scores_local + attention_mask
        
        attn_local = F.softmax(scores_local, dim=-1)
        attn_local = torch.nan_to_num(attn_local, nan=0.0)
        out_local = torch.matmul(self.attn_dropout(attn_local), V_local)
        
        # 3. Global Path (Global tokens see everything)
        # Global tokens use Q_global to see everyone via K_global
        scores_global = torch.matmul(Q_global, K_global.transpose(-2, -1)) / self.scale
        
        # Global mask: only global tokens' rows are active, but they see all columns
        # However, LRA's _get_attention_result for global uses full_global_mask
        # which allows global-to-all and all-to-global.
        # But since we merge at the end, only the rows of global tokens matter here.
        if attention_mask is not None:
            scores_global = scores_global + attention_mask
            
        attn_global = F.softmax(scores_global, dim=-1)
        attn_global = torch.nan_to_num(attn_global, nan=0.0)
        out_global = torch.matmul(self.attn_dropout(attn_global), V_global)
        
        # 4. Merge based on global_attention_mask (Like LRA line 266)
        # (B, H, T, d_k)
        is_global_expanded = is_global.unsqueeze(1).unsqueeze(-1) 
        out = torch.where(is_global_expanded, out_global, out_local)
        
        out = self._merge_heads(out)
        output = self.out_proj(out)

        # BART expects a 3-tuple: (output, attn_weights, past_key_value)
        output_attentions = kwargs.get("output_attentions", False)
        # For output_attentions, we provide the merged weights
        merged_weights = torch.where(is_global_expanded, attn_global, attn_local)

        if TRANSFORMERS_NEW_SIG:
            return (output, merged_weights if output_attentions else None)
        else:
            return (output, merged_weights if output_attentions else None, None)
