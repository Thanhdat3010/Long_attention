"""
LED Attention: Sliding Window with Global Tokens.
A BART-compatible implementation of Longformer-style attention for LED experiments.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

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

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
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
        
        Q = self._split_heads(self.q_proj(hidden_states))
        K = self._split_heads(self.k_proj(hidden_states))
        V = self._split_heads(self.v_proj(hidden_states))

        # Standard Attention Scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # (B, H, T, T)

        # 1. Sliding Window Mask
        # We manually build the mask for simplicity in this baseline
        idx = torch.arange(T, device=device)
        row = idx.unsqueeze(1)
        col = idx.unsqueeze(0)
        # causal window: j <= i AND i - j < W
        # LED usually uses a symmetric window in encoder, but BART can be causal.
        # Proposal says "Sliding window of LED", which is usually symmetric in encoder.
        mask_window = (row - col).abs() < (self.window_size // 2)
        
        # 2. Global Attention Mask
        # If global_attention_mask is not provided, we default to the first token (BOS)
        if global_attention_mask is None:
            global_attention_mask = torch.zeros(B, T, device=device)
            global_attention_mask[:, 0] = 1 # <s> is global
            
        # Global tokens can see everything, and everything can see global tokens
        # mask = window_mask OR (row is global) OR (col is global)
        is_global = global_attention_mask.bool() # (B, T)
        
        # Expand masks to (B, 1, T, T)
        final_mask = mask_window.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) # (B, 1, T, T)
        final_mask = final_mask | is_global.unsqueeze(1).unsqueeze(-1) # Row is global
        final_mask = final_mask | is_global.unsqueeze(1).unsqueeze(-2) # Col is global

        # Apply mask
        scores = scores.masked_fill(~final_mask, float("-inf"))

        # External padding mask
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = self._merge_heads(context)
        output = self.out_proj(context)

        # BART expects a 3-tuple: (output, attn_weights, past_key_value)
        output_attentions = kwargs.get("output_attentions", False)
        return (output, attn_weights if output_attentions else None, None)
