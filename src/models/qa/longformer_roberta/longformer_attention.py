import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LongformerSelfAttention(nn.Module):
    """
    Custom implementation of Longformer Attention (Sliding Window + Global Tokens).
    Implemented via explicit masking for exact comparison without relying on 
    pretrained Longformer weights or custom CUDA kernels.
    
    Fully vectorized (no per-batch for-loop) to ensure correct gradient flow
    and better GPU utilization.
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
        self.window_size = window_size
        self.head_dim = hidden_size // num_heads
        
        if self.head_dim * num_heads != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")

        # Standard projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Global projections (Longformer uses separate K, V for global tokens)
        self.k_proj_global = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj_global = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # 1: global, 0: local, -inf: pad
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = hidden_states.shape
        
        # Project standard Q, K, V
        Q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project Global K, V
        K_g = self.k_proj_global(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V_g = self.v_proj_global(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores for both local and global paths
        scale = math.sqrt(self.head_dim)
        scores_local = torch.matmul(Q, K.transpose(-2, -1)) / scale    # (B, H, T, T)
        scores_global = torch.matmul(Q, K_g.transpose(-2, -1)) / scale # (B, H, T, T)
        
        # Base sliding window mask: (T, T) boolean
        device = hidden_states.device
        idx = torch.arange(T, device=device)
        distance = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        local_mask = (distance <= (self.window_size // 2))  # (T, T)
        
        if attention_mask is not None:
            # attention_mask: (B, T).  1 = global, 0 = local, < -1 = padding
            is_global = (attention_mask > 0)   # (B, T) bool
            is_pad = (attention_mask < -1)     # (B, T) bool
            
            # ---------------------------------------------------------------
            # VECTORIZED Longformer logic (no for-loop, no in-place ops)
            # ---------------------------------------------------------------
            
            # 1. Select scores: use global K scores for global keys, local K scores otherwise
            #    global_key_mask: (B, 1, 1, T) — broadcasts over (H, T_query)
            global_key_mask = is_global.view(B, 1, 1, T)
            combined_scores = torch.where(global_key_mask, scores_global, scores_local)
            
            # 2. Build attention allow mask:
            #    can_attend[b, i, j] = True iff:
            #      - j is NOT padding, AND
            #      - (i is global OR j is global OR |i-j| <= window_size//2)
            global_q_mask = is_global.view(B, 1, T, 1)   # broadcasts over T_key
            pad_k_mask = is_pad.view(B, 1, 1, T)          # broadcasts over T_query
            
            can_attend = ~pad_k_mask & (
                global_q_mask | global_key_mask | local_mask.unsqueeze(0).unsqueeze(0)
            )
            
            final_scores = combined_scores.masked_fill(~can_attend, float("-inf"))
            
            # 3. Blend Values: use V_g for global keys, V for local keys
            #    global_v_mask: (B, 1, T, 1) — broadcasts over (H, d_k)
            global_v_mask = is_global.view(B, 1, T, 1)
            mixed_V = torch.where(global_v_mask, V_g, V)
        else:
            # No attention mask: pure local sliding window
            final_scores = scores_local.masked_fill(
                ~local_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            mixed_V = V

        attn_weights = F.softmax(final_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, mixed_V)
        context = context.transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(context)
        
        return output, (attn_weights if output_attentions else None)
