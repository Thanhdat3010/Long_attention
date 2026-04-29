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

        # We construct a dense attention matrix but mask it to simulate Longformer exactly.
        # This is memory-intensive for >4096 but fits well for our 4096 max_length on A100/RTX3090.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_global = torch.matmul(Q, K_g.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Base sliding window mask
        device = hidden_states.device
        idx = torch.arange(T, device=device)
        distance = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        local_mask = (distance <= (self.window_size // 2))
        
        # Expand masks for batch and heads
        final_scores = scores.clone()
        final_V = V.clone()
        
        # Apply Longformer logic:
        # If token i is global, it attends to all token j.
        # If token j is global, it is attended by all token i (using K_g).
        # Otherwise, use local window.
        if attention_mask is not None:
            # attention_mask: B x T.  1 = global, 0 = local, < 0 = padding
            is_global = (attention_mask > 0).float()
            is_pad = (attention_mask < -1).float()
            
            for b in range(B):
                # Global indices for this batch item
                g_idx = torch.nonzero(is_global[b]).squeeze(-1)
                
                # Copy global scores into final scores for global keys
                final_scores[b, :, :, g_idx] = scores_global[b, :, :, g_idx]
                
                # Combine masks
                b_mask = local_mask.clone()
                b_mask[g_idx, :] = True # Global tokens attend to everything
                b_mask[:, g_idx] = True # Everything attends to global tokens
                
                # Apply padding mask
                pad_idx = torch.nonzero(is_pad[b]).squeeze(-1)
                b_mask[:, pad_idx] = False
                
                # Mask out invalid positions
                final_scores[b] = final_scores[b].masked_fill(~b_mask.unsqueeze(0), float("-inf"))
        else:
            final_scores = final_scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(final_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # For Values, if key j is global, we must use V_g for that column
        # To do this cleanly, we can blend V and V_g based on the global mask
        if attention_mask is not None:
            # V: (B, H, T, D)
            blend_mask = (attention_mask > 0).view(B, 1, T, 1).expand(-1, self.num_heads, -1, self.head_dim)
            mixed_V = torch.where(blend_mask, V_g, V)
            context = torch.matmul(attn_weights, mixed_V)
        else:
            context = torch.matmul(attn_weights, V)
            
        context = context.transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(context)
        
        return output, (attn_weights if output_attentions else None)
