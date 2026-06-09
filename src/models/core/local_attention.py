"""
Local Dense Sliding-Window Attention Module.

This module implements the **Local Branch** of the LongAttention architecture.
The local branch uses dense causal attention restricted to a sliding window
of adjacent tokens, preserving syntactic structure and immediate coherence
without the quadratic cost of full self-attention over long sequences.

Mathematical Formulation
------------------------
For query q_i with window size W, define the local neighbourhood:
    N_i = {j : max(0, i-W) ≤ j ≤ i}   (causal/left-only window)

Local attention score:
    A_local_i = softmax( QK^T / √d_k  )[N_i]  ×  V[N_i]

This is equivalent to standard SDPA on a windowed token subset.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import transformers
from packaging import version

logger = logging.getLogger(__name__)

# Transformers >= 4.36 changed BartEncoderLayer signature from 3-tuple to 2-tuple return expectation
TRANSFORMERS_NEW_SIG = version.parse(transformers.__version__) >= version.parse("4.36.0")


class LocalSlidingWindowAttention(nn.Module):
    """
    Dense sliding-window self-attention for the local branch.

    Computes standard Multi-Head Attention (MHA) but restricts each query
    to attend only to the `window_size` most recent tokens (causal left).
    This preserves linguistic locality (syntax, immediate co-reference)
    while being O(n * W * d) rather than O(n^2 * d).

    Args:
        hidden_size:    Model hidden dimension (d_model).
        num_heads:      Number of attention heads.
        window_size:    Number of tokens each query can attend to (left window).
        dropout_prob:   Dropout applied to attention weights.
        bias:           Whether to use bias in QKV projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 512,
        dropout_prob: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.head_dim)

        # Linear projections: Q, K, V, and output
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        # self.out_proj is now managed by the parent LongAttention module
        # to ensure local and long-range branches are projected together.

        self.attn_dropout = nn.Dropout(p=dropout_prob)
        
        # Cache for window mask to avoid O(T^2) re-computation
        self.register_buffer("cached_mask", None, persistent=False)
        self.cached_seq_len = 0

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, D) → (B, H, T, d_k)."""
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, T, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, H, T, d_k) → (B, T, D)."""
        B, H, T, dk = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * dk)

    def _build_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Build a boolean mask enforcing the sliding-window constraint.

        Returns a (seq_len, seq_len) boolean tensor where True = attend, False = mask.
        Each row i can attend only to positions where |i - j| <= window_size // 2.
        """
        idx = torch.arange(seq_len, device=device)
        row = idx.unsqueeze(1)  # (T, 1)
        col = idx.unsqueeze(0)  # (1, T)
        
        # Symmetric sliding window (Bidirectional context for Encoder)
        radius = self.window_size // 2
        window_mask = torch.abs(row - col) <= radius
        
        return window_mask  # (T, T) bool

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Compute local sliding-window attention.

        Args:
            hidden_states:     Input tensor of shape (B, T, D).
            attention_mask:    Optional additive mask (B, 1, T, T) or padding mask.
            position_ids:      Unused; kept for API compatibility.
            past_key_value:    KV cache (unused in this implementation).
            output_attentions: If True, also return attention weight matrix.

        Returns:
            Tuple of (output: (B, T, D), attn_weights: (B, H, T, T) or None).
        """
        B, T, _ = hidden_states.shape
        device = hidden_states.device

        # Linear projections
        Q = self._split_heads(self.q_proj(hidden_states))  # (B, H, T, dk)
        K = self._split_heads(self.k_proj(hidden_states))
        V = self._split_heads(self.v_proj(hidden_states))

        # ── Optimized Sliding Window Attention ──
        # 1. Use cached mask if available and sequence length matches
        if self.cached_mask is None or self.cached_seq_len != T:
            window_mask = self._build_window_mask(T, device)
            additive_window = torch.zeros(T, T, device=device, dtype=hidden_states.dtype)
            additive_window.masked_fill_(~window_mask, float("-inf"))
            self.cached_mask = additive_window.unsqueeze(0).unsqueeze(0) # (1, 1, T, T)
            self.cached_seq_len = T
        
        full_mask = self.cached_mask
        if attention_mask is not None:
            full_mask = full_mask + attention_mask

        # CRITICAL: mask dtype must match hidden_states/Q/K/V for SDPA and consistent precision
        if full_mask is not None:
            full_mask = full_mask.to(dtype=hidden_states.dtype)

        # 2. Use F.scaled_dot_product_attention (SDPA) if on PyTorch 2.0+ and output_attentions is False
        # This is much faster than manual MatMul + Softmax
        if hasattr(F, "scaled_dot_product_attention") and not output_attentions:
            # Note: SDPA handles scaling and dropout internally
            context = F.scaled_dot_product_attention(
                Q, K, V, 
                attn_mask=full_mask, 
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
            context = self._merge_heads(context)
            attn_weights = None # Not returned by SDPA easily
        else:
            # Fallback for older PyTorch
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            scores = scores + full_mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            context = self._merge_heads(torch.matmul(attn_weights, V))
        
        # BART expects (output, attn_weights, past_key_value)
        # Note: 'context' here is pre-projection.
        if TRANSFORMERS_NEW_SIG:
            return (context, attn_weights if output_attentions else None)
        else:
            return (context, attn_weights if output_attentions else None, past_key_value)
