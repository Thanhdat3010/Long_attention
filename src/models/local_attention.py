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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_dropout = nn.Dropout(p=dropout_prob)

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
        Each row i can attend only to positions in [max(0, i-W+1), i].
        """
        idx = torch.arange(seq_len, device=device)
        # row i, col j: attend if  0 <= j <= i  AND  i - j < window_size
        row = idx.unsqueeze(1)  # (T, 1)
        col = idx.unsqueeze(0)  # (1, T)
        causal_mask = col <= row                        # causal
        window_mask = (row - col) < self.window_size   # within window
        return causal_mask & window_mask  # (T, T) bool

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

        # Scaled dot-product scores: (B, H, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply sliding window mask (additive: -inf for masked positions)
        window_mask = self._build_window_mask(T, device)  # (T, T)
        # Convert bool mask to additive float mask
        additive_window = torch.zeros(T, T, device=device, dtype=scores.dtype)
        additive_window.masked_fill_(~window_mask, float("-inf"))
        scores = scores + additive_window.unsqueeze(0).unsqueeze(0)

        # Optional external attention mask (e.g., padding)
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # NaN guard: positions that are fully masked (all -inf) produce NaN after softmax
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)  # (B, H, T, dk)
        context = self._merge_heads(context)     # (B, T, D)
        output = self.out_proj(context)
        # BART expects a 3-tuple: (output, attn_weights, past_key_value)
        return (output, attn_weights if output_attentions else None, past_key_value)
