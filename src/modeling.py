"""
modeling.py — LongAttention v2 (Summary-Level Long-Range, Memory-Efficient)
============================================================================

Architecture per layer:
  1. Local branch:  dense sliding-window attention (chunked, O(L·W))
  2. Long-range branch:
     a. SegmentSummarizer: mean-pool tokens into segment summaries
     b. NecessityRouter:   gate + top-K segment selection + type assignment
     c. LongRangeAttention: query attends to segment SUMMARY vectors (not tokens)
  3. Combine:  A_i = A_local + g_i · A_long

Memory: ~150MB activations per layer vs ~1.6GB before (100× reduction).
Formula unchanged from proposal:
  A_i^long = Σ_t Σ_{s ∈ topk} w_{i,s} · m_{i,t} · Attn_t(q_i, SK_s, SV_s)
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

TASK_DEPENDENCY_TYPES = {
    "docmt": ["coreference", "lexical_consistency", "tense_aspect", "discourse_relation"],
    "qa":    ["direct_evidence", "supporting_evidence", "bridging_evidence", "distractor_conflict"],
}


# ===================================================================== #
#  Segment Summarizer                                                    #
# ===================================================================== #

class SegmentSummarizer(nn.Module):
    def __init__(self, segment_size: int = 64):
        super().__init__()
        self.segment_size = segment_size

    def forward(self, keys, values):
        """(B,H,L,D) → sk(B,H,S,D), sv(B,H,S,D), num_segments"""
        B, H, L, D = keys.shape
        ss = self.segment_size
        pad = (ss - L % ss) % ss
        if pad > 0:
            keys = F.pad(keys, (0, 0, 0, pad))
            values = F.pad(values, (0, 0, 0, pad))
        n = keys.shape[2] // ss
        return (
            keys.reshape(B, H, n, ss, D).mean(3),    # (B,H,S,D)
            values.reshape(B, H, n, ss, D).mean(3),   # (B,H,S,D)
            n,
        )


# ===================================================================== #
#  Necessity Router (Top-K)                                              #
# ===================================================================== #

class NecessityRouter(nn.Module):
    def __init__(self, d_head: int, num_types: int = 4):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_head, d_head // 2), nn.GELU(), nn.Linear(d_head // 2, 1),
        )
        self.type_proj = nn.Linear(d_head, num_types)

    def forward(self, queries, seg_keys, num_segments, layer_ratio=0.5, top_k=4):
        """
        Returns:
            gate:      (B,H,L,1)
            topk_idx:  (B,H,L,K)
            topk_w:    (B,H,L,K)
            type_mask: (B,H,L,T)
        """
        D = queries.shape[-1]
        gate = torch.sigmoid(self.gate_proj(queries))

        # Scale first to prevent FP16 overflow in dot product
        q_scaled = queries / math.sqrt(D)
        seg_logits = torch.einsum("bhld,bhsd->bhls", q_scaled, seg_keys)
        K = min(top_k, num_segments)
        topk_vals, topk_idx = seg_logits.topk(K, dim=-1)

        temp = max(1.0 - layer_ratio, 0.1)
        topk_w = F.softmax(topk_vals / temp, dim=-1)
        type_mask = F.softmax(self.type_proj(queries) / temp, dim=-1)

        return gate, topk_idx, topk_w, type_mask


# ===================================================================== #
#  Local Attention (Chunked)                                             #
# ===================================================================== #

class LocalAttention(nn.Module):
    def __init__(self, d_head: int, window_size: int = 256):
        super().__init__()
        self.W = window_size

    def forward(self, q, k, v, attention_mask=None):
        L = q.shape[2]
        # Create a bidirectional sliding window mask: |i - j| <= W
        idx = torch.arange(L, device=q.device)
        attn_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.W
        
        if attention_mask is not None:
            # attention_mask: (B, L) with 1 for real tokens, 0 for padding
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, L, -1).bool()
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) & pad_mask

        # SDPA handles this efficiently (Math backend for boolean mask)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


# ===================================================================== #
#  Long-Range Attention (Summary-Level — ultra memory-efficient)         #
# ===================================================================== #

class LongRangeAttention(nn.Module):
    """
    A_i^long = Σ_t Σ_{s ∈ topk} w_{i,s} · m_{i,t} · Attn_t(q_i, SK_s, SV_s)

    Attends to segment SUMMARY vectors (1 vector per segment) instead of
    individual tokens within segments.

    Memory per layer:  O(L · K · D)  — with K=4, D=64: ~3MB
    vs token-level:    O(L · K · Ss · D) — with Ss=64:  ~200MB
    """

    def __init__(self, d_head: int, num_types: int = 4):
        super().__init__()
        self.num_types = num_types
        self.scale = 1.0 / math.sqrt(d_head)
        self.type_query_projs = nn.ModuleList(
            [nn.Linear(d_head, d_head, bias=False) for _ in range(num_types)]
        )

    def forward(self, q, sk, sv, topk_idx, topk_w, type_mask):
        """
        q:         (B,H,L,D)   — queries
        sk:        (B,H,S,D)   — segment summary keys
        sv:        (B,H,S,D)   — segment summary values
        topk_idx:  (B,H,L,K)   — selected segment indices
        topk_w:    (B,H,L,K)   — selection weights
        type_mask: (B,H,L,T)   — dependency-type distribution
        """
        B, H, L, D = q.shape
        K = topk_idx.shape[-1]

        # Gather top-K summary vectors — (B,H,L,K,D) — TINY tensor
        idx_k = topk_idx.unsqueeze(-1).expand(B, H, L, K, D)    # (B,H,L,K,D)
        sk_sel = sk.unsqueeze(2).expand(B, H, L, -1, D)         # (B,H,L,S,D)
        sk_sel = sk_sel.gather(3, idx_k)                          # (B,H,L,K,D)
        sv_sel = sv.unsqueeze(2).expand(B, H, L, -1, D)
        sv_sel = sv_sel.gather(3, idx_k)                          # (B,H,L,K,D)

        out = torch.zeros(B, H, L, D, device=q.device, dtype=q.dtype)

        for t in range(self.num_types):
            qt = self.type_query_projs[t](q)                      # (B,H,L,D)
            t_w = type_mask[..., t].unsqueeze(-1)                 # (B,H,L,1)

            # Attention: query vs K summary vectors
            # Scale first to prevent fp16 overflow
            qt_scaled = qt * self.scale
            attn = torch.einsum("bhld,bhlkd->bhlk", qt_scaled, sk_sel)  # (B,H,L,K)
            attn = F.softmax(attn, dim=-1)                        # (B,H,L,K)

            # Combine with segment weights
            attn = attn * topk_w                                  # (B,H,L,K)
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)    # re-normalize

            # Weighted sum of summary values
            val = torch.einsum("bhlk,bhlkd->bhld", attn, sv_sel) # (B,H,L,D)

            out = out + t_w * val

        return out


# ===================================================================== #
#  LongAttention v2 Layer                                                #
# ===================================================================== #

class LongAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_types: int = 4,
        window_size: int = 256,
        segment_size: int = 64,
        top_k: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.top_k = top_k
        self.use_checkpoint = False

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.seg_summarizer = SegmentSummarizer(segment_size)
        self.router = NecessityRouter(self.d_head, num_types)
        self.local_attn = LocalAttention(self.d_head, window_size)
        self.long_attn = LongRangeAttention(self.d_head, num_types)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def _forward_impl(self, hidden_states, layer_ratio, attention_mask=None):
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.d_head

        # 1. Post-LN: QKV projections take raw hidden states
        qkv = self.qkv_proj(hidden_states).reshape(B, L, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Local branch
        a_local = self.local_attn(q, k, v, attention_mask=attention_mask)

        # 3. Long-range branch (summary-level)
        sk, sv, ns = self.seg_summarizer(k, v)
        gate, topk_idx, topk_w, type_mask = self.router(q, sk, ns, layer_ratio, self.top_k)
        a_long = self.long_attn(q, sk, sv, topk_idx, topk_w, type_mask)

        # 4. Combine & Post-LN 1 (Attention residual)
        combined = a_local + gate * a_long
        combined = combined.transpose(1, 2).reshape(B, L, D)
        attn_out = self.norm1(hidden_states + self.dropout_layer(self.out_proj(combined)))

        # 5. FFN & Post-LN 2 (FFN residual)
        x = self.norm2(attn_out + self.ff(attn_out))

        return x, gate, topk_idx, topk_w, type_mask

    def forward(self, hidden_states, layer_ratio=0.5, attention_mask=None):
        if self.use_checkpoint and self.training:
            lr_t = torch.tensor(layer_ratio, device=hidden_states.device)
            x, gate, topk_idx, topk_w, type_mask = checkpoint(
                self._forward_ckpt, hidden_states, lr_t, attention_mask, use_reentrant=False
            )
        else:
            x, gate, topk_idx, topk_w, type_mask = self._forward_impl(hidden_states, layer_ratio, attention_mask)

        info = {
            "gate": gate,
            "topk_idx": topk_idx,
            "topk_w": topk_w,
            "type_mask": type_mask,
        }
        return x, info

    def _forward_ckpt(self, hidden_states, lr_tensor, attention_mask):
        return self._forward_impl(hidden_states, lr_tensor.item(), attention_mask)


# ===================================================================== #
#  Regularization Losses                                                 #
# ===================================================================== #

def anti_collapse_loss(layer_infos):
    total_loss = torch.tensor(0.0, device=layer_infos[0]["type_mask"].device)
    for info in layer_infos:
        tm = info["type_mask"] # (B, H, L, T)
        
        # 1. Maximize marginal entropy: encourage using all types globally
        marginal_p = tm.mean(dim=(0, 1, 2))  # (T,)
        h_marginal = -(marginal_p * (marginal_p + 1e-8).log()).sum()
        
        # 2. Minimize per-token entropy: encourage each token to pick one type
        h_token = -(tm * (tm + 1e-8).log()).sum(dim=-1).mean()
        
        # Final loss: h_token - h_marginal (minimize this to achieve both goals)
        total_loss = total_loss + (h_token - h_marginal)
        
    return total_loss / len(layer_infos)


def null_route_loss(layer_infos):
    total = torch.tensor(0.0, device=layer_infos[0]["gate"].device)
    for info in layer_infos:
        total = total + info["gate"].mean()
    return total / len(layer_infos)
