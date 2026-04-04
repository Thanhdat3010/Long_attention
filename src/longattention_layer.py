"""
longattention_layer.py — LongAttention v2 Core and Layer Blocks
===============================================================

This module contains only LongAttention building blocks:
- Segment summarization
- Necessity-aware and dependency-typed routing
- Local + long-range attention composition
- LongAttentionLayer and regularization losses
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


TASK_DEPENDENCY_TYPES = {
    "docmt": ["coreference", "lexical_consistency", "tense_aspect", "discourse_relation"],
    "qa": ["direct_evidence", "supporting_evidence", "bridging_evidence", "distractor_conflict"],
}


class SegmentSummarizer(nn.Module):
    def __init__(self, segment_size: int = 64):
        super().__init__()
        self.segment_size = segment_size

    def forward(self, keys, values, attention_mask=None):
        """(B,H,L,D) -> sk(B,H,S,D), sv(B,H,S,D), num_segments, seg_mask"""
        B, H, L, D = keys.shape
        ss = self.segment_size
        pad = (ss - L % ss) % ss
        if pad > 0:
            keys = F.pad(keys, (0, 0, 0, pad))
            values = F.pad(values, (0, 0, 0, pad))
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            if pad > 0:
                mask = F.pad(mask, (0, 0, 0, pad))
        else:
            mask = None

        n = keys.shape[2] // ss
        k = keys.reshape(B, H, n, ss, D)
        v = values.reshape(B, H, n, ss, D)

        if mask is None:
            sk = k.mean(3)
            sv = v.mean(3)
            seg_mask = None
        else:
            m = mask.reshape(B, 1, n, ss, 1)
            denom = m.sum(3).clamp(min=1.0)
            sk = (k * m).sum(3) / denom
            sv = (v * m).sum(3) / denom
            seg_mask = denom.squeeze(-1).squeeze(1) > 0

        return sk, sv, n, seg_mask


class NecessityRouter(nn.Module):
    def __init__(self, d_head: int, num_types: int = 4):
        super().__init__()
        hidden = max(d_head // 2, 16)
        self.q_norm = nn.LayerNorm(d_head)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_head, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.type_proj = nn.Linear(d_head, num_types)

    def forward(self, queries, seg_keys, num_segments, layer_ratio=0.5, top_k=4, seg_mask=None):
        layer_ratio = float(layer_ratio)
        D = queries.shape[-1]
        qn = self.q_norm(queries.float()).to(dtype=queries.dtype)

        gate_logits = self.gate_proj(qn).float().clamp(min=-8.0, max=8.0)
        gate = torch.sigmoid(gate_logits)

        q_scaled = qn.float() / math.sqrt(D)
        seg_logits = torch.einsum("bhld,bhsd->bhls", q_scaled, seg_keys)
        if seg_mask is not None:
            # Use a safe large negative number instead of float min to prevent -inf after division
            mask_val = -10000.0
            seg_logits = seg_logits.masked_fill(~seg_mask.unsqueeze(1).unsqueeze(2), mask_val)

        K = min(top_k, num_segments)
        topk_vals, topk_idx = seg_logits.topk(K, dim=-1)

        # Sharpen routing progressively with depth, while keeping safe floors.
        seg_temp = max(0.90 - 0.45 * layer_ratio, 0.45)
        type_temp = max(0.95 - 0.35 * layer_ratio, 0.55)

        # Cast to float32 before division and softmax to prevent FP16 overflow/NaNs.
        topk_w_f = F.softmax((topk_vals.float() / seg_temp), dim=-1)
        topk_w_f = torch.nan_to_num(topk_w_f, nan=0.0, posinf=1.0, neginf=0.0)
        topk_w_f = topk_w_f / topk_w_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        topk_w_f = topk_w_f.clamp_min(1e-6)
        topk_w_f = topk_w_f / topk_w_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        type_mask_f = F.softmax((self.type_proj(qn).float() / type_temp), dim=-1)
        type_mask_f = torch.nan_to_num(type_mask_f, nan=0.0, posinf=1.0, neginf=0.0)
        type_mask_f = type_mask_f / type_mask_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        type_mask_f = type_mask_f.clamp_min(1e-6)
        type_mask_f = type_mask_f / type_mask_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        topk_w = topk_w_f.to(queries.dtype)
        type_mask = type_mask_f.to(queries.dtype)

        return gate, topk_idx, topk_w, type_mask


class LocalAttention(nn.Module):
    def __init__(self, d_head: int, window_size: int = 256):
        super().__init__()
        self.W = window_size
        self.chunk_size = 512

    def forward(self, q, k, v, attention_mask=None):
        B, H, L, D = q.shape
        out = torch.zeros_like(q)

        token_mask = attention_mask.bool() if attention_mask is not None else None
        scale = 1.0 / math.sqrt(D)

        for qs in range(0, L, self.chunk_size):
            qe = min(qs + self.chunk_size, L)
            ks = max(0, qs - self.W)
            ke = min(L, qe + self.W)

            q_chunk = q[:, :, qs:qe, :]
            k_chunk = k[:, :, ks:ke, :]
            v_chunk = v[:, :, ks:ke, :]

            q_len = qe - qs
            k_len = ke - ks

            q_pos = torch.arange(qs, qe, device=q.device)
            k_pos = torch.arange(ks, ke, device=q.device)
            local_mask = (q_pos[:, None] - k_pos[None, :]).abs() <= self.W

            attn_mask = local_mask.unsqueeze(0).unsqueeze(0)
            if token_mask is not None:
                key_mask = token_mask[:, None, None, ks:ke]
                attn_mask = attn_mask & key_mask

            # Use native SDPA in model dtype for speed; fallback keeps training robust.
            try:
                chunk_out = F.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                )
            except RuntimeError:
                mask_val = -1e4
                scores = torch.matmul(q_chunk.float(), k_chunk.float().transpose(-2, -1)) * scale
                scores = scores.masked_fill(~attn_mask, mask_val)
                attn = F.softmax(scores, dim=-1)
                attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
                chunk_out = torch.matmul(attn.to(dtype=q.dtype), v_chunk)

            if token_mask is not None:
                query_mask = token_mask[:, None, qs:qe, None]
                chunk_out = chunk_out * query_mask

            out[:, :, qs:qe, :] = chunk_out

        return out


class LongRangeAttention(nn.Module):
    def __init__(self, d_head: int, num_types: int = 4):
        super().__init__()
        self.num_types = num_types
        self.scale = 1.0 / math.sqrt(d_head)
        self.type_query_projs = nn.ModuleList([nn.Linear(d_head, d_head, bias=False) for _ in range(num_types)])

    def forward(self, q, sk, sv, topk_idx, topk_w, type_mask):
        B, H, L, D = q.shape
        K = topk_idx.shape[-1]

        idx_k = topk_idx.unsqueeze(-1).expand(B, H, L, K, D)
        sk_sel = sk.unsqueeze(2).expand(B, H, L, -1, D).gather(3, idx_k)
        sv_sel = sv.unsqueeze(2).expand(B, H, L, -1, D).gather(3, idx_k)

        out = torch.zeros(B, H, L, D, device=q.device, dtype=q.dtype)

        for t in range(self.num_types):
            qt = self.type_query_projs[t](q)
            t_w = type_mask[..., t].unsqueeze(-1)

            qt_scaled = qt * self.scale
            attn_logits = torch.einsum("bhld,bhlkd->bhlk", qt_scaled, sk_sel).float()
            # Fuse top-k routing prior in logit space for better FP16 stability.
            log_prior = topk_w.float().clamp_min(1e-6).log()
            attn = F.softmax(attn_logits + log_prior, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
            attn = attn.to(q.dtype)

            val = torch.einsum("bhlk,bhlkd->bhld", attn, sv_sel)
            out = out + t_w * val

        return out


class LongAttentionCore(nn.Module):
    """LongAttention v2 core that operates directly on projected Q/K/V."""

    def __init__(
        self,
        d_head: int,
        num_types: int = 4,
        window_size: int = 256,
        segment_size: int = 64,
        top_k: int = 4,
        alpha_init: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.seg_summarizer = SegmentSummarizer(segment_size)
        self.router = NecessityRouter(d_head, num_types)
        self.local_attn = LocalAttention(d_head, window_size)
        self.long_attn = LongRangeAttention(d_head, num_types)
        self.top_k = top_k
        self.alpha_long = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.constant_(self.router.gate_proj[-1].bias, float(gate_bias_init))

    @staticmethod
    def to_token_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            # Accept both binary masks (0/1) and additive masks used by some backbones
            # (e.g., Longformer-style -10000/0/+10000). We always convert to token-validity
            # mask where 1 means valid token and 0 means padded/invalid token.
            if attention_mask.dtype.is_floating_point:
                if torch.any(attention_mask < 0):
                    return (attention_mask >= 0).to(dtype=torch.long)
                return (attention_mask > 0).to(dtype=torch.long)
            if torch.any(attention_mask < 0):
                return (attention_mask >= 0).to(dtype=torch.long)
            return (attention_mask > 0).to(dtype=torch.long)
        if attention_mask.dim() == 4:
            return (attention_mask[:, 0, 0, :] == 0).to(dtype=torch.long)
        return None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        layer_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        token_mask = self.to_token_mask(attention_mask)

        a_local = self.local_attn(q, k, v, attention_mask=token_mask)
        sk, sv, ns, seg_mask = self.seg_summarizer(k, v, attention_mask=token_mask)
        gate, topk_idx, topk_w, type_mask = self.router(
            q, sk, ns, layer_ratio=layer_ratio, top_k=self.top_k, seg_mask=seg_mask
        )
        a_long = self.long_attn(q, sk, sv, topk_idx, topk_w, type_mask)

        if token_mask is not None:
            a_long = a_long * token_mask.unsqueeze(1).unsqueeze(-1)

        # Bound long-range branch scale to avoid exploding residual magnitude.
        alpha = torch.tanh(self.alpha_long)
        out = a_local + alpha * gate * a_long
        info = {
            "gate": gate,
            "topk_idx": topk_idx,
            "topk_w": topk_w,
            "type_mask": type_mask,
            "alpha": alpha,
        }
        return out, info


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
        alpha_init: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.top_k = top_k
        self.use_checkpoint = False

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.core = LongAttentionCore(
            d_head=self.d_head,
            num_types=num_types,
            window_size=window_size,
            segment_size=segment_size,
            top_k=top_k,
            alpha_init=alpha_init,
            gate_bias_init=gate_bias_init,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def _forward_impl(self, hidden_states, layer_ratio, attention_mask=None):
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.d_head

        qkv = self.qkv_proj(hidden_states).reshape(B, L, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        combined, info = self.core(q, k, v, attention_mask=attention_mask, layer_ratio=layer_ratio)
        combined = combined.transpose(1, 2).reshape(B, L, D)

        attn_out = self.norm1(hidden_states + self.dropout_layer(self.out_proj(combined)))
        x = self.norm2(attn_out + self.ff(attn_out))

        return x, info["gate"], info["topk_idx"], info["topk_w"], info["type_mask"], info["alpha"]

    def forward(self, hidden_states, layer_ratio=0.5, attention_mask=None):
        if self.use_checkpoint and self.training:
            lr_t = torch.tensor(layer_ratio, device=hidden_states.device)
            x, gate, topk_idx, topk_w, type_mask, alpha = checkpoint(
                self._forward_ckpt, hidden_states, lr_t, attention_mask, use_reentrant=False
            )
        else:
            x, gate, topk_idx, topk_w, type_mask, alpha = self._forward_impl(hidden_states, layer_ratio, attention_mask)

        info = {
            "gate": gate,
            "topk_idx": topk_idx,
            "topk_w": topk_w,
            "type_mask": type_mask,
            "alpha": alpha,
        }
        return x, info

    def _forward_ckpt(self, hidden_states, lr_tensor, attention_mask):
        return self._forward_impl(hidden_states, lr_tensor.item(), attention_mask)


def anti_collapse_loss(layer_infos):
    total_loss = torch.tensor(0.0, device=layer_infos[0]["type_mask"].device, dtype=torch.float32)
    for info in layer_infos:
        # Compute entropy in float32; FP16 + tiny eps can produce 0 * -inf -> NaN.
        tm = torch.nan_to_num(info["type_mask"].float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(1e-6)
        marginal_p = tm.mean(dim=(0, 1, 2))
        h_marginal = -(marginal_p * marginal_p.log()).sum()
        h_token = -(tm * tm.log()).sum(dim=-1).mean()
        total_loss = total_loss + (h_token - h_marginal)
    return total_loss / len(layer_infos)


def null_route_loss(layer_infos):
    total = torch.tensor(0.0, device=layer_infos[0]["gate"].device, dtype=torch.float32)
    n_layers = max(len(layer_infos), 1)
    for li, info in enumerate(layer_infos):
        gate = torch.nan_to_num(info["gate"].float(), nan=0.0, posinf=1.0, neginf=0.0)
        layer_ratio = li / max(n_layers - 1, 1)
        target_gate = 0.18 + 0.22 * layer_ratio
        total = total + (gate.mean() - target_gate).pow(2)
    return total / len(layer_infos)
