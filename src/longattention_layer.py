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

    def forward(self, queries, seg_keys, num_segments, layer_ratio=0.5, top_k=4, coarse_candidates=4, seg_mask=None):
        layer_ratio = float(layer_ratio)
        D = queries.shape[-1]
        qn = self.q_norm(queries.float()).to(dtype=queries.dtype)

        # Task-agnostic intent anchor from current token state and global segment context.
        q_anchor = qn.float().mean(dim=2, keepdim=True)
        s_anchor = seg_keys.float().mean(dim=2, keepdim=True)
        intent_anchor = torch.tanh(q_anchor + s_anchor)
        q_norm = F.normalize(qn.float(), dim=-1, eps=1e-6)
        intent_norm = F.normalize(intent_anchor, dim=-1, eps=1e-6)
        token_intent_align = (q_norm * intent_norm).sum(dim=-1, keepdim=True).clamp(min=-1.0, max=1.0)

        gate_logits = self.gate_proj(qn).float().clamp(min=-8.0, max=8.0)
        # Keep intent conditioning mild to avoid over-opening gate (observed in v12).
        centered_align = token_intent_align - token_intent_align.mean(dim=2, keepdim=True)
        gate_align_scale = 0.25 + 0.35 * layer_ratio
        gate_logits = gate_logits + (gate_align_scale * centered_align).clamp(min=-0.8, max=0.8)
        gate = torch.sigmoid(gate_logits)

        q_scaled = qn.float() / math.sqrt(D)
        # Use a lightweight anchor-segment relevance to avoid large normalization overhead.
        anchor_to_seg = torch.einsum(
            "bhid,bhsd->bhis",
            intent_anchor.float() / math.sqrt(D),
            seg_keys.float(),
        ).squeeze(2)
        anchor_bias_scale = 0.08 + 0.14 * layer_ratio

        # Segment pre-pruning: shrink candidate segment set before token-level routing.
        # This substantially reduces LxS router compute/memory on long contexts.
        # Keep more segments in shallow layers for evidence recall, tighten in deeper layers for efficiency.
        base_keep = max(max(coarse_candidates, top_k) * 2, 12)
        depth_bonus = int(round((1.0 - layer_ratio) * 6.0))
        S_keep = min(num_segments, base_keep + depth_bonus)
        if num_segments > S_keep + 2:
            seg_scores = anchor_to_seg
            if seg_mask is not None:
                seg_scores = seg_scores.masked_fill(~seg_mask.unsqueeze(1), -10000.0)
            _, seg_keep_idx = seg_scores.topk(S_keep, dim=-1)
            gather_idx = seg_keep_idx.unsqueeze(-1).expand(-1, -1, -1, D)
            seg_keys_eff = torch.gather(seg_keys, 2, gather_idx)
            anchor_eff = torch.gather(anchor_to_seg, 2, seg_keep_idx)
            coarse_logits = torch.einsum("bhld,bhsd->bhls", q_scaled, seg_keys_eff)
            idx_map = seg_keep_idx
        else:
            coarse_logits = torch.einsum("bhld,bhsd->bhls", q_scaled, seg_keys)
            anchor_eff = anchor_to_seg
            idx_map = None

        if seg_mask is not None:
            # Use a safe large negative number instead of float min to prevent -inf after division
            mask_val = -10000.0
            if idx_map is not None:
                seg_mask_eff = torch.gather(seg_mask, 1, idx_map[:, 0, :])
                coarse_logits = coarse_logits.masked_fill(~seg_mask_eff.unsqueeze(1).unsqueeze(2), mask_val)
            else:
                coarse_logits = coarse_logits.masked_fill(~seg_mask.unsqueeze(1).unsqueeze(2), mask_val)

        M = min(max(coarse_candidates, top_k), coarse_logits.size(-1))
        K = min(top_k, M)

        # Fast path: with few segments, skip coarse stage to reduce overhead.
        if coarse_logits.size(-1) <= M + 2:
            fine_logits = coarse_logits + anchor_bias_scale * anchor_eff.unsqueeze(2)
            topk_vals, topk_idx = fine_logits.topk(K, dim=-1)
        else:
            # Stage-1 (coarse): keep only a small candidate set per token/head.
            coarse_vals, coarse_idx = coarse_logits.topk(M, dim=-1)

            # Stage-2 (fine): refine within coarse candidates with intent-conditioned bias.
            coarse_anchor = torch.take_along_dim(anchor_eff.unsqueeze(2), coarse_idx, dim=-1)
            fine_logits = coarse_vals + anchor_bias_scale * coarse_anchor

            fine_vals, fine_pos = fine_logits.topk(K, dim=-1)
            topk_idx = coarse_idx.gather(-1, fine_pos)
            topk_vals = fine_vals

        if idx_map is not None:
            topk_idx = torch.gather(idx_map.unsqueeze(2).expand(-1, -1, topk_idx.size(2), -1), 3, topk_idx)

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

        aux = {
            "token_intent_align": token_intent_align.mean().detach().to(dtype=queries.dtype),
            "gate_align_scale": torch.tensor(float(gate_align_scale), device=queries.device, dtype=queries.dtype),
            "anchor_bias_scale": torch.tensor(float(anchor_bias_scale), device=queries.device, dtype=queries.dtype),
            "coarse_candidates": torch.tensor(float(M), device=queries.device, dtype=queries.dtype),
            "prepruned_segments": torch.tensor(float(coarse_logits.size(-1)), device=queries.device, dtype=queries.dtype),
        }
        return gate, topk_idx, topk_w, type_mask, aux


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
        coarse_candidates: int = 4,
        hard_budget: int = 1,
        alpha_init: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.seg_summarizer = SegmentSummarizer(segment_size)
        self.router = NecessityRouter(d_head, num_types)
        self.local_attn = LocalAttention(d_head, window_size)
        self.long_attn = LongRangeAttention(d_head, num_types)
        self.top_k = top_k
        self.coarse_candidates = max(int(coarse_candidates), int(top_k))
        self.hard_budget = max(1, int(hard_budget))
        self.alpha_long = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.constant_(self.router.gate_proj[-1].bias, float(gate_bias_init))

    def _effective_top_k(self, layer_ratio: float) -> int:
        # Keep broader exploration in shallow layers, then tighten in deeper layers.
        if self.top_k <= 2:
            return self.top_k
        min_k = max(2, self.top_k - 1)
        ratio = float(max(0.0, min(1.0, layer_ratio)))
        k_float = self.top_k - (self.top_k - min_k) * ratio
        return int(max(min_k, min(self.top_k, round(k_float))))

    @staticmethod
    def _adaptive_margin_threshold(layer_ratio: float) -> float:
        # Deeper layers use stricter confidence to collapse uncertain routes.
        ratio = float(max(0.0, min(1.0, layer_ratio)))
        return 0.16 + 0.12 * ratio

    @staticmethod
    def _gate_threshold(layer_ratio: float) -> float:
        ratio = float(max(0.0, min(1.0, layer_ratio)))
        return 0.46 + 0.10 * ratio

    def _effective_hard_budget(self, layer_ratio: float, k_now: int) -> int:
        if k_now <= 1:
            return 1
        ratio = float(max(0.0, min(1.0, layer_ratio)))
        # Coarse-to-fine hard budget: shallower layers allow +1 route, deeper layers use minimum budget.
        min_budget = min(k_now, self.hard_budget)
        max_budget = min(k_now, self.hard_budget + 1)
        budget_float = max_budget - ratio * (max_budget - min_budget)
        budget = int(max(min_budget, min(max_budget, round(budget_float))))
        return max(1, budget)

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
        layer_ratio = float(max(0.0, min(1.0, layer_ratio)))
        eff_top_k = self._effective_top_k(layer_ratio)
        gate_th = self._gate_threshold(layer_ratio)
        margin_th = self._adaptive_margin_threshold(layer_ratio)

        a_local = self.local_attn(q, k, v, attention_mask=token_mask)
        sk, sv, ns, seg_mask = self.seg_summarizer(k, v, attention_mask=token_mask)
        gate, topk_idx, topk_w, type_mask, aux = self.router(
            q,
            sk,
            ns,
            layer_ratio=layer_ratio,
            top_k=eff_top_k,
            coarse_candidates=self.coarse_candidates,
            seg_mask=seg_mask,
        )

        # Adaptive per-token route budget: keep only the strongest route when top-1 is confident.
        if topk_w.size(-1) > 1:
            topk_w_f = topk_w.float()
            margin = topk_w_f[..., 0] - topk_w_f[..., 1]
            topk_ent = -(topk_w_f.clamp_min(1e-6) * topk_w_f.clamp_min(1e-6).log()).sum(dim=-1)
            ent_th = 0.62 - 0.22 * layer_ratio
            low_gate_th = 0.38 + 0.09 * layer_ratio
            prefer_single = (margin >= margin_th) | (topk_ent <= ent_th) | (gate.squeeze(-1).float() <= low_gate_th)
            keep_mask = torch.ones_like(topk_w_f)
            keep_mask[..., 1:] = (~prefer_single).unsqueeze(-1).to(topk_w_f.dtype)
            topk_w_f = topk_w_f * keep_mask
            topk_w_f = topk_w_f / topk_w_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            topk_w = topk_w_f.to(topk_w.dtype)
            adaptive_single_ratio = prefer_single.float().mean()
        else:
            adaptive_single_ratio = torch.tensor(1.0, device=q.device, dtype=q.dtype)

        # Hard route budget (stage-2): uncertainty-adaptive budget to avoid deep-layer route collapse.
        budget_eff = self._effective_hard_budget(layer_ratio, topk_w.size(-1))
        if topk_w.size(-1) > 1:
            topk_w_f = topk_w.float()
            topk_ent = -(topk_w_f.clamp_min(1e-6) * topk_w_f.clamp_min(1e-6).log()).sum(dim=-1)
            uncertain_th = 0.52 - 0.10 * layer_ratio
            uncertain = topk_ent > uncertain_th
            budget_mask = torch.zeros_like(topk_w_f)
            budget_mask[..., 0] = 1.0
            if topk_w.size(-1) > 1:
                allow_second = uncertain | (budget_eff > 1)
                budget_mask[..., 1] = allow_second.to(dtype=topk_w_f.dtype)
            topk_w_f = topk_w_f * budget_mask
            topk_w_f = topk_w_f / topk_w_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            # Entropy floor on deeper layers to avoid near one-hot collapse across all tokens.
            if layer_ratio >= 0.55:
                cur_ent = -(topk_w_f.clamp_min(1e-6) * topk_w_f.clamp_min(1e-6).log()).sum(dim=-1, keepdim=True)
                ent_floor = 0.08
                mix = ((ent_floor - cur_ent).clamp_min(0.0) / ent_floor).clamp(max=0.20)
                uniform = torch.full_like(topk_w_f, 1.0 / topk_w_f.size(-1))
                topk_w_f = (1.0 - mix) * topk_w_f + mix * uniform
                topk_w_f = topk_w_f / topk_w_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            topk_w = topk_w_f.to(topk_w.dtype)
            budget_eff = float((budget_mask.sum(dim=-1).float().mean().item()))

        a_long = self.long_attn(q, sk, sv, topk_idx, topk_w, type_mask)

        if token_mask is not None:
            a_long = a_long * token_mask.unsqueeze(1).unsqueeze(-1)

        # Bound long-range branch scale to avoid exploding residual magnitude.
        alpha = torch.tanh(self.alpha_long)
        # Smooth thresholding keeps gradients while enforcing deeper-layer selectivity.
        gate_active = torch.sigmoid((gate - gate_th) * 6.0)
        gate_eff = gate * gate_active
        out = a_local + alpha * gate_eff * a_long
        info = {
            "gate": gate,
            "gate_eff": gate_eff,
            "topk_idx": topk_idx,
            "topk_w": topk_w,
            "type_mask": type_mask,
            "alpha": alpha,
            "effective_top_k": torch.tensor(float(eff_top_k), device=q.device, dtype=q.dtype),
            "gate_threshold": torch.tensor(float(gate_th), device=q.device, dtype=q.dtype),
            "adaptive_margin_threshold": torch.tensor(float(margin_th), device=q.device, dtype=q.dtype),
            "adaptive_single_ratio": adaptive_single_ratio.to(dtype=q.dtype),
            "gate_intent_alignment": aux["token_intent_align"],
            "gate_align_scale": aux["gate_align_scale"],
            "anchor_bias_scale": aux["anchor_bias_scale"],
            "coarse_candidates": aux["coarse_candidates"],
            "prepruned_segments": aux["prepruned_segments"],
            "hard_budget": torch.tensor(float(budget_eff), device=q.device, dtype=q.dtype),
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

        return (
            x,
            info["gate"],
            info.get("gate_eff", info["gate"]),
            info["topk_idx"],
            info["topk_w"],
            info["type_mask"],
            info["alpha"],
            info.get("effective_top_k"),
            info.get("gate_threshold"),
        )

    def forward(self, hidden_states, layer_ratio=0.5, attention_mask=None):
        if self.use_checkpoint and self.training:
            lr_t = torch.tensor(layer_ratio, device=hidden_states.device)
            x, gate, gate_eff, topk_idx, topk_w, type_mask, alpha, eff_top_k, gate_th = checkpoint(
                self._forward_ckpt, hidden_states, lr_t, attention_mask, use_reentrant=False
            )
        else:
            x, gate, gate_eff, topk_idx, topk_w, type_mask, alpha, eff_top_k, gate_th = self._forward_impl(
                hidden_states, layer_ratio, attention_mask
            )

        info = {
            "gate": gate,
            "gate_eff": gate_eff,
            "topk_idx": topk_idx,
            "topk_w": topk_w,
            "type_mask": type_mask,
            "alpha": alpha,
            "effective_top_k": eff_top_k,
            "gate_threshold": gate_th,
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


def topk_entropy_penalty(layer_infos):
    total = torch.tensor(0.0, device=layer_infos[0]["topk_w"].device, dtype=torch.float32)
    for info in layer_infos:
        tw = torch.nan_to_num(info["topk_w"].float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(1e-6)
        tw = tw / tw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ent = -(tw * tw.log()).sum(dim=-1).mean()
        total = total + ent
    return total / len(layer_infos)
