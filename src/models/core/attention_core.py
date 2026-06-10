"""
LongAttention v3: Necessity-Aware & Dependency-Typed Long-Context Attention.

This module implements the core architectural contribution of the LongAttention v3 proposal.
It integrates two parallel branches:

  Branch 1 — Local Branch (LocalSlidingWindowAttention):
      Dense attention over a short sliding window to capture syntactic structure
      and immediate coherence. Enhanced with Functional Affix information.

  Branch 2 — Gated & Typed Long-range Branch:
      1. Necessity Gating (g_i): Decides IF long-range information is needed.
      2. Functional Decomposition: Separates hidden states into Semantic Roots and Functional Affixes.
      3. Dependency-Typed Routing: Specialized channels for:
         - Coreference Resolution
         - Lexical Consistency
         - Discourse Relations
      4. Top-K Retrieval: Efficiently routes to the most relevant context segments.

  Output Integration — Gated Interpolation:
      combined = (1 - g_i) * (A_local + A_affix) + g_i * A_long

Changes in v3 (from v2):
  - A_encoded (Affix) is now integrated into the local branch instead of being discarded.
  - Query for long-range retrieval is projected from R_encoded (not raw hidden_states),
    ensuring Q and K share the same semantic feature space.
  - Diversity loss replaced with Static Orthogonality Regularization on projection weights,
    eliminating torch.randperm GPU bottleneck.
  - Branch merging uses Gated Interpolation for balanced activation magnitudes.
"""

import math
import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from packaging import version

logger = logging.getLogger(__name__)

# Transformers >= 4.36 changed BartEncoderLayer signature from 3-tuple to 2-tuple return expectation
TRANSFORMERS_NEW_SIG = version.parse(transformers.__version__) >= version.parse("4.36.0")


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class FunctionalDecomposer(nn.Module):
    """
    Gated Functional Decomposer.
    Separates the input hidden states S into two complementary streams:
      - R (Semantic Root): Tokens carrying high semantic information density.
      - A (Functional Affix): Tokens serving grammatical/syntactic roles.
    """
    def __init__(self, hidden_size: int, bottleneck_ratio: float = 0.25) -> None:
        super().__init__()
        bottleneck = max(1, int(hidden_size * bottleneck_ratio))

        self.gate_proj_local = nn.Sequential(
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, 1, bias=False),
        )
        self.gate_proj_global = nn.Sequential(
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, 1, bias=False),
        )

        self.root_transform = nn.Linear(hidden_size, hidden_size, bias=False)
        self.affix_codebook = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, global_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_logits = self.gate_proj_local(hidden_states)
        global_logits = self.gate_proj_global(global_context)
        
        gate_logits = local_logits + global_logits
        temperature = 0.5
        
        gate_score = torch.sigmoid(gate_logits / temperature)

        root_features = self.root_transform(hidden_states)
        R_encoded = gate_score * root_features

        affix_features = self.affix_codebook(hidden_states)
        A_encoded = (1.0 - gate_score) * affix_features

        return R_encoded, A_encoded, gate_score


class NecessityGate(nn.Module):
    """
    Necessity Gating Mechanism (g_i).
    Computes a per-token scalar in [0, 1] deciding if long-range attention is required.
    """
    def __init__(self, hidden_size: int, bottleneck_ratio: float = 0.25):
        super().__init__()
        bottleneck = max(1, int(hidden_size * bottleneck_ratio))
        # bias=-1.0 → sigmoid(-1)≈0.27: balanced start, not too open (noise), not too closed (dead)
        final_linear = nn.Linear(bottleneck, 1, bias=True)
        nn.init.constant_(final_linear.bias, -1.0)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
            final_linear,
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x) # (B, T, 1)


class DependencyTypedGist(nn.Module):
    """
    Dependency-Typed Gist Builder.
    Creates specialized key/value pairs for different long-range dependency types.
    Types: Coreference, Lexical Consistency, Discourse Relation.
    """
    def __init__(self, hidden_size: int, num_types: int = 3):
        super().__init__()
        self.num_types = num_types
        self.hidden_size = hidden_size
        # Consolidated projection for all types, Keys, and Values: 
        # (hidden_size * 2 for K and V, multiplied by num_types)
        self.multi_type_proj = nn.Linear(hidden_size, hidden_size * 2 * num_types, bias=True)
        
    def forward(self, R_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            K_typed: (B, num_types, T, D)
            V_typed: (B, num_types, T, D)
        """
        B, T, D = R_encoded.shape
        
        # 1. Project all types, keys, and values at once
        kv_all = self.multi_type_proj(R_encoded) # (B, T, num_types * 2 * D)
        
        # 2. Reshape to separate types, and K/V
        # Shape: (B, T, num_types, 2, D)
        kv_all = kv_all.view(B, T, self.num_types, 2, D)
        
        # 3. Extract K and V, transpose to (B, num_types, T, D)
        K_typed = kv_all[:, :, :, 0, :].transpose(1, 2)
        V_typed = kv_all[:, :, :, 1, :].transpose(1, 2)
        
        return K_typed, V_typed

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Static Orthogonality Regularization on projection weight matrices.
        
        Penalizes cosine similarity between the projection weights of different
        dependency types to encourage each type to learn distinct representations.
        This replaces the dynamic TV-Distance diversity loss that caused GPU bottleneck.
        
        No torch.randperm, no CPU-GPU sync — pure static weight computation.
        """
        D = self.hidden_size
        # multi_type_proj.weight shape: (num_types * 2 * D, D)
        # Each type has 2*D rows (D for K, D for V)
        W = self.multi_type_proj.weight  # (num_types * 2D, D)
        chunk_size = 2 * D
        
        # Split into per-type weight matrices
        type_weights = []
        for t in range(self.num_types):
            # Each type's full projection: (2D, D) — flatten to a single vector for comparison
            w_t = W[t * chunk_size : (t + 1) * chunk_size]  # (2D, D)
            type_weights.append(w_t.reshape(-1))  # (2D*D,)
        
        # Compute pairwise cosine similarity penalty
        loss = torch.tensor(0.0, device=W.device, dtype=W.dtype)
        count = 0
        for i in range(self.num_types):
            for j in range(i + 1, self.num_types):
                cos_sim = F.cosine_similarity(type_weights[i].unsqueeze(0), 
                                               type_weights[j].unsqueeze(0))
                loss = loss + cos_sim.abs()  # Penalize both positive and negative similarity
                count += 1
        
        return loss / max(count, 1)


class TypedTopKRetrieval(nn.Module):
    """
    Typed Top-K Retrieval.
    Retrieves and aggregates information from long-range gists across multiple types.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_types: int = 3,
        top_k: int = 64,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_types = num_types
        self.top_k = top_k
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Consolidated Query projection: one giant linear layer for all types
        self.q_proj = nn.Linear(hidden_size, hidden_size * num_types, bias=True)
        
        # Type importance mixer: learns to weight Coref vs Lexical vs Discourse per token
        self.type_mixer = nn.Linear(hidden_size, num_types, bias=False)
        self.attn_dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        query_states: torch.Tensor,
        K_typed: torch.Tensor,
        V_typed: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_states: (B, T, D) - R_encoded (v3: aligned with K space)
            K_typed: (B, num_types, T, D)
            V_typed: (B, num_types, T, D)
            
        Returns:
            A_long: (B, T, D) - Aggregated long-range output
            orthogonality_loss: Scalar - Static regularization (always 0 in forward; computed separately)
        """
        B, T, D = query_states.shape
        effective_k = min(self.top_k, T)
        
        # 1. Compute Type Mixing Weights (from query_states which is R_encoded)
        type_weights = F.softmax(self.type_mixer(query_states), dim=-1) # (B, T, num_types)
        
        # 2. Parallel Retrieval for all types (Vectorized Forward Pass)
        # Project all queries at once: (B, T, num_types * D)
        Q_all = self.q_proj(query_states)
        
        # Reshape Q: (B, T, num_types, H, d_k) -> permute -> (B, num_types, H, T, d_k)
        Q = Q_all.view(B, T, self.num_types, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        
        # K_typed, V_typed are (B, num_types, T, D)
        # Reshape to (B, num_types, T, H, d_k) -> permute -> (B, num_types, H, T, d_k)
        K = K_typed.view(B, self.num_types, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        V = V_typed.view(B, self.num_types, T, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        
        # 2. Typed Top-K Retrieval (As per Proposal)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if attention_mask is not None:
            # Ensure mask dtype matches scores for consistent precision
            attention_mask = attention_mask.to(dtype=scores.dtype)
            if attention_mask.dim() == 4:
                scores = scores + attention_mask.unsqueeze(1)
            else:
                scores = scores + attention_mask

        # --- THE ROUTING STEP (Top-K) ---
        if effective_k < T:
            topk_values, _ = torch.topk(scores, k=effective_k, dim=-1)
            threshold = topk_values[..., -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # v3: No dynamic diversity loss computation here — orthogonality loss is static
        # and computed via compute_orthogonality_loss() on weight matrices.

        # 3. Context Aggregation
        out = torch.matmul(self.attn_dropout(attn_weights), V) 
        out = out.permute(0, 3, 1, 2, 4).reshape(B, T, self.num_types, D)

        # 4. Combine outputs via weights: O = sum_t w_t * O_t
        A_long = (out * type_weights.unsqueeze(-1)).sum(dim=-2) # (B, T, D)
        
        return A_long

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Static Orthogonality Regularization on Q projection weight matrices.
        
        Penalizes cosine similarity between the query projection weights of different
        dependency types. Combined with DependencyTypedGist's orthogonality loss.
        """
        D = self.hidden_size
        W = self.q_proj.weight  # (num_types * D, D)
        
        # Split into per-type weight matrices
        type_weights = []
        for t in range(self.num_types):
            w_t = W[t * D : (t + 1) * D]  # (D, D)
            type_weights.append(w_t.reshape(-1))  # (D*D,)
        
        # Compute pairwise cosine similarity penalty
        loss = torch.tensor(0.0, device=W.device, dtype=W.dtype)
        count = 0
        for i in range(self.num_types):
            for j in range(i + 1, self.num_types):
                cos_sim = F.cosine_similarity(type_weights[i].unsqueeze(0),
                                               type_weights[j].unsqueeze(0))
                loss = loss + cos_sim.abs()
                count += 1
        
        return loss / max(count, 1)


# ---------------------------------------------------------------------------
# LongAttention v3: Main Module
# ---------------------------------------------------------------------------

class LongAttention(nn.Module):
    """
    LongAttention v3: Necessity-aware, Dependency-typed Attention.
    
    v3 improvements over v2:
      - Affix integration into local branch
      - Q/K space alignment (query from R_encoded)
      - Static orthogonality regularization (no GPU bottleneck)
      - Gated interpolation for branch merging
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        local_window_size: int = 512,
        top_k: int = 64,
        num_types: int = 3,
        bottleneck_ratio: float = 0.25,
        dropout_prob: float = 0.1,
        layer_idx: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.num_types = num_types

        # Branch 1: Local Sliding Window
        from .local_attention import LocalSlidingWindowAttention
        self.local_attention = LocalSlidingWindowAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=local_window_size,
            dropout_prob=dropout_prob,
            bias=bias,
        )

        # Branch 2: Long-range (Necessity-aware & Typed)
        self.necessity_gate = NecessityGate(hidden_size, bottleneck_ratio)
        
        # S -> R (Root) + A (Affix)
        self.decomposer = FunctionalDecomposer(hidden_size, bottleneck_ratio)
        
        self.typed_gist = DependencyTypedGist(hidden_size, num_types)
        
        self.typed_retrieval = TypedTopKRetrieval(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_types=num_types,
            top_k=top_k,
            dropout_prob=dropout_prob
        )

        # Final output projection (to be initialized with BART's pretrained weights)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Track metric for research analysis
        self.last_gate_val = 0.0
        self.last_diversity_loss = 0.0

    def compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Aggregate static orthogonality loss from both typed_gist and typed_retrieval.
        This replaces the dynamic diversity loss from v2.
        """
        return (self.typed_gist.compute_orthogonality_loss() + 
                self.typed_retrieval.compute_orthogonality_loss()) * 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        
        # ── Local Branch ──
        local_outputs = self.local_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        if TRANSFORMERS_NEW_SIG:
            A_local, local_attn_weights = local_outputs
        else:
            A_local, local_attn_weights, _ = local_outputs

        # ── Long-range Branch ──
        # 1. Necessity Gate g_i
        g_i = self.necessity_gate(hidden_states) # (B, T, 1)
        self.last_gate_val = g_i.mean() # Keep as tensor for backprop/logging

        # 2. Decompose: S -> R (Semantic Root) + A (Functional Affix)
        global_ctx = hidden_states.mean(dim=1, keepdim=True)
        R_encoded, A_encoded, _ = self.decomposer(hidden_states, global_ctx)
        
        # 3. Build Typed Gists (K_t, V_t) — Only from Semantic Roots
        # Per Proposal: only semantically important tokens enter long-range memory
        K_typed, V_typed = self.typed_gist(R_encoded)
        
        # 4. Typed Retrieval — v3: Query from R_encoded (aligned Q/K space)
        A_long = self.typed_retrieval(
            R_encoded,   # v3: Query from decomposed semantic roots (not raw hidden_states)
            K_typed, 
            V_typed, 
            attention_mask
        )
        
        # v3: Static orthogonality loss (computed from weights, not attention distributions)
        if self.training:
            ortho_loss = self.compute_orthogonality_loss()
            self.last_diversity_loss = ortho_loss
        else:
            self.last_diversity_loss = torch.tensor(0.0, device=hidden_states.device)
        
        # ── Output Integration (v3: Gated Interpolation with Affix) ──
        # A_encoded enriches local branch with syntactic/grammatical information
        A_local_combined = A_local + A_encoded
        
        # Gated Interpolation: balanced branch merging
        # Formula: O = OutProj( (1 - g_i) * LocalCombined + g_i * LongContext )
        combined_context = (1.0 - g_i) * A_local_combined + g_i * A_long
        output = self.out_proj(combined_context)
        
        # Track metric for research analysis / Trainer logging
        self.last_gate_val = g_i.mean() 
        
        # Hook for trainer to collect losses (used in metrics and callbacks)
        if self.training:
            output.diversity_loss = self.last_diversity_loss
            output.gate_val = g_i.mean()
            
        if TRANSFORMERS_NEW_SIG:
            return (output, local_attn_weights if output_attentions else None)
        else:
            return (output, local_attn_weights if output_attentions else None, past_key_value)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_types={self.num_types}, "
            f"layer_idx={self.layer_idx}"
        )
