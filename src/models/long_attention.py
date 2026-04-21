"""
LongAttention v2: Necessity-Aware & Dependency-Typed Long-Context Attention.

This module implements the core architectural contribution of the LongAttention v2 proposal.
It integrates two parallel branches:

  Branch 1 — Local Branch (LocalSlidingWindowAttention):
      Dense attention over a short sliding window to capture syntactic structure
      and immediate coherence.

  Branch 2 — Gated & Typed Long-range Branch:
      1. Necessity Gating (g_i): Decides IF long-range information is needed.
      2. Functional Decomposition: Separates hidden states into Semantic Roots and Functional Affixes.
      3. Dependency-Typed Routing: Specialized channels for:
         - Coreference Resolution
         - Lexical Consistency
         - Discourse Relations
      4. Top-K Retrieval: Efficiently routes to the most relevant context segments.
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
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, 1, bias=False),
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
        # Each type has its own semantic projection
        self.type_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size * 2, bias=True) 
            for _ in range(num_types)
        ])
        
    def forward(self, R_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            K_typed: (B, num_types, T, D)
            V_typed: (B, num_types, T, D)
        """
        B, T, D = R_encoded.shape
        ks, vs = [], []
        for proj in self.type_projs:
            kv = proj(R_encoded) # (B, T, 2*D)
            k, v = kv.chunk(2, dim=-1)
            ks.append(k.unsqueeze(1))
            vs.append(v.unsqueeze(1))
        
        K_typed = torch.cat(ks, dim=1) # (B, num_types, T, D)
        V_typed = torch.cat(vs, dim=1) # (B, num_types, T, D)
        return K_typed, V_typed


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

        # Query projections: one per type to allow type-specific relevance
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(num_types)
        ])
        
        # Type importance mixer: learns to weight Coref vs Lexical vs Discourse per token
        self.type_mixer = nn.Linear(hidden_size, num_types, bias=False)
        self.attn_dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        K_typed: torch.Tensor,
        V_typed: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A_long: (B, T, D) - Aggregated long-range output
            diversity_loss: Scalar - Regularization to ensure types don't collapse
        """
        B, T, D = hidden_states.shape
        effective_k = min(self.top_k, T)
        
        # 1. Compute Type Mixing Weights
        type_weights = F.softmax(self.type_mixer(hidden_states), dim=-1) # (B, T, num_types)
        
        # 2. Parallel Retrieval for each type
        type_outputs = []
        all_attn_maps = []
        
        for t_idx in range(self.num_types):
            # Project query for this type
            Q_t = self.q_projs[t_idx](hidden_states) # (B, T, D)
            Q_t = Q_t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, d_k)
            
            K_t = K_typed[:, t_idx].view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, d_k)
            V_t = V_typed[:, t_idx].view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, d_k)
            
            # Scores: (B, H, T, T)
            scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) / self.scale
            
            # Apply padding mask (e.g., -inf on <PAD> tokens)
            if attention_mask is not None:
                # attention_mask is usually (B, 1, T, T) or (B, 1, 1, T) provided by HF
                scores = scores + attention_mask

            # Top-K
            if effective_k < T:
                topk_values, _ = torch.topk(scores, k=effective_k, dim=-1)
                threshold = topk_values[..., -1].unsqueeze(-1)
                scores = scores.masked_fill(scores < threshold, float("-inf"))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            all_attn_maps.append(attn_weights.mean(dim=1)) # Store avg head attention for diversity loss
            
            out_t = torch.matmul(self.attn_dropout(attn_weights), V_t) # (B, H, T, d_k)
            out_t = out_t.transpose(1, 2).reshape(B, T, D)
            type_outputs.append(out_t.unsqueeze(-2)) # (B, T, 1, D)

        # 3. Combine outputs via weights: O = sum_t w_t * O_t
        A_long_stacked = torch.cat(type_outputs, dim=-2) # (B, T, num_types, D)
        A_long = (A_long_stacked * type_weights.unsqueeze(-1)).sum(dim=-2) # (B, T, D)
        
        # 4. Compute Diversity Loss (Cosine similarity between attention maps)
        # We want to minimize similarity between different type attention patterns
        diversity_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.num_types > 1:
            for i in range(self.num_types):
                for j in range(i + 1, self.num_types):
                    # Flatten T,T to compare patterns
                    sim = F.cosine_similarity(all_attn_maps[i].view(B, -1), all_attn_maps[j].view(B, -1), dim=-1)
                    diversity_loss += sim.mean()
        
        return A_long, diversity_loss


# ---------------------------------------------------------------------------
# LongAttention v2: Main Module
# ---------------------------------------------------------------------------

class LongAttention(nn.Module):
    """
    LongAttention v2: Necessity-aware, Dependency-typed Attention.
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

        # 2. Decompose Gist
        global_ctx = hidden_states.mean(dim=1, keepdim=True)
        decomp_outputs = self.decomposer(hidden_states, global_ctx)
        
        # Internal modules like Decomposer should return consistent 3-tuples or we handle versioning
        # FunctionalDecomposer.forward returns 3 values currently, let's keep it safe.
        R_encoded, A_encoded, _ = decomp_outputs
        
        # 3. Build Typed Gists (K_t, V_t)
        K_typed, V_typed = self.typed_gist(R_encoded + A_encoded)
        
        # 4. Typed Retrieval
        A_long, diversity_loss = self.typed_retrieval(
            hidden_states=hidden_states, 
            K_typed=K_typed, 
            V_typed=V_typed, 
            attention_mask=attention_mask
        )
        self.last_diversity_loss = diversity_loss # Keep as tensor for backprop
        
        # ── Output Integration ──
        # Formula: O = OutProj( LocalContext + g_i * LongContext )
        # This perfectly matches standard Transformer output projection logic.
        combined_context = A_local + g_i * A_long
        output = self.out_proj(combined_context)
        
        # Track metric for research analysis / Trainer logging
        self.last_gate_val = g_i.mean() 
        self.last_diversity_loss = diversity_loss 
        
        # Hook for trainer to collect losses (used in metrics and callbacks)
        # We attach these as attributes to the output tensor so the trainer can find them
        if self.training:
            output.diversity_loss = diversity_loss
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
