"""
LongAttention: Gated Functional Information Compression Module.

This is the core architectural contribution of the LongAttention proposal.
It implements a two-branch attention mechanism:

  Branch 1 — Local Branch (``LocalSlidingWindowAttention``):
      Dense attention over a short sliding window to capture syntactic
      structure and immediate coherence.

  Branch 2 — Gated Functional Compression Branch (this module):
      1. Gated Functional Decomposer  — separates hidden states into:
            R  (Semantic Root)    ← compressed, semantically rich tokens
            A  (Functional Affix) ← syntactic/grammatical glue tokens
      2. Gating Modulation Mechanism — query-dependent sigmoid gate:
            G_score = σ(X @ W_θ)
      3. Compressed Gist & Reservoir — builds gisted key/value pairs:
            K_gist, V_gist = (G_score ⊙ f_θ(R)) + Codebook(A)
      4. Bidirectional Top-K Retrieval — long-range affinities:
            A_long_i = Σ_{j ∈ TopK(i)} Corr(q_i, K_gist_j) · V_gist_j
      5. Output Integration — adaptive combination with local output:
            O_i = LayerNorm(A_local_i + α_i · A_long_i)

Mathematical Notation (from Proposal §4.1)
------------------------------------------
    R, A = Decompose(S)
    G_score = σ(X W_θ)                          (gating signal)
    K_gist, V_gist = (G_score ⊙ f_θ(R)) + Codebook(A)
    A_long_i = Σ_{j ∈ TopK} Corr(q_i, K_gist_j) · V_gist_j
    O_i = LayerNorm(A_local_i + α_i · A_long_i)
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_attention import LocalSlidingWindowAttention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class FunctionalDecomposer(nn.Module):
    """
    Gated Functional Decomposer.

    Separates the input hidden states ``S`` into two complementary streams:
      - ``R`` (Semantic Root):    Tokens carrying high semantic information density.
                                  These will be gated, compressed, and retrieved long-range.
      - ``A`` (Functional Affix): Tokens serving grammatical/syntactic roles.
                                  These are encoded via a lightweight Codebook embedding.

    Implementation Strategy
    -----------------------
    We use a learned single-layer gating network that, for each token position,
    emits a scalar in (0, 1) indicating its "semantic root probability".
    Hard routing (top-p selection) is approximated by straight-through soft gating
    to remain differentiable during training.

    Args:
        hidden_size:        Model hidden dimension.
        bottleneck_ratio:   Fraction of hidden_size used for the gate projection.
    """

    def __init__(self, hidden_size: int, bottleneck_ratio: float = 0.25) -> None:
        super().__init__()
        bottleneck = max(1, int(hidden_size * bottleneck_ratio))

        # Gate head: projects hidden → scalar routing score
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

        # Root transformation: the "f_θ" in the proposal
        self.root_transform = nn.Linear(hidden_size, hidden_size, bias=False)

        # Affix transformation: lightweight codebook mapping
        # In the full implementation this would be a learned VQ codebook;
        # here we use a linear projection as a differentiable approximation.
        self.affix_codebook = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, global_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose hidden states into Semantic Root and Functional Affix streams.

        Args:
            hidden_states: Tensor of shape (B, T, D).
            global_context: Tensor of shape (B, 1, D).

        Returns:
            Tuple:
                R_encoded:   Semantic Root encoding  (B, T, D).
                A_encoded:   Functional Affix encoding (B, T, D).
                gate_score:  Per-token gate probability in [0,1] (B, T, 1).
        """
        # Context-Aware Gate Score with Temperature Sharpening
        local_logits = self.gate_proj_local(hidden_states)     # (B, T, 1)
        global_logits = self.gate_proj_global(global_context)  # (B, 1, 1)
        
        gate_logits = local_logits + global_logits
        temperature = 0.5
        
        # gate_score ∈ (0, 1): high = Semantic Root, low = Functional Affix
        gate_score = torch.sigmoid(gate_logits / temperature)  # (B, T, 1)

        # Semantic Root stream — only activated tokens contribute significantly
        root_features = self.root_transform(hidden_states)   # (B, T, D) — f_θ(S)
        R_encoded = gate_score * root_features               # (B, T, D) — gated root

        # Functional Affix stream — complement of root gate
        affix_features = self.affix_codebook(hidden_states)  # (B, T, D)
        A_encoded = (1.0 - gate_score) * affix_features      # (B, T, D)

        return R_encoded, A_encoded, gate_score


class GistReservoir(nn.Module):
    """
    Compressed Gist & Reservoir Builder.

    Combines the gated Semantic Root (R) and Functional Affix Codebook (A)
    to build compressed gist key/value pairs used for long-range retrieval.

    From the proposal (§4.1):
        K_gist, V_gist = (G_score ⊙ f_θ(R)) + Codebook(A)

    Since `R_encoded` already incorporates the gate (``G_score ⊙ f_θ(R)``),
    the gist is:
        K_gist = R_encoded + A_encoded   (additive fusion of both streams)
        V_gist = a separate projection of the same fusion

    Args:
        hidden_size:  Model hidden dimension.
        num_heads:    Number of attention heads (for per-head projections).
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Project fused representation into gist keys and values
        self.k_gist_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_gist_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        R_encoded: torch.Tensor,
        A_encoded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build compressed gist key/value pairs.

        Args:
            R_encoded:  Semantic Root stream (B, T, D).
            A_encoded:  Functional Affix stream (B, T, D).

        Returns:
            K_gist: Compressed Keys  (B, T, D).
            V_gist: Compressed Values (B, T, D).
        """
        # Additive fusion: K_gist, V_gist = (G_score ⊙ f_θ(R)) + Codebook(A)
        fused = R_encoded + A_encoded  # (B, T, D)

        K_gist = self.k_gist_proj(fused)  # (B, T, D)
        V_gist = self.v_gist_proj(fused)  # (B, T, D)

        return K_gist, V_gist


class BidirectionalTopKRetrieval(nn.Module):
    """
    Bidirectional Top-K Long-Range Retrieval.

    For each query q_i, computes long-range attention by:
      1. Computing correlation scores against ALL gist keys.
      2. Masking to keep only the Top-K highest scoring positions.
      3. Normalising via softmax over the kept positions.
      4. Computing the weighted sum of gist values.

    From the proposal (§4.1):
        A_long_i = Σ_{j ∈ TopK} Corr(q_i, K_gist_j) · V_gist_j

    "Bidirectional" means queries can attend both backward and forward,
    unlike the causal-only local branch. This mirrors cross-sentence and
    cross-paragraph reasoning.

    Args:
        hidden_size:  Model hidden dimension.
        num_heads:    Number of attention heads.
        top_k:        Number of positions to retrieve per query.
        dropout_prob: Dropout on retrieval attention weights.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        top_k: int = 64,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.top_k = top_k
        self.scale = math.sqrt(self.head_dim)

        # Query projection for long-range retrieval (separate from local Q)
        self.q_long_proj = nn.Linear(hidden_size, hidden_size, bias=False)
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
        K_gist: torch.Tensor,
        V_gist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bidirectional Top-K long-range attention output.

        Args:
            hidden_states:  Input states for query projection (B, T, D).
            K_gist:         Compressed gist keys (B, T, D).
            V_gist:         Compressed gist values (B, T, D).

        Returns:
            A_long: Long-range contextual output (B, T, D).
        """
        B, T, _ = hidden_states.shape
        effective_k = min(self.top_k, T)

        # Project queries for long-range retrieval
        Q_long = self._split_heads(self.q_long_proj(hidden_states))  # (B, H, T, dk)
        K_gist_h = self._split_heads(K_gist)                         # (B, H, T, dk)
        V_gist_h = self._split_heads(V_gist)                         # (B, H, T, dk)

        # Correlation scores: (B, H, T_q, T_k) — bidirectional (no causal mask here)
        corr_scores = torch.matmul(Q_long, K_gist_h.transpose(-2, -1)) / self.scale

        # Top-K masking: keep only the top-k positions per query
        if effective_k < T:
            # Get threshold for each query: the k-th largest score
            topk_values, _ = torch.topk(corr_scores, k=effective_k, dim=-1)
            threshold = topk_values[..., -1].unsqueeze(-1)  # (B, H, T, 1)
            # Mask positions below threshold to -inf
            mask = corr_scores < threshold
            corr_scores = corr_scores.masked_fill(mask, float("-inf"))

        # Normalise over retrieved positions with Temperature Sharpening
        attn_temperature = 0.5
        attn_weights = F.softmax(corr_scores / attn_temperature, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of gist values
        A_long = torch.matmul(attn_weights, V_gist_h)  # (B, H, T, dk)
        A_long = self._merge_heads(A_long)              # (B, T, D)

        return A_long


# ---------------------------------------------------------------------------
# LongAttention: Main Module
# ---------------------------------------------------------------------------

class LongAttention(nn.Module):
    """
    LongAttention: Gated Functional Information Compression Module.

    Integrates two parallel branches to optimise the trade-off between
    local syntactic preservation and global semantic retrieval:

    **Local Branch** (``LocalSlidingWindowAttention``):
        Dense causal attention restricted to `local_window_size` adjacent
        tokens. Captures grammar, named-entity coherence, and immediate
        co-reference. Output: ``A_local``.

    **Gated Functional Compression Branch** (this class body):
        1. ``FunctionalDecomposer`` separates S → R (root) + A (affix).
        2. Query-dependent sigmoid gate: ``G_score = σ(X W_θ)``.
        3. ``GistReservoir`` fuses R and A into compressed K_gist / V_gist.
        4. ``BidirectionalTopKRetrieval`` retrieves ``A_long`` via Top-K.

    **Output Integration**:
        ``O_i = LayerNorm(A_local_i + α_i · A_long_i)``
        where ``α_i`` is a learned scalar mixing weight (per-position).

    Args:
        hidden_size:        D — model hidden dimension.
        num_heads:          H — number of attention heads.
        local_window_size:  W — local branch window size.
        top_k:              K — number of positions retrieved in long-range branch.
        bottleneck_ratio:   Ratio for FunctionalDecomposer gate projection.
        dropout_prob:       Dropout applied to attention weights.
        layer_idx:          Layer index (for logging/analysis hooks).
        bias:               Whether to add bias to projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        local_window_size: int = 512,
        top_k: int = 64,
        bottleneck_ratio: float = 0.25,
        dropout_prob: float = 0.0,
        layer_idx: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx

        # ── Branch 1: Local sliding-window dense attention ──────────────────
        self.local_attention = LocalSlidingWindowAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=local_window_size,
            dropout_prob=dropout_prob,
            bias=bias,
        )

        # ── Branch 2: Gated Functional Compression ──────────────────────────

        # Step 1 — Gated Functional Decomposer: S → (R, A, G_score)
        self.decomposer = FunctionalDecomposer(
            hidden_size=hidden_size,
            bottleneck_ratio=bottleneck_ratio,
        )

        # Step 2 — Gating Modulation Mechanism: G_score = σ(X W_θ)
        # We split the modulation into local and global context weights
        self.W_theta_local = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_theta_global = nn.Linear(hidden_size, hidden_size, bias=False)

        # Step 3 — Compressed Gist & Reservoir
        self.gist_reservoir = GistReservoir(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        # Step 4 — Bidirectional Top-K Retrieval
        self.topk_retrieval = BidirectionalTopKRetrieval(
            hidden_size=hidden_size,
            num_heads=num_heads,
            top_k=top_k,
            dropout_prob=dropout_prob,
        )

        # ── Step 5 — Output Integration ─────────────────────────────────────
        # α_i: per-position learned scalar mixing weight (broadcast over D)
        # Initialised near 0.5 so local and long contributions start balanced.
        self.alpha_proj = nn.Linear(hidden_size, 1, bias=False)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of LongAttention.

        Steps
        -----
        1. Local Branch  → A_local
        2. Decompose     → R, A, G_score
        3. Explicit gate modulation:  G_score = σ(X @ W_θ)
        4. Build K_gist, V_gist from gist reservoir
        5. Top-K long-range retrieval → A_long
        6. Output integration: O = LayerNorm(A_local + α ⊙ A_long)

        Args:
            hidden_states:      Input tensor (B, T, D).
            attention_mask:     Optional additive attention mask (B, 1, T, T).
            position_ids:       Position IDs (unused by default but kept for compatibility).
            past_key_value:     KV cache (passed through, not currently used).
            output_attentions:  If True, return local attention weights.
            use_cache:          If True, return past_key_value (no-op here).
            cache_position:     Cache position tensor (unused, for compatibility).
            **kwargs:           Absorbs any extra BART forward kwargs.

        Returns:
            Tuple: (output, local_attn_weights or None, past_key_value or None)
        """
        # ── Step 1: Local Branch ─────────────────────────────────────────────
        A_local, local_attn_weights = self.local_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        # A_local: (B, T, D)

        # ── Step 2: Gated Functional Decomposer ─────────────────────────────
        # Extract global context to empower routing mechanisms
        global_context = hidden_states.mean(dim=1, keepdim=True) # (B, 1, D)

        # R = Semantic Root encoding, A = Functional Affix encoding
        R_encoded, A_encoded, gate_score_decomp = self.decomposer(hidden_states, global_context)
        # R_encoded: (B, T, D), A_encoded: (B, T, D), gate_score_decomp: (B, T, 1)

        # ── Step 3: Explicit Gating Modulation G_score = σ(X W_θ) ──────────
        # Context-Aware explicit routing gate
        mod_local = self.W_theta_local(hidden_states)     # (B, T, D)
        mod_global = self.W_theta_global(global_context)  # (B, 1, D)
        
        mod_logits = mod_local + mod_global
        temperature = 0.5
        G_score = torch.sigmoid(mod_logits / temperature) # (B, T, D)

        # Apply the explicit gate to the semantic root encoding
        R_gated = G_score * R_encoded  # (B, T, D)

        # ── Step 4: Compressed Gist & Reservoir ─────────────────────────────
        # K_gist, V_gist = (G_score ⊙ f_θ(R)) + Codebook(A)
        K_gist, V_gist = self.gist_reservoir(R_gated, A_encoded)
        # K_gist: (B, T, D), V_gist: (B, T, D)

        # ── Step 5: Bidirectional Top-K Retrieval ───────────────────────────
        # A_long_i = Σ_{j ∈ TopK(i)} Corr(q_i, K_gist_j) · V_gist_j
        A_long = self.topk_retrieval(hidden_states, K_gist, V_gist)
        # A_long: (B, T, D)

        # ── Step 6: Output Integration ───────────────────────────────────────
        # α_i: per-position mixing weight from temporary latent factor
        # O_i = LayerNorm(A_local_i + α_i · A_long_i)
        alpha = torch.sigmoid(self.alpha_proj(hidden_states))  # (B, T, 1)
        combined = A_local + alpha * A_long                    # (B, T, D)
        output = self.output_norm(combined)                    # (B, T, D)

        # Build return tuple matching BART's expected format
        outputs = (output, local_attn_weights if output_attentions else None)
        if use_cache:
            outputs += (past_key_value,)
        return outputs

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"layer_idx={self.layer_idx}, "
            f"window={self.local_attention.window_size}, "
            f"top_k={self.topk_retrieval.top_k}"
        )
