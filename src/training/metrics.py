"""
Translation Metrics for LongAttention NMT Experiments.

Provides computation functions for:
  - SacreBLEU  (corpus-level BLEU via the ``sacrebleu`` standard)
  - ChrF++     (character n-gram F-score with word order)
  - COMET      (neural reference-based / reference-free MT evaluation)
  - Attention Sink Ratio (percentage of attention mass on the first token)

All functions are designed to be composable and callable inside a
HuggingFace ``compute_metrics`` callback or standalone evaluation loops.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — avoids hard failures if optional dependencies are missing
# ---------------------------------------------------------------------------

def _load_evaluate_metric(name: str):
    """Load an ``evaluate`` metric, with a clear error on missing deps."""
    try:
        import evaluate
        return evaluate.load(name)
    except Exception as exc:
        raise ImportError(
            f"Could not load evaluate metric '{name}'. "
            f"Install with: pip install evaluate sacrebleu\nOriginal error: {exc}"
        ) from exc


def _load_comet_model(model_name: str = "Unbabel/wmt22-comet-da"):
    """Load a COMET model from unbabel-comet, with a clear error on missing deps."""
    try:
        from comet import load_from_checkpoint, download_model
        checkpoint = download_model(model_name)
        return load_from_checkpoint(checkpoint)
    except Exception as exc:
        raise ImportError(
            f"Could not load COMET model '{model_name}'. "
            f"Install with: pip install unbabel-comet\nOriginal error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# SacreBLEU
# ---------------------------------------------------------------------------

def compute_sacrebleu(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute corpus-level SacreBLEU score.

    SacreBLEU tokenises consistently and produces standardised BLEU scores
    comparable across papers.

    Args:
        predictions: List of model hypotheses (detokenised strings).
        references:  List of reference translations (one ref per hypothesis).

    Returns:
        Dict with key ``"sacrebleu"`` → BLEU score (0–100).
    """
    metric = _load_evaluate_metric("sacrebleu")
    # evaluate expects references as list-of-lists
    result = metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
    )
    score = result["score"]
    logger.debug("SacreBLEU: %.2f", score)
    return {"sacrebleu": round(score, 4)}


# ---------------------------------------------------------------------------
# ChrF++
# ---------------------------------------------------------------------------

def compute_chrf(
    predictions: List[str],
    references: List[str],
    word_order: int = 2,
) -> Dict[str, float]:
    """
    Compute ChrF++ score (character n-gram F-score with word order penalties).

    ChrF++ (word_order=2) is the recommended variant for MT evaluation and
    correlates well with human judgements, especially for morphologically
    rich languages.

    Args:
        predictions: List of model hypotheses.
        references:  List of reference translations.
        word_order:  Word n-gram order (2 for ChrF++, 0 for ChrF).

    Returns:
        Dict with key ``"chrf"`` → ChrF++ score (0–100).
    """
    metric = _load_evaluate_metric("chrf")
    result = metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
        word_order=word_order,
    )
    score = result["score"]
    logger.debug("ChrF++ (word_order=%d): %.2f", word_order, score)
    return {"chrf": round(score, 4)}


# ---------------------------------------------------------------------------
# COMET
# ---------------------------------------------------------------------------

def compute_comet(
    sources: List[str],
    predictions: List[str],
    references: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 16,
    gpus: int = 1,
) -> Dict[str, float]:
    """
    Compute COMET score using Unbabel's WMT22 COMET-DA model.

    COMET is a neural metric that correlates strongly with human DA (Direct
    Assessment) scores. It requires source sentences alongside hypotheses
    and references.

    Args:
        sources:      Source sentences (original language).
        predictions:  Model hypotheses.
        references:   Reference translations.
        model_name:   COMET model checkpoint to use.
        batch_size:   Batch size for COMET inference.
        gpus:         Number of GPUs for COMET inference (0 = CPU).

    Returns:
        Dict with keys ``"comet_mean"`` and ``"comet_std"``.
    """
    try:
        comet_model = _load_comet_model(model_name)
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, predictions, references)
        ]
        output = comet_model.predict(
            data,
            batch_size=batch_size,
            gpus=gpus,
            progress_bar=False,
        )
        scores: List[float] = output.scores
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        logger.debug("COMET: mean=%.4f std=%.4f", mean_score, std_score)
        return {"comet_mean": round(mean_score, 4), "comet_std": round(std_score, 4)}
    except Exception as exc:
        logger.warning("COMET computation failed: %s. Returning NaN.", exc)
        return {"comet_mean": float("nan"), "comet_std": float("nan")}


# ---------------------------------------------------------------------------
# Attention Sink Ratio
# ---------------------------------------------------------------------------

def compute_attention_sink_ratio(
    attention_weights: torch.Tensor,
    sink_token_index: int = 0,
) -> float:
    """
    Compute the Attention Sink Ratio for a batch of attention maps.

    The "attention sink" phenomenon refers to the tendency of Transformer
    models to over-focus on the first token (index 0), regardless of its
    semantic content. This metric measures that bias.

    Definition (from LongAttention proposal §5):
        Attention Sink Ratio = mean over all (heads, query positions) of the
        attention weight assigned to token ``sink_token_index``.

    Args:
        attention_weights:  Attention weight tensor of shape
                            (batch, heads, seq_len_q, seq_len_k).
        sink_token_index:   Which key position to treat as the "sink" (default: 0).

    Returns:
        Mean attention sink ratio as a float in [0, 1].
        e.g. 0.467 means 46.7% of attention is focused on token 0.
    """
    # attention_weights: (B, H, T_q, T_k)
    # Weight on the sink token for every (batch, head, query) triple
    sink_weights = attention_weights[..., sink_token_index]  # (B, H, T_q)
    ratio = sink_weights.mean().item()
    return ratio


def aggregate_sink_ratios(
    per_layer_ratios: List[float],
) -> Dict[str, float]:
    """
    Aggregate per-layer attention sink ratios into summary statistics.

    Args:
        per_layer_ratios: List of sink ratios, one per attention layer.

    Returns:
        Dict with ``"sink_ratio_mean"``, ``"sink_ratio_max"``, ``"sink_ratio_min"``.
    """
    if not per_layer_ratios:
        return {"sink_ratio_mean": 0.0, "sink_ratio_max": 0.0, "sink_ratio_min": 0.0}
    arr = np.array(per_layer_ratios)
    return {
        "sink_ratio_mean": round(float(arr.mean()), 6),
        "sink_ratio_max": round(float(arr.max()), 6),
        "sink_ratio_min": round(float(arr.min()), 6),
    }


# ---------------------------------------------------------------------------
# Root Fidelity Score (New Metric — Proposal §7.3)
# ---------------------------------------------------------------------------

def compute_root_fidelity_score(
    gate_scores_compressed: torch.Tensor,
    gate_scores_uncompressed: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute the Root Fidelity Score (RFS) — a new metric from the proposal.

    RFS measures what fraction of "Semantic Root" tokens (gate_score > threshold)
    identified by the uncompressed reference model are retained in the
    compressed Dynamic Semantic Reservoir.

    Definition:
        Let Roots_ref   = {i : gate_score_uncompressed[i] > threshold}
        Let Roots_comp  = {i : gate_score_compressed[i]  > threshold}
        RFS = |Roots_ref ∩ Roots_comp| / |Roots_ref|

    Args:
        gate_scores_compressed:   Gate probabilities after compression (B, T).
        gate_scores_uncompressed: Gate probabilities without compression (B, T).
        threshold:                Scalar threshold for root classification.

    Returns:
        Dict with key ``"root_fidelity_score"`` ∈ [0, 1].
    """
    roots_ref  = gate_scores_uncompressed > threshold   # (B, T) bool
    roots_comp = gate_scores_compressed  > threshold    # (B, T) bool

    num_ref   = roots_ref.float().sum()
    num_both  = (roots_ref & roots_comp).float().sum()

    if num_ref.item() == 0:
        return {"root_fidelity_score": 1.0}  # No roots → trivially perfect

    rfs = (num_both / num_ref).item()
    return {"root_fidelity_score": round(rfs, 4)}


# ---------------------------------------------------------------------------
# Metric Factory for HuggingFace Trainer
# ---------------------------------------------------------------------------

def make_compute_metrics(
    tokenizer: Any,
    sources_for_comet: Optional[List[str]] = None,
    use_comet: bool = True,
):
    """
    Factory that returns a compute_metrics function for HF Trainer.
    
    Args:
        tokenizer: Tokenizer used to decode predictions and labels.
        sources_for_comet: Optional list of source strings, required if use_comet=True.
        use_comet: Whether to attempt computing the COMET metric.
    """
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        
        # If logits is a tuple (standard for some models), the first element is the actual logits
        if isinstance(logits, tuple):
            logits = logits[0]
            
        # Decode predictions (greedy decoding from logits)
        # Note: HF Seq2SeqTrainer with predict_with_generate=True sends token IDs, not logits
        preds = logits
        if len(preds.shape) == 3: # Logits case
            preds = np.argmax(preds, axis=-1)
            
        # Replace -100 in labels and preds as the fast tokenizer can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        
        # Decode to strings
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        
        # 1. SacreBLEU
        results = compute_sacrebleu(decoded_preds, decoded_labels)
        
        # 2. ChrF++
        results.update(compute_chrf(decoded_preds, decoded_labels))
        
        # 3. COMET (Requires sources)
        if use_comet and sources_for_comet is not None:
            # Match current batch size if trainer sends subsets? 
            # Usually eval_preds contains the full eval set.
            if len(sources_for_comet) == len(decoded_preds):
                results.update(compute_comet(sources_for_comet, decoded_preds, decoded_labels))
            else:
                logger.warning("COMET: source/hypo length mismatch. Skipping COMET.")
                
        return results

    return compute_metrics
