"""
metrics.py — Evaluation Metrics
================================

End-task:    BLEU / COMET (DocMT),  EM / F1 (QA)
Efficiency:  latency, memory, FLOPs
Routing:     gate activation, type distribution, segment entropy
             (maps to Proposal §8 "Routing behavior" + "Specialization" axes)
Faithfulness: router vs gold evidence overlap
             (maps to Proposal §8 "Faithfulness" axis)

QA normalization follows SQuAD v2 / TriviaQA official eval script:
  lowercase → remove punctuation → remove articles → collapse whitespace
  Required so "The Amazon" vs "Amazon" counts as correct.

EM / F1 with aliases:
  compute_em_aliases / compute_f1_aliases take List[List[str]] as refs
  and score a prediction against ALL aliases, taking the best match.
  This is the correct protocol for TriviaQA evaluation.
"""

from __future__ import annotations

import re
import string
import time
import warnings
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# SQuAD / TriviaQA standard text normalization
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


# ---------------------------------------------------------------------------
# QA metrics — single-ref versions (used internally)
# ---------------------------------------------------------------------------


def _em_single(pred: str, ref: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(ref))


def _f1_single(pred: str, ref: str) -> float:
    pred_tokens  = _normalize_answer(pred).split()
    truth_tokens = _normalize_answer(ref).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    common   = Counter(pred_tokens) & Counter(truth_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Public QA metric functions
# ---------------------------------------------------------------------------


def compute_em(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """EM against a single ref string per sample (used for DocMT-style refs)."""
    c = sum(_em_single(p, r) for p, r in zip(preds, refs))
    return {"em": c / max(len(preds), 1) * 100}


def compute_f1(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Token-level F1 against a single ref string per sample."""
    scores = [_f1_single(p, r) for p, r in zip(preds, refs)]
    return {"f1": float(np.mean(scores)) * 100}


def compute_em_aliases(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    """
    EM scored against ALL answer aliases — correct TriviaQA protocol.
    refs[i] is the list of all valid answers for sample i.
    Takes best match (max EM across aliases).
    """
    scores = []
    for pred, alias_list in zip(preds, refs):
        best = max((_em_single(pred, a) for a in alias_list), default=0.0)
        scores.append(best)
    return {"em": float(np.mean(scores)) * 100}


def compute_f1_aliases(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    """
    Token F1 scored against ALL answer aliases — correct TriviaQA protocol.
    Takes best match (max F1 across aliases).
    """
    scores = []
    for pred, alias_list in zip(preds, refs):
        best = max((_f1_single(pred, a) for a in alias_list), default=0.0)
        scores.append(best)
    return {"f1": float(np.mean(scores)) * 100}


# ---------------------------------------------------------------------------
# DocMT metrics
# ---------------------------------------------------------------------------


def compute_bleu(preds: List[str], refs: List[str]) -> Dict[str, float]:
    import sacrebleu
    b = sacrebleu.corpus_bleu(preds, [refs])
    return {"bleu": b.score, "bleu_bp": b.bp}


def compute_comet(preds, refs, srcs, model_name="Unbabel/wmt22-comet-da"):
    try:
        from comet import download_model, load_from_checkpoint
        path  = download_model(model_name)
        model = load_from_checkpoint(path)
        data  = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
        use_gpu = torch.cuda.is_available()
        gpus = 1 if use_gpu else 0
        try:
            out = model.predict(data, batch_size=8, gpus=gpus)
        except Exception as gpu_exc:
            if use_gpu:
                warnings.warn(f"COMET GPU scoring failed, fallback to CPU: {gpu_exc}")
                out = model.predict(data, batch_size=8, gpus=0)
            else:
                raise
        return {"comet": float(out.system_score)}
    except Exception as exc:
        warnings.warn(f"COMET scoring failed: {exc}")
        return {"comet": -1.0}


def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        r1, r2, rl = [], [], []
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return {
            "rouge1": float(np.mean(r1)) * 100,
            "rouge2": float(np.mean(r2)) * 100,
            "rougeL": float(np.mean(rl)) * 100,
        }
    except ImportError:
        return {"rouge1": -1.0, "rouge2": -1.0, "rougeL": -1.0}


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


def compute_efficiency(model, input_ids, attention_mask=None, n=5):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        if attention_mask is None:
            model(input_ids)  # warmup
        else:
            model(input_ids, attention_mask=attention_mask)  # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            if attention_mask is None:
                model(input_ids)
            else:
                model(input_ids, attention_mask=attention_mask)
    ms  = (time.perf_counter() - t0) / n * 1000
    mem = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 2
        if device.type == "cuda" else 0
    )
    params = sum(p.numel() for p in model.parameters())
    return {
        "latency_ms":       round(ms,  2),
        "peak_memory_mb":   round(mem, 2),
        "estimated_gflops": round(2 * params * input_ids.shape[1] / 1e9, 4),
    }


# ---------------------------------------------------------------------------
# Routing analysis  (Proposal §8 — Routing behavior + Specialization axes)
# ---------------------------------------------------------------------------


def compute_routing_stats(layer_infos):
    """
    gate_activation_rate : mean g_i — close to 1 → gate always open (bad),
                           close to 0 → null-route too aggressive (bad).
    type_distribution    : marginal distribution over dependency types;
                           used to detect Type Collapse.
    routing_entropy      : entropy of top-k segment weights; low = confident.
    """
    gates, types, entropies = [], [], []
    for info in layer_infos:
        g  = info["gate"]
        tm = info["type_mask"]
        gates.append(float(g.mean()))
        types.append(tm.mean(dim=(0, 1, 2)).tolist())
        if "topk_w" in info:
            tw = info["topk_w"]
            entropies.append(
                float(-(tw * (tw + 1e-8).log()).sum(-1).mean())
            )
        else:
            entropies.append(0.0)
    return {
        "gate_activation_rate": float(np.mean(gates)),
        "gate_per_layer":       gates,
        "type_distribution":    types,
        "routing_entropy":      entropies,
    }


# ---------------------------------------------------------------------------
# Faithfulness  (Proposal §8 — Faithfulness axis)
# ---------------------------------------------------------------------------


def compute_faithfulness(selected, gold):
    """
    Overlap between routed segments (selected) and gold evidence segments.
    Both are lists of segment-index sets per sample.
    """
    ps, rs, fs = [], [], []
    for sel, g in zip(selected, gold):
        ss, gs = set(sel), set(g)
        if not ss and not gs:
            ps.append(1); rs.append(1); fs.append(1); continue
        if not ss or not gs:
            ps.append(0); rs.append(0); fs.append(0); continue
        o = ss & gs
        p = len(o) / len(ss)
        r = len(o) / len(gs)
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        ps.append(p); rs.append(r); fs.append(f)
    return {
        "faithfulness_precision": float(np.mean(ps)),
        "faithfulness_recall":    float(np.mean(rs)),
        "faithfulness_f1":        float(np.mean(fs)),
    }
