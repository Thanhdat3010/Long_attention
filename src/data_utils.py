"""
data_utils.py — Data Loading for DocMT & QA
=============================================

  Progress bars are shown during dataset tokenization via tqdm so you
  can track preprocessing instead of staring at a frozen terminal.
=============================================

QA CSV schema (required columns):
    id          : sample identifier (e.g. "train_0")
    context     : long passage with [PARA] markers, MUST contain answer_text
    question    : question string
    answer_text : exact answer string as it appears in context
    start_char  : character index where answer_text begins in context
                  i.e. context[start_char : start_char+len(answer_text)] == answer_text
    answers     : JSON list of all valid answer aliases (used for EM/F1 eval)
                  e.g. '["William Shakespeare", "Shakespeare"]'

DocMT CSV schema:
    source, target

Token alignment (SQuAD-style, CRITICAL):
    We read start_char directly from the CSV (pre-computed by your converter).
    We then use offset_mapping from a Fast tokenizer to map:
        start_char  →  start_token_idx
        end_char    →  end_token_idx
    This is the ONLY correct approach for BPE tokenizers. The old _find_span
    trick (matching token sub-sequences after separate tokenization) fails
    because BPE splits context differently than standalone text.
"""

from __future__ import annotations

import ast
import csv
import json
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tokenizer  —  MUST be Fast to support offset_mapping
# ---------------------------------------------------------------------------


def get_tokenizer(name_or_path: str, max_length: int = 4096):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    tok.model_max_length = max_length
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Generic file loaders
# ---------------------------------------------------------------------------


def _load_rows(path: str, cols: List[str]) -> List[Dict[str, str]]:
    """Load rows from CSV / TSV / JSON / JSONL."""
    if path.endswith(".json") or path.endswith(".jsonl"):
        rows = []
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    obj = json.loads(line)
                    rows.append({c: obj[c] for c in cols})
            else:
                for obj in json.load(f):
                    rows.append({c: obj[c] for c in cols})
        return rows

    delim = "\t" if path.endswith(".tsv") else ","
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=delim):
            rows.append({c: row[c] for c in cols})
    return rows


def _parse_answers(raw: str) -> List[str]:
    """Parse the answers column — accepts JSON list or Python list literal."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except Exception:
        return [raw]


# ---------------------------------------------------------------------------
# DocMT Dataset
# ---------------------------------------------------------------------------


class DocMTDataset(Dataset):
    """source, target parallel corpus."""

    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = _load_rows(path, ["source", "target"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        src = self.tokenizer(r["source"], max_length=self.max_length,
                             truncation=True, padding="max_length", return_tensors="pt")
        tgt = self.tokenizer(r["target"], max_length=self.max_length,
                             truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids":      src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels":         tgt["input_ids"].squeeze(0),
            "source_text":    r["source"],
            "target_text":    r["target"],
        }


# ---------------------------------------------------------------------------
# QA Dataset — SQuAD-style, reads start_char from CSV
# ---------------------------------------------------------------------------


class QADataset(Dataset):
    """
    Extractive QA dataset.

    Reads pre-computed start_char from CSV so no str.find() heuristic is
    needed.  Token positions are derived via offset_mapping (Fast tokenizer).

    Each sample returned:
        input_ids             : (max_length,)  token ids
        attention_mask        : (max_length,)  1 for real tokens
        global_attention_mask : (max_length,)  1 for CLS + question tokens
        labels                : (2,)           [start_tok, end_tok]
        answer_text           : str            primary gold answer
        answers               : List[str]      all valid aliases (for eval)
    """

    QA_COLS = ["id", "context", "question", "answer_text", "start_char", "answers"]

    def __init__(self, path: str, tokenizer, max_length: int = 4096):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, Any]] = []
        self._build(path)

    # ------------------------------------------------------------------
    def _build(self, path: str):
        raw = _load_rows(path, self.QA_COLS)

        n_bad_char  = 0   # start_char mismatch / invalid
        n_truncated = 0   # answer truncated by max_length

        for row in tqdm(raw, desc=f"[QADataset] Tokenizing {path}", unit="sample"):
            sample_id   = row["id"]
            context     = row["context"]
            question    = row["question"]
            answer_text = row["answer_text"]
            answers     = _parse_answers(row["answers"])

            # ── Read pre-computed start_char from CSV ───────────────────
            try:
                start_char = int(row["start_char"])
            except (ValueError, TypeError):
                n_bad_char += 1
                continue

            end_char = start_char + len(answer_text)  # exclusive

            # Sanity check: context[start_char:end_char] must match answer
            actual = context[start_char:end_char]
            if actual.lower() != answer_text.lower():
                n_bad_char += 1
                continue

            # ── Tokenize (question, context) jointly ────────────────────
            # truncation="only_second" → question is never cut, only context
            enc = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                truncation="only_second",
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            # sequence_ids: 0=question, 1=context, None=special/padding
            seq_ids        = enc.sequence_ids(0)
            offset_mapping = enc["offset_mapping"].squeeze(0).tolist()

            # ── Find first/last context token indices ───────────────────
            ctx_start = next(
                (i for i, s in enumerate(seq_ids) if s == 1), None
            )
            ctx_end = next(
                (i for i in range(len(seq_ids) - 1, -1, -1) if seq_ids[i] == 1),
                None,
            )

            if ctx_start is None or ctx_end is None:
                n_truncated += 1
                continue

            # ── Check answer is not truncated away ──────────────────────
            ctx_char_lo = offset_mapping[ctx_start][0]
            ctx_char_hi = offset_mapping[ctx_end][1]
            if not (ctx_char_lo <= start_char and ctx_char_hi >= end_char):
                n_truncated += 1
                continue

            # ── Map start_char/end_char → token indices ─────────────────
            # tok_start: last context token whose char_start <= start_char
            tok_start = ctx_start
            for i in range(ctx_start, ctx_end + 1):
                if offset_mapping[i][0] <= start_char:
                    tok_start = i
                else:
                    break

            # tok_end: first context token whose char_end >= end_char
            tok_end = ctx_end
            for i in range(ctx_end, ctx_start - 1, -1):
                if offset_mapping[i][1] >= end_char:
                    tok_end = i
                else:
                    break

            # ── Global attention mask (Longformer requirement) ───────────
            # Longformer needs global attention on question tokens so that
            # every context token can attend to the question via global path.
            input_ids   = enc["input_ids"].squeeze(0)
            global_attn = torch.zeros_like(input_ids)
            for i, sid in enumerate(seq_ids):
                if sid == 0 or i == 0:   # question token or CLS
                    global_attn[i] = 1

            # Ensure answers always includes the primary answer_text
            all_answers = list(dict.fromkeys([answer_text] + answers))

            self.samples.append({
                "input_ids":             input_ids,
                "attention_mask":        enc["attention_mask"].squeeze(0),
                "global_attention_mask": global_attn,
                "labels":                torch.tensor(
                    [tok_start, tok_end], dtype=torch.long
                ),
                "answer_text": answer_text,   # primary gold (for loss display)
                "answers":     all_answers,   # all aliases (for EM/F1 eval)
                "sample_id":   sample_id,
            })

        total = len(raw)
        kept  = len(self.samples)
        print(
            f"[QADataset] {path}\n"
            f"  total={total}  kept={kept}  "
            f"dropped_bad_char={n_bad_char}  dropped_truncated={n_truncated}"
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------


def get_dataloader(task, path, tokenizer, batch_size=2, max_length=4096,
                   shuffle=True, num_workers=0):
    if task == "qa":
        ds = QADataset(path, tokenizer, max_length)
    elif task == "docmt":
        ds = DocMTDataset(path, tokenizer, max_length)
    else:
        raise ValueError(f"Unknown task: {task}")

    def collate(batch):
        out = {}
        for k in batch[0]:
            if isinstance(batch[0][k], torch.Tensor):
                out[k] = torch.stack([b[k] for b in batch])
            else:
                out[k] = [b[k] for b in batch]
        return out

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate)
