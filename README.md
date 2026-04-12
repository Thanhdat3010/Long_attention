# LongAttention: Gated Functional Information Compression for Scalable Context Understanding

A complete, modular research codebase for evaluating the **LongAttention** attention mechanism against standard Qwen2 baselines on Neural Machine Translation (WMT14).

---

## 📁 Project Structure

```
Long_attention/
├── config/
│   └── default_config.yaml        # Reference for all hyperparameters
├── data/                          # Auto-populated with WMT14 CSV files
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── outputs/                       # Model checkpoints and metrics (auto-generated)
│   └── {model_name}/
│       └── {attention_type}_lr{lr}_bs{bs}/
│           ├── pytorch_model.safetensors
│           ├── tokenizer files
│           ├── metrics.json
│           ├── args.json
│           ├── attention_sink_log.json
│           └── gate_entropy_log.json
├── scripts/
│   ├── run_experiment.py          # ← Main entry point
│   └── evaluate.py                # ← Standalone evaluation
└── src/
    ├── data/
    │   └── data_preparation.py    # WMT14 download + CSV caching
    ├── models/
    │   ├── long_attention.py      # Core LongAttention nn.Module
    │   ├── local_attention.py     # Sliding-window local branch
    │   └── model_factory.py       # Backbone loading + injection logic
    ├── training/
    │   ├── trainer.py             # Seq2SeqTrainer wrapper
    │   ├── metrics.py             # BLEU / ChrF++ / COMET / Sink Ratio
    │   └── callbacks.py           # AttentionSink + GateDiversity hooks
    └── utils/
        └── io_utils.py            # Output dir management, metrics saving
```

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd Long_attention

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU requirement:** A CUDA-capable GPU with at least 16 GB VRAM is recommended for Qwen2-1.5B, and ≥40 GB for Qwen2-7B.

---

## 🚀 Quick Start

### Run Standard Baseline (Qwen2-1.5B)

```bash
python scripts/run_experiment.py \
    --backbone Qwen/Qwen2-1.5B \
    --attention_type standard \
    --lang_pair en-fr \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5
```

### Run LongAttention Experiment (Qwen2-1.5B)

```bash
python scripts/run_experiment.py \
    --backbone Qwen/Qwen2-1.5B \
    --attention_type long_attention \
    --lang_pair en-fr \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --local_window_size 512 \
    --top_k 64
```

### Scale to 7B

```bash
python scripts/run_experiment.py \
    --backbone Qwen/Qwen2-7B \
    --attention_type long_attention \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_train_samples 500000
```

### Evaluate a Saved Checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint_dir outputs/qwen2-1.5b/long_attention_lr2e-5_bs4 \
    --lang_pair en-fr \
    --split test \
    --compute_sink_ratio \
    --batch_size 8
```

### Quick Smoke Test (Minimal Data)

```bash
python scripts/run_experiment.py \
    --attention_type long_attention \
    --max_train_samples 512 \
    --max_val_samples 128 \
    --epochs 1 \
    --batch_size 2 \
    --no_comet
```

---

## 🧠 Architecture Overview

### LongAttention (Gated Functional Information Compression)

```
Input Hidden States S  (B, T, D)
         │
         ├─────────────────────────────────────────────┐
         │                                             │
   ┌─────▼────────────────────────────────────┐  ┌────▼──────────────────────────────┐
   │   LOCAL BRANCH (Branch 1)                 │  │   GATED FUNCTIONAL COMPRESSION    │
   │   LocalSlidingWindowAttention             │  │   BRANCH (Branch 2)               │
   │   Dense causal attention, window W        │  │                                   │
   │   Captures syntax & co-reference          │  │  1. FunctionalDecomposer          │
   │                                           │  │     S → R (Semantic Root)         │
   │   A_local = softmax(QK^T/√d)[N_i] · V    │  │       + A (Functional Affix)      │
   └─────────────────────┬─────────────────────┘  │                                   │
                         │                         │  2. Gating: G = σ(X W_θ)         │
                         │                         │  3. GistReservoir                 │
                         │                         │     K_gist, V_gist =              │
                         │                         │     (G⊙f_θ(R)) + Codebook(A)     │
                         │                         │                                   │
                         │                         │  4. BiDirectional Top-K           │
                         │                         │     A_long = Σ_{TopK} Corr·V     │
                         │                         └──────────────────┬────────────────┘
                         │                                            │
                         └──────────────┬─────────────────────────────┘
                                        │
                                 ┌──────▼──────────────────────────────────┐
                                 │   OUTPUT INTEGRATION                     │
                                 │   α = sigmoid(W_α · S)                   │
                                 │   O = LayerNorm(A_local + α · A_long)    │
                                 └─────────────────────────────────────────┘
```

---

## 📊 Metrics

| Metric | Description |
|---|---|
| **SacreBLEU** | Standardised corpus BLEU (0–100) |
| **ChrF++** | Character n-gram F-score with word order (0–100) |
| **COMET** | Neural DA metric via `Unbabel/wmt22-comet-da` (≈0–1) |
| **Attention Sink Ratio** | % of attention on token 0 (target: <5%) |
| **Root Fidelity Score** | % of semantic roots retained post-compression (new) |
| **Gate Entropy** | Diversity of the functional decomposer gate scores |

---

## 🔧 Full Argument Reference

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Directory for WMT14 CSVs |
| `--lang_pair` | `en-fr` | Language pair (e.g., `en-fr`, `de-en`) |
| `--backbone` | `Qwen/Qwen2-1.5B` | HuggingFace model ID |
| `--attention_type` | `standard` | `standard` or `long_attention` |
| `--batch_size` | `4` | Per-device batch size |
| `--learning_rate` | `2e-5` | Peak LR for AdamW |
| `--epochs` | `3` | Training epochs |
| `--max_source_length` | `256` | Max source prompt tokens |
| `--max_target_length` | `256` | Max target tokens |
| `--local_window_size` | `512` | Local branch window size |
| `--top_k` | `64` | Top-K positions for long-range retrieval |
| `--bottleneck_ratio` | `0.25` | Gate head bottleneck ratio |
| `--dropout_prob` | `0.0` | Attention dropout |
| `--freeze_backbone` | `False` | Freeze backbone, train only LongAttention |
| `--output_dir` | `./outputs` | Base output directory |
| `--no_comet` | `False` | Skip COMET computation |
| `--dtype` | `float16` | Weight precision |
| `--seed` | `42` | Random seed |

---

## 📂 Output Directory Format

```
outputs/
└── qwen2-1.5b/
    ├── standard_lr2e-5_bs4/
    │   ├── metrics.json
    │   ├── args.json
    │   └── attention_sink_log.json
    └── long_attention_lr2e-5_bs4/
        ├── metrics.json
        ├── args.json
        ├── attention_sink_log.json
        └── gate_entropy_log.json
```

---

## 📜 Reference

This codebase implements the **LongAttention** proposal:
> *LongAttention: Gated Functional Information Compression for Scalable Context Understanding*

Key references:
- Qwen2 Technical Report (Qwen Team, 2024)
- FlashAttention-2 (Dao et al., 2023)
- RULER Benchmark (Hsieh et al., 2024)
- WMT14 En-Fr Translation Benchmark
