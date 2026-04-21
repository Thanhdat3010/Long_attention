# LongAttention: Necessity-Aware, Dependency-Typed Attention for Long-Context Modeling (v2)

A modular research codebase for evaluating the **LongAttention v2** mechanism—featuring necessity-aware gating and dependency-typed retrieval—against **LED (Longformer-Encoder-Decoder)** and **BART** baselines on document-level translation tasks (WMT14).

---

## 📁 Project Structure

```bash
Long_attention/
├── scripts/
│   ├── run_experiment.py          # ← Main entry point (Two-stage: SPT + Fine-tuning)
│   ├── evaluate.py                # ← Standalone evaluation
│   └── prepare_data.py            # ← Manual data pre-processing script
├── src/
│   ├── models/
│   │   ├── long_attention.py      # Gated & Typed Long-range branch
│   │   ├── local_attention.py     # Bidirectional sliding-window local branch
│   │   ├── led_attention.py       # Official LED (Dedicated Global QKV) baseline
│   │   └── model_factory.py       # Backbone loading + Weight Inheritance logic
│   ├── training/
│   │   ├── trainer.py             # Seq2SeqTrainer + Research Loss computation
│   │   ├── metrics.py             # BLEU / ChrF++ / COMET
│   │   └── callbacks.py           # Logging hooks
│   └── data/
│       └── data_preparation.py    # Document-level concatenation & caching
├── data/                          # Auto-populated with cached WMT14 documents
└── outputs/                       # Experiment logs and checkpoints
```

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd Long_attention

# 2. Install dependencies
pip install -r requirements.txt
```

> **Hardware Note:** Optimized for A100/H100 GPUs using `bfloat16`. 

---

## 🚀 Quick Start

The pipeline supports **Two-Stage Training**:
1. **Stage 1 (SPT):** Self-Pre-training via Text Infilling (MLM) to stabilize the new attention gates.
2. **Stage 2 (Fine-tuning):** Document-level translation on WMT14.

### 1. Run LongAttention Experiment (v2)
This command performs both SPT and Fine-tuning with **Weight Inheritance** from pre-trained BART.

```bash
python scripts/run_experiment.py \
    --attention_type long_attention \
    --dataset wmt14 \
    --lang_pair en-fr \
    --max_train_samples 1000000 \
    --group_size 50 \
    --run_spt \
    --spt_epochs 3 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --diversity_weight 0.1 \
    --null_weight 0.01 \
    --dtype bfloat16 \
    --output_dir ./outputs/long_attention_v2
```

### 2. Run LED Baseline (Official Architecture)
Uses dedicated global projections for the `<s>` token, inheriting weights from BART.

```bash
python scripts/run_experiment.py \
    --attention_type led \
    --dataset wmt14 \
    --lang_pair en-fr \
    --run_spt \
    --spt_epochs 3 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --dtype bfloat16 \
    --output_dir ./outputs/led_baseline
```

---

## 🧠 Architecture: LongAttention v2

LongAttention is designed to be **necessity-aware** (deciding *if* to look far) and **dependency-typed** (deciding *why* to look far).

### The Multi-Branch Operator
```
Query Token q_i
     │
     ├─────────────────────────────────────────────┐
     │                                             │
┌────▼──────────────────────────┐      ┌───────────▼────────────────────────────┐
│ LOCAL BRANCH                  │      │ LONG-RANGE BRANCH (Necessity/Typed)    │
│ Sliding Window (Symmetric)    │      │                                        │
│ Window Size W = 512           │      │ 1. Necessity Gate g_i ∈ [0, 1]         │
│                               │      │ 2. Dependency Typed Gists (K_t, V_t)    │
│ A_local = Attn(q, K_win, V_win)│      │ 3. Typed Top-K Retrieval               │
└────────────────┬──────────────┘      │    - Coreference / Lexical / Discourse │
                 │                     │    - O_long = Σ_t w_t * Attn_t(q, K, V)│
                 │                     └───────────┬────────────────────────────┘
                 │                                 │
                 └──────────────┬──────────────────┘
                                │
                  ┌─────────────▼───────────────┐
                  │ FINAL PROJECTION            │
                  │ O = OutProj(Local + g_i*Long)│
                  └─────────────────────────────┘
```

### Key Principles
- **Weight Inheritance:** Newly injected layers (Local, Typed) are initialized by copying pre-trained BART weights to ensure a warm start.
- **Null-Route Calibration:** A sparsity penalty (`null_weight`) encourages the gate to stay closed when local context is sufficient.
- **Diversity Regularization:** A cosine similarity penalty (`diversity_weight`) forces dependency types to specialize in different context patterns.

---

## 📊 Metrics

| Metric | Description |
|---|---|
| **SacreBLEU** | Corpus-level translation quality. |
| **COMET** | Neural semantic embedding similarity (Unbabel/wmt22-comet-da). |
| **Gate Activity** | % of tokens activating the long-range branch. |
| **Type Diversity** | Measure of how distinct the dependency types are. |

---

## 🔧 Argument Reference (Key Flags)

| Flag | Default | Description |
|---|---|---|
| `--attention_type` | `vanilla` | `vanilla`, `led`, or `long_attention`. |
| `--run_spt` | `False` | Enables Stage 1 (Text Infilling). |
| `--spt_epochs` | `1` | Duration of Stage 1. |
| `--group_size` | `50` | Sentence count for document-level concatenation. |
| `--null_weight` | `0.01` | Penalty for opening the long-range gate. |
| `--diversity_weight` | `0.1` | Penalty for overlap between dependency types. |
| `--top_k` | `64` | Num of segments retrieved in the long-range branch. |

---

## 📜 Acknowledgements
This codebase is built upon **HuggingFace Transformers** and uses **facebook/bart-base** as the default backbone. Special credit to the **Longformer (LED)** paper for the sliding window baseline.
