# LongAttention: Necessity-Aware, Dependency-Typed Attention for Long-Context Modeling (v2)

A modular research codebase for evaluating the **LongAttention v2** mechanism—featuring necessity-aware gating and dependency-typed retrieval—against **LED (Longformer-Encoder-Decoder)** and **BART** baselines on document-level translation tasks.

Following the **Self Pre-Training (SPT)** protocol from *"Never Train from Scratch"* (ICLR 2024 Outstanding Paper) for fair architectural comparison.

---

## 📁 Project Structure

```bash
Long_attention/
├── scripts/
│   ├── run_experiment.py          # ← Main entry point (Two-stage: SPT + Fine-tuning)
│   ├── run_eval.py                # ← Standalone evaluation
│   └── prepare_data.py            # ← Manual data pre-processing script
├── src/
│   ├── models/
│   │   ├── long_attention.py      # Gated & Typed Long-range branch
│   │   ├── local_attention.py     # Bidirectional sliding-window local branch
│   │   ├── led_attention.py       # Official LED (Dedicated Global QKV) baseline
│   │   └── model_factory.py       # Backbone loading + Weight Inheritance logic
│   ├── training/
│   │   ├── trainer.py             # Seq2SeqTrainer + SPT Collator + Research Loss
│   │   ├── metrics.py             # BLEU / ChrF++ / COMET / GFLOPS
│   │   └── callbacks.py           # Attention sink / Gate entropy logging
│   ├── data/
│   │   └── data_preparation.py    # Document-level concatenation & caching
│   └── utils/
│       └── io_utils.py            # Output dirs, metrics persistence, logging
├── data/                          # Auto-populated with cached datasets
└── outputs/                       # Experiment logs and checkpoints
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> **Hardware Note:** Optimized for A100/H100 GPUs using `bfloat16`. Use `--gradient_checkpointing` for long sequences.

---

## 🧪 Experimental Protocol (SPT — Fair Comparison)

This codebase follows the **Self Pre-Training (SPT)** methodology:

```
                    ┌───────────────────────┐
                    │  BART-base pretrained  │
                    │  (facebook/bart-base)  │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
            Model A         Model B         Model C
           (Vanilla)         (LED)         (LongAttention)
                │               │               │
    ════════════╪═══════════════╪═══════════════╪════════════
    Stage 1:    │    SPT — Masked Denoising (15% masking)  │
    SPT         │    Data: downstream task train set only   │
                │    Same LR, epochs, batch for all models  │
    ════════════╪═══════════════╪═══════════════╪════════════
    Stage 2:    │    Fine-tuning — Translation (Seq2Seq)    │
    FT          │    Same hyperparameters for all models    │
    ════════════╪═══════════════╪═══════════════╪════════════
                │               │               │
            BLEU A          BLEU B          BLEU C
```

**Why SPT?** Without it, models with more new parameters (LongAttention > LED > BART) start from an unfair disadvantage due to random initialization. SPT calibrates all new modules before competition begins.

---

## 🚀 Quick Start

### 1. Smoke Test (CPU, tiny data — verify pipeline works)

```bash
# Model A: Vanilla BART
python scripts/run_experiment.py \
    --attention_type vanilla \
    --dataset iwslt2017 \
    --lang_pair en-fr \
    --max_train_samples 100 \
    --max_val_samples 50 \
    --max_test_samples 50 \
    --group_size 5 \
    --run_spt \
    --spt_epochs 1 \
    --spt_mask_ratio 0.15 \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --dtype float32 \
    --device_map cpu \
    --no_comet \
    --output_dir ./outputs/smoke_test/vanilla

# Model B: LED
python scripts/run_experiment.py \
    --attention_type led \
    --dataset iwslt2017 \
    --lang_pair en-fr \
    --max_train_samples 100 \
    --max_val_samples 50 \
    --max_test_samples 50 \
    --group_size 5 \
    --run_spt \
    --spt_epochs 1 \
    --spt_mask_ratio 0.15 \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --dtype float32 \
    --device_map cpu \
    --no_comet \
    --output_dir ./outputs/smoke_test/led

# Model C: LongAttention
python scripts/run_experiment.py \
    --attention_type long_attention \
    --dataset iwslt2017 \
    --lang_pair en-fr \
    --max_train_samples 100 \
    --max_val_samples 50 \
    --max_test_samples 50 \
    --group_size 5 \
    --run_spt \
    --spt_epochs 1 \
    --spt_mask_ratio 0.15 \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --dtype float32 \
    --device_map cpu \
    --no_comet \
    --diversity_weight 0.1 \
    --null_weight 0.01 \
    --output_dir ./outputs/smoke_test/long_attention
```

---

### 2. Full Experiment (GPU, WMT14 1M samples, group_size=50)

> **IMPORTANT:** Run all 3 models with **identical hyperparameters** for fair comparison.

```bash
# ═══════════════════════════════════════════════════════
# Model A: Vanilla BART (Full Attention Baseline)
# ═══════════════════════════════════════════════════════
python scripts/run_experiment.py \
    --attention_type vanilla \
    --dataset wmt14 \
    --lang_pair en-fr \
    --max_train_samples 1000000 \
    --group_size 50 \
    --run_spt \
    --spt_epochs 3 \
    --spt_mask_ratio 0.15 \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --no_comet \
    --output_dir ./outputs/full/vanilla

# ═══════════════════════════════════════════════════════
# Model B: LED (Sliding Window + Global Attention)
# ═══════════════════════════════════════════════════════
python scripts/run_experiment.py \
    --attention_type led \
    --dataset wmt14 \
    --lang_pair en-fr \
    --max_train_samples 1000000 \
    --group_size 50 \
    --run_spt \
    --spt_epochs 3 \
    --spt_mask_ratio 0.15 \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --no_comet \
    --output_dir ./outputs/full/led

# ═══════════════════════════════════════════════════════
# Model C: LongAttention v2 (Proposed)
# ═══════════════════════════════════════════════════════
python scripts/run_experiment.py \
    --attention_type long_attention \
    --dataset wmt14 \
    --lang_pair en-fr \
    --max_train_samples 1000000 \
    --group_size 50 \
    --run_spt \
    --spt_epochs 3 \
    --spt_mask_ratio 0.15 \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --no_comet \
    --diversity_weight 0.1 \
    --null_weight 0.01 \
    --output_dir ./outputs/full/long_attention
```

---

## 🧠 Architecture: LongAttention v2

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

### Key Design Principles
- **Weight Inheritance:** All Q/K/V projections initialized from pretrained BART weights.
- **Null-Route Calibration:** Gate sparsity penalty keeps long-range branch closed when local context suffices. 
- **Diversity Regularization:** Cosine penalty forces dependency types to specialize.

---

## 📊 Metrics

| Metric | Description |
|---|---|
| **SacreBLEU** | Corpus-level translation quality (primary). |
| **ChrF++** | Character n-gram F-score with word order. |
| **COMET** | Neural semantic similarity (Unbabel/wmt22-comet-da). |
| **Gate Activity** | % of tokens activating the long-range branch. |
| **Type Diversity** | Cosine distance between dependency type attention maps. |
| **Attention Sink** | % of attention mass on position 0 (sink ratio). |
| **GFLOPS** | Estimated computational cost per forward pass. |

---

## 🔧 Full Argument Reference

### Data
| Flag | Default | Description |
|---|---|---|
| `--dataset` | `iwslt2017` | HuggingFace dataset name. |
| `--lang_pair` | `en-fr` | Language pair. |
| `--group_size` | `50` | Sentences per document. |
| `--max_train_samples` | `None` | Limit training rows. |

### Model
| Flag | Default | Description |
|---|---|---|
| `--backbone` | `facebook/bart-base` | Pretrained checkpoint. |
| `--attention_type` | `vanilla` | `vanilla`, `led`, or `long_attention`. |
| `--dtype` | `bfloat16` | Model precision. |

### Training
| Flag | Default | Description |
|---|---|---|
| `--epochs` | `3` | Fine-tuning epochs. |
| `--batch_size` | `4` | Per-device batch size. |
| `--learning_rate` | `2e-5` | Fine-tuning learning rate. |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation. |
| `--gradient_checkpointing` | `False` | Trade compute for VRAM. |
| `--freeze_backbone` | `False` | Only train injected attention. |

### SPT (Self Pre-Training)
| Flag | Default | Description |
|---|---|---|
| `--run_spt` | `False` | Enable Stage 1 (denoising). |
| `--spt_epochs` | `1` | SPT duration. |
| `--spt_learning_rate` | `None` | SPT learning rate (defaults to `--learning_rate`). |
| `--spt_mask_ratio` | `0.15` | Token masking ratio (paper: 15% for text). |

### LongAttention-Specific
| Flag | Default | Description |
|---|---|---|
| `--local_window_size` | `512` | Sliding window size. |
| `--top_k` | `64` | Top-K retrieval positions. |
| `--num_types` | `3` | Dependency types. |
| `--diversity_weight` | `0.1` | Type diversity loss weight. |
| `--null_weight` | `0.01` | Gate sparsity penalty weight. |

---

## 📜 References

- **SPT Protocol:** Amos et al., *"Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors"*, ICLR 2024 (Outstanding Paper Award).
- **BART:** Lewis et al., *"BART: Denoising Sequence-to-Sequence Pre-training"*, ACL 2020.
- **LED:** Beltagy et al., *"Longformer: The Long-Document Transformer"*, arXiv 2020.
