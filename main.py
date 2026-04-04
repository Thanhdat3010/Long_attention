"""
main.py — LongAttention v2 Experiment Runner
==============================================

Usage examples:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  EXPERIMENT 1: Baseline Longformer on QA                           │
  │                                                                    │
  │  python main.py --mode train --task qa --model baseline \          │
  │      --backbone allenai/longformer-base-4096 \                     │
  │      --dataset_path data/qa --output_dir outputs/baseline_qa       │
  │                                                                    │
  │  EXPERIMENT 2: LongAttention v2 + Longformer on QA                 │
  │                                                                    │
  │  python main.py --mode train --task qa --model longattention \     │
  │      --backbone allenai/longformer-base-4096 \                     │
  │      --dataset_path data/qa --output_dir outputs/longatt_qa        │
  │                                                                    │
  │  EXPERIMENT 3: Swap backbone to BigBird                            │
  │                                                                    │
  │  python main.py --mode train --task qa --model longattention \     │
  │      --backbone google/bigbird-roberta-base \                      │
  │      --dataset_path data/qa --output_dir outputs/longatt_bigbird   │
  │                                                                    │
    │  EXPERIMENT 4: DocMT with BART                                     │
  │                                                                    │
    │  python main.py --mode train --task docmt --model longattention \  │
    │      --backbone facebook/bart-base \                               │
  │      --dataset_path data/docmt --output_dir outputs/longatt_docmt  │
  │                                                                    │
  │  EVALUATE any experiment:                                          │
  │                                                                    │
  │  python main.py --mode eval --task qa --model baseline \           │
  │      --backbone allenai/longformer-base-4096 \                     │
  │      --dataset_path data/qa --output_dir outputs/baseline_qa       │
  └──────────────────────────────────────────────────────────────────────┘

Supported backbones (any HuggingFace encoder):
  - allenai/longformer-base-4096
  - allenai/longformer-large-4096
  - google/bigbird-roberta-base
  - roberta-base / roberta-large
  - bert-base-uncased / bert-large-uncased
  - microsoft/deberta-v3-base
  - ... any AutoModel-compatible model
"""

import argparse
import sys
import os

# Add project root to path so `from src.xxx` works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(
        description="LongAttention v2 — Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Execution ----
    p.add_argument("--mode", choices=["train", "eval", "all"], default="train")
    p.add_argument("--task", choices=["qa", "docmt"], default="qa")
    p.add_argument("--model", choices=["baseline", "longattention"], default="longattention",
                   help="baseline = vanilla backbone;  longattention = backbone + LongAttn v2")

    # ---- Paths ----
    p.add_argument("--dataset_path", required=True,
                   help="Directory containing train.csv and test.csv")
    p.add_argument("--output_dir", default="./outputs",
                   help="Checkpoints + JSON summaries")

    # ---- Backbone (swap freely) ----
    p.add_argument("--backbone", default="allenai/longformer-base-4096",
                   help="Any HuggingFace encoder model name")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer (defaults to --backbone)")

    # ---- Regularization (LongAttention v2 only) ----
    p.add_argument("--anti_collapse_weight", type=float, default=0.01)
    p.add_argument("--null_route_weight", type=float, default=0.01)

    # ---- LongAttention v2 config ----
    p.add_argument("--window_size", type=int, default=512)
    p.add_argument("--segment_size", type=int, default=64)
    p.add_argument("--num_types", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2,
             help="Top-K segments to route to (lower=less memory, default 2)")
    p.add_argument("--gradient_checkpoint", action="store_true",
                   help="Enable gradient checkpointing (saves ~2-3× memory, slower)")
    p.add_argument("--alpha_init", type=float, default=0.02,
             help="Initial ReZero scale for long-range branch")
    p.add_argument("--gate_bias_init", type=float, default=0.0,
             help="Initial bias for necessity gate (sigmoid domain)")

    # ---- Training ----
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--gen_max_length", type=int, default=256,
             help="Max length for MT generation during eval")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.backbone

    print("=" * 64)
    print("  LongAttention v2 — Experiment Runner")
    print("=" * 64)
    print(f"  mode       : {args.mode}")
    print(f"  task       : {args.task}")
    print(f"  model      : {args.model}")
    print(f"  backbone   : {args.backbone}")
    print(f"  dataset    : {args.dataset_path}")
    print(f"  output     : {args.output_dir}")
    if args.model == "longattention":
        print(f"  window     : {args.window_size}")
        print(f"  segment    : {args.segment_size}")
        print(f"  types      : {args.num_types}")
        print(f"  top_k      : {args.top_k}")
        print(f"  grad_ckpt  : {args.gradient_checkpoint}")
        print(f"  alpha_init : {args.alpha_init}")
        print(f"  gate_bias  : {args.gate_bias_init}")
        print(f"  α(collapse): {args.anti_collapse_weight}")
        print(f"  β(null_rt) : {args.null_route_weight}")
    print(f"  epochs     : {args.epochs}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  lr         : {args.lr}")
    print(f"  max_length : {args.max_length}")
    print(f"  amp (fp16) : auto (cuda only)")
    print("=" * 64)

    from src.runner import run
    run(args)


if __name__ == "__main__":
    main()
