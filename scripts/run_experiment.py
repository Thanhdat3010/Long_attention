#!/usr/bin/env python3
"""
run_experiment.py — Main Entry Point for LongAttention NMT Experiments.

This script orchestrates the full experimental pipeline:
  1. Parse command-line arguments.
  2. Set up logging and output directory.
  3. Download / load WMT14 data (with CSV caching).
  4. Load Qwen backbone and (optionally) inject LongAttention.
  5. Build datasets and run training.
  6. Save model, tokenizer, and final metrics.

Usage Examples
--------------
# Standard Transformer baseline (1.5B):
python scripts/run_experiment.py \\
    --backbone Qwen/Qwen2-1.5B \\
    --attention_type standard \\
    --lang_pair en-fr \\
    --epochs 3 \\
    --batch_size 4 \\
    --learning_rate 2e-5

# LongAttention experiment (1.5B):
python scripts/run_experiment.py \\
    --backbone Qwen/Qwen2-1.5B \\
    --attention_type long_attention \\
    --lang_pair en-fr \\
    --epochs 3 \\
    --batch_size 4 \\
    --learning_rate 2e-5 \\
    --top_k 64 \\
    --local_window_size 512

# 7B scale with lower batch size:
python scripts/run_experiment.py \\
    --backbone Qwen/Qwen2-7B \\
    --attention_type long_attention \\
    --batch_size 2 \\
    --learning_rate 1e-5 \\
    --max_train_samples 100000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_preparation import download_and_cache_dataset
from src.models.model_factory import build_model, build_tokenizer
from src.training.trainer import Seq2SeqDocumentDataset, run_training
from src.utils.io_utils import (
    build_output_dir,
    save_metrics,
    save_model_artifacts,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the full argument parser for the experiment pipeline.

    All defaults are chosen to produce a quick smoke-test run that can
    be scaled up by the user for real experiments.
    """
    parser = argparse.ArgumentParser(
        prog="run_experiment.py",
        description="LongAttention NMT training pipeline on WMT14.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store/load WMT14 CSV files.",
    )
    data_grp.add_argument(
        "--dataset",
        type=str,
        default="iwslt2017",
        help="HuggingFace dataset name (e.g., 'iwslt2017' or 'wmt14').",
    )
    data_grp.add_argument(
        "--lang_pair",
        type=str,
        default="en-fr",
        help="Language pair to use (e.g., 'en-fr', 'fr-en', 'de-en').",
    )
    data_grp.add_argument(
        "--group_size",
        type=int,
        default=50,
        help="Number of consecutive sentences to concatenate to form a long document.",
    )
    data_grp.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit training rows to N (None = use all). Useful for quick tests.",
    )
    data_grp.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit validation rows to N.",
    )
    data_grp.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit test rows to N.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--backbone",
        type=str,
        default="facebook/bart-base",
        help="HuggingFace model ID for the backbone.",
    )
    model_grp.add_argument(
        "--attention_type",
        type=str,
        choices=["vanilla", "sparse", "long_attention"],
        default="vanilla",
        help=(
            "Attention type to use. "
            "'vanilla' = unmodified BART; "
            "'sparse' = sliding window local attention; "
            "'long_attention' = LongAttention layers."
        ),
    )
    model_grp.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HuggingFace device_map for model loading ('auto', 'cpu', 'cuda').",
    )
    model_grp.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Floating point precision for model weights. bfloat16 recommended for A100.",
    )

    # ── LongAttention-specific ───────────────────────────────────────────────
    la_grp = parser.add_argument_group("LongAttention")
    la_grp.add_argument(
        "--local_window_size",
        type=int,
        default=512,
        help="Local branch sliding window size (tokens).",
    )
    la_grp.add_argument(
        "--top_k",
        type=int,
        default=64,
        help="Number of Top-K positions for bidirectional long-range retrieval.",
    )
    la_grp.add_argument(
        "--bottleneck_ratio",
        type=float,
        default=0.25,
        help="Bottleneck ratio for the FunctionalDecomposer gate head.",
    )
    la_grp.add_argument(
        "--dropout_prob",
        type=float,
        default=0.0,
        help="Dropout applied to attention weights inside LongAttention.",
    )

    # ── Sequence Lengths ────────────────────────────────────────────────────
    seq_grp = parser.add_argument_group("Sequence Lengths")
    seq_grp.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="Maximum token length for source (prompt) sequences.",
    )
    seq_grp.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="Maximum token length for target sequences.",
    )

    # ── Training ─────────────────────────────────────────────────────────────
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training and evaluation batch size.",
    )
    train_grp.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate for the AdamW optimiser.",
    )
    train_grp.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    train_grp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    train_grp.add_argument(
        "--no_comet",
        action="store_true",
        default=False,
        help="Disable COMET metric (faster evaluation, no unbabel-comet needed).",
    )
    train_grp.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help=(
            "Freeze all backbone weights and only train the injected "
            "LongAttention layers. Requires --attention_type long_attention."
        ),
    )

    # ── Output ───────────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Base directory for saving model artifacts and logs.",
    )
    out_grp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Resolve output directory FIRST (needed for log file) ────────────────
    output_dir = build_output_dir(args)

    # ── Logging ─────────────────────────────────────────────────────────────
    import logging as _logging
    log_level = _logging.DEBUG if args.verbose else _logging.INFO
    setup_logging(level=log_level, log_file=f"{output_dir}/run.log")
    logger.info("=" * 60)
    logger.info("LongAttention Experiment Pipeline")
    logger.info("=" * 60)
    logger.info("Backbone        : %s", args.backbone)
    logger.info("Attention Type  : %s", args.attention_type)
    logger.info("Language Pair   : %s", args.lang_pair)
    logger.info("Output Dir      : %s", output_dir)
    logger.info("=" * 60)

    # ── Reproducibility ──────────────────────────────────────────────────────
    import torch, random, numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── dtype ────────────────────────────────────────────────────────────────
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # ── Step 1: Data Preparation ─────────────────────────────────────────────
    logger.info("[1/5] Preparing %s data (%s)…", args.dataset, args.lang_pair)
    dataframes = download_and_cache_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        lang_pair=args.lang_pair,
        max_train_rows=args.max_train_samples,
        max_val_rows=args.max_val_samples,
        max_test_rows=args.max_test_samples,
        group_size=args.group_size,
    )
    logger.info(
        "Data ready — train: %d | val: %d | test: %d",
        len(dataframes["train"]),
        len(dataframes["val"]),
        len(dataframes["test"]),
    )

    # ── Step 2: Tokenizer ────────────────────────────────────────────────────
    logger.info("[2/5] Loading tokenizer…")
    tokenizer = build_tokenizer(args.backbone)

    # ── Step 3: Model (+ optional injection) ─────────────────────────────────
    logger.info("[3/5] Building model (attention_type=%s)…", args.attention_type)
    long_attention_config = {
        "local_window_size": args.local_window_size,
        "top_k": args.top_k,
        "bottleneck_ratio": args.bottleneck_ratio,
        "dropout_prob": args.dropout_prob,
    }
    model = build_model(
        backbone=args.backbone,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        attention_type=args.attention_type,
        long_attention_config=long_attention_config,
    )

    if args.freeze_backbone and args.attention_type != "vanilla":
        logger.info("Freezing backbone — only Injected layers will be trained.")

    # ── Step 4: Datasets ──────────────────────────────────────────────────────
    logger.info("[4/5] Building PyTorch datasets…")
    train_ds = Seq2SeqDocumentDataset(
        dataframe=dataframes["train"],
        tokenizer=tokenizer,
        src_max_len=args.max_source_length,
        tgt_max_len=args.max_target_length,
    )
    val_ds = Seq2SeqDocumentDataset(
        dataframe=dataframes["val"],
        tokenizer=tokenizer,
        src_max_len=args.max_source_length,
        tgt_max_len=args.max_target_length,
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # Attach raw dataframe for COMET source retrieval
    val_ds.data = dataframes["val"]

    # ── Step 5: Train & Evaluate ──────────────────────────────────────────────
    logger.info("[5/5] Starting training pipeline…")
    args.use_comet = not args.no_comet  # forward to trainer
    final_metrics = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        args=args,
        output_dir=output_dir,
    )

    # ── Save Artifacts ────────────────────────────────────────────────────────
    logger.info("Saving final model artifacts…")
    save_model_artifacts(model, tokenizer, output_dir=output_dir, args=args)
    save_metrics(final_metrics, output_dir=output_dir, filename="metrics.json")

    logger.info("=" * 60)
    logger.info("Experiment complete. Results in: %s", output_dir)
    for k, v in sorted(final_metrics.items()):
        logger.info("  %-30s : %s", k, v)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
