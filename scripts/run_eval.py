#!/usr/bin/env python3
"""
evaluate.py — Standalone Evaluation Script for LongAttention Checkpoints.

Usage
-----
python scripts/evaluate.py \
    --checkpoint_dir outputs/vanilla_lr2e-5_bs4 \
    --lang_pair en-fr \
    --split test \
    --batch_size 8
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nmt.data_preparation import download_and_cache_dataset
from src.nmt.metrics import (
    compute_sacrebleu,
    compute_chrf,
    compute_comet,
    compute_attention_sink_ratio,
    aggregate_sink_ratios,
)
from src.utils.io_utils import save_metrics, setup_logging

logger = logging.getLogger(__name__)


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Standalone evaluation for LongAttention NMT checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Path to the saved model/tokenizer directory."
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory for WMT14 CSV cache."
    )
    parser.add_argument(
        "--dataset", type=str, default="iwslt2017", help="HuggingFace dataset name."
    )
    parser.add_argument(
        "--lang_pair", type=str, default="en-fr", help="Language pair used in the experiment."
    )
    parser.add_argument(
        "--group_size", type=int, default=50, help="Number of sentences to concatenate."
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], default="test", help="Dataset split to evaluate on."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Inference batch size."
    )
    parser.add_argument(
        "--max_source_length", type=int, default=1024, help="Max source token length."
    )
    parser.add_argument(
        "--max_target_length", type=int, default=256, help="Max generation length for decoding."
    )
    parser.add_argument(
        "--num_beams", type=int, default=4, help="Beam search width for generation."
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, metavar="N", help="Limit to N samples (None = all)."
    )
    parser.add_argument(
        "--no_comet", action="store_true", default=False, help="Skip COMET computation (faster)."
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="JSON file to write evaluation results."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float32"
    )
    parser.add_argument(
        "--attention_type", type=str, choices=["vanilla", "led", "long_attention"], default=None,
        help="Override attention type (defaults to reading from args.json in checkpoint)."
    )
    parser.add_argument(
        "--local_window_size", type=int, default=512, help="Override local branch sliding window size."
    )
    parser.add_argument(
        "--top_k", type=int, default=64, help="Override number of Top-K positions for long-range retrieval."
    )
    parser.add_argument(
        "--num_types", type=int, default=3, help="Override number of dependency types."
    )
    parser.add_argument(
        "--bottleneck_ratio", type=float, default=0.25, help="Override bottleneck ratio for gating modules."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Override dropout probability."
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def generate_translations(
    model,
    tokenizer,
    sources: List[str],
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> List[str]:
    """Generate translations for a list of source sentences via Seq2Seq standard."""
    model.eval()
    all_preds: List[str] = []

    with torch.no_grad():
        for batch_start in range(0, len(sources), batch_size):
            batch_sources = sources[batch_start : batch_start + batch_size]

            inputs = tokenizer(
                batch_sources,
                max_length=max_source_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            target_ids = model.generate(
                **inputs,
                max_length=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            all_preds.extend([d.strip() for d in decoded])

            if batch_start % (batch_size * 10) == 0:
                logger.info("Generated %d / %d", batch_start + len(batch_sources), len(sources))

    return all_preds


def main() -> None:
    parser = build_eval_parser()
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        logger.error("Checkpoint directory not found: %s", checkpoint_dir)
        sys.exit(1)

    logger.info("Loading checkpoint from: %s", checkpoint_dir)
    from transformers import AutoTokenizer
    from src.models import build_model

    # Load args.json from checkpoint to reconstruct the exact model parameters
    args_json_path = checkpoint_dir / "args.json"
    attention_type = "vanilla"
    long_attention_config = None
    if args_json_path.is_file():
        try:
            with open(args_json_path, "r", encoding="utf-8") as f:
                saved_args = json.load(f)
                attention_type = saved_args.get("attention_type", "vanilla")
                long_attention_config = {
                    "local_window_size": saved_args.get("local_window_size", 512),
                    "top_k": saved_args.get("top_k", 64),
                    "num_types": saved_args.get("num_types", 3),
                    "bottleneck_ratio": saved_args.get("bottleneck_ratio", 0.25),
                    "dropout_prob": saved_args.get("dropout_prob", 0.1),
                    "max_length": max(saved_args.get("max_source_length", 1024), saved_args.get("max_target_length", 1024)),
                }
                logger.info("Reconstructed config from args.json: attention_type=%s", attention_type)
        except Exception as e:
            logger.warning("Could not read args.json from checkpoint directory: %s.", e)

    # CLI Override for attention configuration
    if args.attention_type is not None:
        attention_type = args.attention_type
        logger.info("CLI Override: attention_type=%s", attention_type)
        
    if attention_type != "vanilla":
        if long_attention_config is None:
            long_attention_config = {}
        # Override with CLI arguments if specified, else keep checkpoint value or defaults
        long_attention_config.update({
            "local_window_size": args.local_window_size if args.local_window_size is not None else long_attention_config.get("local_window_size", 512),
            "top_k": args.top_k if args.top_k is not None else long_attention_config.get("top_k", 64),
            "num_types": args.num_types if args.num_types is not None else long_attention_config.get("num_types", 3),
            "bottleneck_ratio": args.bottleneck_ratio if args.bottleneck_ratio is not None else long_attention_config.get("bottleneck_ratio", 0.25),
            "dropout_prob": args.dropout_prob if args.dropout_prob is not None else long_attention_config.get("dropout_prob", 0.1),
            "max_length": max(args.max_source_length, args.max_target_length),
        })
        logger.info("Final LongAttention config for loading: %s", long_attention_config)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    device_map = args.device if args.device != "auto" else "auto"
    model = build_model(
        task="nmt",
        backbone=str(checkpoint_dir),
        device_map=device_map,
        torch_dtype=torch_dtype,
        attention_type=attention_type,
        long_attention_config=long_attention_config,
    )

    if args.device == "auto":
        device = next(model.parameters()).device
    else:
        device = torch.device(args.device)

    logger.info("Loading %s data (split=%s)…", args.dataset, args.split)
    dataframes = download_and_cache_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        lang_pair=args.lang_pair,
        max_test_rows=args.max_samples if args.split == "test" else None,
        max_val_rows=args.max_samples if args.split == "val" else None,
        max_train_rows=args.max_samples if args.split == "train" else None,
        group_size=args.group_size,
    )
    # Enforce truncation even if loaded from a larger cached CSV
    if args.max_samples is not None:
        for split in dataframes:
            dataframes[split] = dataframes[split].head(args.max_samples)

    df = dataframes[args.split]

    sources = df["source"].tolist()
    references = df["target"].tolist()
    logger.info("Evaluating on %d examples.", len(sources))

    logger.info("Generating translations…")
    predictions = generate_translations(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_target_length,
        num_beams=args.num_beams,
        device=device,
    )

    logger.info("Computing metrics…")
    metrics = {}
    metrics.update(compute_sacrebleu(predictions, references))
    metrics.update(compute_chrf(predictions, references))

    if not args.no_comet:
        metrics.update(compute_comet(sources, predictions, references))

    out_file = args.output_file or str(checkpoint_dir / "eval_metrics.json")
    save_metrics(metrics, output_dir=str(Path(out_file).parent), filename=Path(out_file).name)

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    for k, v in sorted(metrics.items()):
        logger.info("  %-20s : %s", k, v)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
