#!/usr/bin/env python
"""
Entry point for LongAttention Question Answering Experiments.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import set_seed

from src.models import build_model, build_tokenizer
from src.qa.data_preparation import load_qa_dataset
from src.qa.trainer import run_qa_training
from src.utils.io_utils import build_output_dir, setup_logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run QA Experiment")
    # Setup
    parser.add_argument("--output_dir", type=str, default="./outputs/qa", help="Base output dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model
    parser.add_argument("--attention_type", type=str, required=True, choices=["vanilla", "longformer", "long_attention"])
    parser.add_argument("--backbone", type=str, default="roberta-base", help="Model backbone")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    
    # Data
    parser.add_argument("--dataset", type=str, default="hotpotqa/hotpot_qa", help="Dataset on HuggingFace")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length for RoBERTa")
    parser.add_argument("--doc_stride", type=int, default=128, help="Stride for sliding window over long documents")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples (debug)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit val samples (Full eval at the end)")
    parser.add_argument("--max_val_samples_during_train", type=int, default=None, help="Limit val samples ONLY during training steps to speed up")
    
    # Training
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # SPT
    parser.add_argument("--run_spt", action="store_true", help="Run Masked Language Modeling first")
    parser.add_argument("--spt_epochs", type=float, default=2.0)
    
    # LongAttention Specific
    parser.add_argument("--diversity_weight", type=float, default=1.0)
    parser.add_argument("--null_weight", type=float, default=0.05)
    parser.add_argument("--num_types", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--local_window_size", type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Deterministic behavior
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dynamic output directory
    output_dir = build_output_dir(args)
    setup_logging(log_file=str(Path(output_dir) / "train.log"))
    logger.info("Starting QA Experiment with args:\n%s", vars(args))

    # 1. Tokenizer
    tokenizer = build_tokenizer(args.backbone)

    # 2. Dataset
    train_dataset, val_dataset = load_qa_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_train_rows=args.max_train_samples,
        max_val_rows=args.max_val_samples,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
    )

    # 3. Model
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]
    
    long_attention_config = {
        "local_window_size": args.local_window_size,
        "top_k": args.top_k,
        "num_types": args.num_types,
        "max_length": args.max_length,
    }

    model = build_model(
        task="qa",
        backbone=args.backbone,
        torch_dtype=torch_dtype,
        attention_type=args.attention_type,
        long_attention_config=long_attention_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. Training (SPT Phase)
    if args.run_spt:
        spt_dir = str(Path(output_dir) / "spt")
        run_qa_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            output_dir=spt_dir,
            is_spt=True,
        )

    # 5. Training (Fine-Tuning Phase)
    run_qa_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
        output_dir=output_dir,
        is_spt=False,
    )


if __name__ == "__main__":
    main()
