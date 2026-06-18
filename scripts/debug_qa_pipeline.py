#!/usr/bin/env python3
"""
debug_qa_pipeline.py — Diagnostic utility to audit QA data preparation and prediction alignment.
Can be run locally on CPU to verify data prep, or on server to analyze predictions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from src.qa.data_preparation import prepare_hotpotqa_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("debug_qa_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="QA Pipeline Diagnostic Tool")
    parser.add_argument("--mode", type=str, default="data", choices=["data", "preds"],
                        help="data: Verify raw data tokenization. preds: Audit saved prediction files.")
    parser.add_argument("--predictions_file", type=str, default="outputs/outputs_qa_longformer/roberta-base/longformer_lr2e-5_bs4/predictions.json",
                        help="Path to predictions.json saved during trainer execution.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize.")
    parser.add_argument("--max_length", type=int, default=2048, help="Sequence length used.")
    parser.add_argument("--doc_stride", type=int, default=128, help="Stride used.")
    return parser.parse_args()

def debug_data_preparation(args):
    logger.info("Loading HotpotQA validation split (distractor)...")
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info("Loading RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Select samples
    samples = ds.select(range(min(args.num_samples * 2, len(ds))))
    
    logger.info(f"Processing first {len(samples)} examples with stride={args.doc_stride} and max_length={args.max_length}...")
    features = prepare_hotpotqa_features(
        samples,
        tokenizer,
        max_length=args.max_length,
        doc_stride=args.doc_stride
    )

    logger.info("\n" + "="*80)
    logger.info("  QA DATA PREPARATION VISUAL ALIGNMENT AUDIT")
    logger.info("="*80)
    
    printed = 0
    for idx, feature in enumerate(features):
        if printed >= args.num_samples:
            break
            
        t_start = feature["start_positions"].item()
        t_end = feature["end_positions"].item()
        t_type = feature["answer_types"].item()
        input_ids = feature["input_ids"].tolist()
        
        # Check if this chunk actually contains the answer
        if t_type == 0 and t_start >= 0 and t_end >= 0:
            # Decode the true answer from tokens
            decoded_true_answer = tokenizer.decode(input_ids[t_start : t_end + 1]).strip()
            
            print(f"\n--- Feature Chunk {idx} (Sample Index mapping) ---")
            print(f"Answer Type   : {t_type} (0: Span, 1: Yes, 2: No)")
            print(f"Target Span   : Token [{t_start} to {t_end}]")
            print(f"Decoded Span  : '{decoded_true_answer}'")
            
            # Print a snippet of context around the answer
            start_context = max(0, t_start - 10)
            end_context = min(len(input_ids), t_end + 10)
            context_tokens = input_ids[start_context:end_context]
            context_decoded = tokenizer.decode(context_tokens)
            print(f"Context Area  : ... {context_decoded.strip()} ...")
            print("-" * 50)
            printed += 1
            
        elif t_type != 0:
            # Yes/No sample
            ans_str = "Yes" if t_type == 1 else "No"
            print(f"\n--- Feature Chunk {idx} ---")
            print(f"Answer Type   : {t_type} ({ans_str})")
            print("-" * 50)
            printed += 1

def debug_predictions(args):
    pred_path = Path(args.predictions_file)
    if not pred_path.is_file():
        logger.error(f"Predictions file not found at: {pred_path}")
        logger.info("Please run the training pipeline first to save predictions.")
        return

    logger.info(f"Loading predictions from {pred_path}...")
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    logger.info(f"Total predictions found: {len(preds)}")
    
    em_count = 0
    span_count = 0
    yn_count = 0
    yn_correct = 0
    
    mismatches = []
    
    for item in preds:
        true_ans = item.get("true_answer", "").lower().strip()
        pred_ans = item.get("pred_answer", "").lower().strip()
        true_type = item.get("true_type", 0)
        pred_type = item.get("pred_type", 0)
        is_em = item.get("is_exact_match", False)
        
        if true_type == 0:
            span_count += 1
            if is_em:
                em_count += 1
            else:
                mismatches.append(item)
        else:
            yn_count += 1
            if pred_type == true_type:
                yn_correct += 1
                
    logger.info("\n" + "="*80)
    logger.info("  QA PREDICTIONS AUDIT REPORT")
    logger.info("="*80)
    logger.info(f"Total Span Questions     : {span_count}")
    logger.info(f"  - Exact Match (EM)     : {em_count}/{span_count} ({100*em_count/max(1, span_count):.2f}%)")
    logger.info(f"Total Yes/No Questions   : {yn_count}")
    logger.info(f"  - Accuracy             : {yn_correct}/{yn_count} ({100*yn_correct/max(1, yn_count):.2f}%)")
    logger.info("-" * 80)
    
    if mismatches:
        logger.info(f"Sample Mismatches (showing first {min(args.num_samples, len(mismatches))}):")
        for i, item in enumerate(mismatches[:args.num_samples]):
            print(f"\nMismatch #{i+1} (Sample {item['sample_idx']}):")
            print(f"  True Answer : '{item['true_answer']}'")
            print(f"  Pred Answer : '{item['pred_answer']}'")
            print(f"  True Type   : {item['true_type']} (0: Span, 1: Yes, 2: No)")
            print(f"  Pred Type   : {item['pred_type']}")
            print("-" * 50)
            
    logger.info("="*80 + "\n")

def main():
    args = parse_args()
    if args.mode == "data":
        debug_data_preparation(args)
    elif args.mode == "preds":
        debug_predictions(args)

if __name__ == "__main__":
    main()
