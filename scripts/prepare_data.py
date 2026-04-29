#!/usr/bin/env python3
"""
prepare_data.py — Standalone Data Downloader for LongAttention (NMT & QA).
Use this script to download and cache datasets before training.
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nmt.data_preparation import download_and_cache_dataset as download_nmt
from src.qa.data_preparation import load_qa_dataset as download_qa
from src.models import build_tokenizer
from src.utils.io_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Universal Data Downloader for NMT and QA.")
    parser.add_argument("--task", type=str, choices=["nmt", "qa", "both"], default="both", help="Task to prepare data for.")
    
    # NMT Args
    parser.add_argument("--nmt_dataset", type=str, default="FiveC/CoCoDoc-MT-20k", help="NMT Dataset name.")
    parser.add_argument("--lang_pair", type=str, default="en-fr", help="Language pair for NMT.")
    parser.add_argument("--nmt_dir", type=str, default="./data_cocodoc", help="NMT data directory.")
    
    # QA Args
    parser.add_argument("--qa_dataset", type=str, default="hotpotqa/hotpot_qa", help="QA Dataset name.")
    parser.add_argument("--backbone", type=str, default="roberta-base", help="Tokenizer to use for QA processing.")
    
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    if args.task in ["nmt", "both"]:
        logger.info("Preparing NMT Data (%s)...", args.nmt_dataset)
        download_nmt(
            data_dir=args.nmt_dir,
            dataset_name=args.nmt_dataset,
            lang_pair=args.lang_pair
        )
        logger.info("NMT Data Ready.")

    if args.task in ["qa", "both"]:
        logger.info("Preparing QA Data (%s)...", args.qa_dataset)
        # QA preparation requires a tokenizer for stride processing
        tokenizer = build_tokenizer(args.backbone)
        download_qa(
            dataset_name=args.qa_dataset,
            tokenizer=tokenizer,
            max_length=4096 # Trigger full processing logic
        )
        logger.info("QA Data Ready.")

if __name__ == "__main__":
    main()
