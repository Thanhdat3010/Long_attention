#!/usr/bin/env python3
"""
prepare_data.py — Standalone Data Downloader for LongAttention.

Use this script to strictly download, group sentences into documents, 
and cache the result to CSV BEFORE running the training pipeline.
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_preparation import download_and_cache_dataset
from src.utils.io_utils import setup_logging
import logging

def main():
    parser = argparse.ArgumentParser(description="Download and Cache IWSLT2017 Dataset.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory cache.")
    parser.add_argument("--dataset", type=str, default="iwslt2017", help="Dataset name.")
    parser.add_argument("--lang_pair", type=str, default="en-fr", help="Language pair.")
    parser.add_argument("--group_size", type=int, default=50, help="Sentences per document.")
    parser.add_argument("--max_train", type=int, default=None, metavar="N")
    parser.add_argument("--max_val", type=int, default=None, metavar="N")
    parser.add_argument("--max_test", type=int, default=None, metavar="N")
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("DATA PREPARATION TOOL")
    logger.info("Dataset : %s", args.dataset)
    logger.info("Pair    : %s", args.lang_pair)
    logger.info("Grouping: %s (Simulating Document Level NMT)", args.group_size)
    logger.info("=" * 50)

    # Trigger the download and caching
    download_and_cache_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        lang_pair=args.lang_pair,
        max_train_rows=args.max_train,
        max_val_rows=args.max_val,
        max_test_rows=args.max_test,
        group_size=args.group_size,
    )

    logger.info("SUCCESS: All splits cached in '%s' successfully.", args.data_dir)

if __name__ == "__main__":
    main()
