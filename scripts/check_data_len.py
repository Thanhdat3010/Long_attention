
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import os
import argparse

def check_lengths(csv_path, backbone="facebook/bart-base", sample_size=1000):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    print(f"Loading tokenizer {backbone}...")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    
    print(f"Calculating token lengths for {len(df)} samples...")
    # Tokenize and get length
    lengths = []
    for text in df['source'].astype(str):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))
    
    lengths = np.array(lengths)
    
    print("\n" + "="*40)
    print("DATA LENGTH STATISTICS")
    print("="*40)
    print(f"Mean length    : {lengths.mean():.2f}")
    print(f"Median length  : {np.median(lengths):.2f}")
    print(f"Max length     : {lengths.max()}")
    print(f"Min length     : {lengths.min()}")
    print("-"*40)
    
    over_512 = (lengths > 512).sum() / len(lengths) * 100
    over_1024 = (lengths > 1024).sum() / len(lengths) * 100
    over_2048 = (lengths > 2048).sum() / len(lengths) * 100
    
    print(f"Over 512 tokens  : {over_512:.2f}%")
    print(f"Over 1024 tokens : {over_1024:.2f}% (TRUNCATED currently)")
    print(f"Over 2048 tokens : {over_2048:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/wmt14/en-fr/train.csv")
    parser.add_argument("--backbone", type=str, default="facebook/bart-base")
    parser.add_argument("--sample_size", type=int, default=2000)
    args = parser.parse_args()
    
    check_lengths(args.csv_path, args.backbone, args.sample_size)
