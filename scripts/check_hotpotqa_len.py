import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

print("Loading HotpotQA...")
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

print("Loading RoBERTa tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Sample 5000 random rows to get a representative distribution quickly
ds = ds.shuffle(seed=42).select(range(5000))

lengths = []
for item in tqdm(ds, desc="Calculating lengths"):
    # Format the input exactly as the data preparation does:
    # Question </s></s> Title: Sentences...
    question = item["question"]
    contexts = item["context"]
    
    context_str = ""
    for title, sentences in zip(contexts["title"], contexts["sentences"]):
        context_str += f" {title}: " + " ".join(sentences)
    
    # Tokenize
    text = question + " </s></s> " + context_str.strip()
    tokens = tokenizer.encode(text, add_special_tokens=True)
    lengths.append(len(tokens))

lengths = np.array(lengths)

print("\n" + "="*40)
print("HOTPOTQA DATA LENGTH STATISTICS (RoBERTa Tokens)")
print("="*40)
print(f"Mean length    : {lengths.mean():.2f}")
print(f"Median length  : {np.median(lengths):.2f}")
print(f"Max length     : {lengths.max()}")
print(f"Min length     : {lengths.min()}")
print("-"*40)

over_512 = (lengths > 512).sum() / len(lengths) * 100
over_1024 = (lengths > 1024).sum() / len(lengths) * 100
over_2048 = (lengths > 2048).sum() / len(lengths) * 100

print(f"Over 512 tokens  : {over_512:.2f}% (Fails on standard RoBERTa)")
print(f"Over 1024 tokens : {over_1024:.2f}%")
print(f"Over 2048 tokens : {over_2048:.2f}% (Truncated by our 2048 limit)")
print("="*40)
