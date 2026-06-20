import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class HotpotQADataset(Dataset):
    """
    PyTorch Dataset for Extractive QA with Span and Yes/No targets.
    Expects pre-tokenized features including start/end positions and answer_type.
    """
    def __init__(self, features: List[Dict[str, torch.Tensor]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.features[idx]


def prepare_hotpotqa_features(
    dataset_split,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 4096,
    doc_stride: int = 128,
    max_examples: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Process HotpotQA examples into tokenized features with sliding window stride.
    Resolves the exact answer span using `supporting_facts` to avoid `.find()` errors.
    """
    features = []
    
    for i, example in enumerate(dataset_split):
        if max_examples and i >= max_examples:
            break
            
        question = example["question"]
        answer_text = example["answer"]
        
        # 1. Answer Type Classification
        # 0: Span, 1: Yes, 2: No
        answer_type = 0
        ans_lower = answer_text.strip().lower()
        if ans_lower == "yes":
            answer_type = 1
        elif ans_lower == "no":
            answer_type = 2
            
        # 2. Build Flattened Context and Find Answer Char Span
        context_dict = example["context"]
        titles = context_dict["title"]
        sentences_lists = context_dict["sentences"]
        
        supp_facts = example["supporting_facts"]
        # Convert supporting facts to a quick lookup set: (title, x_id)
        # In datasets version, supporting_facts is a dict of lists
        supp_fact_set = set(zip(supp_facts["title"], supp_facts["sent_id"]))
        
        flattened_context = ""
        start_char_idx = -1
        end_char_idx = -1
        
        for t_idx, title in enumerate(titles):
            for s_idx, sentence in enumerate(sentences_lists[t_idx]):
                current_len = len(flattened_context)
                
                # Check if this sentence is a supporting fact
                is_supp = (title, s_idx) in supp_fact_set
                
                # If it's a supporting fact, AND we haven't found the answer yet, AND it's a span answer
                if is_supp and answer_type == 0 and start_char_idx == -1:
                    local_idx = sentence.find(answer_text)
                    if local_idx != -1:
                        start_char_idx = current_len + local_idx
                        end_char_idx = start_char_idx + len(answer_text)
                
                flattened_context += sentence + " "
                
        # 3. Tokenization with Stride
        tokenized = tokenizer(
            question,
            flattened_context,
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        offset_mapping = tokenized.pop("offset_mapping")
        for chunk_idx in range(len(offset_mapping)):
            offsets = offset_mapping[chunk_idx]
            sequence_ids = tokenized.sequence_ids(chunk_idx)
            
            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1
                
            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1
                
            start_position = -1
            end_position = -1
            
            if answer_type == 0 and start_char_idx != -1:
                chunk_start_char = offsets[context_start][0]
                chunk_end_char = offsets[context_end][1]
                
                if start_char_idx >= chunk_start_char and end_char_idx <= chunk_end_char:
                    idx = context_start
                    while idx <= context_end and offsets[idx][0] <= start_char_idx:
                        idx += 1
                    start_position = idx - 1
                    
                    idx = context_end
                    while idx >= context_start and offsets[idx][1] >= end_char_idx:
                        idx -= 1
                    end_position = idx + 1

            feature = {
                "input_ids": torch.tensor(tokenized["input_ids"][chunk_idx], dtype=torch.long),
                "attention_mask": torch.tensor(tokenized["attention_mask"][chunk_idx], dtype=torch.long),
                "start_positions": torch.tensor(start_position, dtype=torch.long),
                "end_positions": torch.tensor(end_position, dtype=torch.long),
                "answer_types": torch.tensor(answer_type, dtype=torch.long),
                "answer_text": answer_text,
            }
            
            global_mask = torch.zeros_like(feature["input_ids"])
            for idx, seq_id in enumerate(sequence_ids):
                if seq_id == 0 or idx == 0:
                    global_mask[idx] = 1
            feature["global_attention_mask"] = global_mask
            
            features.append(feature)
    
    n_valid = sum(1 for f in features if f["start_positions"].item() >= 0)
    n_total = len(features)
    logger.info(f"Processed {n_total} chunks. Valid spans: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    return features


def load_qa_dataset(
    dataset_name: str = "hotpotqa/hotpot_qa",
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    max_train_rows: Optional[int] = None,
    max_val_rows: Optional[int] = None,
    max_length: int = 4096,
    doc_stride: int = 128,
) -> Tuple[Dataset, Dataset]:
    
    cache_dir = Path("./data/processed/qa")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_cache_path = cache_dir / f"train_len{max_length}_stride{doc_stride}_rows{max_train_rows}_v2.pt"
    val_cache_path = cache_dir / f"val_len{max_length}_stride{doc_stride}_rows{max_val_rows}_v2.pt"
    
    if train_cache_path.exists() and val_cache_path.exists():
        logger.info(f"Loading cached QA dataset from {cache_dir}...")
        train_features = torch.load(train_cache_path)
        val_features = torch.load(val_cache_path)
        return HotpotQADataset(train_features), HotpotQADataset(val_features)
        
    logger.info(f"Downloading/Loading {dataset_name}...")
    ds = load_dataset(dataset_name, "distractor")
    
    logger.info("Processing train split...")
    train_features = prepare_hotpotqa_features(
        ds["train"].select(range(min(max_train_rows, len(ds["train"])))) if max_train_rows else ds["train"],
        tokenizer, max_length, doc_stride
    )
    
    logger.info("Processing validation split...")
    val_features = prepare_hotpotqa_features(
        ds["validation"].select(range(min(max_val_rows, len(ds["validation"])))) if max_val_rows else ds["validation"],
        tokenizer, max_length, doc_stride
    )
    
    logger.info("Saving processed features to cache...")
    torch.save(train_features, train_cache_path)
    torch.save(val_features, val_cache_path)
    
    return HotpotQADataset(train_features), HotpotQADataset(val_features)
