"""
Data Preparation Module for LongAttention NMT Experiments.

This module handles:
1. Downloading WMT14 translation dataset via HuggingFace `datasets`.
2. Parsing nested dataset structures for configurable language pairs.
3. Converting splits to Pandas DataFrames.
4. Caching DataFrames as CSV files to avoid redundant downloads.
5. Providing a clean DataLoader-ready interface for training/evaluation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPLIT_NAMES: Dict[str, str] = {
    "train": "train",
    "val": "validation",
    "test": "test",
}

CSV_NAMES: Dict[str, str] = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}


# ---------------------------------------------------------------------------
# CSV Caching Utilities
# ---------------------------------------------------------------------------

def _csv_path(data_dir: str, dataset_name: str, lang_pair: str, split: str) -> Path:
    """Return the canonical CSV path for a given split, safely nested."""
    safe_ds = dataset_name.replace("/", "_")
    return Path(data_dir) / safe_ds / lang_pair / CSV_NAMES[split]


def _all_csvs_exist(data_dir: str, dataset_name: str, lang_pair: str) -> bool:
    """Return True if all three split CSVs already exist on disk."""
    return all(_csv_path(data_dir, dataset_name, lang_pair, s).is_file() for s in CSV_NAMES)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame as CSV with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved %d rows → %s", len(df), path)


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV from disk and return a DataFrame."""
    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded %d rows ← %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# WMT14 Download & Parsing
# ---------------------------------------------------------------------------

def _parse_lang_pair(lang_pair: str) -> Tuple[str, str]:
    """
    Parse a language-pair string such as 'en-fr' or 'fr-en'.

    Args:
        lang_pair: Hyphen-separated language pair string.

    Returns:
        (source_lang, target_lang) tuple.

    Raises:
        ValueError: If the format is incorrect.
    """
    parts = lang_pair.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid lang_pair '{lang_pair}'. Expected format: 'en-fr'."
        )
    return parts[0].strip(), parts[1].strip()


def _extract_translation_rows(
    hf_split,
    src_lang: str,
    tgt_lang: str,
    max_examples: Optional[int] = None,
    group_size: int = 1,
    is_flat: bool = False, # True for datasets like CoCoDoc where langs are direct columns
) -> pd.DataFrame:
    """
    Extract source/target text from a HuggingFace split and optionally group
    them into long documents.

    Args:
        hf_split:     A HuggingFace dataset split.
        src_lang:     Source language code (e.g., 'en').
        tgt_lang:     Target language code (e.g., 'fr').
        max_examples: If set, stop reading from the source after this many SENTENCES (or docs if group_size=1).
        group_size:   Number of consecutive sentences to concatenate into a document.
        is_flat:      If True, assume src_lang and tgt_lang are top-level columns.
    """
    sources, targets = [], []
    current_src_group, current_tgt_group = [], []
    
    examples_read = 0
    
    for example in hf_split:
        if max_examples is not None and examples_read >= max_examples:
            break
            
        if is_flat:
            src_text = example.get(src_lang, "").strip()
            tgt_text = example.get(tgt_lang, "").strip()
        else:
            translation: Dict[str, str] = example.get("translation", {})
            src_text = translation.get(src_lang, "").strip()
            tgt_text = translation.get(tgt_lang, "").strip()
        
        if src_text and tgt_text:
            current_src_group.append(src_text)
            current_tgt_group.append(tgt_text)
            examples_read += 1
            
            if len(current_src_group) == group_size:
                sources.append(" ".join(current_src_group))
                targets.append(" ".join(current_tgt_group))
                current_src_group, current_tgt_group = [], []

    # Handle remaining sentences if any
    if current_src_group:
        sources.append(" ".join(current_src_group))
        targets.append(" ".join(current_tgt_group))

    df = pd.DataFrame({"source": sources, "target": targets})
    logger.info(
        "Extracted %d documents from %d source samples (Group Size: %d, Flat: %s)", 
        len(df), examples_read, group_size, is_flat
    )
    return df


def download_and_cache_dataset(
    data_dir: str,
    dataset_name: str = "iwslt2017",
    lang_pair: str = "en-fr",
    max_train_rows: Optional[int] = None,
    max_val_rows: Optional[int] = None,
    max_test_rows: Optional[int] = None,
    group_size: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Download/Stream a dataset, group sentences into docs, and cache.
    """
    src_lang, tgt_lang = _parse_lang_pair(lang_pair)

    # --- Fast path: load from cache ---
    if _all_csvs_exist(data_dir, dataset_name, lang_pair):
        safe_ds = dataset_name.replace("/", "_")
        logger.info( "Loading from cache: %s", data_dir)
        return {
            split: _load_dataframe(_csv_path(data_dir, dataset_name, lang_pair, split))
            for split in CSV_NAMES
        }

    logger.info("Cache not found. Processing %s...", dataset_name)
    
    # Decide between streaming and full download
    use_streaming = (dataset_name == "wmt14") or (dataset_name == "HPLT/DocHPLT")
    is_doc_level = (dataset_name == "FiveC/CoCoDoc-MT-20k")

    if is_doc_level:
        group_size = 1 # Already document level
        logger.info("Dataset %s is already at document level. Setting group_size=1.", dataset_name)

    if dataset_name == "iwslt2017":
        hf_subset = f"iwslt2017-{src_lang}-{tgt_lang}"
        kwargs = {"trust_remote_code": True}
    elif is_doc_level:
        hf_subset = None # No config for this one
        kwargs = {}
    else:
        hf_subset = f"{src_lang}-{tgt_lang}"
        kwargs = {}

    try:
        raw_datasets = load_dataset(dataset_name, hf_subset, streaming=use_streaming, **kwargs)
    except Exception as e:
        if is_doc_level: raise e # Don't retry if it's doc level and fails
        # Fallback for reversed language pairs (e.g., fr-en instead of en-fr)
        if dataset_name == "iwslt2017":
            hf_subset_reversed = f"iwslt2017-{tgt_lang}-{src_lang}"
        else:
            hf_subset_reversed = f"{tgt_lang}-{src_lang}"
        
        logger.warning("Subset '%s' failed, retrying with '%s'...", hf_subset, hf_subset_reversed)
        raw_datasets = load_dataset(dataset_name, hf_subset_reversed, streaming=use_streaming, **kwargs)

    max_examples_map = {
        "train": max_train_rows,
        "val": max_val_rows,
        "test": max_test_rows,
    }
    
    dataframes: Dict[str, pd.DataFrame] = {}
    
    # Split names mapping
    hf_split_key_map = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }

    for split_name, hf_key in hf_split_key_map.items():
        # Handle cases where splits are missing in streaming DatasetDict
        try:
            split_data = raw_datasets[hf_key]
        except KeyError:
            logger.warning(f"Split {hf_key} not found in {dataset_name}. Skipping.")
            dataframes[split_name] = pd.DataFrame(columns=["source", "target"])
            continue

        logger.info(f"Processing {split_name} split...")
        df = _extract_translation_rows(
            split_data,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_examples=max_examples_map[split_name],
            group_size=group_size,
            is_flat=is_doc_level,
        )
        _save_dataframe(df, _csv_path(data_dir, dataset_name, lang_pair, split_name))
        dataframes[split_name] = df

    return dataframes


# ---------------------------------------------------------------------------
# PyTorch Dataset Wrapper
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """
    PyTorch Dataset wrapping a translation DataFrame.

    Each item is a dictionary with 'source' and 'target' strings,
    ready to be tokenised by a HuggingFace tokenizer via a DataCollator.

    Args:
        dataframe:    DataFrame with columns ['source', 'target'].
        tokenizer:    HuggingFace tokenizer (e.g., Qwen2Tokenizer).
        src_max_len:  Maximum token length for source sequences.
        tgt_max_len:  Maximum token length for target sequences.
        src_lang:     Source language ISO code (for mBART-style tokenizers).
        tgt_lang:     Target language ISO code.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        src_max_len: int = 128,
        tgt_max_len: int = 128,
        src_lang: str = "en",
        tgt_lang: str = "fr",
    ) -> None:
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Tokenise a single source/target pair.

        Returns a dict with:
            input_ids, attention_mask  — tokenised source.
            labels                     — tokenised target (for Seq2Seq loss).
        """
        row = self.data.iloc[idx]
        src_text: str = str(row["source"])
        tgt_text: str = str(row["target"])

        # Tokenise source (dynamic padding will be done by data collator)
        model_inputs = self.tokenizer(
            src_text,
            max_length=self.src_max_len,
            padding=False,
            truncation=True,
        )

        # Tokenise target as labels
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_text,
                max_length=self.tgt_max_len,
                padding=False,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def build_datasets(
    dataframes: Dict[str, pd.DataFrame],
    tokenizer,
    src_max_len: int,
    tgt_max_len: int,
    lang_pair: str = "en-fr",
) -> Dict[str, TranslationDataset]:
    """
    Wrap DataFrames into TranslationDataset objects.

    Args:
        dataframes:   Dict of split name → DataFrame (from download_and_cache_wmt14).
        tokenizer:    HuggingFace tokenizer.
        src_max_len:  Max source token length.
        tgt_max_len:  Max target token length.
        lang_pair:    Language pair string (e.g. 'en-fr').

    Returns:
        Dict of split name → TranslationDataset.
    """
    src_lang, tgt_lang = _parse_lang_pair(lang_pair)
    datasets: Dict[str, TranslationDataset] = {}
    for split, df in dataframes.items():
        datasets[split] = TranslationDataset(
            dataframe=df,
            tokenizer=tokenizer,
            src_max_len=src_max_len,
            tgt_max_len=tgt_max_len,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
    return datasets