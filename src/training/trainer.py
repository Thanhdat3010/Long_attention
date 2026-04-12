"""
Training Pipeline for LongAttention NMT Experiments.

Wraps HuggingFace ``Seq2SeqTrainer`` (or a causal-LM generation loop for
Qwen decoder-only models) with:
  - Full metric integration (SacreBLEU, ChrF++, COMET, Sink Ratio).
  - Generation-based decoding during evaluation.
  - Callback injection (AttentionSinkCallback, GateDiversityCallback).
  - Dynamic output directory management.

Design Note
-----------
Qwen2 is a decoder-only Causal LM, not an encoder-decoder.
For NMT fine-tuning we use a **prompt-based seq2seq** approach:
    Prompt: "Translate English to French:\n{source}\n\nTranslation:"
    Target: "{reference translation}"

This allows us to use standard Causal LM loss on the target tokens only
(labels are -100 for prompt positions), while the generation API handles
beam search / greedy decoding at inference time.
"""

import logging
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

from .metrics import (
    compute_sacrebleu,
    compute_chrf,
    compute_comet,
    compute_attention_sink_ratio,
)
from .callbacks import (
    AttentionSinkCallback,
    GateDiversityCallback,
    CheckpointMetadataCallback,
)
from ..utils.io_utils import save_metrics, save_model_artifacts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Native Seq2Seq Dataset for Encoder-Decoder Models (e.g., BART)
# ---------------------------------------------------------------------------

class Seq2SeqDocumentDataset(Dataset):
    """
    PyTorch Dataset for standard Encoder-Decoder Seq2Seq models.
    
    Source text is encoded as 'input_ids' for the Encoder.
    Target text is encoded as 'labels' for the Decoder.
    """

    def __init__(
        self,
        dataframe,
        tokenizer: PreTrainedTokenizerBase,
        src_max_len: int = 1024,
        tgt_max_len: int = 256,
        src_lang_name: str = "English",
        tgt_lang_name: str = "French",
    ) -> None:
        import pandas as pd
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        src_text: str = str(row["source"])
        tgt_text: str = str(row["target"])

        # Tokenise source for encoder
        source_enc = self.tokenizer(
            src_text,
            max_length=self.src_max_len,
            truncation=True,
        )

        # Tokenise target for decoder labels
        target_enc = self.tokenizer(
            text_target=tgt_text,
            max_length=self.tgt_max_len,
            truncation=True,
        )

        return {
            "input_ids": torch.tensor(source_enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(source_enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(target_enc["input_ids"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Compute Metrics Factory
# ---------------------------------------------------------------------------

def make_compute_metrics(
    tokenizer: PreTrainedTokenizerBase,
    sources_for_comet: Optional[List[str]] = None,
    use_comet: bool = True,
) -> Callable:
    """
    Build a ``compute_metrics`` function compatible with HuggingFace Trainer.

    The returned function decodes model predictions and computes:
      - SacreBLEU
      - ChrF++
      - COMET (if ``use_comet=True`` and sources provided)

    Args:
        tokenizer:          Tokenizer for decoding predictions.
        sources_for_comet:  Source sentences for COMET (needed for DA model).
        use_comet:          Whether to include COMET computation.

    Returns:
        A callable ``compute_metrics(eval_pred) → Dict[str, float]``.
    """

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred

        # Convert logits → token IDs if needed
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = predictions.argmax(-1)

        # Replace -100 (padding) in labels with pad_token_id for decoding
        labels_for_decode = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and references
        decoded_preds: List[str] = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels: List[str] = tokenizer.batch_decode(
            labels_for_decode, skip_special_tokens=True
        )

        # Strip leading/trailing whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        metrics: Dict[str, float] = {}
        metrics.update(compute_sacrebleu(decoded_preds, decoded_labels))
        metrics.update(compute_chrf(decoded_preds, decoded_labels))

        if use_comet and sources_for_comet is not None:
            # COMET expects a source list matching the eval batch size
            batch_sources = sources_for_comet[: len(decoded_preds)]
            metrics.update(
                compute_comet(batch_sources, decoded_preds, decoded_labels)
            )

        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Training Arguments Builder
# ---------------------------------------------------------------------------

def build_training_args(args: Namespace, output_dir: str) -> Seq2SeqTrainingArguments:
    """
    Build ``Seq2SeqTrainingArguments`` from an argparse Namespace.

    Args:
        args:       Parsed argparse arguments.
        output_dir: Resolved experiment output directory.

    Returns:
        Configured ``Seq2SeqTrainingArguments``.
    """
    # A100 supports bfloat16 natively — no gradient scaler issues.
    # fp16 + pre-loaded float16 model = ValueError (cannot unscale FP16 grads).
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = False  # always off — model loaded in bf16/fp16 already

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="sacrebleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=4,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=2,
        report_to=["none"],
        save_total_limit=3,
        push_to_hub=False,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        gradient_checkpointing=False,
        label_smoothing_factor=0.1,
    )


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def run_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: Namespace,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Execute the full training and evaluation pipeline.

    Steps:
    1. Build ``Seq2SeqTrainingArguments``.
    2. Construct ``compute_metrics`` with source sentences for COMET.
    3. Register callbacks: AttentionSink, GateDiversity, CheckpointMetadata.
    4. Initialise and run ``Seq2SeqTrainer``.
    5. Run final evaluation and return metric dict.

    Args:
        model:         The (possibly patched) Qwen model.
        tokenizer:     Corresponding tokenizer.
        train_dataset: Training split Dataset.
        val_dataset:   Validation split Dataset.
        args:          Full argparse Namespace.
        output_dir:    Resolved output directory path.

    Returns:
        Dict of final evaluation metrics.
    """
    training_args = build_training_args(args, output_dir)

    # Collect source sentences from val set for COMET
    val_sources: Optional[List[str]] = None
    if hasattr(val_dataset, "data"):
        val_sources = val_dataset.data["source"].tolist()

    compute_metrics = make_compute_metrics(
        tokenizer=tokenizer,
        sources_for_comet=val_sources,
        use_comet=getattr(args, "use_comet", True),
    )

    # Collator: standard seq2seq padding
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Callbacks
    metadata_dict = vars(args)
    callbacks = [
        AttentionSinkCallback(output_dir=output_dir, log_to_file=True),
        CheckpointMetadataCallback(metadata=metadata_dict),
    ]
    # Only add GateDiversity callback when LongAttention is active
    if args.attention_type == "long_attention":
        callbacks.append(GateDiversityCallback(output_dir=output_dir, log_to_file=True))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ------ Train ------
    logger.info("Starting training… output_dir=%s", output_dir)
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    # Save the final model and tokenizer
    logger.info("Saving model and tokenizer to %s", output_dir)
    trainer.save_model(output_dir)

    # ------ Final Evaluation ------
    logger.info("Running final evaluation…")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)

    return {**train_result.metrics, **eval_metrics}
