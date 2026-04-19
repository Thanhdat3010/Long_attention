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
    TrainerCallback,
)
from torch.utils.data import Dataset

from ..utils.io_utils import save_metrics, save_model_artifacts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPUMemoryCallback: Tracks and logs VRAM usage
# ---------------------------------------------------------------------------

class GPUMemoryCallback(TrainerCallback):
    """
    Callback that logs GPU memory usage at the end of each log step.
    Helps diagnose OOM issues by showing 'Reserved' vs 'Allocated' memory.
    """
    def on_log(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            # Get current device
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            
            logger.info(
                f"[GPU Memory] Step {state.global_step}: "
                f"Allocated: {allocated:.2f}GB | "
                f"Reserved: {reserved:.2f}GB | "
                f"Peak: {max_allocated:.2f}GB"
            )



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
        tgt_max_len: int = 1024,
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
            padding="max_length",
        )

        # Tokenise target for decoder labels
        target_enc = self.tokenizer(
            text_target=tgt_text,
            max_length=self.tgt_max_len,
            truncation=True,
            padding="max_length",
        )

        return {
            "input_ids": torch.tensor(source_enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(source_enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(target_enc["input_ids"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Compute Metrics Factory
# ---------------------------------------------------------------------------

from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    EarlyStoppingCallback,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BART Denoising Collator for SPT (Text Infilling)
# ---------------------------------------------------------------------------

class BARTDenoisingCollator:
    """
    Data Collator for BART Self Pre-training (SPT).
    Implements Text Infilling: random spans of tokens are replaced with a single [MASK].
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask_ratio: float = 0.3, poisson_lambda: float = 3.0):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.poisson_lambda = poisson_lambda

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Handle pre-tokenized inputs: Pad them to the same length in the batch
        if "input_ids" in examples[0]:
            # For SPT denoising, we reconstruct labels from input_ids. 
            # We strip existing 'labels' to avoid padding conflicts if they have different lengths.
            features = [
                {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]} 
                for ex in examples
            ]
            batch = self.tokenizer.pad(
                features, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=self.tokenizer.model_max_length
            )
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
        else:
            # Fallback for raw text inputs
            texts = [str(ex.get("source", ex.get("text", ""))) for ex in examples]
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.tokenizer.model_max_length
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

        labels = input_ids.clone()
        
        # Create mask
        # Simplified Text Infilling: Mask tokens with mask_ratio probability
        # In a real BART SPT, you'd use Poisson distribution for spans.
        # Here we use a standard MLM-like collator for simplicity but targeting BART.
        probability_matrix = torch.full(labels.shape, self.mask_ratio)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # BART specific: labels stay original, input_ids get [MASK]
        input_ids[masked_indices] = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.convert_tokens_to_ids("<mask>")
        
        # Replace -100 in labels for positions that are NOT masked? 
        # No, for BART denoising, the decoder predicts the FULL uncorrupted sequence.
        # So labels = original input_ids.
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ---------------------------------------------------------------------------
# LongAttentionTrainer: Custom Trainer with Research Losses
# ---------------------------------------------------------------------------

class LongAttentionTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer for LongAttention research.
    Integrates Diversity Regularization and Null-Route Calibration.
    """
    def __init__(self, *args, diversity_weight: float = 0.1, null_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_weight = diversity_weight
        self.null_weight = null_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute standard Seq2Seq loss + LongAttention research losses.
        """
        outputs = model(**inputs)
        # Standard CrossEntropy loss
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        # Collect Research Losses from layers
        # We look for 'diversity_loss' and 'gate_val' attached to encoder outputs
        diversity_loss = torch.tensor(0.0, device=loss.device)
        gate_val = torch.tensor(0.0, device=loss.device)
        count = 0
        
        # Search in the model's encoder for LongAttention layers
        for module in model.modules():
            from ..models.long_attention import LongAttention
            if isinstance(module, LongAttention):
                if hasattr(module, 'last_diversity_loss'):
                    diversity_loss += module.last_diversity_loss
                    gate_val += module.last_gate_val
                    count += 1
        
        if count > 0:
            diversity_loss /= count
            gate_val /= count
            
            # 1. Diversity Loss: Encourage types to stay separate
            loss += self.diversity_weight * diversity_loss
            
            # 2. Null-Route Calibration: Penalize large gate values (encourage sparsity)
            # This is the "Reward for not opening routes"
            loss += self.null_weight * gate_val
            
            if self.state.global_step % 50 == 0:
                logger.debug(f"Step {self.state.global_step} | DivLoss: {diversity_loss:.4f} | GateVal: {gate_val:.4f}")

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def build_training_args(
    args: Namespace, 
    output_dir: str, 
    gradient_accumulation_steps: int = 1
) -> Seq2SeqTrainingArguments:
    """
    Construct HuggingFace Seq2SeqTrainingArguments from argparse Namespace.
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        predict_with_generate=True, # Critical for BLEU evaluation
        generation_max_length=args.max_target_length,
        fp16=(args.dtype == "float16"),
        bf16=(args.dtype == "bfloat16"),
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_sacrebleu",
        greater_is_better=True,
        report_to="tensorboard",
        disable_tqdm=False, # Ensure progress bars are visible
    )


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def run_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: Namespace,
    output_dir: str,
    test_dataset: Optional[Dataset] = None,
    gradient_accumulation_steps: int = 1,
    is_spt: bool = False,
) -> Dict[str, Any]:
    """
    Execute the training pipeline (SPT or Fine-tuning).
    """
    training_args = build_training_args(args, output_dir, gradient_accumulation_steps)
    
    # SPT uses a different collator and objective
    # Lazy imports to break circular dependency
    from .metrics import make_compute_metrics
    from .callbacks import (
        AttentionSinkCallback,
        CheckpointMetadataCallback,
        GateDiversityCallback
    )

    if is_spt:
        logger.info("Configuring for Self Pre-training (SPT)...")
        collator = BARTDenoisingCollator(tokenizer)
        training_args.predict_with_generate = False # Don't evaluate with BLEU during SPT
        compute_metrics = None
    else:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        val_sources = val_dataset.data["source"].tolist() if hasattr(val_dataset, "data") else None
        compute_metrics = make_compute_metrics(
            tokenizer=tokenizer,
            sources_for_comet=val_sources,
            use_comet=getattr(args, "use_comet", True),
        )

    # Callbacks
    callbacks = [
        AttentionSinkCallback(output_dir=output_dir, log_to_file=True),
        GPUMemoryCallback(),
        CheckpointMetadataCallback(metadata=vars(args)),
        EarlyStoppingCallback(early_stopping_patience=3),
    ]
    if args.attention_type == "long_attention":
        callbacks.append(GateDiversityCallback(output_dir=output_dir, log_to_file=True))

    trainer = LongAttentionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        diversity_weight=getattr(args, "diversity_weight", 0.1),
        null_weight=getattr(args, "null_weight", 0.01),
    )

    # ------ Train ------
    logger.info("Starting %s training...", "SPT" if is_spt else "Fine-tuning")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_model(output_dir)

    # ------ Evaluation ------
    eval_metrics = {}
    if not is_spt:
        logger.info("Running final evaluation (Val)...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        if test_dataset is not None:
            logger.info("Running final evaluation (Test)...")
            test_sources = test_dataset.data["source"].tolist() if hasattr(test_dataset, "data") else None
            trainer.compute_metrics = make_compute_metrics(
                tokenizer=tokenizer,
                sources_for_comet=test_sources,
                use_comet=getattr(args, "use_comet", True),
            )
            test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
            eval_metrics.update(test_metrics)

    return {**train_result.metrics, **eval_metrics}
