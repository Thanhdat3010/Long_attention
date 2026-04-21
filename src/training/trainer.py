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
import time
import transformers
from packaging import version
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
from .metrics import make_compute_metrics, estimate_model_gflops

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


import sys

from tqdm.auto import tqdm

class SmoothProgressCallback(TrainerCallback):
    """
    A custom TQDM progress bar that replaces the default HuggingFace one.
    It guarantees real-time updates for loss, gate, and diversity without JSON spam.
    """
    def __init__(self):
        self.pbar = None
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        is_spt = getattr(args, "metric_for_best_model", "") == "eval_loss"
        stage_name = "Stage 1 (SPT)" if is_spt else "Stage 2 (Fine-Tuning)"
        
        self.pbar = tqdm(total=state.max_steps, desc=stage_name, dynamic_ncols=True, leave=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.update(1)
            trainer = self.trainer
            if trainer and hasattr(trainer, "latest_loss"):
                loss = trainer.latest_loss
                gate = getattr(trainer, "latest_gate_val", 0.0)
                div = getattr(trainer, "latest_diversity_loss", 0.0)
                
                postfix = {"loss": f"{loss:.4f}"}
                if gate > 0 or div > 0:
                    postfix["gate"] = f"{gate:.3f}"
                    postfix["div"] = f"{div:.3f}"
                
                self.pbar.set_postfix(postfix)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

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
            # Hardcode max_length to 1024 to prevent OverflowError in some environments
            # where model_max_length defaults to a huge number.
            batch = self.tokenizer.pad(
                features, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=1024
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
        self.latest_loss = 0.0 # Store loss for real-time display
        self.latest_gate_val = 0.0 
        self.latest_diversity_loss = 0.0

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Override log to inject custom metrics into the progress bar and training logs."""
        if hasattr(self, "latest_gate_val") and self.latest_gate_val > 0.0:
            logs["gate"] = round(self.latest_gate_val, 4)
        if hasattr(self, "latest_diversity_loss") and self.latest_diversity_loss > 0.0:
            logs["div"] = round(self.latest_diversity_loss, 4)
        super().log(logs, *args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and collect profiling metrics (Latency, Memory, GFLOPS).
        """
        # Reset peak memory tracker
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.time()
        
        # Run standard evaluation
        metrics = super().evaluate(
            eval_dataset=eval_dataset, 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix
        )
        
        duration = time.time() - start_time
        num_samples = len(eval_dataset) if eval_dataset is not None else 1
        
        # 1. Latency (ms per sample)
        metrics[f"{metric_key_prefix}_latency_ms"] = (duration / num_samples) * 1000
        
        # 2. Peak Memory (MB)
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            metrics[f"{metric_key_prefix}_peak_memory_mb"] = peak_mem
            
        # 3. Estimated GFLOPS (Inference)
        # We assume max_length from trainer config or model
        seq_len = getattr(self.args, "max_source_length", 1024)
        attn_type = getattr(self.args, "attention_type", "standard")
        
        gflops = estimate_model_gflops(
            model=self.model,
            seq_len=seq_len,
            batch_size=1, # Report per-sample GFLOPS
            attention_type=attn_type,
            window_size=512 # Default window size
        )
        metrics[f"{metric_key_prefix}_estimated_gflops"] = gflops
        
        return metrics

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
            
            # 2. Null-Route Calibration (With Dynamic Epoch-based Warmup)
            # Dần dần áp dụng hình phạt trong 0.5 Epoch đầu tiên để tránh chết Cổng sớm
            current_epoch = self.state.epoch if self.state.epoch is not None else 0.0
            warmup_epochs = 0.5 
            
            if current_epoch < warmup_epochs:
                warmup_factor = current_epoch / warmup_epochs
            else:
                warmup_factor = 1.0
                
            effective_null_weight = self.null_weight * warmup_factor
            loss += effective_null_weight * gate_val
            
            self.latest_gate_val = gate_val.item()
            self.latest_diversity_loss = diversity_loss.item()
            
            if self.state.global_step % 50 == 0:
                logger.debug(f"Step {self.state.global_step} | DivLoss: {diversity_loss:.4f} | GateVal: {gate_val:.4f}")

        self.latest_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
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
    # Build kwargs dynamically for version compatibility
    training_kwargs = {
        "output_dir": output_dir,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "epoch",  # Revert to epoch to avoid JSON spam
        "save_strategy": "epoch",
        "predict_with_generate": True,
        "generation_max_length": 1024, # Force sequence generation up to 1024 to prevent BLEU 0
        "generation_num_beams": 4,     # Use beam search for higher translation quality
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": "linear",
        "warmup_steps": 100,
        "fp16": (args.dtype == "float16"),
        "bf16": (args.dtype == "bfloat16"),
        "seed": args.seed,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_sacrebleu",
        "greater_is_better": True,
        "report_to": "tensorboard",
        "disable_tqdm": True, # Disable default TQDM to let our custom bar run smoothly
    }

    # Handle evaluation_strategy vs eval_strategy
    v_info = version.parse(transformers.__version__)
    if v_info >= version.parse("4.41.0"):
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    return Seq2SeqTrainingArguments(**training_kwargs)


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
        training_args.metric_for_best_model = "eval_loss" # Use loss for SPT
        training_args.greater_is_better = False
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
    progress_callback = SmoothProgressCallback()
    callbacks = [
        AttentionSinkCallback(output_dir=output_dir, log_to_file=True),
        GPUMemoryCallback(),
        CheckpointMetadataCallback(metadata=vars(args)),
        EarlyStoppingCallback(early_stopping_patience=3),
        progress_callback,
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
    
    # Assign trainer reference to custom progress callback for real-time loss tracking
    progress_callback.trainer = trainer

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
