import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from packaging import version
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction

from ..utils.io_utils import save_metrics, save_model_artifacts
from .metrics import make_qa_compute_metrics

logger = logging.getLogger(__name__)


class QATrainer(Trainer):
    """
    Custom Trainer for Question Answering tasks.
    Collects diversity_loss and gate_val from LongAttention layers.
    """
    def __init__(self, *args, diversity_weight: float = 0.1, null_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_weight = diversity_weight
        self.null_weight = null_weight
        self.latest_loss = 0.0
        self.latest_gate_val = 0.0
        self.latest_diversity_loss = 0.0
        self.latest_span_loss = 0.0
        self.latest_yn_loss = 0.0

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if hasattr(self, "latest_span_loss") and self.latest_span_loss > 0.0:
            logs["span"] = round(self.latest_span_loss, 4)
        if hasattr(self, "latest_yn_loss") and self.latest_yn_loss > 0.0:
            logs["yn"] = round(self.latest_yn_loss, 4)
        if hasattr(self, "latest_gate_val") and self.latest_gate_val > 0.0:
            logs["gate"] = round(self.latest_gate_val, 4)
        if hasattr(self, "latest_diversity_loss") and self.latest_diversity_loss > 0.0:
            logs["div"] = round(self.latest_diversity_loss, 4)
        super().log(logs, *args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and collect profiling metrics (Latency, Memory, GFLOPS).
        """
        # Determine eval phase dynamically for progress bar description
        old_phase = getattr(self, "eval_phase", None)
        if metric_key_prefix == "test":
            self.eval_phase = "Testing (Test Set)"
        elif metric_key_prefix == "eval":
            if old_phase is None or "Validation" in old_phase:
                if self.state.global_step == 0:
                    self.eval_phase = "Validation (Pre-check)"
                elif self.state.global_step >= self.state.max_steps:
                    self.eval_phase = "Validation (Final)"
                else:
                    self.eval_phase = f"Validation (Epoch {int(self.state.epoch) if self.state.epoch is not None else 0})"

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
        seq_len = getattr(self.args, "max_length", 4096)
        attn_type = getattr(self.args, "attention_type", "standard")
        
        try:
            from ..nmt.metrics import estimate_model_gflops
            gflops = estimate_model_gflops(
                model=self.model,
                seq_len=seq_len,
                batch_size=1, # Report per-sample GFLOPS
                attention_type=attn_type,
                window_size=getattr(self.args, "local_window_size", 512)
            )
            metrics[f"{metric_key_prefix}_estimated_gflops"] = gflops
        except Exception as e:
            logger.warning("GFLOPS estimation failed: %s", e)
            
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation to add LongAttention research losses.
        """
        # MultiTaskQAOutput contains the base loss (Span + YesNo)
        outputs = model(**inputs)
        
        # Loss from the QA head
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        # Track span vs yn loss from the model output
        if hasattr(outputs, "start_logits") and "start_positions" in inputs:
            start_positions = inputs["start_positions"]
            end_positions = inputs["end_positions"]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            span_l = (loss_fct(outputs.start_logits, start_positions) + loss_fct(outputs.end_logits, end_positions)) / 2
            self.latest_span_loss = span_l.item() if not torch.isnan(span_l) else 0.0
            
            if "answer_types" in inputs and hasattr(outputs, "yes_no_logits"):
                yn_l = nn.CrossEntropyLoss()(outputs.yes_no_logits, inputs["answer_types"])
                self.latest_yn_loss = yn_l.item()

        # In MLM phase (SPT), model might return standard MaskedLMOutput
        if hasattr(outputs, "diversity_loss") and outputs.diversity_loss is not None:
            # Extracted cleanly in the wrapper
            diversity_loss = outputs.diversity_loss
            gate_val = outputs.gate_val
            
            # Apply weights
            loss += self.diversity_weight * diversity_loss
            
            # Null-route calibration with warmup
            current_epoch = self.state.epoch if self.state.epoch is not None else 0.0
            warmup_epochs = 0.5 
            warmup_factor = current_epoch / warmup_epochs if current_epoch < warmup_epochs else 1.0
            effective_null_weight = self.null_weight * warmup_factor
            
            loss += effective_null_weight * gate_val
            
            self.latest_gate_val = gate_val.item()
            self.latest_diversity_loss = diversity_loss.item()
        
        self.latest_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        return (loss, outputs) if return_outputs else loss


class SmoothQAProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.trainer = None
        self.eval_pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        from tqdm.auto import tqdm
        is_spt = "MLM" in str(getattr(args, "label_names", "")) or getattr(args, "metric_for_best_model", "") == "eval_loss"
        stage_name = "Stage 1 (SPT - MLM)" if is_spt else "Stage 2 (QA Fine-Tuning)"
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
                span = getattr(trainer, "latest_span_loss", 0.0)
                yn = getattr(trainer, "latest_yn_loss", 0.0)
                if span > 0 or yn > 0:
                    postfix["span"] = f"{span:.3f}"
                    postfix["yn"] = f"{yn:.3f}"
                if gate > 0 or div > 0:
                    postfix["gate"] = f"{gate:.3f}"
                    postfix["div"] = f"{div:.3f}"
                self.pbar.set_postfix(postfix)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def on_evaluate(self, args, state, control, **kwargs):
        from tqdm.auto import tqdm
        eval_dataloader = kwargs.get("eval_dataloader", None)
        if eval_dataloader is not None:
            phase = getattr(self.trainer, "eval_phase", "Validation")
            self.eval_pbar = tqdm(
                total=len(eval_dataloader),
                desc=phase,
                dynamic_ncols=True,
                leave=False
            )

    def on_predict(self, args, state, control, **kwargs):
        return self.on_evaluate(args, state, control, **kwargs)

    def on_prediction_step(self, args, state, control, **kwargs):
        if hasattr(self, "eval_pbar") and self.eval_pbar is not None:
            self.eval_pbar.update(1)

    def on_evaluate_end(self, args, state, control, **kwargs):
        if hasattr(self, "eval_pbar") and self.eval_pbar is not None:
            self.eval_pbar.close()
            self.eval_pbar = None

    def on_predict_end(self, args, state, control, **kwargs):
        return self.on_evaluate_end(args, state, control, **kwargs)


def run_qa_training(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    args,
    output_dir: str,
    is_spt: bool = False,
):
    """
    Main execution loop for QA tasks.
    """
    from ..utils.callbacks import GPUMemoryCallback, CheckpointMetadataCallback

    effective_lr = args.learning_rate
    
    training_kwargs = {
        "output_dir": output_dir,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
        "learning_rate": effective_lr,
        "weight_decay": 0.01,
        "num_train_epochs": args.epochs if not is_spt else getattr(args, "spt_epochs", 2),
        "lr_scheduler_type": "linear",
        "warmup_steps": 100,
        "fp16": (args.dtype == "float16"),
        "bf16": (args.dtype == "bfloat16"),
        "seed": getattr(args, "seed", 42),
        "load_best_model_at_end": True,
        "report_to": "tensorboard",
        "disable_tqdm": True, 
        "gradient_checkpointing": getattr(args, 'gradient_checkpointing', False),
    }

    # Remove parameters that HuggingFace expects to receive via collator in custom loops
    training_args = TrainingArguments(
        label_names=["start_positions", "end_positions", "answer_types"] if not is_spt else ["labels"],
        metric_for_best_model="eval_loss" if is_spt else "eval_text_f1",
        greater_is_better=not is_spt,
        **training_kwargs
    )

    if is_spt:
        logger.info("Setting up Masked Language Modeling (MLM) for SPT...")
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        compute_metrics = None
        # Strip QA-specific keys so DataCollatorForLanguageModeling doesn't crash
        class MLMDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                item = self.dataset[idx]
                return {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
        train_dataset = MLMDatasetWrapper(train_dataset)
        val_dataset = MLMDatasetWrapper(val_dataset)
        eval_dataset_in_trainer = val_dataset
    else:
        # Custom collator to stack tensors
        def qa_collator(features):
            batch = {}
            for k in features[0].keys():
                if isinstance(features[0][k], torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        collator = qa_collator

        # --- Fast Eval Logic (Subset) ---
        eval_dataset_in_trainer = val_dataset
        val_subset_size = getattr(args, "max_val_samples_during_train", None)
        if val_subset_size and val_subset_size < len(val_dataset):
            logger.info(f"Fast Eval: Slicing val_dataset to {val_subset_size} for intermediate steps.")
            from .data_preparation import HotpotQADataset
            eval_dataset_in_trainer = HotpotQADataset(val_dataset.features[:val_subset_size])

        compute_metrics = make_qa_compute_metrics(tokenizer, eval_dataset_in_trainer)

    progress_callback = SmoothQAProgressCallback()
    callbacks = [
        GPUMemoryCallback(),
        CheckpointMetadataCallback(metadata=vars(args)),
        progress_callback,
    ]

    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_in_trainer,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        diversity_weight=getattr(args, "diversity_weight", 0.1),
        null_weight=getattr(args, "null_weight", 0.01),
    )
    
    progress_callback.trainer = trainer

    logger.info("Starting QA training...")
    train_result = trainer.train()
    trainer.save_model(output_dir)

    eval_metrics = {}
    if not is_spt:
        logger.info("Running final FULL QA evaluation...")
        trainer.eval_phase = "Validation (Final)"
        trainer.compute_metrics = make_qa_compute_metrics(tokenizer, val_dataset)
        eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
        logger.info(f"Final QA Metrics: {eval_metrics}")
        
        logger.info("Generating and saving predictions...")
        import numpy as np
        import json
        import os
        
        trainer.eval_phase = "Testing (Predictions)"
        preds = trainer.predict(val_dataset)
        # Handle tuple of predictions safely
        predictions_tuple = preds.predictions
        if isinstance(predictions_tuple, tuple) and len(predictions_tuple) >= 3:
            start_logits = predictions_tuple[0]
            end_logits = predictions_tuple[1]
            yn_logits = predictions_tuple[2]
            
            pred_starts = np.argmax(start_logits, axis=-1)
            pred_ends = np.argmax(end_logits, axis=-1)
            pred_yns = np.argmax(yn_logits, axis=-1)
            
            results = []
            for i in range(len(val_dataset)):
                item = val_dataset[i]
                
                # Get input_ids
                input_ids = item["input_ids"].tolist() if isinstance(item["input_ids"], torch.Tensor) else item["input_ids"]
                
                p_start, p_end, p_yn = pred_starts[i], pred_ends[i], pred_yns[i]
                
                # Handle true values
                t_start = item.get("start_positions", 0)
                t_end = item.get("end_positions", 0)
                t_type = item.get("answer_types", 0)
                
                if isinstance(t_start, torch.Tensor): t_start = t_start.item()
                if isinstance(t_end, torch.Tensor): t_end = t_end.item()
                if isinstance(t_type, torch.Tensor): t_type = t_type.item()
                
                # Decode predicted answer
                if p_yn == 1: pred_answer = "yes"
                elif p_yn == 2: pred_answer = "no"
                else:
                    if p_start <= p_end and p_end < len(input_ids):
                        pred_ids = input_ids[p_start : p_end + 1]
                        pred_answer = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
                    else:
                        pred_answer = ""
                        
                # Decode true answer
                if t_type == 1: true_answer = "yes"
                elif t_type == 2: true_answer = "no"
                else:
                    if t_start <= t_end and t_end < len(input_ids):
                        true_ids = input_ids[t_start : t_end + 1]
                        true_answer = tokenizer.decode(true_ids, skip_special_tokens=True).strip()
                    else:
                        true_answer = ""
                
                results.append({
                    "sample_idx": i,
                    "true_answer": true_answer,
                    "pred_answer": pred_answer,
                    "true_type": t_type,
                    "pred_type": int(p_yn),
                    "is_exact_match": bool(pred_answer.lower() == true_answer.lower())
                })
                
            out_file = os.path.join(output_dir, "predictions.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved {len(results)} predictions to {out_file}")

    return {**train_result.metrics, **eval_metrics}
