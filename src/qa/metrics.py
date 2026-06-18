import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)

def make_qa_compute_metrics():
    """
    Factory that returns a compute_metrics function for the QA Trainer.
    Evaluates Token-level Exact Match, Span F1, and Yes/No Accuracy,
    along with detailed diagnostic logging to audit the QA pipeline.
    """
    def compute_metrics(eval_preds) -> Dict[str, float]:
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        
        # Extract predictions safely
        if isinstance(preds, tuple) and len(preds) >= 3:
            start_logits = preds[0]
            end_logits = preds[1]
            yes_no_logits = preds[2]
        else:
            logger.warning("Unexpected prediction format in QA metrics. Returning 0.")
            return {"exact_match": 0.0, "f1": 0.0, "yes_no_acc": 0.0}

        # Extract labels safely
        if isinstance(labels, tuple) and len(labels) >= 3:
            start_positions = labels[0]
            end_positions = labels[1]
            answer_types = labels[2]
        elif isinstance(labels, dict):
            start_positions = labels.get("start_positions")
            end_positions = labels.get("end_positions")
            answer_types = labels.get("answer_types")
        else:
            logger.warning("Could not unpack labels properly. Metrics might be skipped.")
            return {"exact_match": 0.0, "f1": 0.0, "yes_no_acc": 0.0}

        # Convert to numpy arrays if they are tensors
        if hasattr(start_positions, "numpy"): start_positions = start_positions.numpy()
        if hasattr(end_positions, "numpy"): end_positions = end_positions.numpy()
        if hasattr(answer_types, "numpy"): answer_types = answer_types.numpy()

        pred_start = np.argmax(start_logits, axis=-1)
        pred_end = np.argmax(end_logits, axis=-1)
        pred_yn = np.argmax(yes_no_logits, axis=-1)

        total_samples = len(answer_types)
        total_span = 0
        total_span_valid = 0  # Chunks that actually contain the answer (start/end >= 0)
        total_span_invalid = 0  # Chunks that do not contain the answer (start/end == -1)
        
        em_count_all = 0
        f1_sum_all = 0.0
        
        em_count_valid = 0
        f1_sum_valid = 0.0
        
        total_yn = 0
        yn_correct = 0
        
        # Predict type distribution counters
        pred_types_count = {0: 0, 1: 0, 2: 0} # 0: Span, 1: Yes, 2: No

        for i in range(total_samples):
            true_type = answer_types[i]
            p_yn = pred_yn[i]
            pred_types_count[int(p_yn)] += 1
            
            if true_type == 0:  # Span answer
                total_span += 1
                t_start = start_positions[i]
                t_end = end_positions[i]
                
                is_valid_span = (t_start >= 0 and t_end >= 0)
                if is_valid_span:
                    total_span_valid += 1
                else:
                    total_span_invalid += 1
                
                # Check prediction bounds
                p_start = pred_start[i]
                p_end = pred_end[i]
                
                # Compute Exact Match (EM)
                is_em = (p_start == t_start and p_end == t_end)
                if is_em:
                    em_count_all += 1
                    if is_valid_span:
                        em_count_valid += 1
                
                # Compute Token-level F1
                if is_valid_span:
                    pred_span = set(range(p_start, p_end + 1))
                    true_span = set(range(t_start, t_end + 1))
                    num_same = len(pred_span.intersection(true_span))
                    if num_same > 0:
                        precision = 1.0 * num_same / len(pred_span)
                        recall = 1.0 * num_same / len(true_span)
                        f1 = (2 * precision * recall) / (precision + recall)
                        f1_sum_valid += f1
                        f1_sum_all += f1
                else:
                    # For invalid spans (where ground truth is -1):
                    # If model predicts start > end, it is an empty span prediction, which matches ground truth.
                    if p_start > p_end:
                        f1_sum_all += 1.0
                    else:
                        f1_sum_all += 0.0
            else:  # Yes/No answer
                total_yn += 1
                if p_yn == true_type:
                    yn_correct += 1

        # Calculate metrics
        metrics = {}
        
        # Standard overall metrics (over all chunks, including invalid ones)
        if total_span > 0:
            metrics["exact_match"] = round((em_count_all / total_span) * 100, 2)
            metrics["f1"] = round((f1_sum_all / total_span) * 100, 2)
        else:
            metrics["exact_match"] = 0.0
            metrics["f1"] = 0.0
            
        if total_yn > 0:
            metrics["yes_no_acc"] = round((yn_correct / total_yn) * 100, 2)
        else:
            metrics["yes_no_acc"] = 0.0
            
        # Valid-only metrics (only evaluated on chunks where answer is present)
        valid_em = round((em_count_valid / max(1, total_span_valid)) * 100, 2) if total_span_valid > 0 else 0.0
        valid_f1 = round((f1_sum_valid / max(1, total_span_valid)) * 100, 2) if total_span_valid > 0 else 0.0
        metrics["span_valid_exact_match"] = valid_em
        metrics["span_valid_f1"] = valid_f1
        
        # Compute Logit Statistics
        start_min, start_max, start_mean = float(np.min(start_logits)), float(np.max(start_logits)), float(np.mean(start_logits))
        end_min, end_max, end_mean = float(np.min(end_logits)), float(np.max(end_logits)), float(np.mean(end_logits))
        yn_min, yn_max, yn_mean = float(np.min(yes_no_logits)), float(np.max(yes_no_logits)), float(np.mean(yes_no_logits))
        
        # Print a beautiful evaluation diagnostic card
        logger.info("\n" + "=" * 60)
        logger.info("  QUESTION ANSWERING EVALUATION DIAGNOSTIC SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Chunks Processed       : {total_samples}")
        logger.info(f"  - Yes/No Chunks            : {total_yn}")
        logger.info(f"  - Span Chunks              : {total_span}")
        logger.info(f"    * Valid (has answer)     : {total_span_valid} ({100*total_span_valid/max(1, total_span):.1f}%)")
        logger.info(f"    * Invalid (no answer)    : {total_span_invalid} ({100*total_span_invalid/max(1, total_span):.1f}%)")
        logger.info("-" * 60)
        logger.info("Model Predictions Distribution:")
        logger.info(f"  - Span Predictions         : {pred_types_count[0]} ({100*pred_types_count[0]/total_samples:.1f}%)")
        logger.info(f"  - Yes Predictions          : {pred_types_count[1]} ({100*pred_types_count[1]/total_samples:.1f}%)")
        logger.info(f"  - No Predictions           : {pred_types_count[2]} ({100*pred_types_count[2]/total_samples:.1f}%)")
        logger.info("-" * 60)
        logger.info("Overall Chunks Metrics (Traditional):")
        logger.info(f"  - Exact Match (EM)         : {metrics['exact_match']:.2f}%")
        logger.info(f"  - F1 Score                 : {metrics['f1']:.2f}%")
        logger.info(f"  - Yes/No Accuracy          : {metrics['yes_no_acc']:.2f}%")
        logger.info("-" * 60)
        logger.info("Valid-only Chunks Metrics (Answer is present):")
        logger.info(f"  - EM (Valid-only)          : {valid_em:.2f}%")
        logger.info(f"  - F1 (Valid-only)          : {valid_f1:.2f}%")
        logger.info("-" * 60)
        logger.info("Logits Range and Distribution:")
        logger.info(f"  - Start Logits             : min={start_min:.2f}, max={start_max:.2f}, mean={start_mean:.2f}")
        logger.info(f"  - End Logits               : min={end_min:.2f}, max={end_max:.2f}, mean={end_mean:.2f}")
        logger.info(f"  - Yes/No Logits            : min={yn_min:.2f}, max={yn_max:.2f}, mean={yn_mean:.2f}")
        logger.info("=" * 60 + "\n")
        
        return metrics

    return compute_metrics
