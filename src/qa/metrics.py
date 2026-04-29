import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

def make_qa_compute_metrics():
    """
    Factory that returns a compute_metrics function for the QA Trainer.
    Evaluates Token-level Exact Match, Span F1, and Yes/No Accuracy.
    """
    def compute_metrics(eval_preds) -> Dict[str, float]:
        # eval_preds.predictions is a tuple containing whatever the model returns.
        # Our MultiTaskQAOutput returns (start_logits, end_logits, yes_no_logits)
        # However, HuggingFace Trainer by default gathers all tensors returned.
        # Let's handle the extraction carefully.
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        
        # Depending on how Trainer collates, preds might be a tuple of 3 arrays
        if isinstance(preds, tuple) and len(preds) >= 3:
            start_logits = preds[0]
            end_logits = preds[1]
            yes_no_logits = preds[2]
        else:
            logger.warning("Unexpected prediction format in QA metrics. Returning 0.")
            return {"exact_match": 0.0, "f1": 0.0, "yes_no_acc": 0.0}

        # Labels are also tricky if multiple were passed. The Trainer usually only passes the 
        # first one defined in the signature or we use a custom eval loop.
        # But we can assume the trainer has been configured to return the dict of labels.
        # If label_ids is a tuple (start_positions, end_positions, answer_types)
        if isinstance(labels, tuple) and len(labels) >= 3:
            start_positions = labels[0]
            end_positions = labels[1]
            answer_types = labels[2]
        elif isinstance(labels, dict):
            start_positions = labels.get("start_positions")
            end_positions = labels.get("end_positions")
            answer_types = labels.get("answer_types")
        else:
            # Fallback if trainer only captured one label
            logger.warning("Could not unpack labels properly. Metrics might be skipped.")
            return {"exact_match": 0.0, "f1": 0.0, "yes_no_acc": 0.0}

        pred_start = np.argmax(start_logits, axis=-1)
        pred_end = np.argmax(end_logits, axis=-1)
        pred_yn = np.argmax(yes_no_logits, axis=-1)

        total_span = 0
        em_count = 0
        f1_sum = 0.0
        
        total_yn = 0
        yn_correct = 0

        for i in range(len(answer_types)):
            true_type = answer_types[i]
            if true_type == 0: # Span answer
                total_span += 1
                
                # EM
                if pred_start[i] == start_positions[i] and pred_end[i] == end_positions[i]:
                    em_count += 1
                    
                # Token-level F1
                pred_span = set(range(pred_start[i], pred_end[i] + 1))
                true_span = set(range(start_positions[i], end_positions[i] + 1))
                
                num_same = len(pred_span.intersection(true_span))
                if num_same > 0:
                    precision = 1.0 * num_same / len(pred_span)
                    recall = 1.0 * num_same / len(true_span)
                    f1 = (2 * precision * recall) / (precision + recall)
                    f1_sum += f1
            else:
                total_yn += 1
                if pred_yn[i] == true_type:
                    yn_correct += 1

        metrics = {}
        if total_span > 0:
            metrics["exact_match"] = round((em_count / total_span) * 100, 2)
            metrics["f1"] = round((f1_sum / total_span) * 100, 2)
            
        if total_yn > 0:
            metrics["yes_no_acc"] = round((yn_correct / total_yn) * 100, 2)
            
        return metrics

    return compute_metrics
