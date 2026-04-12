"""
Training Callbacks for LongAttention Experiments.

Provides HuggingFace ``TrainerCallback`` subclasses for:
  1. ``AttentionSinkCallback`` — Registers forward hooks on attention layers
     during eval to measure and log the Attention Sink Ratio per layer.
  2. ``GateDiversityCallback`` — Monitors the entropy of gate_score
     distributions to detect attention/gate collapse.
  3. ``CheckpointMetadataCallback`` — Saves experiment metadata alongside
     each checkpoint for reproducibility.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .metrics import compute_attention_sink_ratio, aggregate_sink_ratios

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AttentionSinkCallback
# ---------------------------------------------------------------------------

class AttentionSinkCallback(TrainerCallback):
    """
    Forward-hook based callback to measure Attention Sink Ratio during eval.

    How it works
    ------------
    1. ``on_evaluate_begin``:  Registers forward hooks on every attention
       layer's ``local_attention`` (LongAttention) or ``attn_weights``
       attribute. Each hook captures the returned attention weight tensor.

    2. ``on_evaluate_end``:    Reads captured tensors, computes per-layer
       Attention Sink Ratio, aggregates statistics, and logs them.

    Attributes:
        sink_ratios_history:  Dict mapping eval step → aggregated sink stats.
        log_to_file:          If True, dumps results to a JSON file in
                              the output directory.

    Args:
        output_dir:    Directory where the JSON log is saved.
        log_to_file:   Whether to persist results to disk.
        sink_index:    Token index to treat as the attention sink (default: 0).
    """

    def __init__(
        self,
        output_dir: str,
        log_to_file: bool = True,
        sink_index: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.log_to_file = log_to_file
        self.sink_index = sink_index

        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._captured_weights: Dict[str, List[torch.Tensor]] = {}
        self.sink_ratios_history: Dict[int, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Hook registration helpers
    # ------------------------------------------------------------------

    def _make_hook(self, layer_key: str):
        """Factory: return a forward hook that captures attention weights."""

        def hook(module, input, output):
            # LongAttention returns (output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
            else:
                # Fallback for standard MHA that may return weights directly
                attn_weights = output if isinstance(output, torch.Tensor) else None

            if attn_weights is not None:
                # Detach and move to CPU to avoid OOM on GPU
                self._captured_weights.setdefault(layer_key, []).append(
                    attn_weights.detach().cpu()
                )

        return hook

    def _register_hooks(self, model: torch.nn.Module) -> None:
        """Walk model, register hooks on every self_attn layer."""
        self._captured_weights.clear()
        self._hooks.clear()

        for name, module in model.named_modules():
            if name.endswith("self_attn"):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
                logger.debug("Registered sink hook on: %s", name)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Callback methods
    # ------------------------------------------------------------------

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> None:
        """Register hooks just before evaluation begins."""
        if model is not None:
            self._register_hooks(model)

    def on_evaluate_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Compute sink ratios and log results after evaluation."""
        self._remove_hooks()

        if not self._captured_weights:
            logger.warning("AttentionSinkCallback: no attention weights captured.")
            return

        per_layer_ratios: List[float] = []
        for layer_key, weight_list in self._captured_weights.items():
            # Concatenate batches along the batch dimension
            all_weights = torch.cat(weight_list, dim=0)  # (B_total, H, T_q, T_k)
            ratio = compute_attention_sink_ratio(all_weights, self.sink_index)
            per_layer_ratios.append(ratio)
            logger.info("  Sink ratio [%s]: %.4f (%.1f%%)", layer_key, ratio, ratio * 100)

        aggregated = aggregate_sink_ratios(per_layer_ratios)
        step = state.global_step
        self.sink_ratios_history[step] = aggregated

        logger.info(
            "Attention Sink Ratio @ step %d — mean: %.4f (%.1f%%), "
            "max: %.4f, min: %.4f",
            step,
            aggregated["sink_ratio_mean"],
            aggregated["sink_ratio_mean"] * 100,
            aggregated["sink_ratio_max"],
            aggregated["sink_ratio_min"],
        )

        # Persist to file
        if self.log_to_file:
            log_path = Path(self.output_dir) / "attention_sink_log.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {str(k): v for k, v in self.sink_ratios_history.items()},
                    f,
                    indent=2,
                )

        self._captured_weights.clear()


# ---------------------------------------------------------------------------
# GateDiversityCallback
# ---------------------------------------------------------------------------

class GateDiversityCallback(TrainerCallback):
    """
    Monitor gate_score entropy in LongAttention layers.

    High entropy = gates are firing diversely (healthy compression).
    Low entropy  = gate collapse (all tokens treated as root or affix).

    Hooks into forward pass of ``FunctionalDecomposer`` to capture gate scores,
    then computes Shannon entropy averaged over batch and positions.
    """

    def __init__(self, output_dir: str, log_to_file: bool = True) -> None:
        self.output_dir = output_dir
        self.log_to_file = log_to_file
        self._hooks: List[Any] = []
        self._captured_gates: Dict[str, List[torch.Tensor]] = {}
        self.entropy_history: Dict[int, Dict[str, float]] = {}

    def _make_gate_hook(self, key: str):
        def hook(module, input, output):
            # FunctionalDecomposer returns (R_encoded, A_encoded, gate_score)
            if isinstance(output, tuple) and len(output) == 3:
                gate_score = output[2].detach().cpu()  # (B, T, 1)
                self._captured_gates.setdefault(key, []).append(gate_score)

        return hook

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        self._captured_gates.clear()
        self._hooks.clear()
        for name, module in model.named_modules():
            if type(module).__name__ == "FunctionalDecomposer":
                h = module.register_forward_hook(self._make_gate_hook(name))
                self._hooks.append(h)

    def on_evaluate_end(self, args, state, control, **kwargs):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        if not self._captured_gates:
            return

        entropies: List[float] = []
        for key, gate_list in self._captured_gates.items():
            gates = torch.cat(gate_list, dim=0).squeeze(-1)  # (B_total, T)
            # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
            eps = 1e-8
            p = gates.clamp(eps, 1 - eps)
            entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean().item()
            entropies.append(entropy)

        mean_entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0
        step = state.global_step
        self.entropy_history[step] = {"gate_entropy_mean": round(mean_entropy, 6)}
        logger.info("Gate entropy @ step %d: %.4f", step, mean_entropy)

        if self.log_to_file:
            path = Path(self.output_dir) / "gate_entropy_log.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump({str(k): v for k, v in self.entropy_history.items()}, f, indent=2)

        self._captured_gates.clear()


# ---------------------------------------------------------------------------
# CheckpointMetadataCallback
# ---------------------------------------------------------------------------

class CheckpointMetadataCallback(TrainerCallback):
    """
    Save experiment metadata alongside each checkpoint.

    Writes a ``experiment_metadata.json`` file into every checkpoint
    directory, capturing the training arguments and experiment config.
    This ensures every checkpoint is fully self-describing for reproducibility.

    Args:
        metadata: Dict of metadata to serialize (e.g., argparse Namespace dict).
    """

    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Write metadata into the just-saved checkpoint directory."""
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta_path = ckpt_dir / "experiment_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info("Saved experiment metadata → %s", meta_path)
