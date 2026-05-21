"""
I/O Utilities for LongAttention Experiments.

Handles:
- Dynamic output directory construction (model/attention_type/lr/bs).
- Saving final metrics as ``metrics.json``.
- Saving tokenizer and model weights.
- Logging setup.
"""

import json
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output Directory Management
# ---------------------------------------------------------------------------

def build_output_dir(args: Namespace) -> str:
    """
    Build a unique, descriptive output directory path from experiment args.

    Format:
        {args.output_dir}/{model_name}/{attention_type}_lr{lr}_bs{batch_size}/

    Example:
        outputs/qwen2-1.5b/long_attention_lr2e-5_bs16/

    Args:
        args: Parsed argparse Namespace with fields:
              output_dir, backbone, attention_type, learning_rate, batch_size.

    Returns:
        Absolute string path to the experiment output directory.
    """
    # Sanitise model name: 'Qwen/Qwen2-1.5B' → 'qwen2-1.5b'
    model_name = args.backbone.split("/")[-1].lower().replace("_", "-")

    # Format learning rate in scientific notation without trailing zeros
    lr_str = f"{args.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e")

    run_name = f"{args.attention_type}_lr{lr_str}_bs{args.batch_size}"

    output_path = Path(args.output_dir) / model_name / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Metrics Persistence
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
    filename: str = "metrics.json",
) -> None:
    """
    Serialize and save evaluation metrics to a JSON file.

    Args:
        metrics:    Dict of metric name → value.
        output_dir: Directory to write the JSON file into.
        filename:   Output filename (default: ``metrics.json``).
    """
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert float32/int tensors to Python native types for JSON serialisation
    serialisable = {k: _to_serialisable(v) for k, v in metrics.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics → %s", path)


def _to_serialisable(value: Any) -> Any:
    """Convert numpy/torch scalars to Python-native types."""
    try:
        import numpy as np
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.tolist()
    except ImportError:
        pass
    return value


def load_metrics(output_dir: str, filename: str = "metrics.json") -> Dict[str, Any]:
    """
    Load saved metrics from a JSON file.

    Args:
        output_dir: Directory containing the JSON file.
        filename:   Filename to load (default: ``metrics.json``).

    Returns:
        Dict of metric name → value, or empty dict if file not found.
    """
    path = Path(output_dir) / filename
    if not path.is_file():
        logger.warning("Metrics file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model & Tokenizer Persistence
# ---------------------------------------------------------------------------

def save_model_artifacts(
    model,
    tokenizer,
    output_dir: str,
    args: Optional[Namespace] = None,
) -> None:
    """
    Save model weights, tokenizer, and experiment config to ``output_dir``.

    Args:
        model:      Trained HuggingFace model.
        tokenizer:  Corresponding tokenizer.
        output_dir: Destination directory.
        args:       Optional argparse Namespace to save as ``args.json``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Saving model weights → %s", out)
    model.save_pretrained(str(out))

    logger.info("Saving tokenizer → %s", out)
    tokenizer.save_pretrained(str(out))

    if args is not None:
        args_path = out / "args.json"
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
        logger.info("Saved args → %s", args_path)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure root logger with a coloured console handler and optional file handler.

    Args:
        level:    Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: If provided, also write logs to this file path.
    """
    handlers = []

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    try:
        import colorlog
        fmt = colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s[%(levelname)s]%(reset)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console.setFormatter(fmt)
    except ImportError:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console.setFormatter(fmt)
    handlers.append(console)

    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    # Suppress noisy third-party loggers
    for noisy in ("transformers", "datasets", "tokenizers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
