#!/usr/bin/env python3
"""
run_experiment.py — Main Entry Point for LongAttention NMT Experiments.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nmt.data_preparation import download_and_cache_dataset
from src.models import build_model, build_tokenizer
from src.nmt.trainer import Seq2SeqDocumentDataset, run_training
from src.utils.io_utils import (
    build_output_dir,
    save_metrics,
    save_model_artifacts,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the full argument parser for the experiment pipeline.

    All defaults are chosen to produce a quick smoke-test run that can
    be scaled up by the user for real experiments.
    """
    parser = argparse.ArgumentParser(
        prog="run_experiment.py",
        description="LongAttention NMT training pipeline on WMT14.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store/load WMT14 CSV files.",
    )
    data_grp.add_argument(
        "--dataset",
        type=str,
        default="iwslt2017",
        help="HuggingFace dataset name (e.g., 'iwslt2017' or 'wmt14').",
    )
    data_grp.add_argument(
        "--lang_pair",
        type=str,
        default="en-fr",
        help="Language pair to use (e.g., 'en-fr', 'fr-en', 'de-en').",
    )
    data_grp.add_argument(
        "--group_size",
        type=int,
        default=50,
        help="Number of consecutive sentences to concatenate to form a long document.",
    )
    data_grp.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit training rows to N (None = use all). Useful for quick tests.",
    )
    data_grp.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit validation rows to N (Full dataset used for final evaluation).",
    )
    data_grp.add_argument(
        "--max_val_samples_during_train",
        type=int,
        default=None,
        metavar="N",
        help="Limit validation samples to N ONLY during training epochs to speed up.",
    )
    data_grp.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit test rows to N.",
    )
    data_grp.add_argument(
        "--use_comet_during_train",
        action="store_true",
        help="If set, run COMET evaluation during training epochs (Warning: slow).",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--backbone",
        type=str,
        default="facebook/bart-base",
        help="HuggingFace model ID for the backbone.",
    )
    model_grp.add_argument(
        "--attention_type",
        type=str,
        choices=["vanilla", "led", "long_attention"],
        default="vanilla",
        help=(
            "Attention type to use. "
            "'vanilla' = unmodified BART; "
            "'led' = sliding window + global tokens; "
            "'long_attention' = LongAttention v2 layers."
        ),
    )
    model_grp.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HuggingFace device_map for model loading ('auto', 'cpu', 'cuda').",
    )
    model_grp.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Floating point precision for model weights. bfloat16 recommended for A100.",
    )

    # ── LongAttention-specific ───────────────────────────────────────────────
    la_grp = parser.add_argument_group("LongAttention")
    la_grp.add_argument(
        "--local_window_size",
        type=int,
        default=512,
        help="Local branch sliding window size (tokens).",
    )
    la_grp.add_argument(
        "--top_k",
        type=int,
        default=64,
        help="Number of Top-K positions for long-range retrieval.",
    )
    la_grp.add_argument(
        "--num_types",
        type=int,
        default=3,
        help="Number of dependency types for LongAttention v2.",
    )
    la_grp.add_argument(
        "--bottleneck_ratio",
        type=float,
        default=0.25,
        help="Bottleneck ratio for gating modules.",
    )
    la_grp.add_argument(
        "--dropout_prob",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )

    # ── Sequence Lengths ────────────────────────────────────────────────────
    seq_grp = parser.add_argument_group("Sequence Lengths")
    seq_grp.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="Maximum token length for source (prompt) sequences.",
    )
    seq_grp.add_argument(
        "--max_target_length",
        type=int,
        default=1024,
        help="Maximum token length for target sequences.",
    )

    # ── Training ─────────────────────────────────────────────────────────────
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training and evaluation batch size.",
    )
    train_grp.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate for the AdamW optimiser.",
    )
    train_grp.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    train_grp.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights.",
    )
    train_grp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    train_grp.add_argument(
        "--no_comet",
        action="store_true",
        default=False,
        help="Disable COMET metric (faster evaluation, no unbabel-comet needed).",
    )
    train_grp.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help=(
            "Freeze all backbone weights and only train the injected "
            "LongAttention layers. Requires --attention_type long_attention."
        ),
    )
    train_grp.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing to trade compute for VRAM savings.",
    )
    train_grp.add_argument(
        "--no_val_during_train",
        action="store_true",
        default=False,
        help="Disable validation and checkpoint selection during training epochs to save time.",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Base directory for saving model artifacts and logs.",
    )
    out_grp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    # ── Research Losses ──────────────────────────────────────────────────────
    res_grp = parser.add_argument_group("Research Regularization")
    res_grp.add_argument(
        "--diversity_weight",
        type=float,
        default=0.1,
        help="Weight for type diversity loss.",
    )
    res_grp.add_argument(
        "--null_weight",
        type=float,
        default=0.01,
        help="Weight for null-route (gate sparsity) calibration.",
    )
    res_grp.add_argument(
        "--run_spt",
        action="store_true",
        default=False,
        help="Run Self Pre-training (Text Infilling) before fine-tuning.",
    )
    res_grp.add_argument(
        "--spt_epochs",
        type=int,
        default=1,
        help="Number of epochs for SPT phase.",
    )
    res_grp.add_argument(
        "--spt_learning_rate",
        type=float,
        default=None,
        help="Learning rate for SPT phase. Defaults to --learning_rate if not set.",
    )
    res_grp.add_argument(
        "--spt_mask_ratio",
        type=float,
        default=0.15,
        help="Token masking ratio for SPT denoising (paper default: 0.15 for text).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Resolve output directory FIRST (needed for log file) ────────────────
    output_dir = build_output_dir(args)

    # ── Logging ─────────────────────────────────────────────────────────────
    import logging as _logging
    log_level = _logging.DEBUG if args.verbose else _logging.INFO
    setup_logging(level=log_level, log_file=f"{output_dir}/run.log")
    logger.info("=" * 60)
    logger.info("LongAttention Experiment Pipeline")
    logger.info("=" * 60)
    logger.info("Backbone        : %s", args.backbone)
    logger.info("Attention Type  : %s", args.attention_type)
    logger.info("Language Pair   : %s", args.lang_pair)
    logger.info("Output Dir      : %s", output_dir)
    logger.info("=" * 60)

    # ── Reproducibility ──────────────────────────────────────────────────────
    import torch, random, numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── dtype ────────────────────────────────────────────────────────────────
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # ── Step 1: Data Preparation ─────────────────────────────────────────────
    logger.info("[1/5] Preparing %s data (%s)…", args.dataset, args.lang_pair)
    dataframes = download_and_cache_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        lang_pair=args.lang_pair,
        max_train_rows=args.max_train_samples,
        max_val_rows=args.max_val_samples,
        max_test_rows=args.max_test_samples,
        group_size=args.group_size,
    )
    
    # Enforce truncation even if loaded from a larger cached CSV
    if args.max_train_samples is not None:
        dataframes["train"] = dataframes["train"].head(args.max_train_samples)
    if args.max_val_samples is not None:
        dataframes["val"] = dataframes["val"].head(args.max_val_samples)
    if args.max_test_samples is not None:
        dataframes["test"] = dataframes["test"].head(args.max_test_samples)
    logger.info(
        "Data ready — train: %d | val: %d | test: %d",
        len(dataframes["train"]),
        len(dataframes["val"]),
        len(dataframes["test"]),
    )

    # ── Step 2: Tokenizer ────────────────────────────────────────────────────
    logger.info("[2/5] Loading tokenizer…")
    tokenizer = build_tokenizer(args.backbone)
    # Ensure tokenizer respects the extended length
    tokenizer.model_max_length = args.max_source_length

    # ── Step 3: Model (+ optional injection) ─────────────────────────────────
    logger.info("[3/5] Building model (attention_type=%s)…", args.attention_type)
    long_attention_config = {
        "local_window_size": args.local_window_size,
        "top_k": args.top_k,
        "num_types": args.num_types,
        "bottleneck_ratio": args.bottleneck_ratio,
        "dropout_prob": args.dropout_prob,
        "max_length": max(args.max_source_length, args.max_target_length), # Extend pos embeds for both encoder & decoder
    }
    model = build_model(
        backbone=args.backbone,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        attention_type=args.attention_type,
        long_attention_config=long_attention_config,
        freeze_backbone=args.freeze_backbone,
    )

    # ── Parameter Count Breakdown (important for paper) ─────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_attn_params = 0
    encoder_other_params = 0
    decoder_params = 0
    embed_params = 0
    for name, p in model.named_parameters():
        if "embed" in name or "shared" in name:
            embed_params += p.numel()
        elif "decoder" in name:
            decoder_params += p.numel()
        elif "encoder" in name and "self_attn" in name:
            encoder_attn_params += p.numel()
        elif "encoder" in name:
            encoder_other_params += p.numel()
    # Format parameters nicely
    def fmt_num(val):
        return f"{val:>15,}"
        
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " MODEL PARAMETER BREAKDOWN ".center(58) + "║")
    logger.info("╠" + "═" * 38 + "╦" + "═" * 19 + "╣")
    logger.info("║ {:<36} ║ {:<17} ║".format("Component", "Parameters"))
    logger.info("╠" + "═" * 38 + "╬" + "═" * 19 + "╣")
    logger.info("║ {:<36} ║ {} ║".format("Embedding/Shared", fmt_num(embed_params)))
    logger.info("║ {:<36} ║ {} ║".format("Encoder self_attn (Variable attn)", fmt_num(encoder_attn_params)))
    logger.info("║ {:<36} ║ {} ║".format("Encoder other", fmt_num(encoder_other_params)))
    logger.info("║ {:<36} ║ {} ║".format("Decoder", fmt_num(decoder_params)))
    logger.info("╠" + "═" * 38 + "╬" + "═" * 19 + "╣")
    logger.info("║ {:<36} ║ {} ║".format("Total Parameters", fmt_num(total_params)))
    logger.info("║ {:<36} ║ {} ║".format("Trainable Parameters", fmt_num(trainable_params)))
    logger.info("╚" + "═" * 38 + "╩" + "═" * 19 + "╝")

    if args.freeze_backbone and args.attention_type != "vanilla":
        logger.info("Freezing backbone — only Injected layers will be trained.")

    # ── Step 4: Datasets ──────────────────────────────────────────────────────
    logger.info("[4/5] Building PyTorch datasets…")
    train_ds = Seq2SeqDocumentDataset(
        dataframe=dataframes["train"],
        tokenizer=tokenizer,
        src_max_len=args.max_source_length,
        tgt_max_len=args.max_target_length,
    )
    val_ds = Seq2SeqDocumentDataset(
        dataframe=dataframes["val"],
        tokenizer=tokenizer,
        src_max_len=args.max_source_length,
        tgt_max_len=args.max_target_length,
    )
    test_ds = Seq2SeqDocumentDataset(
        dataframe=dataframes["test"],
        tokenizer=tokenizer,
        src_max_len=args.max_source_length,
        tgt_max_len=args.max_target_length,
    )
    logger.info("Train: %d | Val: %d | Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # Attach raw dataframe for COMET source retrieval
    val_ds.data = dataframes["val"]
    test_ds.data = dataframes["test"]

    # ── Step 5 & 6: Train & Evaluate ──────────────────────────────────────────
    # Stage 1: SPT (Optional)
    if args.run_spt:
        logger.info("[5/6] Starting Stage 1: Self Pre-training (SPT)…")
        spt_output_dir = f"{output_dir}/spt"
        
        # Override epochs for SPT phase
        original_epochs = args.epochs
        args.epochs = args.spt_epochs
        
        # Determine SPT learning rate
        spt_lr = args.spt_learning_rate if args.spt_learning_rate is not None else args.learning_rate
        logger.info("SPT config: epochs=%d | lr=%s | mask_ratio=%.2f", 
                    args.spt_epochs, spt_lr, args.spt_mask_ratio)
        
        run_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            args=args,
            output_dir=spt_output_dir,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            is_spt=True,
            learning_rate_override=spt_lr,
        )
        # Restore epochs for Stage 2
        args.epochs = original_epochs
        
        # Free up memory (GC + CUDA Cache) by deleting model and re-loading it from SPT checkpoint
        del model
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("SPT completed. Re-loading calibrated model from: %s", spt_output_dir)
        model = build_model(
            task="nmt",
            backbone=spt_output_dir,
            device_map=args.device_map,
            torch_dtype=args.dtype,
            attention_type=args.attention_type,
            long_attention_config=long_attention_config,
            freeze_backbone=args.freeze_backbone,
        )
        logger.info("Moving to Stage 2: Fine-tuning.")
    else:
        logger.info("[5/6] Skipping Stage 1 (SPT).")

    # Stage 2: Fine-tuning
    logger.info("[6/6] Starting Stage 2: Fine-tuning…")
    args.use_comet = not args.no_comet  # forward to trainer
    final_metrics = run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        args=args,
        output_dir=output_dir,
        test_dataset=test_ds,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        is_spt=False, # FT stage
    )

    # ── Save Artifacts ────────────────────────────────────────────────────────
    logger.info("Saving final model artifacts…")
    save_model_artifacts(model, tokenizer, output_dir=output_dir, args=args)
    save_metrics(final_metrics, output_dir=output_dir, filename="metrics.json")

    # ── Architectural Diagnostic Report ──────────────────────────────────────
    logger.info("=" * 80)
    logger.info("ARCHITECTURAL DIAGNOSTIC REPORT")
    logger.info("=" * 80)

    if args.attention_type in ("led", "long_attention"):
        try:
            import json as _json
            import traceback as _tb
            model.eval()
            diag_samples = 5
            from transformers import DataCollatorForSeq2Seq
            collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            diag_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
            )
            diag_results = {"per_layer": {}, "summary": {}}

            # Collect per-layer info via forward hooks
            layer_gate_vals = {}    # layer_idx -> list of gate means
            layer_div_losses = {}   # layer_idx -> list of div losses
            layer_type_weights = {} # layer_idx -> list of type weight distributions
            layer_decomp_gates = {} # layer_idx -> list of decomposer gate means

            hooks = []
            encoder_layers = model.model.encoder.layers

            for idx, layer in enumerate(encoder_layers):
                attn = layer.self_attn

                if args.attention_type == "long_attention":
                    # Hook on necessity gate
                    def make_gate_hook(layer_id):
                        def hook(module, inp, out):
                            layer_gate_vals.setdefault(layer_id, []).append(out.mean().item())
                        return hook
                    hooks.append(attn.necessity_gate.register_forward_hook(make_gate_hook(idx)))

                    # Hook on decomposer for gate_score
                    def make_decomp_hook(layer_id):
                        def hook(module, inp, out):
                            if isinstance(out, tuple) and len(out) == 3:
                                layer_decomp_gates.setdefault(layer_id, []).append(out[2].mean().item())
                        return hook
                    hooks.append(attn.decomposer.register_forward_hook(make_decomp_hook(idx)))

                    # Hook on typed retrieval for type_weights and ortho loss (v3)
                    def make_retrieval_hook(layer_id):
                        def hook(module, inp, out):
                            # v3: forward returns single tensor A_long, ortho loss is static
                            # Compute ortho loss from weight matrices
                            if hasattr(module, 'compute_orthogonality_loss'):
                                ortho = module.compute_orthogonality_loss()
                                layer_div_losses.setdefault(layer_id, []).append(ortho.item())
                            elif isinstance(out, tuple) and len(out) == 2:
                                # v2 fallback
                                layer_div_losses.setdefault(layer_id, []).append(out[1].item())
                            # Capture type mixer weights from input
                            if len(inp) > 0:
                                hidden = inp[0]  # query_states (R_encoded in v3)
                                tw = torch.nn.functional.softmax(module.type_mixer(hidden), dim=-1)
                                layer_type_weights.setdefault(layer_id, []).append(tw.mean(dim=(0, 1)).detach().cpu().tolist())
                            else:
                                logger.warning(f"TypedTopKRetrieval hook got empty input tuple at layer {layer_id}")
                        return hook
                    hooks.append(attn.typed_retrieval.register_forward_hook(make_retrieval_hook(idx)))

            # Run forward pass on a few samples
            logger.info("Running diagnostic forward pass on %d val samples…", diag_samples)
            with torch.no_grad():
                for i, batch in enumerate(diag_loader):
                    if i >= diag_samples:
                        break
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    model(**batch)

            # Remove hooks
            for h in hooks:
                h.remove()

            # ── Log results (plain-text, copy-paste friendly) ─────────────
            if args.attention_type == "long_attention":
                num_layers = len(encoder_layers)
                logger.info("")
                logger.info("+" + "-" * 7 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 32 + "+")
                logger.info("| {:^5} | {:^12} | {:^12} | {:^10} | {:^30} |".format(
                    "Layer", "NecessGate", "DecompGate", "DivLoss", "TypeWeights"))
                logger.info("+" + "-" * 7 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 32 + "+")

                all_gates = []
                all_divs = []
                for idx in range(num_layers):
                    g_vals = layer_gate_vals.get(idx, [0])
                    g_mean = sum(g_vals) / max(len(g_vals), 1)
                    d_vals = layer_div_losses.get(idx, [0])
                    d_mean = sum(d_vals) / max(len(d_vals), 1)
                    dg_vals = layer_decomp_gates.get(idx, [0])
                    decomp_mean = sum(dg_vals) / max(len(dg_vals), 1)
                    tw_list = layer_type_weights.get(idx, [[]])
                    if tw_list and tw_list[0]:
                        tw_avg = [sum(x) / len(tw_list) for x in zip(*tw_list)]
                        tw_str = " | ".join(f"T{t}={w:.2f}" for t, w in enumerate(tw_avg))
                    else:
                        tw_str = "N/A"
                        tw_avg = []

                    all_gates.append(g_mean)
                    all_divs.append(d_mean)

                    logger.info("| {:^5} | {:>12.4f} | {:>12.4f} | {:>10.4f} | {:<30} |".format(
                        idx, g_mean, decomp_mean, d_mean, tw_str))

                    diag_results["per_layer"][f"layer_{idx}"] = {
                        "gate_mean": round(g_mean, 4),
                        "diversity_loss": round(d_mean, 4),
                        "decomposer_gate": round(decomp_mean, 4),
                        "type_weights": [round(w, 4) for w in tw_avg],
                    }

                logger.info("+" + "-" * 7 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 32 + "+")

                # Summary
                g_mean_all = sum(all_gates) / len(all_gates)
                d_mean_all = sum(all_divs) / len(all_divs)
                gate_open_pct = sum(1 for g in all_gates if g > 0.5) / len(all_gates) * 100

                logger.info("")
                logger.info("DIAGNOSTIC SUMMARY (LongAttention v3)")
                logger.info("  Gate Activity (Mean)   : %.4f", g_mean_all)
                logger.info("  Diversity Loss (Mean)  : %.4f", d_mean_all)
                logger.info("  Layers Gate > 0.5      : %.1f%% (%d/%d)",
                            gate_open_pct,
                            sum(1 for g in all_gates if g > 0.5),
                            len(all_gates))

                diag_results["summary"] = {
                    "gate_mean": round(g_mean_all, 4),
                    "diversity_loss_mean": round(d_mean_all, 4),
                    "layers_gate_over_0.5": f"{sum(1 for g in all_gates if g > 0.5)}/{len(all_gates)}",
                }

            elif args.attention_type == "led":
                logger.info("LED attention: no per-layer gate diagnostics (standard sliding window).")

            # Save diagnostic report JSON
            diag_path = Path(output_dir) / "diagnostic_report.json"
            with open(diag_path, "w") as f:
                _json.dump(diag_results, f, indent=2)
            logger.info("Diagnostic report saved -> %s", diag_path)

        except Exception as e:
            import traceback as _tb
            logger.error("Diagnostic report FAILED:")
            logger.error(_tb.format_exc())

    else:
        logger.info("Attention type '%s' — no architectural diagnostics.", args.attention_type)

    # ── Final Experiment Metrics (plain-text table) ──────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL EXPERIMENT METRICS")
    logger.info("=" * 60)
    logger.info("+" + "-" * 38 + "+" + "-" * 19 + "+")
    logger.info("| {:<36} | {:>17} |".format("Metric", "Value"))
    logger.info("+" + "-" * 38 + "+" + "-" * 19 + "+")
    for k, v in sorted(final_metrics.items()):
        if isinstance(v, float):
            v_str = f"{v:.4f}"
        else:
            v_str = str(v)
        if len(k) > 36:
            k = k[:33] + "..."
        if len(v_str) > 17:
            v_str = v_str[:14] + "..."
        logger.info("| {:<36} | {:>17} |".format(k, v_str))
    logger.info("+" + "-" * 38 + "+" + "-" * 19 + "+")
    logger.info("")


if __name__ == "__main__":
    main()
