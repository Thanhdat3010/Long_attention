#!/usr/bin/env python3
"""
generate_diagnostic_report.py — Generate architectural diagnostic report from a saved checkpoint.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import torch

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nmt.data_preparation import download_and_cache_dataset
from src.models import build_model
from src.nmt.trainer import Seq2SeqDocumentDataset
from src.utils.io_utils import setup_logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic report for LongAttention checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="./data_cocodoc", help="Data directory.")
    parser.add_argument("--dataset", type=str, default="FiveC/CoCoDoc-MT-20k", help="Dataset name.")
    parser.add_argument("--lang_pair", type=str, default="en-fr", help="Language pair.")
    parser.add_argument("--max_source_length", type=int, default=2048, help="Max source length.")
    parser.add_argument("--max_target_length", type=int, default=4096, help="Max target length.")
    parser.add_argument("--group_size", type=int, default=1, help="Group size (1 for CoCoDoc).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run diagnostics on.")
    
    args = parser.parse_args()
    setup_logging(level=logging.INFO)
    
    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.is_dir():
        logger.error(f"Checkpoint directory not found: {checkpoint_path}")
        sys.exit(1)
        
    logger.info(f"Loading tokenizer and config from {checkpoint_path}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    
    # Load parameters from args.json to rebuild the correct model structure
    args_json_path = checkpoint_path / "args.json"
    attention_type = "long_attention"
    long_attention_config = None
    if args_json_path.is_file():
        try:
            with open(args_json_path, "r", encoding="utf-8") as f:
                saved_args = json.load(f)
                attention_type = saved_args.get("attention_type", "long_attention")
                long_attention_config = {
                    "local_window_size": saved_args.get("local_window_size", 512),
                    "top_k": saved_args.get("top_k", 64),
                    "num_types": saved_args.get("num_types", 3),
                    "bottleneck_ratio": saved_args.get("bottleneck_ratio", 0.25),
                    "dropout_prob": saved_args.get("dropout_prob", 0.1),
                    "max_length": max(saved_args.get("max_source_length", 2048), saved_args.get("max_target_length", 4096)),
                }
        except Exception as e:
            logger.warning(f"Could not read args.json: {e}")
            
    logger.info(f"Building model (attention_type={attention_type})...")
    model = build_model(
        task="nmt",
        backbone=str(checkpoint_path),
        device_map=args.device,
        torch_dtype=torch.float32,
        attention_type=attention_type,
        long_attention_config=long_attention_config,
    )
    model.eval()
    
    logger.info("Loading validation dataset...")
    dataframes = download_and_cache_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        lang_pair=args.lang_pair,
        group_size=args.group_size,
    )
    val_df = dataframes["val"].head(5) # Run on 5 samples
    
    val_ds = Seq2SeqDocumentDataset(
        dataframe=val_df,
        tokenizer=tokenizer,
        src_max_len=args.max_source_length,
        tgt_max_len=args.max_target_length,
    )
    
    from transformers import DataCollatorForSeq2Seq
    collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    diag_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    
    logger.info("Setting up forward hooks...")
    layer_gate_vals = {}
    layer_div_losses = {}
    layer_type_weights = {}
    layer_decomp_gates = {}
    
    hooks = []
    encoder_layers = model.model.encoder.layers
    
    for idx, layer in enumerate(encoder_layers):
        attn = layer.self_attn
        if attention_type == "long_attention":
            def make_gate_hook(layer_id):
                def hook(module, inp, out):
                    layer_gate_vals.setdefault(layer_id, []).append(out.mean().item())
                return hook
            hooks.append(attn.necessity_gate.register_forward_hook(make_gate_hook(idx)))
            
            def make_decomp_hook(layer_id):
                def hook(module, inp, out):
                    if isinstance(out, tuple) and len(out) == 3:
                        layer_decomp_gates.setdefault(layer_id, []).append(out[2].mean().item())
                return hook
            hooks.append(attn.decomposer.register_forward_hook(make_decomp_hook(idx)))
            
            def make_retrieval_hook(layer_id):
                def hook(module, inp, out):
                    if isinstance(out, tuple) and len(out) == 2:
                        layer_div_losses.setdefault(layer_id, []).append(out[1].item())
                    if len(inp) > 0:
                        hidden = inp[0]
                        tw = torch.nn.functional.softmax(module.type_mixer(hidden), dim=-1)
                        layer_type_weights.setdefault(layer_id, []).append(tw.mean(dim=(0, 1)).detach().cpu().tolist())
                return hook
            hooks.append(attn.typed_retrieval.register_forward_hook(make_retrieval_hook(idx)))
            
    logger.info("Running diagnostic forward pass...")
    with torch.no_grad():
        for batch in diag_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
            
    for h in hooks:
        h.remove()
        
    logger.info("Processing results...")
    diag_results = {"per_layer": {}, "summary": {}}
    
    if attention_type == "long_attention":
        num_layers = len(encoder_layers)
        print("\n+" + "-" * 7 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 32 + "+")
        print("| {:^5} | {:^12} | {:^12} | {:^10} | {:^30} |".format(
            "Layer", "NecessGate", "DecompGate", "DivLoss", "TypeWeights"))
        print("+" + "-" * 7 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 32 + "+")
        
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
            
            print("| {:^5} | {:>12.4f} | {:>12.4f} | {:>10.4f} | {:<30} |".format(
                idx, g_mean, decomp_mean, d_mean, tw_str))
                
            diag_results["per_layer"][f"layer_{idx}"] = {
                "gate_mean": round(g_mean, 4),
                "diversity_loss": round(d_mean, 4),
                "decomposer_gate": round(decomp_mean, 4),
                "type_weights": [round(w, 4) for w in tw_avg],
            }
        print("+" + "-" * 7 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 32 + "+")
        
        g_mean_all = sum(all_gates) / len(all_gates)
        d_mean_all = sum(all_divs) / len(all_divs)
        gate_open_pct = sum(1 for g in all_gates if g > 0.5) / len(all_gates) * 100
        
        print("\nDIAGNOSTIC SUMMARY (LongAttention v2)")
        print(f"  Gate Activity (Mean)   : {g_mean_all:.4f}")
        print(f"  Diversity Loss (Mean)  : {d_mean_all:.4f}")
        print(f"  Layers Gate > 0.5      : {gate_open_pct:.1f}% ({sum(1 for g in all_gates if g > 0.5)}/{len(all_gates)})")
        
        diag_results["summary"] = {
            "gate_mean": round(g_mean_all, 4),
            "diversity_loss_mean": round(d_mean_all, 4),
            "layers_gate_over_0.5": f"{sum(1 for g in all_gates if g > 0.5)}/{len(all_gates)}",
        }
        
        # Save diagnostic report JSON
        diag_path = Path(checkpoint_path) / "diagnostic_report.json"
        with open(diag_path, "w") as f:
            json.dump(diag_results, f, indent=2)
        logger.info(f"Diagnostic report saved -> {diag_path}")

if __name__ == "__main__":
    main()
