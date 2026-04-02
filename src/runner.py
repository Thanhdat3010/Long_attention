"""
runner.py — Train & Evaluate (with Mixed Precision + Gradient Checkpointing)
=============================================================================

Memory optimizations:
  B. Mixed Precision: torch.cuda.amp with fp16/bf16 → ~2× memory saving
  C. Gradient Checkpointing: enabled via --gradient_checkpoint flag
"""

from __future__ import annotations

import json
import os
from argparse import Namespace
from typing import Any, Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from src.backbone import build_model
from src.data_utils import get_dataloader, get_tokenizer
from src.logging_utils import MetricLogger, RoutingTracker, SpecializationTracker
from src.metrics import (
    compute_bleu, compute_comet, compute_efficiency,
    compute_em, compute_f1,
    compute_em_aliases, compute_f1_aliases,
    compute_rouge, compute_routing_stats,
)
from src.modeling import anti_collapse_loss, null_route_loss
from transformers import get_linear_schedule_with_warmup


def qa_loss(start_logits, end_logits, start_pos, end_pos):
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    return (ce(start_logits, start_pos) + ce(end_logits, end_pos)) / 2


# ===================================================================== #
#  Train                                                                 #
# ===================================================================== #

def train(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_la = (args.model == "longattention")
    use_amp = (device.type == "cuda")
    print(f"\n[TRAIN] model={args.model}  backbone={args.backbone}  task={args.task}  device={device}  amp={use_amp}")

    print(f"  [1/4] Loading tokenizer: {args.tokenizer} ...")
    tokenizer = get_tokenizer(args.tokenizer, args.max_length)
    print(f"  [1/4] Tokenizer ready.")

    # Data
    train_path = os.path.join(args.dataset_path, "train.csv")
    test_path  = os.path.join(args.dataset_path, "test.csv")
    print(f"  [2/4] Building train dataset (tokenizing all samples) ...")
    loader = get_dataloader(args.task, train_path, tokenizer,
                            batch_size=args.batch_size, max_length=args.max_length)
    print(f"  [2/4] Train dataset ready: {len(loader.dataset)} samples, {len(loader)} batches.")

    # Model
    print(f"  [3/4] Loading model: {args.backbone} ...")
    model = build_model(args).to(device)
    print(f"  [3/4] Model ready.")
    
    # 2. AdamW chuẩn paper (hạ beta2 để ổn định gradients, eps=1e-8)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01, 
        eps=1e-8, 
        betas=(0.9, 0.98)
    )

    # 3. Kích thước batch mục tiêu (Effective Batch Size = 16)
    target_batch_size = 16
    accum_steps = max(1, target_batch_size // args.batch_size)
    print(f"  → Accumulation steps: {accum_steps} (Effective batch size: {args.batch_size * accum_steps})")

    # 4. Learning Rate Scheduler có Warmup (10% tổng số bước)
    num_training_steps = (len(loader) // accum_steps) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # DocMT loss
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    docmt_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Loggers
    os.makedirs(args.output_dir, exist_ok=True)
    ml = MetricLogger(args.output_dir)
    rt = RoutingTracker() if is_la else None
    st = SpecializationTracker() if is_la else None

    step = 0
    model.train()
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, nb = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step_idx, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            global_attn = batch.get("global_attention_mask")
            if global_attn is not None:
                global_attn = global_attn.to(device)

            # ---- Forward with mixed precision ----
            with torch.cuda.amp.autocast(enabled=use_amp):
                if is_la:
                    out, layer_infos = model(input_ids=ids, attention_mask=mask)
                else:
                    out = model(input_ids=ids, attention_mask=mask, global_attention_mask=global_attn)
                    layer_infos = None

                # Task loss
                if args.task == "qa":
                    sl, el = out["start_logits"], out["end_logits"]
                    sp = labels[:, 0]
                    ep = labels[:, 1]
                    sp = torch.where(sp >= 0, sp.clamp(max=ids.size(1) - 1), sp)
                    ep = torch.where(ep >= 0, ep.clamp(max=ids.size(1) - 1), ep)
                    task_l = qa_loss(sl, el, sp, ep)
                else:
                    logits = out["logits"]
                    shift_l = logits[:, :-1, :].contiguous()
                    shift_t = labels[:, 1:].contiguous()
                    task_l = docmt_criterion(shift_l.view(-1, shift_l.size(-1)), shift_t.view(-1))

                # Regularization (LongAttention only)
                loss = task_l
                if is_la and layer_infos:
                    ac = anti_collapse_loss(layer_infos)
                    nr = null_route_loss(layer_infos)
                    loss = loss + args.anti_collapse_weight * ac + args.null_route_weight * nr
                
                # Chia loss cho accum_steps vì PyTorch cộng dồn thay vì lấy trung bình
                loss = loss / accum_steps

            # ---- Backward with scaler ----
            scaler.scale(loss).backward()
            
            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            total_loss += loss.item()
            nb += 1

            # Log
            entry = {"loss": loss.item(), "task_loss": task_l.item()}
            if is_la and layer_infos:
                entry["anti_collapse"] = ac.item()
                entry["null_route"] = nr.item()
                rt.record(layer_infos, step)
                st.record(layer_infos, step)
            ml.log_step(entry, step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(nb, 1)
        print(f"  → Epoch {epoch}  avg_loss={avg:.4f}")

        # ---- Eval on test set after each epoch ----
        eval_results = eval_epoch(model, tokenizer, args, device, use_amp, is_la)
        epoch_log = {"avg_loss": avg, **eval_results}
        ml.log_epoch(epoch_log, epoch)
        if args.task == "qa":
            em = eval_results.get("em", 0)
            f1 = eval_results.get("f1", 0)
            print(f"    [test] EM={em:.2f}  F1={f1:.2f}")
        else:
            print(f"    [test] BLEU={eval_results.get('bleu', 0):.2f}")

        # Checkpoint
        ckpt = os.path.join(args.output_dir, f"ckpt_epoch{epoch}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                     "eval": eval_results, "args": vars(args)}, ckpt)

    # Save logs
    ml.save("train_metrics.json")
    if rt: rt.save(os.path.join(args.output_dir, "routing_log.json"))
    if st: st.save(os.path.join(args.output_dir, "specialization.json"))
    print(f"[TRAIN] done → {args.output_dir}\n")


# ===================================================================== #
#  Eval after each epoch (lightweight, no efficiency profiling)          #
# ===================================================================== #

def eval_epoch(model, tokenizer, args, device, use_amp, is_la):
    """Quick eval on test set — returns EM/F1 or BLEU dict."""
    model.eval()
    test_path = os.path.join(args.dataset_path, "test.csv")
    if not os.path.isfile(test_path):
        model.train()
        return {}

    loader = get_dataloader(args.task, test_path, tokenizer,
                            batch_size=args.batch_size, max_length=args.max_length, shuffle=False)

    preds, refs, aliases, srcs = [], [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for batch in tqdm(loader, desc="Eval (Epoch)"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            global_attn = batch.get("global_attention_mask")
            if global_attn is not None: global_attn = global_attn.to(device)

            if is_la:
                out, _ = model(input_ids=ids, attention_mask=mask)
            else:
                out = model(input_ids=ids, attention_mask=mask, global_attention_mask=global_attn)

            if args.task == "qa":
                si = out["start_logits"].argmax(-1)
                ei = out["end_logits"].argmax(-1)
                for b in range(ids.size(0)):
                    s, e = si[b].item(), ei[b].item()
                    if e < s: e = s
                    preds.append(tokenizer.decode(ids[b, s:e+1], skip_special_tokens=True))
                refs.extend(batch.get("answer_text", [""] * ids.size(0)))
                # answers = List[List[str]] — all aliases per sample
                batch_aliases = batch.get("answers", None)
                if batch_aliases is not None:
                    aliases.extend(batch_aliases)
                else:
                    aliases.extend([[r] for r in batch.get("answer_text", [""] * ids.size(0))])
            else:
                pid = out["logits"].argmax(-1)
                preds.extend(tokenizer.batch_decode(pid, skip_special_tokens=True))
                refs.extend(batch.get("target_text", [""] * ids.size(0)))
                srcs.extend(batch.get("source_text", [""] * ids.size(0)))

    results = {}
    if args.task == "qa":
        # Use alias-aware metrics (correct for TriviaQA)
        results.update(compute_em_aliases(preds, aliases))
        results.update(compute_f1_aliases(preds, aliases))
        results.update(compute_rouge(preds, refs))
    else:
        results.update(compute_bleu(preds, refs))

    model.train()
    return results


# ===================================================================== #
#  Evaluate                                                              #
# ===================================================================== #

def evaluate(args: Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_la = (args.model == "longattention")
    print(f"\n[EVAL] model={args.model}  backbone={args.backbone}  task={args.task}  device={device}")

    tokenizer = get_tokenizer(args.tokenizer, args.max_length)

    model = build_model(args).to(device)
    ckpts = sorted(
        [f for f in os.listdir(args.output_dir) if f.startswith("ckpt_")],
        key=lambda x: int(x.split("epoch")[1].split(".")[0]),
    )
    if ckpts:
        path = os.path.join(args.output_dir, ckpts[-1])
        sd = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"  loaded {path}")
    else:
        print("  WARNING: no checkpoint found")

    model.eval()

    test_path = os.path.join(args.dataset_path, "test.csv")
    loader = get_dataloader(args.task, test_path, tokenizer,
                            batch_size=args.batch_size, max_length=args.max_length, shuffle=False)

    preds, refs, aliases, srcs = [], [], [], []
    all_infos = []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for batch in tqdm(loader, desc="Eval"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            global_attn = batch.get("global_attention_mask")
            if global_attn is not None: global_attn = global_attn.to(device)

            if is_la:
                out, li = model(input_ids=ids, attention_mask=mask)
                all_infos.append({k: v.cpu() for k, v in li[-1].items()})
            else:
                out = model(input_ids=ids, attention_mask=mask, global_attention_mask=global_attn)

            if args.task == "qa":
                si = out["start_logits"].argmax(-1)
                ei = out["end_logits"].argmax(-1)
                for b in range(ids.size(0)):
                    s, e = si[b].item(), ei[b].item()
                    if e < s: e = s
                    preds.append(tokenizer.decode(ids[b, s:e+1], skip_special_tokens=True))
                refs.extend(batch.get("answer_text", [""]*ids.size(0)))
                batch_aliases = batch.get("answers", None)
                if batch_aliases is not None:
                    aliases.extend(batch_aliases)
                else:
                    aliases.extend([[r] for r in batch.get("answer_text", [""]*ids.size(0))])
            else:
                pid = out["logits"].argmax(-1)
                preds.extend(tokenizer.batch_decode(pid, skip_special_tokens=True))
                refs.extend(batch.get("target_text", [""]*ids.size(0)))
                srcs.extend(batch.get("source_text", [""]*ids.size(0)))

    # Metrics
    results: Dict[str, Any] = {"model": args.model, "backbone": args.backbone, "task": args.task}
    if args.task == "qa":
        # Alias-aware metrics — correct for TriviaQA/HotpotQA
        results.update(compute_em_aliases(preds, aliases))
        results.update(compute_f1_aliases(preds, aliases))
        results.update(compute_rouge(preds, refs))
    else:
        results.update(compute_bleu(preds, refs))
        results.update(compute_comet(preds, refs, srcs))

    # Efficiency
    sample = torch.randint(0, 100, (1, min(args.max_length, 512)), device=device)
    results.update(compute_efficiency(model, sample))

    # Routing (LongAttention only)
    if all_infos:
        results["routing"] = compute_routing_stats(all_infos[-1:])

    out_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[EVAL] saved {out_path}")
    print(json.dumps(results, indent=2, default=str))
    return results


def run(args):
    if args.mode in ("train", "all"):
        train(args)
    if args.mode in ("eval", "all"):
        evaluate(args)
