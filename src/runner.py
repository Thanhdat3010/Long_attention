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
import random
from argparse import Namespace
from collections import Counter
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.backbone import build_model
from src.data_utils import get_dataloader, get_tokenizer
from src.logging_utils import MetricLogger, RoutingTracker, SpecializationTracker
from src.metrics import (
    compute_bleu, compute_comet, compute_efficiency,
    compute_em, compute_f1,
    compute_em_aliases, compute_f1_aliases,
    compute_faithfulness, compute_rouge, compute_routing_stats,
)
from src.longattention_layer import anti_collapse_loss, null_route_loss
from transformers import get_linear_schedule_with_warmup


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def qa_loss(start_logits, end_logits, start_pos, end_pos):
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    return (ce(start_logits, start_pos) + ce(end_logits, end_pos)) / 2


def layerwise_topk_entropy_penalty(layer_infos: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Entropy penalty with depth weighting: shallow layers lighter, deep layers stronger."""
    n_layers = len(layer_infos)
    total = torch.tensor(0.0, device=layer_infos[0]["topk_w"].device, dtype=torch.float32)
    w_sum = torch.tensor(0.0, device=layer_infos[0]["topk_w"].device, dtype=torch.float32)

    for li, info in enumerate(layer_infos):
        tw = torch.nan_to_num(info["topk_w"].float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(1e-6)
        tw = tw / tw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ent = -(tw * tw.log()).sum(dim=-1).mean()

        ratio = li / max(n_layers - 1, 1)
        depth_w = 0.5 + ratio  # 0.5 (early) -> 1.5 (late)
        total = total + depth_w * ent
        w_sum = w_sum + depth_w

    return total / w_sum.clamp_min(1e-6)


def layerwise_topk_margin_penalty(layer_infos: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Penalize low top1-top2 margin, with stronger pressure in deeper layers."""
    n_layers = len(layer_infos)
    total = torch.tensor(0.0, device=layer_infos[0]["topk_w"].device, dtype=torch.float32)
    w_sum = torch.tensor(0.0, device=layer_infos[0]["topk_w"].device, dtype=torch.float32)

    for li, info in enumerate(layer_infos):
        tw = torch.nan_to_num(info["topk_w"].float(), nan=0.0, posinf=1.0, neginf=0.0)
        if tw.size(-1) < 2:
            continue
        tw = tw / tw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        sorted_w, _ = tw.sort(dim=-1, descending=True)
        margin = sorted_w[..., 0] - sorted_w[..., 1]

        ratio = li / max(n_layers - 1, 1)
        target_margin = 0.10 + 0.20 * ratio
        depth_w = 0.5 + ratio

        total = total + depth_w * F.relu(target_margin - margin).mean()
        w_sum = w_sum + depth_w

    if w_sum.item() <= 0:
        return torch.zeros((), device=layer_infos[0]["topk_w"].device, dtype=torch.float32)
    return total / w_sum.clamp_min(1e-6)


def layerwise_route_budget_penalty(layer_infos: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Penalize overly-open long-range routing; deeper layers are encouraged to be leaner."""
    n_layers = len(layer_infos)
    total = torch.tensor(0.0, device=layer_infos[0]["gate"].device, dtype=torch.float32)
    w_sum = torch.tensor(0.0, device=layer_infos[0]["gate"].device, dtype=torch.float32)

    for li, info in enumerate(layer_infos):
        ratio = li / max(n_layers - 1, 1)
        depth_w = 0.6 + ratio
        # Softer V11 budget targets: preserve task quality while still discouraging over-routing.
        target_active = 0.52 - 0.14 * ratio
        gate_eff = info.get("gate_eff", info["gate"]).float()
        active = gate_eff.mean()
        total = total + depth_w * F.relu(active - target_active).pow(2)
        w_sum = w_sum + depth_w

    return total / w_sum.clamp_min(1e-6)


def layerwise_route_consistency_penalty(layer_infos: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Encourage adjacent layers to keep routing distributions reasonably consistent."""
    if len(layer_infos) < 2:
        return torch.zeros((), device=layer_infos[0]["topk_w"].device, dtype=torch.float32)

    total = torch.tensor(0.0, device=layer_infos[0]["topk_w"].device, dtype=torch.float32)
    w_sum = torch.tensor(0.0, device=layer_infos[0]["topk_w"].device, dtype=torch.float32)
    n_pairs = len(layer_infos) - 1

    for li in range(n_pairs):
        p = torch.nan_to_num(layer_infos[li]["topk_w"].float(), nan=0.0, posinf=1.0, neginf=0.0)
        q = torch.nan_to_num(layer_infos[li + 1]["topk_w"].float(), nan=0.0, posinf=1.0, neginf=0.0)
        k = min(p.size(-1), q.size(-1))
        if k <= 0:
            continue
        p = p[..., :k]
        q = q[..., :k]
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        kl_pq = (p * (p.clamp_min(1e-6).log() - q.clamp_min(1e-6).log())).sum(dim=-1)
        kl_qp = (q * (q.clamp_min(1e-6).log() - p.clamp_min(1e-6).log())).sum(dim=-1)
        sym_kl = 0.5 * (kl_pq + kl_qp).mean()

        ratio = li / max(n_pairs - 1, 1)
        depth_w = 0.5 + ratio
        total = total + depth_w * sym_kl
        w_sum = w_sum + depth_w

    return total / w_sum.clamp_min(1e-6)


# ===================================================================== #
#  Train                                                                 #
# ===================================================================== #

def train(args: Namespace):
    set_seed(getattr(args, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_la = (args.model == "longattention")
    use_amp = (device.type == "cuda")
    route_entropy_weight = 5e-4
    route_margin_weight = 2e-4
    route_budget_weight = float(getattr(args, "route_budget_weight", 1.1e-3))
    route_consistency_weight = float(getattr(args, "route_consistency_weight", 1e-4))
    route_budget_warmup_ratio = float(getattr(args, "route_budget_warmup_ratio", 0.35))
    route_budget_warmup_ratio = max(0.0, min(1.0, route_budget_warmup_ratio))
    routing_reg_warmup_ratio = float(getattr(args, "routing_reg_warmup_ratio", 0.35))
    routing_reg_warmup_ratio = max(0.0, min(1.0, routing_reg_warmup_ratio))
    print(f"\n[TRAIN] model={args.model}  backbone={args.backbone}  task={args.task}  device={device}  amp={use_amp}")

    print(f"  [1/4] Loading tokenizer: {args.tokenizer} ...")
    tokenizer = get_tokenizer(args.tokenizer, args.max_length)
    effective_max_length = int(getattr(tokenizer, "model_max_length", args.max_length))
    if effective_max_length < int(args.max_length):
        print(
            f"  [warn] max_length={args.max_length} exceeds tokenizer/backbone capacity; "
            f"using max_length={effective_max_length}"
        )
    print(f"  [1/4] Tokenizer ready.")

    # Data
    train_path = os.path.join(args.dataset_path, "train.csv")
    test_path  = os.path.join(args.dataset_path, "test.csv")
    print(f"  [2/4] Building train dataset (tokenizing all samples) ...")
    loader = get_dataloader(args.task, train_path, tokenizer,
                            batch_size=args.batch_size, max_length=effective_max_length)
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
    num_training_steps = ((len(loader) + accum_steps - 1) // accum_steps) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # DocMT loss handled by seq2seq models

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
        epoch_alphas: List[float] = []
        ac_warn_count = 0
        nr_warn_count = 0
        ent_warn_count = 0
        margin_warn_count = 0
        budget_warn_count = 0
        consistency_warn_count = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step_idx, batch in enumerate(pbar):
            global_forward_step = (epoch - 1) * len(loader) + (step_idx + 1)
            total_forward_steps = max(args.epochs * len(loader), 1)
            if routing_reg_warmup_ratio > 0.0:
                reg_warmup_steps = max(int(total_forward_steps * routing_reg_warmup_ratio), 1)
                routing_reg_scale = min(global_forward_step / reg_warmup_steps, 1.0)
            else:
                routing_reg_scale = 1.0
            if route_budget_warmup_ratio > 0.0:
                warmup_steps = max(int(total_forward_steps * route_budget_warmup_ratio), 1)
                budget_scale = min(global_forward_step / warmup_steps, 1.0)
            else:
                budget_scale = 1.0
            route_entropy_weight_eff = route_entropy_weight * routing_reg_scale
            route_margin_weight_eff = route_margin_weight * routing_reg_scale
            route_budget_weight_eff = route_budget_weight * routing_reg_scale * budget_scale
            route_consistency_weight_eff = route_consistency_weight * routing_reg_scale

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            context_mask = batch.get("context_mask")
            if context_mask is not None:
                context_mask = context_mask.to(device)
            
            global_attn = batch.get("global_attention_mask")
            if global_attn is not None:
                global_attn = global_attn.to(device)

            # ---- Forward with mixed precision ----
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                if args.task == "docmt":
                    out = model(input_ids=ids, attention_mask=mask, labels=labels)
                    task_l = out["loss"]
                    layer_infos = out.get("layer_infos") if isinstance(out, dict) else None
                else:
                    if is_la:
                        out, layer_infos = model(input_ids=ids, attention_mask=mask)
                    else:
                        out = model(input_ids=ids, attention_mask=mask, global_attention_mask=global_attn)
                        layer_infos = None

                    # Task loss
                    sl, el = out["start_logits"], out["end_logits"]
                    sl = torch.nan_to_num(sl, nan=-1e4, posinf=1e4, neginf=-1e4)
                    el = torch.nan_to_num(el, nan=-1e4, posinf=1e4, neginf=-1e4)
                    if context_mask is not None:
                        # Avoid dtype min in AMP; very large magnitude can destabilize softmax/CE.
                        mask_val = -1e4
                        sl = sl.masked_fill(context_mask == 0, mask_val)
                        el = el.masked_fill(context_mask == 0, mask_val)
                    sp = labels[:, 0]
                    ep = labels[:, 1]
                    sp = torch.where(sp >= 0, sp.clamp(max=ids.size(1) - 1), sp)
                    ep = torch.where(ep >= 0, ep.clamp(max=ids.size(1) - 1), ep)
                    task_l = qa_loss(sl.float(), el.float(), sp, ep)

                # Regularization (LongAttention only)
                raw_loss = task_l
                if is_la and layer_infos:
                    ac = anti_collapse_loss(layer_infos)
                    nr = null_route_loss(layer_infos)
                    ent = layerwise_topk_entropy_penalty(layer_infos)
                    margin_l = layerwise_topk_margin_penalty(layer_infos)
                    budget_l = layerwise_route_budget_penalty(layer_infos)
                    consistency_l = layerwise_route_consistency_penalty(layer_infos)
                    if not torch.isfinite(ac):
                        ac_warn_count += 1
                        if ac_warn_count <= 5 or ac_warn_count % 100 == 0:
                            print(
                                f"[warn] non-finite anti_collapse at epoch={epoch} step={step_idx}; "
                                f"set to 0 (count={ac_warn_count})"
                            )
                        ac = torch.zeros((), device=task_l.device, dtype=task_l.dtype)
                    if not torch.isfinite(nr):
                        nr_warn_count += 1
                        if nr_warn_count <= 5 or nr_warn_count % 100 == 0:
                            print(
                                f"[warn] non-finite null_route at epoch={epoch} step={step_idx}; "
                                f"set to 0 (count={nr_warn_count})"
                            )
                        nr = torch.zeros((), device=task_l.device, dtype=task_l.dtype)
                    if not torch.isfinite(ent):
                        ent_warn_count += 1
                        if ent_warn_count <= 5 or ent_warn_count % 100 == 0:
                            print(
                                f"[warn] non-finite route entropy penalty at epoch={epoch} step={step_idx}; "
                                f"set to 0 (count={ent_warn_count})"
                            )
                        ent = torch.zeros((), device=task_l.device, dtype=task_l.dtype)
                    if not torch.isfinite(margin_l):
                        margin_warn_count += 1
                        if margin_warn_count <= 5 or margin_warn_count % 100 == 0:
                            print(
                                f"[warn] non-finite route margin penalty at epoch={epoch} step={step_idx}; "
                                f"set to 0 (count={margin_warn_count})"
                            )
                        margin_l = torch.zeros((), device=task_l.device, dtype=task_l.dtype)
                    if not torch.isfinite(budget_l):
                        budget_warn_count += 1
                        if budget_warn_count <= 5 or budget_warn_count % 100 == 0:
                            print(
                                f"[warn] non-finite route budget penalty at epoch={epoch} step={step_idx}; "
                                f"set to 0 (count={budget_warn_count})"
                            )
                        budget_l = torch.zeros((), device=task_l.device, dtype=task_l.dtype)
                    if not torch.isfinite(consistency_l):
                        consistency_warn_count += 1
                        if consistency_warn_count <= 5 or consistency_warn_count % 100 == 0:
                            print(
                                f"[warn] non-finite route consistency penalty at epoch={epoch} step={step_idx}; "
                                f"set to 0 (count={consistency_warn_count})"
                            )
                        consistency_l = torch.zeros((), device=task_l.device, dtype=task_l.dtype)
                    raw_loss = (
                        raw_loss
                        + args.anti_collapse_weight * ac
                        + args.null_route_weight * nr
                        + route_entropy_weight_eff * ent
                        + route_margin_weight_eff * margin_l
                        + route_budget_weight_eff * budget_l
                        + route_consistency_weight_eff * consistency_l
                    )

                # Chia loss cho accum_steps vì PyTorch cộng dồn thay vì lấy trung bình
                loss = raw_loss / accum_steps

            if not torch.isfinite(loss):
                print(
                    f"[warn] non-finite loss at epoch={epoch} step={step_idx}: "
                    f"raw_loss={float(raw_loss.detach().float().cpu())}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

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
            total_loss += raw_loss.item()
            nb += 1

            # Log
            entry = {"loss": raw_loss.item(), "task_loss": task_l.item()}
            if is_la and layer_infos:
                entry["anti_collapse"] = ac.item()
                entry["null_route"] = nr.item()
                entry["route_entropy_penalty"] = ent.item()
                entry["route_margin_penalty"] = margin_l.item()
                entry["route_budget_penalty"] = budget_l.item()
                entry["route_budget_weight_eff"] = float(route_budget_weight_eff)
                entry["route_consistency_penalty"] = consistency_l.item()
                entry["route_entropy_weight_eff"] = float(route_entropy_weight_eff)
                entry["route_margin_weight_eff"] = float(route_margin_weight_eff)
                entry["route_consistency_weight_eff"] = float(route_consistency_weight_eff)
                gate_eff_means = [float(info.get("gate_eff", info["gate"]).detach().float().mean().cpu().item()) for info in layer_infos if "gate" in info]
                if gate_eff_means:
                    entry["gate_eff_mean"] = float(sum(gate_eff_means) / len(gate_eff_means))
                adaptive_single = [float(info["adaptive_single_ratio"].detach().float().cpu().item()) for info in layer_infos if "adaptive_single_ratio" in info]
                if adaptive_single:
                    entry["adaptive_single_ratio_mean"] = float(sum(adaptive_single) / len(adaptive_single))
                intent_align = [float(info["gate_intent_alignment"].detach().float().cpu().item()) for info in layer_infos if "gate_intent_alignment" in info]
                if intent_align:
                    entry["gate_intent_alignment_mean"] = float(sum(intent_align) / len(intent_align))
                coarse_candidates = [float(info["coarse_candidates"].detach().float().cpu().item()) for info in layer_infos if "coarse_candidates" in info]
                if coarse_candidates:
                    entry["coarse_candidates_mean"] = float(sum(coarse_candidates) / len(coarse_candidates))
                hard_budgets = [float(info["hard_budget"].detach().float().cpu().item()) for info in layer_infos if "hard_budget" in info]
                if hard_budgets:
                    entry["hard_budget_mean"] = float(sum(hard_budgets) / len(hard_budgets))
                alphas = [float(info["alpha"].detach().cpu().item()) for info in layer_infos if "alpha" in info]
                if alphas:
                    entry["alpha_mean"] = float(sum(alphas) / len(alphas))
                    epoch_alphas.extend(alphas)
                rt.record(layer_infos, step)
                st.record(layer_infos, step)
            ml.log_step(entry, step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(nb, 1)
        if epoch_alphas:
            alpha_mean_epoch = sum(epoch_alphas) / len(epoch_alphas)
            alpha_min = min(epoch_alphas)
            alpha_max = max(epoch_alphas)
            print(
                f"  → Epoch {epoch}  avg_loss={avg:.4f}  "
                f"alpha_mean_epoch={alpha_mean_epoch:.6f}  "
                f"alpha_min={alpha_min:.6f}  alpha_max={alpha_max:.6f}"
            )
        else:
            print(f"  → Epoch {epoch}  avg_loss={avg:.4f}")

        # Log only training loss per epoch; eval runs in evaluate()
        ml.log_epoch({"avg_loss": avg}, epoch)

        # Checkpoint
        ckpt = os.path.join(args.output_dir, f"ckpt_epoch{epoch}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                 "args": vars(args)}, ckpt)

    # Save logs
    ml.save("train_metrics.json")
    if rt: rt.save(os.path.join(args.output_dir, "routing_log.json"))
    if st: st.save(os.path.join(args.output_dir, "specialization.json"))
    print(f"[TRAIN] done → {args.output_dir}\n")


# ===================================================================== #
#  Evaluate                                                              #
# ===================================================================== #

def evaluate(args: Namespace):
    set_seed(getattr(args, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_la = (args.model == "longattention")
    faithfulness_conf_ratio = float(getattr(args, "faithfulness_conf_ratio", 0.5))
    faithfulness_vote_ratio = float(getattr(args, "faithfulness_vote_ratio", 0.025))
    print(f"\n[EVAL] model={args.model}  backbone={args.backbone}  task={args.task}  device={device}")

    tokenizer = get_tokenizer(args.tokenizer, args.max_length)
    effective_max_length = int(getattr(tokenizer, "model_max_length", args.max_length))
    if effective_max_length < int(args.max_length):
        print(
            f"  [warn] max_length={args.max_length} exceeds tokenizer/backbone capacity; "
            f"using max_length={effective_max_length}"
        )

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
    if not os.path.isfile(test_path):
        print(f"  WARNING: test.csv not found at {test_path}")
        return {}
    loader = get_dataloader(args.task, test_path, tokenizer,
                            batch_size=args.batch_size, max_length=effective_max_length, shuffle=False)

    preds, refs, aliases, srcs = [], [], [], []
    selected_segments, gold_segments = [], []
    active_route_counts, selected_segment_counts = [], []
    all_infos = []
    sample_ids = None
    sample_mask = None

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        for batch in tqdm(loader, desc="Eval"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            context_mask = batch.get("context_mask")
            if context_mask is not None:
                context_mask = context_mask.to(device)
            if sample_ids is None:
                sample_ids = ids[:1].detach().clone()
                sample_mask = mask[:1].detach().clone()
            global_attn = batch.get("global_attention_mask")
            if global_attn is not None: global_attn = global_attn.to(device)

            if args.task == "docmt":
                gen_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=min(args.max_length, getattr(args, "gen_max_length", 256)),
                )
                preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
                refs.extend(batch.get("target_text", [""] * ids.size(0)))
                srcs.extend(batch.get("source_text", [""] * ids.size(0)))
                inner = getattr(model, "model", None)
                if inner is not None:
                    enc_infos = getattr(inner.model.encoder, "_last_infos", None)
                    if enc_infos:
                        # Memory optimization: Keep only current batch info
                        all_infos = [{k: v.cpu() for k, v in info.items()} for info in enc_infos]
            else:
                if is_la:
                    out, li = model(input_ids=ids, attention_mask=mask)
                    # Memory optimization: Keep only current batch info to avoid RAM OOM
                    all_infos = [{k: v.cpu() for k, v in layer.items()} for layer in li]
                    last_info = li[-1]
                else:
                    out = model(input_ids=ids, attention_mask=mask, global_attention_mask=global_attn)

                sl, el = out["start_logits"], out["end_logits"]
                if context_mask is not None:
                    mask_val = -1e4
                    sl = sl.masked_fill(context_mask == 0, mask_val)
                    el = el.masked_fill(context_mask == 0, mask_val)
                si = sl.argmax(-1)
                ei = el.argmax(-1)
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

                if is_la:
                    # Faithfulness: compare routed segments vs gold answer segments
                    topk_idx = last_info["topk_idx"]  # (B,H,L,K)
                    topk_w = last_info.get("topk_w")
                    gate = last_info.get("gate_eff", last_info.get("gate"))
                    gate_th = 0.15
                    gold = batch["labels"].to(device)
                    for b in range(ids.size(0)):
                        gs, ge = gold[b, 0].item(), gold[b, 1].item()
                        if ge < gs:
                            ge = gs
                        seg_lo = gs // args.segment_size
                        seg_hi = ge // args.segment_size
                        gold_segments.append(list(range(seg_lo, seg_hi + 1)))

                        # Fix Faithfulness: check if Question tokens route to the gold answer segment
                        if global_attn is not None:
                            question_tokens = (global_attn[b] == 1).nonzero(as_tuple=True)[0]
                        else:
                            # Fallback (e.g. first 50 tokens are typically the question)
                            question_tokens = torch.arange(min(50, ids.size(1)), device=device)

                        votes = Counter()
                        active_route_pairs = 0
                        for t in question_tokens:
                            head_idx = torch.arange(topk_idx.size(1), device=device)

                            if gate is not None:
                                gate_bt = gate[b, :, t, 0].float()
                                head_idx = head_idx[gate_bt >= gate_th]
                                if head_idx.numel() == 0:
                                    continue

                            seg_ids = topk_idx[b, head_idx, t, :]
                            if topk_w is not None:
                                w = topk_w[b, head_idx, t, :].float()
                                w_max = w.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
                                conf = w >= (faithfulness_conf_ratio * w_max)
                                active_route_pairs += int(conf.sum().item())
                                seg_ids = seg_ids[conf]
                            else:
                                active_route_pairs += int(seg_ids.numel())
                                seg_ids = seg_ids.reshape(-1)

                            for sid in seg_ids.tolist():
                                votes[int(sid)] += 1

                        if votes:
                            min_votes = max(1, int(max(active_route_pairs, 1) * faithfulness_vote_ratio))
                            sel = sorted([sid for sid, c in votes.items() if c >= min_votes])
                            if not sel:
                                sel = [sid for sid, _ in votes.most_common(1)]
                        else:
                            sel = []

                        active_route_counts.append(active_route_pairs)
                        selected_segment_counts.append(len(sel))
                        selected_segments.append(sel)

    # Metrics
    results: Dict[str, Any] = {"model": args.model, "backbone": args.backbone, "task": args.task}
    if args.task == "qa":
        # Alias-aware metrics — correct for TriviaQA/HotpotQA
        results.update(compute_em_aliases(preds, aliases))
        results.update(compute_f1_aliases(preds, aliases))
        results.update(compute_rouge(preds, refs))
        if is_la and selected_segments and gold_segments:
            results.update(compute_faithfulness(selected_segments, gold_segments))
            results["faithfulness_avg_active_routes"] = float(sum(active_route_counts) / max(len(active_route_counts), 1))
            results["faithfulness_avg_selected_segments"] = float(sum(selected_segment_counts) / max(len(selected_segment_counts), 1))
    else:
        results.update(compute_bleu(preds, refs))
        results.update(compute_comet(preds, refs, srcs))

    # Efficiency
    if sample_ids is None:
        sample_ids = torch.randint(0, 100, (1, min(args.max_length, 512)), device=device)
        sample_mask = torch.ones_like(sample_ids)
    results.update(compute_efficiency(model, sample_ids, attention_mask=sample_mask))

    # Routing (LongAttention only)
    if all_infos:
        # all_infos contains info for all layers of the last batch
        results["routing"] = compute_routing_stats(all_infos)

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
