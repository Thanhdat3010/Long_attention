"""
logging_utils.py — Tracking & JSON Logging
============================================

MetricLogger        : per-step + per-epoch scalars → JSON
RoutingTracker      : per-layer gate, segment, type info
SpecializationTracker: head-wise type usage & persistence
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import torch


class MetricLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.step_logs: List[Dict] = []
        self.epoch_logs: List[Dict] = []

    def log_step(self, metrics: Dict[str, float], step: int):
        self.step_logs.append({"step": step, **metrics})

    def log_epoch(self, metrics: Dict[str, float], epoch: int):
        self.epoch_logs.append({"epoch": epoch, **metrics})

    def save(self, filename="metrics.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump({"step_logs": self.step_logs, "epoch_logs": self.epoch_logs}, f, indent=2)
        return path


class RoutingTracker:
    def __init__(self):
        self.records: List[Dict] = []

    def record(self, layer_infos: List[Dict[str, torch.Tensor]], step: int):
        snap = {"step": step, "layers": []}
        for i, info in enumerate(layer_infos):
            g = info["gate"]
            tm = info["type_mask"]
            entry = {
                "layer": i,
                "gate_mean": float(g.mean()),
                "gate_std": float(g.std()),
                "gate_active_frac": float((g > 0.5).float().mean()),
                "type_dist": tm.mean(dim=(0, 1, 2)).tolist(),
            }
            if "topk_w" in info:
                tw = info["topk_w"]
                entry["routing_entropy"] = float(-(tw * (tw + 1e-8).log()).sum(-1).mean())
            snap["layers"].append(entry)
        self.records.append(snap)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)


class SpecializationTracker:
    def __init__(self):
        self._hist: Dict[tuple, List[List[float]]] = defaultdict(list)

    def record(self, layer_infos: List[Dict[str, torch.Tensor]], step: int):
        for li, info in enumerate(layer_infos):
            per_head = info["type_mask"].mean(dim=(0, 2))  # (H, T)
            for h in range(per_head.shape[0]):
                self._hist[(li, h)].append(per_head[h].tolist())

    def summary(self):
        out = {}
        for (l, h), dists in self._hist.items():
            doms = [int(np.argmax(d)) for d in dists]
            same = sum(1 for i in range(1, len(doms)) if doms[i] == doms[i-1])
            pers = same / max(len(doms)-1, 1)
            out[f"L{l}_H{h}"] = {
                "dominant_type": Counter(doms).most_common(1)[0][0],
                "persistence": round(pers, 4),
                "avg_dist": np.mean(dists, axis=0).tolist(),
            }
        return out

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
