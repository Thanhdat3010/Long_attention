# Experiment Run History

This file tracks all training and evaluation runs manually. For each run, we name the output folder with a version suffix (e.g. `_v0`, `_v1`, etc.) to prevent overwriting.

## NMT Runs: CoCoDoc-MT-20k (en-fr)

### 1. Translation Quality

| Version | Split | Attention Type | Hyperparameters | Train Loss | Eval/Test Loss | SacreBLEU | ChrF++ | COMET | Output Directory |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **v0** | Val | `led` | Default: LR=2e-5, BS=2, Epochs=3 | 3.8800 | 3.1563 | 8.4522 | 31.8047 | 0.4317 | `outputs/outputs_led_enfr_v0` |
| | Test | `led` | Default: LR=2e-5, BS=2, Epochs=3 | | 3.3713 | 3.6824 | 40.3282 | 0.3782 | |
| **v1** | Val | `long_attention` | Optimized: LR=2e-5, BS=2, Epochs=3, K=128, null=0.002, div=0.25 | 4.1195 | - | 8.6084 | 34.1482 | 0.4210 | `outputs/outputs_long_attn_enfr_optimized` |
| | Test | `long_attention` | Optimized: LR=2e-5, BS=2, Epochs=3, K=128, null=0.002, div=0.25 | | - | 3.5040 | 40.0668 | 0.3763 | |
| **v3** | Val | `long_attention` | v3 Arch: LR=2e-5, BS=2, Epochs=3, K=64, null=0.002, div=0.1 | 3.9775 | - | 8.7196 | 33.9851 | 0.405 | `outputs/outputs_long_attn_enfr_v3` |
| | Test | `long_attention` | v3 Arch: (Killed - OOM during eval) | | - | - | - | - | |
| **v3.1** | Val | `led` | LED GC Baseline: LR=2e-5, BS=1, accum=16, Epochs=3 | 3.8988 | 3.1793 | 8.0534 | 30.8506 | 0.4396 | `outputs/outputs_led_enfr_gc` |
| | Test | `led` | LED GC Baseline: LR=2e-5, BS=1, accum=16, Epochs=3 | | - | 3.9266 | 39.2422 | 0.3849 | |
| | Val | `long_attention` | v3.1 Arch: LR=2e-5, BS=1, accum=16, Epochs=3, K=64, null=0.001, div=1.0 | 4.8179 | 3.1863 | 8.6482 | 32.4925 | 0.4094 | `outputs/outputs_long_attn_enfr_gc` |
| | Test | `long_attention` | v3.1 Arch: (No test) | | | | | | |
| **v3.2** | Val | `long_attention` | v3.2 Arch: LR=2e-5, BS=1, accum=16, Epochs=3, K=128, null=0.0005, div=0.5 | 4.3288 | 3.1467 | 8.6448 | 33.8814 | 0.4069 | `outputs/outputs_long_attn_enfr_v3_2` |
| | Test | `long_attention` | v3.2 Arch: (No test) | | | | | | |

### 2. Training Resource & Speed

| Version | Attention Type | Runtime (s) | Samples/sec | Steps/sec | Gate (Mean) | Div Loss (Mean) | Train Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v0** | `led` | 4693.0456 | 12.785 | 0.799 | N/A | N/A | 3.8800 |
| **v1** | `long_attention` | 7920.5765 | 7.575 | 0.473 | 0.2401 | N/A | 4.1195 |
| **v3** | `long_attention` | 8103.8701 | 7.404 | 0.463 | 0.266 | 0.9897 | 3.9775 |
| **v3.1 (LED)** | `led` | 6971.1842 | 8.607 | 0.538 | N/A | N/A | 3.8988 |
| **v3.1 (LA)** | `long_attention` | 13429.7582 | 4.468 | 0.279 | 0.2682 | 0.9134 | 4.8179 |
| **v3.2 (LA)** | `long_attention` | 13373.6519 | 4.486 | 0.280 | 0.2689 | 0.9154 | 4.3288 |
