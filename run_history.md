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

### 2. Training Resource & Speed

| Version | Attention Type | Runtime (s) | Samples/sec | Steps/sec | Gate (Mean) | Train Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v0** | `led` | 4693.0456 | 12.785 | 0.799 | N/A | 3.8800 |
| **v1** | `long_attention` | 7920.5765 | 7.575 | 0.473 | 0.2401 | 4.1195 |
