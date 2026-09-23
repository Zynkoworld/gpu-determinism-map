# GPU determinism oracle — NVIDIA A100 80GB PCIe / tf32-dispatch bit-reproducibility

Per-stack, byte-locked bit-repro decider. Stack: NVIDIA A100 80GB PCIe / CUDA 12.8 / driver 580.159.04 / 2.8.0+cu128.

- partition: 0 GREY (non-repro) / 216 repro (216 config, S=64)
- rule_byte_lock (hardware-independent): ee285a698f4bee7208eba9add7a18a65a6ab20b02e68bb35568c578104ac93e8
- ground_truth_byte_lock (per-stack): a77eb3a523b58944817840cf36ed610b17c6d2ce4fbfef99230cc7dd76865deb

## Verification WITHOUT a GPU
```
cd /srv/cuda/oracles/gpu-a100-determinism && python3 verify.py
```

## Scope
MEASURED per-stack lookup (FP0 + recall 1.0 on the enumerated grid), NOT a portable formula.
