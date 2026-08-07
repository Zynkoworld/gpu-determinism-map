# GPU-determinizmus orakulum — NVIDIA GeForce RTX 3090 / tf32-dispatch bit-reprodukalhatosag

Per-stack, byte-lockolt bit-repro decider. Stack: NVIDIA GeForce RTX 3090 / CUDA 12.8 / driver 580.126.20 / 2.8.0+cu128.

- partition: 0 GREY (non-repro) / 216 repro (216 config, S=64)
- rule_byte_lock (hardver-fuggetlen): ee285a698f4bee7208eba9add7a18a65a6ab20b02e68bb35568c578104ac93e8
- ground_truth_byte_lock (per-stack): 5e98280c1bf64b4f82623e02268d0f69d92d26c00dc22fe354b00c3db0a8f8e2

## Verifikalas GPU NELKUL
```
cd /srv/cuda/oracles/gpu-3090-determinism && python3 verify.py
```

## Scope
MERT per-stack lookup (FP0 + recall 1.0 az enumeralt gridre), NEM hordozhato formula.
