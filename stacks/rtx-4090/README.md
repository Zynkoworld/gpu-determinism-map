# GPU-determinizmus orakulum — NVIDIA GeForce RTX 4090 / tf32-dispatch bit-reprodukalhatosag

Per-stack, byte-lockolt bit-repro decider. Stack: NVIDIA GeForce RTX 4090 / CUDA 12.8 / driver 580.126.20 / 2.8.0+cu128.

- partition: 25 GREY (non-repro) / 191 repro (216 config, S=64)
- rule_byte_lock (hardver-fuggetlen): ee285a698f4bee7208eba9add7a18a65a6ab20b02e68bb35568c578104ac93e8
- ground_truth_byte_lock (per-stack): d0afe64e06f6d17f6b8837d72d4678ed3d11deb14355337cb6281a8de02937db

## Verifikalas GPU NELKUL
```
cd /srv/cuda/oracles/gpu-4090-determinism && python3 verify.py
```

## Scope
MERT per-stack lookup (FP0 + recall 1.0 az enumeralt gridre), NEM hordozhato formula.
