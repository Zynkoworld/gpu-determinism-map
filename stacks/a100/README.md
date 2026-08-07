# GPU-determinizmus orakulum — NVIDIA A100 80GB PCIe / tf32-dispatch bit-reprodukalhatosag

Per-stack, byte-lockolt bit-repro decider. Stack: NVIDIA A100 80GB PCIe / CUDA 12.8 / driver 580.159.04 / 2.8.0+cu128.

- partition: 0 GREY (non-repro) / 216 repro (216 config, S=64)
- rule_byte_lock (hardver-fuggetlen): ee285a698f4bee7208eba9add7a18a65a6ab20b02e68bb35568c578104ac93e8
- ground_truth_byte_lock (per-stack): a77eb3a523b58944817840cf36ed610b17c6d2ce4fbfef99230cc7dd76865deb

## Verifikalas GPU NELKUL
```
cd /srv/cuda/oracles/gpu-a100-determinism && python3 verify.py
```

## Scope
MERT per-stack lookup (FP0 + recall 1.0 az enumeralt gridre), NEM hordozhato formula.
