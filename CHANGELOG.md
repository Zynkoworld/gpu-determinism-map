# Changelog

All results are deterministic and byte-locked: a released version's verdicts are reproducible forever
(the oracles re-check without a GPU).

## [1.0.0] — 2026-08-07
### Added
- First release: `gpu-determinism-map` — measured, byte-locked bit-reproducibility oracles across
  **4 GPUs / 2 NVIDIA architectures** (216 configs each, seed S=64, CUDA 12.8).
- **Ada (sm_89)** — NVIDIA **L4**, **RTX 4090**: valid oracles, **25/216** value-dependent non-reproducible configs each.
- **Ampere (sm_80/86)** — **A100 80GB**, **RTX 3090**: fully deterministic (GREY=0) → documented **negative results**
  (degenerate → intentionally not published as oracles; `verify.py` returns `HIBA` by design).
- **Finding:** TF32 tensor-core dispatch value-dependent non-determinism is **Ada-architecture-specific**.
- Every stack re-checkable **without a GPU** via `verify.py`.
