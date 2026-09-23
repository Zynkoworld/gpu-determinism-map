# A100 — NEGATIVE RESULT (degenerate oracle): NO non-determinism

Stack: NVIDIA A100 80GB PCIe (Ampere, sm_80) / CUDA 12.8 / driver 580.159.04 / torch 2.8.0+cu128.

MEASURED (S=64, 216 configs): NEVER=91, ALWAYS=125, **GREY=0**. On the A100 the tf32-tensor-core dispatch is
deterministic for EVERY config (no value-dependent GREY zone) -> the bit-reproducibility oracle is DEGENERATE
(trivially everything is reproducible). `verify.py` -> VERDICT ERROR (non_degenerate=False), INTENTIONALLY:
a degenerate oracle is NOT publishable (there is no negative case).

FINDING (valuable, honest): the value-dependent non-determinism of the tf32 dispatch is ADA-SPECIFIC (4090/L4, sm_89:
25 GREY); on the A100 (Ampere sm_80) it does NOT exist. The oracle's validity is per-arch: discriminating on Ada,
nothing to detect on the A100 (the stack is already bit-reproducible on this grid).
