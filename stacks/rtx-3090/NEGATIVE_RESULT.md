# RTX 3090 — NEGATIVE RESULT (degenerate oracle): NO non-determinism

Stack: NVIDIA GeForce RTX 3090 (Ampere, sm_86) / CUDA 12.8 / driver 580.126.20 / torch 2.8.0+cu128.

MEASURED (S=64, 216 configs): GREY=0 (NEVER + ALWAYS, no GREY) -> the tf32-tensor-core dispatch is deterministic for
EVERY config -> the bit-reproducibility oracle is DEGENERATE. verify.py -> VERDICT ERROR (non_degenerate=False),
INTENTIONALLY: a degenerate oracle is NOT publishable.

FINDING (confirms the A100): the value-dependent non-determinism of the tf32 dispatch is ADA-SPECIFIC. On two Ampere
GPUs TOO (A100 sm_80 AND 3090 sm_86) GREY=0 -> the phenomenon is absent on Ampere. Ada (4090/L4 sm_89) = 25 GREY.
