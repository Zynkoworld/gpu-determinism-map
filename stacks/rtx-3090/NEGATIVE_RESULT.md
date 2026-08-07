# RTX 3090 — NEGATIV EREDMENY (degenerate oracle): NINCS nemdeterminizmus

Stack: NVIDIA GeForce RTX 3090 (Ampere, sm_86) / CUDA 12.8 / driver 580.126.20 / torch 2.8.0+cu128.

MERT (S=64, 216 config): GREY=0 (NEVER + ALWAYS, nincs SZURKE) -> a tf32-tensor-core dispatch MINDEN konfigra
determinista -> a bit-reprodukalhatosag-orakulum DEGENERATE. verify.py -> VERDICT HIBA (non_degenerate=False),
SZANDEKOSAN: degenerate oracle NEM publikalhato.

LELET (megerositi az A100-at): a tf32-dispatch ertek-fuggo nemdeterminizmusa ADA-SPECIFIKUS. Ket Ampere GPU-n
IS (A100 sm_80 ES 3090 sm_86) GREY=0 -> Ampere-en nincs a jelenseg. Ada (4090/L4 sm_89) = 25 GREY.
