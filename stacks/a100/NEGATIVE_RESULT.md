# A100 — NEGATIV EREDMENY (degenerate oracle): NINCS nemdeterminizmus

Stack: NVIDIA A100 80GB PCIe (Ampere, sm_80) / CUDA 12.8 / driver 580.159.04 / torch 2.8.0+cu128.

MERT (S=64, 216 config): NEVER=91, ALWAYS=125, **GREY=0**. Az A100-on a tf32-tensor-core dispatch MINDEN
konfigra determinista (nincs ertek-fuggo SZURKE zona) -> a bit-reprodukalhatosag-orakulum DEGENERATE
(trivialisan minden reprodukalhato). `verify.py` -> VERDICT HIBA (non_degenerate=False), SZANDEKOSAN:
degenerate oracle NEM publikalhato (nincs negativ eset).

LELET (ertekes, becsuletes): a tf32-dispatch ertek-fuggo nemdeterminizmusa ADA-SPECIFIKUS (4090/L4, sm_89:
25 GREY), az A100-on (Ampere sm_80) NEM letezik. Az orakulum ervenyessege per-arch: Ada-n diszkrimalo,
A100-on nincs mit detektalni (a stack mar eleve bit-reprodukalhato ezen a gridon).
