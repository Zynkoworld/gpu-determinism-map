# zynko-oracle · `gpu-determinism-map`

**Where does GPU TF32 matmul come out bit-identical every time — and where doesn't it?**
A measured, byte-locked, re-checkable map across GPU architectures — with per-stack oracles you can rerun **without a GPU**.

An **oracle** *deterministically decides* the truth of a case; it doesn't guess. This one decides, for a given
matmul configuration on a **specific GPU stack**, whether the result is **bit-reproducible** (identical every run)
or **non-reproducible** (varies) — a property that matters to anyone who needs reproducible ML training/inference.

## The finding
The value-dependent **non-determinism in TF32 tensor-core dispatch is architecture-specific.**
Measured on 4 GPUs across 2 NVIDIA architectures (216 configs each, seed S=64, CUDA 12.8):

| GPU | Arch | Non-reproducible configs (GREY / 216) | Result |
|---|---|---|---|
| **NVIDIA L4** | Ada (sm_89) | **25** | ✅ valid oracle |
| **NVIDIA RTX 4090** | Ada (sm_89) | **25** | ✅ valid oracle |
| **NVIDIA A100 80GB** | Ampere (sm_80) | **0** | ⚪ negative (fully deterministic) |
| **NVIDIA RTX 3090** | Ampere (sm_86) | **0** | ⚪ negative (fully deterministic) |

**On Ada (sm_89), 25 of 216 configs are value-dependent non-reproducible. On Ampere (sm_80/86), none are —
the stack is already bit-reproducible on this grid.** No external tool decides this; we measured it.

## What each stack folder holds
`stacks/<gpu>/` — the deterministic oracle module, the measured 216-config ground-truth table, a
`BYTE_LOCK_SPEC.json` pinning the full stack (GPU · CUDA · driver · cuBLAS), and a `verify.py`.

## Proof — rerun it yourself (no GPU needed)
Each stack's `verify.py` re-derives the byte-locks and checks the lookup-oracle against its table:
```
python3 stacks/rtx-4090/verify.py
```
- **Ada stacks (L4, RTX 4090):** `VERDICT OK` — a non-degenerate oracle (25 reproducible-vs-not cases both present),
  recall 1.0 / FP 0 by construction on the measured grid, byte-locked.
- **Ampere stacks (A100, RTX 3090):** `VERDICT HIBA` **by design** — the stack is fully deterministic, so a
  bit-reproducibility oracle here would be *degenerate* (trivially all-reproducible, no negative case). We refuse to
  publish a degenerate oracle; instead each ships a `NEGATIVE_RESULT.md` documenting the honest finding. The `HIBA`
  is the guard working, not a bug.

## Scope (honest)
- **Per-stack.** Each oracle is valid for its exact stack (GPU model + CUDA + driver + cuBLAS) and the 216-config
  grid. A different GPU is a different measurement — the map grows one stack at a time.
- The valid oracles are **measured lookups**, not a generalizing predictor: on Ada no simple rule achieves FP 0
  (the boundary is genuinely position-/hardware-dependent), so the honest oracle is the measured ground-truth table.
- The Ampere entries are **negative findings**, not oracles.

## License
**Apache-2.0** — free to use, including commercially.

---
Part of **[Zynko](https://zynko.dev)** — deterministic, provable AI.
Zynko doesn't guess — it *establishes truth*. See the full set: **https://zynko.dev/oracles.html**
