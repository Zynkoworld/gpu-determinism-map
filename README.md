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
```
python3 stacks/rtx-4090/verify.py     # one stack
python3 verify_all.py                 # all four stacks + the positive/negative control pair
```

**What `verify.py` checks, and what it deliberately does not claim.**
1. `rule_byte_lock` — pins the *code* of the portable rule.
2. `ground_truth_byte_lock` — pins the *rows* of the measured table (config + regime). Flip one regime and this
   breaks: that is what protects table integrity.
3. **An independent cross-check:** the portable rule (`oracle_reproducible`, a pure function of `(M, N, K)` that
   **does not read the table**) against the **measured table**. This one can — and does — disagree.
4. The lookup self-consistency count is still printed, but it is **not evidence**: the lookup reads the very table
   it is compared against. It only shows the loader parses the table correctly.

> **An earlier version of this check compared the lookup to the table and reported 216/216 — a tautology.**
> It would have reported a perfect score even if every row were wrong. It has been replaced by (3), and the
> label on (4) says what it is worth.

**Every run also executes a mandatory negative control**: one `regime` is flipped in memory; the ground-truth
byte-lock **must** break and the rule cross-check **must** change. If either does not, the verifier reports
`ERROR` — a measure that cannot say no is not a measure.

**Three verdicts, not two:**
- `OK` (rc 0) — **Ada stacks (L4, RTX 4090)**: the table discriminates (25 GREY of 216), byte-locks match,
  negative control passed.
- `DEGENERATE` (rc 2) — **Ampere stacks (A100, RTX 3090)**: **a negative result, not an error.** The stack is
  fully bit-reproducible on this grid (GREY 0/216), so an oracle here would be a constant "yes" — nothing to
  decide. The measurement is sound; the *lane* is not publishable as a decider. Each ships a `NEGATIVE_RESULT.md`.
- `ERROR` (rc 1) — a byte-lock mismatch, or the negative control failed.

## The table is the oracle — the rule is only a portable approximation
The **measured ground-truth table is the oracle.** The portable rule (derived from RTX 4090 measurements) is a
convenience approximation, and it has a **known, measured error** — stated here rather than hidden:

| stack | rule vs measured table | false positives | false negatives |
|---|---|---|---|
| NVIDIA L4 | 200 / 216 (0.926) | **FP = 8** | FN = 8 |
| NVIDIA RTX 4090 | 200 / 216 (0.926) | **FP = 8** | FN = 8 |
| NVIDIA A100 | 191 / 216 (0.884) | FP = 0 | FN = 25 |
| NVIDIA RTX 3090 | 191 / 216 (0.884) | FP = 0 | FN = 25 |

**`FP = 8` means: on the Ada stacks the rule calls 8 configurations bit-reproducible that the measurement shows
to be value-dependent (GREY).** That is the dangerous direction — a false promise of reproducibility. This is why
the rule is *not* the oracle and must not be used as one: **decide from the table.** A non-zero FP does not fail
`verify.py`; it is reported as a known residual of the approximation, while the table's integrity is enforced by
the byte-lock.

## Scope (honest)
- **Per-stack.** Each oracle is valid for its exact stack (GPU model + CUDA + driver + cuBLAS) and the 216-config
  grid. A different GPU is a different measurement — the map grows one stack at a time.
- The valid oracles are **measured lookups**, not a generalizing predictor: on Ada no simple rule achieves FP 0
  (the boundary is genuinely position-/hardware-dependent, and the measured FP = 8 above shows it), so the honest
  oracle is the measured ground-truth table.
- The Ampere entries are **negative findings**, not oracles.

## License
**Apache-2.0** — free to use, including commercially.

---
Part of **[Zynko](https://zynko.dev)** — deterministic, provable AI.
Zynko doesn't guess — it *establishes truth*. See the full set: **https://zynko.dev/oracles.html**
