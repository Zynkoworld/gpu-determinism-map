# GPU determinism oracle — L4 / tf32-dispatch bit-reproducibility

The **first GPU-domain oracle** (Polaris #11706, delegated by János). A deterministic **decider** that, for a
GEMM/matmul config (M, N, K; fp32 + `allow_tf32`), decides **whether it is bit-reproducible** on the target GPU — i.e.
whether the tf32-tensor-core **dispatch** (and thus the mantissa truncation, hence the output bits) is determined by
the config (reproducible from the spec), or depends on the input **values** (not reproducible).

## The artefact (per-stack, byte-locked)

| file | what |
|---|---|
| `gpu_determinism_oracle.py` | the module: `oracle_lookup()` (per-stack decider), `ground_truth()` (GPU adjudicator), `gate()`, `rule_byte_lock()` |
| `l4_anchor_s64.json` | the **measured discriminating table**: 216 configs → regime (NEVER/ALWAYS/GREY), S=64 seeds, on L4 |
| `BYTE_LOCK_SPEC.json` | the byte-lock spec: full stack + partition + gate metrics + both byte-locks |
| `verify.py` | **GPU-FREE** verification: consistency of table ↔ byte-lock ↔ lookup |

## Stack (per-stack validity)

```
GPU=NVIDIA L4 (sm_89) | CUDA runtime 12.8 (torch 2.8.0+cu128) | driver 580.159.04
ground_truth_byte_lock (S=64) = 5070945ab9cc2204090253ec6f029fbd58df615a9dbc2de183c587d2dd59afe6
rule_byte_lock (hardware-independent) = ee285a698f4bee72...
```

## Verification — WITHOUT a GPU (the pod is no longer needed for this)

```
cd oracles/gpu-l4-determinism && python3 verify.py
```
Recomputes both byte-locks from the table + checks that the lookup oracle matches the measurement
(FP0 + recall 1.0 by construction), and that it is non-degenerate.

## Re-anchor — on a NEW stack (GPU required)

A different GPU/CUDA/driver = a different anchor (the byte-exact matmul output is cuBLAS-internal, **not portable**).
Anchoring a new stack:
```
scp -i ~/.ssh/id_ed25519 -P <port> gpu_determinism_oracle.py root@<ip>:/workspace/
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip> 'cd /workspace && python3 -c "
import gpu_determinism_oracle as o, json
tab=[{\"config\":list(c),**{k:o.ground_truth(*c,seeds=64)[k] for k in (\"engaged\",\"regime\",\"reproducible\")}} for c in o.probe_grid()]
json.dump({\"stack\":o.stack_fingerprint(),\"seeds\":64,\"table\":tab}, open(\"anchor.json\",\"w\"))"'
```
then pull down the `anchor.json` and pin the byte-lock following the pattern of `BYTE_LOCK_SPEC.json`.

## Honest scope (the finding itself)

- The **lookup oracle** (the measured partition as the decider) is **FP0 + recall 1.0** by construction on the
  **enumerated 216-config space**, for this stack. Publishable as a **per-stack L4 bit-repro decider**.
- A **portable formula does NOT exist with FP0**: the 4090-derived min-dim rule gives recall 0.68 / FP=8 on L4, and
  no single simple min/max/K rule reaches FP0 (the best is 0.48 recall). The boundary of dispatch determinism is
  **hardware-specific and position-dependent** (the K contraction dim behaves differently from M/N; `[3,3,3]` is
  GREY on L4, ALWAYS on 4090). **This is itself an oracle-validity finding:** bit-reproducibility is per-stack, not
  generalisable by a simple rule.
