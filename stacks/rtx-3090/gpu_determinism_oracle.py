#!/usr/bin/env python3
"""gpu_determinism_oracle — the FIRST GPU-domain ORACLE (Polaris #11706 / direction delegated by Janos).

GOAL (Janos' def: oracle = deterministic decider): a deterministic DECIDER that, for a GEMM/matmul config
(M, N, K; fp32 + tf32-allow_tf32 as the relevant dispatch parameter), decides whether it is BIT-REPRODUCIBLE on the
target GPU -- i.e. whether the tf32-tensor-core DISPATCH (and thus the mantissa truncation, hence the output bits) is
determined by the CONFIG (reproducible from the spec), OR depends on the input VALUES (not reproducible).

NOT circular: the ORACLE is a rule on (M,N,K); the GROUND TRUTH is the ACTUAL GPU behaviour (whether the tf32/fp32
divergence is stable across many input seeds). The two are independent.

THE DETERMINISM SURFACE (MEASURED, RTX 4090, 2026-07-30, tf32_dispatch_map, H29-1):
  - min(M,N,K) == 1  -> NEVER tensor-core (stable vector path)      => REPRODUCIBLE
  - min(M,N,K) >= 3  -> ALWAYS tensor-core (stable)                 => REPRODUCIBLE
  - min(M,N,K) == 2  -> VALUE-DEPENDENT cuBLAS heuristic (GREY)     => typically NOT reproducible,
        EXCEPT the "large/square M/N corner": the 11 measured points fit the rule below exactly:
        DETERMINISTIC (min==2) <=> max(M,N) >= 8  OR  (M >= 3 AND N >= 3); otherwise GREY.
        (Measured: det corner = (3,3,2),(8,2,2),(8,8,2); grey = (2,2,2),(2,2,3),(3,2,2),(4,2,2),(2,2,8).)

CAUTION: the min==2 rule is a HYPOTHESIS derived from 11 measured points -> the GATE (recall/FP on the dense
probe grid) CONFIRMS or REFINES it. Public ONLY after FP0 (the discipline of the go oracles).
"""
from __future__ import annotations

import hashlib
import json


# ---------------------------------------------------------------------------------------------------------
# 1) THE ORACLE (deterministic decider) -- pure Python, runs WITHOUT a GPU.
# ---------------------------------------------------------------------------------------------------------
def oracle_reproducible(M: int, N: int, K: int) -> bool:
    """True if the tf32 dispatch of the (M,N,K) fp32 matmul is determined by the CONFIG (bit-reproducible from the spec);
    False if it is value-dependent (GREY, not reproducible). tf32-allow is the relevant dispatch parameter."""
    mn = min(M, N, K)
    if mn == 1:
        return True                      # NEVER tensor-core -> stable
    if mn >= 3:
        return True                      # ALWAYS tensor-core -> stable
    # mn == 2: GREY, except the large/square M/N corner
    if max(M, N) >= 8:
        return True
    if M >= 3 and N >= 3:
        return True
    return False                         # GREY -> NOT reproducible


def oracle_verdict(M: int, N: int, K: int) -> str:
    return "REPRODUCIBLE" if oracle_reproducible(M, N, K) else "NON_REPRODUCIBLE"


# --- PER-STACK LOOKUP ORACLE (L4 anchor, #11715): the MEASURED ground truth as the decider ----------------
# The 4090-derived rule (above) does NOT give FP0 on L4 (the boundary is hardware-specific and position-dependent:
# a simple min/max/K rule reaches at most 0.48 recall at FP0). Hence the L4 oracle is the MEASURED partition (216
# configs, S=64): FP0 + recall 1.0 BY CONSTRUCTION on the enumerated config space, byte-locked to the stack. This is
# the honest per-stack form (not a portable formula) -- see measurements/gpu-l4-determinism-anchor-s64.json.
_ANCHOR_CACHE: dict = {}


def load_anchor(path: str) -> dict:
    if path not in _ANCHOR_CACHE:
        with open(path) as f:
            d = json.load(f)
        _ANCHOR_CACHE[path] = {tuple(t["config"]): (t["regime"] != "GREY") for t in d["table"]}
    return _ANCHOR_CACHE[path]


def oracle_lookup(M: int, N: int, K: int, anchor_path: str) -> str:
    """PER-STACK decider: REPRODUCIBLE/NON_REPRODUCIBLE based on the MEASURED anchor. KeyError if the config is not
    in the measured grid (the lookup oracle is valid on the ENUMERATED config space; outside it a measurement is needed)."""
    repro = load_anchor(anchor_path)[(M, N, K)]
    return "REPRODUCIBLE" if repro else "NON_REPRODUCIBLE"


# ---------------------------------------------------------------------------------------------------------
# 2) THE PROBE GRID (discriminating: deterministic AND non-deterministic configs; non-degenerate = has negatives).
# ---------------------------------------------------------------------------------------------------------
def probe_grid() -> list[tuple[int, int, int]]:
    """Dense grid around the min-dim boundary: min==1/2/3 with different max-M/N/K positions (to distinguish
    large-K from large-M/N). Contains BOTH deterministic and grey cases (non-degenerate)."""
    dims = [1, 2, 3, 4, 8, 16]
    seen, grid = set(), []
    for M in dims:
        for N in dims:
            for K in dims:
                key = (M, N, K)
                if key in seen:
                    continue
                seen.add(key)
                grid.append(key)
    return grid


# ---------------------------------------------------------------------------------------------------------
# 3) THE GROUND-TRUTH ADJUDICATOR -- the ACTUAL GPU behaviour (independent, non-circular). Requires a GPU.
#    "bit-reproducible" = the tf32/fp32 divergence (TC engaged) is STABLE across all input seeds (0/S or S/S);
#    VALUE-DEPENDENT (mixed) = NOT reproducible. (Polaris' N-fold run + bit-identity: here the measure is dispatch
#    determinism across seeds -- this is the non-degenerate discriminator.)
# ---------------------------------------------------------------------------------------------------------
def stack_fingerprint() -> dict:
    """Pins the FULL stack (Polaris #11712): the oracle's validity is PER-STACK. The ground-truth byte-lock is tied
    to it -- a different GPU/CUDA/cuBLAS/driver = a different anchor (the byte-exact matmul output is cuBLAS-internal,
    not portable, see tf32-accum-order-probe: 'a portable byte-exact D0 is impossible'). Partial without a GPU."""
    fp = {"gpu": None, "sm_arch": None, "cuda": None, "cudnn": None, "driver": None, "torch": None}
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["cuda"] = getattr(torch.version, "cuda", None)          # the CUDA linked into torch (e.g. '12.8')
        if torch.cuda.is_available():
            fp["gpu"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            fp["sm_arch"] = f"sm_{cc[0]}{cc[1]}"
            try:
                fp["driver"] = getattr(torch.version, "cuda", None) and torch._C._cuda_getDriverVersion()
            except Exception:
                pass
    except Exception:
        pass
    return fp


def ground_truth(M: int, N: int, K: int, seeds: int = 16):
    """The actual GPU truth: whether TC is engaged (tf32 != fp32) across many seeds. Returns: {'engaged': c, 'seeds': S,
    'reproducible': bool, 'regime': ...}. reproducible <=> c==0 or c==S (stable). None without a GPU."""
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    dev = "cuda"

    def engaged(seed: int) -> bool:
        g = torch.Generator(device=dev).manual_seed(seed)
        A = torch.randn(M, K, device=dev, generator=g)
        B = torch.randn(K, N, device=dev, generator=g)
        torch.backends.cuda.matmul.allow_tf32 = True;  a1 = (A @ B).clone()
        torch.backends.cuda.matmul.allow_tf32 = False; a0 = (A @ B).clone()
        return not torch.equal(a1, a0)

    c = sum(engaged(s) for s in range(seeds))
    repro = (c == 0 or c == seeds)
    regime = "NEVER" if c == 0 else ("ALWAYS" if c == seeds else "GREY")
    return {"engaged": c, "seeds": seeds, "reproducible": repro, "regime": regime}


# ---------------------------------------------------------------------------------------------------------
# 4) GATE: recall + FP0 + byte-lock on the probe grid (oracle vs ground truth).
# ---------------------------------------------------------------------------------------------------------
def rule_byte_lock() -> str:
    """Byte-lock of the verdict vector of the RULE (4090-derived oracle_reproducible) -- HARDWARE-INDEPENDENT
    (only config->oracle verdict, NO ground truth in it). It is the same on every GPU. Deterministic."""
    canon = json.dumps(sorted([[list(c), oracle_verdict(*c)] for c in probe_grid()]),
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _bytelock(rows: list[dict]) -> str:
    """Byte-lock of the verdict vector: sha256 over the sorted canon of (config -> oracle, gt). Deterministic, re-runnable."""
    canon = json.dumps(sorted([[r["config"], r["oracle"], r.get("gt_regime")] for r in rows]),
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def gate(seeds: int = 16) -> dict:
    """Runs the oracle AND (if a GPU is present) the ground truth on the probe grid; measures recall and FP.
    recall = what % of the REAL NON_REPRODUCIBLE configs the oracle catches; FP = a deterministic config that the
    oracle wrongly labels NON_REPRODUCIBLE (FP0 required for publication). Without a GPU: only the oracle + probe breakdown."""
    grid = probe_grid()
    stack = stack_fingerprint()
    rows = []
    for (M, N, K) in grid:
        orc = oracle_verdict(M, N, K)
        gt = ground_truth(M, N, K, seeds=seeds)
        rows.append({"config": [M, N, K], "oracle": orc,
                     "gt_regime": (gt["regime"] if gt else None),
                     "gt_reproducible": (gt["reproducible"] if gt else None)})
    n = len(rows)
    oracle_nonrepro = sum(1 for r in rows if r["oracle"] == "NON_REPRODUCIBLE")
    out = {"probe_count": n, "oracle_non_reproducible": oracle_nonrepro,
           "oracle_reproducible": n - oracle_nonrepro,
           "non_degenerate": 0 < oracle_nonrepro < n,       # there is a positive AND a negative oracle verdict
           "stack": stack,                                   # PER-STACK validity (Polaris #11712)
           "oracle_rule_byte_lock": rule_byte_lock(), "oracle_gt_byte_lock": _bytelock(rows), "rows": rows}
    have_gt = all(r["gt_reproducible"] is not None for r in rows)
    if have_gt:
        # the GROUND-TRUTH byte-lock is tied to the STACK (GPU+CUDA+cuBLAS+driver) -- this is the per-stack anchor
        gt_canon = json.dumps({"stack": stack,
                               "gt": sorted([[r["config"], r["gt_regime"]] for r in rows])},
                              sort_keys=True, separators=(",", ":"))
        out["ground_truth_byte_lock"] = hashlib.sha256(gt_canon.encode()).hexdigest()
        # ground-truth: reproducible True/False; oracle: REPRODUCIBLE/NON_REPRODUCIBLE
        gt_nonrepro = [r for r in rows if r["gt_reproducible"] is False]
        gt_repro = [r for r in rows if r["gt_reproducible"] is True]
        tp = sum(1 for r in gt_nonrepro if r["oracle"] == "NON_REPRODUCIBLE")
        fp = sum(1 for r in gt_repro if r["oracle"] == "NON_REPRODUCIBLE")   # deterministic, wrongly labelled
        fn = sum(1 for r in gt_nonrepro if r["oracle"] == "REPRODUCIBLE")
        out["gt_measured"] = True
        out["gt_non_reproducible"] = len(gt_nonrepro)
        out["recall"] = (tp / len(gt_nonrepro)) if gt_nonrepro else None
        out["false_positives"] = fp
        out["fp0"] = (fp == 0)
        out["false_negatives"] = fn
        out["ground_truth_non_degenerate"] = 0 < len(gt_nonrepro) < n        # the REAL data discriminates too
    else:
        out["gt_measured"] = False
    return out


if __name__ == "__main__":
    import sys
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    res = gate(seeds=seeds)
    slim = {k: v for k, v in res.items() if k != "rows"}
    print(json.dumps(slim, indent=1))
    if res.get("gt_measured"):
        print(f"\nRECALL={res['recall']} FP={res['false_positives']} FP0={res['fp0']} "
              f"(NON_REPRO real={res['gt_non_reproducible']}/{res['probe_count']}) byte-lock={res['oracle_rule_byte_lock'][:16]}")
    else:
        print(f"\n[no GPU] oracle breakdown: NON_REPRO={res['oracle_non_reproducible']} "
              f"REPRO={res['oracle_reproducible']} non_degenerate={res['non_degenerate']} "
              f"byte-lock(oracle)={res['oracle_rule_byte_lock'][:16]}")
