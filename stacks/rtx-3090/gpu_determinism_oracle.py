#!/usr/bin/env python3
"""gpu_determinism_oracle — az ELSO GPU-domain ORAKULUM (Polaris #11706 / Janos delegalt irany).

CEL (Janos def: orakulum = determinista eldonto): egy determinista DECIDER, ami egy GEMM/matmul-konfigrol
(M, N, K; fp32 + a tf32-allow_tf32 mint releváns dispatch-parameter) eldonti, hogy BIT-REPRODUKALHATO-e a
cel-GPU-n -- azaz a tf32-tensor-core DISPATCH-ot (es igy a mantissza-levagast, tehat a kimeneti biteket) a
KONFIG hatarozza-e meg (reprodukalhato a spec-bol), VAGY az input ERTEKEKTOL fugg (nem reprodukalhato).

NEM cirkularis: az ORAKULUM egy szabaly a (M,N,K)-n; a GROUND-TRUTH a TENYLEGES GPU-viselkedes (a tf32/fp32
divergencia stabil-e sok input-seeden). A ketto fuggetlen.

A DETERMINIZMUS-FELULET (MERT, RTX 4090, 2026-07-30, tf32_dispatch_map, H29-1):
  - min(M,N,K) == 1  -> SOHA tensor-core (stabil vektor-ut)      => REPRODUKALHATO
  - min(M,N,K) >= 3  -> MINDIG tensor-core (stabil)              => REPRODUKALHATO
  - min(M,N,K) == 2  -> ERTEK-FUGGO cuBLAS-heurisztika (SZURKE)  => tipikusan NEM reprodukalhato,
        KIVEVE a "nagy/negyzetes M/N sarok": a 11 mert pont pontosan illik az alabbi szabalyra:
        DETERMINISTA (min==2) <=> max(M,N) >= 8  VAGY  (M >= 3 ES N >= 3); egyebkent SZURKE.
        (Mert: det sarok = (3,3,2),(8,2,2),(8,8,2); szurke = (2,2,2),(2,2,3),(3,2,2),(4,2,2),(2,2,8).)

FIGYELEM: a min==2 szabaly 11 mert pontbol szarmazo HIPOTEZIS -> a GATE (recall/FP a suru probe-gridon)
IGAZOLJA vagy FINOMITJA. Publikusra CSAK FP0 utan (a go-orakulumok fegyelme).
"""
from __future__ import annotations

import hashlib
import json


# ---------------------------------------------------------------------------------------------------------
# 1) AZ ORAKULUM (determinista decider) -- tiszta Python, GPU NELKUL fut.
# ---------------------------------------------------------------------------------------------------------
def oracle_reproducible(M: int, N: int, K: int) -> bool:
    """True, ha a (M,N,K) fp32-matmul tf32-dispatch-e a KONFIGBOL determinalt (bit-reprodukalhato a spec-bol);
    False, ha ertek-fuggo (SZURKE, nem reprodukalhato). A tf32-allow a releváns dispatch-parameter."""
    mn = min(M, N, K)
    if mn == 1:
        return True                      # NEVER tensor-core -> stabil
    if mn >= 3:
        return True                      # ALWAYS tensor-core -> stabil
    # mn == 2: SZURKE, kiveve a nagy/negyzetes M/N sarok
    if max(M, N) >= 8:
        return True
    if M >= 3 and N >= 3:
        return True
    return False                         # SZURKE -> NEM reprodukalhato


def oracle_verdict(M: int, N: int, K: int) -> str:
    return "REPRODUCIBLE" if oracle_reproducible(M, N, K) else "NON_REPRODUCIBLE"


# --- PER-STACK LOOKUP-ORAKULUM (L4 anchor, #11715): a MERT ground-truth mint decider ------------------
# A 4090-derivalt szabaly (fent) L4-en NEM ad FP0-t (a hatar hardver-specifikus, pozicio-fuggo: egy egyszeru
# min/max/K szabaly max 0.48 recall FP0-nal). Ezert az L4 orakuluma a MERT partitio (216 config, S=64):
# FP0 + recall 1.0 KONSTRUKCIO SZERINT az enumeralt config-terre, a stack-hez byte-lockolva. Ez a per-stack
# becsuletes forma (nem hordozhato formula) -- lásd measurements/gpu-l4-determinism-anchor-s64.json.
_ANCHOR_CACHE: dict = {}


def load_anchor(path: str) -> dict:
    if path not in _ANCHOR_CACHE:
        with open(path) as f:
            d = json.load(f)
        _ANCHOR_CACHE[path] = {tuple(t["config"]): (t["regime"] != "GREY") for t in d["table"]}
    return _ANCHOR_CACHE[path]


def oracle_lookup(M: int, N: int, K: int, anchor_path: str) -> str:
    """PER-STACK decider: a MERT anchor alapjan REPRODUCIBLE/NON_REPRODUCIBLE. KeyError, ha a config nincs a
    mert gridben (a lookup-orakulum az ENUMERALT config-terre ervenyes; kivul meres kell)."""
    repro = load_anchor(anchor_path)[(M, N, K)]
    return "REPRODUCIBLE" if repro else "NON_REPRODUCIBLE"


# ---------------------------------------------------------------------------------------------------------
# 2) A PROBE-GRID (diszkriminalo: determinista ES nemdeterminista konfigok; nem-degenerate = van negativ).
# ---------------------------------------------------------------------------------------------------------
def probe_grid() -> list[tuple[int, int, int]]:
    """Suru grid a min-dim hatar korul: min==1/2/3, kulonbozo max-M/N/K pozicioval (a nagy-K vs nagy-M/N
    megkulonboztetesre). Determinista + szurke eseteket IS tartalmaz (nem-degenerate)."""
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
# 3) A GROUND-TRUTH ADJUDIKATOR -- a TENYLEGES GPU-viselkedes (fuggetlen, nem-cirkularis). GPU-t igenyel.
#    "bit-reprodukalhato" = a tf32/fp32 divergencia (TC engaged) STABIL minden input-seeden (0/S vagy S/S);
#    ERTEK-FUGGO (mixed) = NEM reprodukalhato. (Polaris N-szeri futas + bit-azonossag: itt a dispatch-
#    determinizmus a merteke, seedek kozott -- ez a nem-degenerate diszkriminator.)
# ---------------------------------------------------------------------------------------------------------
def stack_fingerprint() -> dict:
    """A TELJES stack rogzitese (Polaris #11712): az orakulum ervenyessege PER-STACK. A ground-truth byte-lock
    ehhez kotodik -- mas GPU/CUDA/cuBLAS/driver = mas horgony (a byte-exact matmul-kimenet cuBLAS-belso, nem
    hordozhato, ld. tf32-accum-order-probe: 'portabilis byte-pontos D0 lehetetlen'). GPU nelkul reszleges."""
    fp = {"gpu": None, "sm_arch": None, "cuda": None, "cudnn": None, "driver": None, "torch": None}
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["cuda"] = getattr(torch.version, "cuda", None)          # a torch-hoz linkelt CUDA (pl. '12.8')
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
    """A tenyleges GPU-igazsag: TC engaged-e (tf32 != fp32) sok seeden. Visszaad: {'engaged': c, 'seeds': S,
    'reproducible': bool, 'regime': ...}. reproducible <=> c==0 vagy c==S (stabil). GPU nelkul None."""
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
# 4) GATE: recall + FP0 + byte-lock, a probe-gridon (oracle vs ground-truth).
# ---------------------------------------------------------------------------------------------------------
def rule_byte_lock() -> str:
    """A SZABALY (4090-derivalt oracle_reproducible) verdikt-vektoranak byte-lockja -- HARDVER-FUGGETLEN
    (csak a config->oracle-verdikt, NINCS benne ground-truth). Ez ugyanaz minden GPU-n. Determinista."""
    canon = json.dumps(sorted([[list(c), oracle_verdict(*c)] for c in probe_grid()]),
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _bytelock(rows: list[dict]) -> str:
    """A verdikt-vektor byte-lockja: sha256 a (config -> oracle,gt) rendezett kanonjan. Determinista, re-runnable."""
    canon = json.dumps(sorted([[r["config"], r["oracle"], r.get("gt_regime")] for r in rows]),
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def gate(seeds: int = 16) -> dict:
    """Lefuttatja az orakulumot ES (ha van GPU) a ground-truth-t a probe-gridon; meri a recall-t es az FP-t.
    recall = a VALODI NON_REPRODUCIBLE-ok hany %-at fogja meg az orakulum; FP = determinista konfig, amit az
    orakulum tevesen NON_REPRODUCIBLE-nak jelol (FP0 kell publikusra). GPU nelkul: csak az oracle + probe-bontas."""
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
           "non_degenerate": 0 < oracle_nonrepro < n,       # van pozitiv ES negativ oracle-verdikt
           "stack": stack,                                   # PER-STACK ervenyesseg (Polaris #11712)
           "oracle_rule_byte_lock": rule_byte_lock(), "oracle_gt_byte_lock": _bytelock(rows), "rows": rows}
    have_gt = all(r["gt_reproducible"] is not None for r in rows)
    if have_gt:
        # a GROUND-TRUTH byte-lock a STACK-hez kotott (GPU+CUDA+cuBLAS+driver) -- ez a per-stack horgony
        gt_canon = json.dumps({"stack": stack,
                               "gt": sorted([[r["config"], r["gt_regime"]] for r in rows])},
                              sort_keys=True, separators=(",", ":"))
        out["ground_truth_byte_lock"] = hashlib.sha256(gt_canon.encode()).hexdigest()
        # ground-truth: reproducible True/False; oracle: REPRODUCIBLE/NON_REPRODUCIBLE
        gt_nonrepro = [r for r in rows if r["gt_reproducible"] is False]
        gt_repro = [r for r in rows if r["gt_reproducible"] is True]
        tp = sum(1 for r in gt_nonrepro if r["oracle"] == "NON_REPRODUCIBLE")
        fp = sum(1 for r in gt_repro if r["oracle"] == "NON_REPRODUCIBLE")   # determinista, tevesen jelolve
        fn = sum(1 for r in gt_nonrepro if r["oracle"] == "REPRODUCIBLE")
        out["gt_measured"] = True
        out["gt_non_reproducible"] = len(gt_nonrepro)
        out["recall"] = (tp / len(gt_nonrepro)) if gt_nonrepro else None
        out["false_positives"] = fp
        out["fp0"] = (fp == 0)
        out["false_negatives"] = fn
        out["ground_truth_non_degenerate"] = 0 < len(gt_nonrepro) < n        # a VALODI adat is diszkriminal
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
              f"(NON_REPRO valodi={res['gt_non_reproducible']}/{res['probe_count']}) byte-lock={res['oracle_rule_byte_lock'][:16]}")
    else:
        print(f"\n[GPU nelkul] oracle-bontas: NON_REPRO={res['oracle_non_reproducible']} "
              f"REPRO={res['oracle_reproducible']} non_degenerate={res['non_degenerate']} "
              f"byte-lock(oracle)={res['oracle_rule_byte_lock'][:16]}")
