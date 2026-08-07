# GPU-determinizmus orákulum — L4 / tf32-dispatch bit-reprodukálhatóság

Az **első GPU-domain orákulum** (Polaris #11706, János delegálta). Determinista **decider**, ami egy
GEMM/matmul-konfigról (M, N, K; fp32 + `allow_tf32`) eldönti, **bit-reprodukálható-e** a cél-GPU-n — azaz a
tf32-tensor-core **dispatch** (és így a mantissza-levágás, tehát a kimeneti bitek) a konfigból determinált-e
(reprodukálható a spec-ből), vagy az input **értékektől** függ (nem reprodukálható).

## Az artefaktum (per-stack, byte-lockolt)

| fájl | mi |
|---|---|
| `gpu_determinism_oracle.py` | a modul: `oracle_lookup()` (per-stack decider), `ground_truth()` (GPU-adjudikátor), `gate()`, `rule_byte_lock()` |
| `l4_anchor_s64.json` | a **mért diszkrimináló tábla**: 216 config → regime (NEVER/ALWAYS/GREY), S=64 seed, L4-en |
| `BYTE_LOCK_SPEC.json` | a byte-lock spec: teljes stack + partíció + gate-metrikák + mindkét byte-lock |
| `verify.py` | **GPU-NÉLKÜLI** verifikáció: a tábla ↔ byte-lock ↔ lookup konzisztenciája |

## Stack (per-stack érvényesség)

```
GPU=NVIDIA L4 (sm_89) | CUDA runtime 12.8 (torch 2.8.0+cu128) | driver 580.159.04
ground_truth_byte_lock (S=64) = 5070945ab9cc2204090253ec6f029fbd58df615a9dbc2de183c587d2dd59afe6
rule_byte_lock (hardver-független) = ee285a698f4bee72...
```

## Verifikálás — GPU NÉLKÜL (ehhez már nem kell a pod)

```
cd oracles/gpu-l4-determinism && python3 verify.py
```
Újraszámolja mindkét byte-lockot a táblából + ellenőrzi, hogy a lookup-orákulum egyezik a méréssel
(FP0 + recall 1.0 konstrukció szerint), és hogy nem-degenerate.

## Re-anchor — ÚJ stacken (GPU kell)

Más GPU/CUDA/driver = más horgony (a byte-exact matmul-kimenet cuBLAS-belső, **nem hordozható**). Új stack
anchorolása:
```
scp -i ~/.ssh/id_ed25519 -P <port> gpu_determinism_oracle.py root@<ip>:/workspace/
ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip> 'cd /workspace && python3 -c "
import gpu_determinism_oracle as o, json
tab=[{\"config\":list(c),**{k:o.ground_truth(*c,seeds=64)[k] for k in (\"engaged\",\"regime\",\"reproducible\")}} for c in o.probe_grid()]
json.dump({\"stack\":o.stack_fingerprint(),\"seeds\":64,\"table\":tab}, open(\"anchor.json\",\"w\"))"'
```
majd húzd le az `anchor.json`-t és rögzítsd a byte-lockot a `BYTE_LOCK_SPEC.json` mintájára.

## Becsületes scope (a lelet maga)

- A **lookup-orákulum** (mért partíció mint decider) **FP0 + recall 1.0** konstrukció szerint az **enumerált
  216-config-térre**, ehhez a stackhez. Publikálható mint **per-stack L4 bit-repro decider**.
- Egy **hordozható formula NEM létezik FP0-val**: a 4090-derivált min-dim szabály L4-en recall 0.68 / FP=8, és
  egyetlen egyszerű min/max/K szabály sem ér FP0-t (a legjobb 0.48 recall). A dispatch-determinizmus határa
  **hardver-specifikus és pozíció-függő** (a K kontrakciós dim máshogy viselkedik, mint M/N; `[3,3,3]` L4-en
  GREY, 4090-en ALWAYS). **Ez maga is oracle-validitás lelet:** a bit-reprodukálhatóság per-stack, nem
  általánosítható egyszerű szabállyal.
