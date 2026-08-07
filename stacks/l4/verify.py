#!/usr/bin/env python3
"""verify.py — a per-stack L4 bit-reprodukalhatosag-orakulum GPU-NELKULI verifikacioja (#11724).

A lookup-orakulum verifikalasa NEM igenyel GPU-t: csak a mert tabla + a byte-lock KONZISZTENCIAJAT ellenorzi.
Ujraszamolja mindket byte-lockot a tablabol es a spec-bol, es osszeveti a rogzitett ertekekkel.
Futtatas:  cd oracles/gpu-l4-determinism && python3 verify.py
"""
import json
import hashlib
import sys

sys.path.insert(0, ".")
import gpu_determinism_oracle as o

d = json.load(open("l4_anchor_s64.json"))
spec = json.load(open("BYTE_LOCK_SPEC.json"))
tab = d["table"]

# 1) rule_byte_lock (hardver-fuggetlen) -- ujraszamolva a szabalybol
rule_bl = o.rule_byte_lock()

# 2) ground_truth_byte_lock (per-stack) -- ujraszamolva a tablabol + a HASHELT stackbol
stack = spec["stack_HASHED"]
canon = json.dumps({"stack": stack, "gt": sorted([[t["config"], t["regime"]] for t in tab])},
                   sort_keys=True, separators=(",", ":"))
gt_bl = hashlib.sha256(canon.encode()).hexdigest()

exp_rule = spec["byte_locks"]["rule_byte_lock"]
exp_gt = spec["byte_locks"]["ground_truth_byte_lock"]

# 3) a lookup-orakulum konzisztens a tablaval (FP0 + recall 1.0 by construction)
mismatch = 0
for t in tab:
    M, N, K = t["config"]
    want = "REPRODUCIBLE" if t["regime"] != "GREY" else "NON_REPRODUCIBLE"
    got = o.oracle_lookup(M, N, K, "l4_anchor_s64.json")
    if got != want:
        mismatch += 1

grey = sum(1 for t in tab if t["regime"] == "GREY")
ok_rule = (rule_bl == exp_rule)
ok_gt = (gt_bl == exp_gt)
ok_lookup = (mismatch == 0)
non_degenerate = 0 < grey < len(tab)

print(f"rule_byte_lock:         {rule_bl[:24]}  {'OK' if ok_rule else 'MISMATCH!'}")
print(f"ground_truth_byte_lock: {gt_bl[:24]}  {'OK' if ok_gt else 'MISMATCH!'} (per-stack: {stack.get('gpu')} / CUDA {stack.get('cuda')} / driver {stack.get('driver')})")
print(f"lookup-oracle vs table: {len(tab) - mismatch}/{len(tab)} egyezik  {'OK' if ok_lookup else 'MISMATCH!'}")
print(f"partition: GREY(non-repro)={grey} / repro={len(tab) - grey}  non_degenerate={non_degenerate}")
allok = ok_rule and ok_gt and ok_lookup and non_degenerate
print("\nVERDICT:", "OK -- per-stack lookup-orakulum konzisztens (GPU NELKUL verifikalt)" if allok else "HIBA -- konzisztencia serult")
sys.exit(0 if allok else 1)
