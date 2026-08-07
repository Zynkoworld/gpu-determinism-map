#!/usr/bin/env python3
"""verify.py — a per-stack bit-reprodukalhatosag-orakulum GPU-NELKULI verifikacioja.
Ujraszamolja mindket byte-lockot a tablabol + a lookup konzisztenciajat. Futtatas: python3 verify.py"""
import json, hashlib, sys
sys.path.insert(0, ".")
import gpu_determinism_oracle as o
d = json.load(open("anchor.json")); spec = json.load(open("BYTE_LOCK_SPEC.json")); tab = d["table"]
rule_bl = o.rule_byte_lock()
stack = spec["stack_HASHED"]
canon = json.dumps({"stack": stack, "gt": sorted([[t["config"], t["regime"]] for t in tab])},
                   sort_keys=True, separators=(",", ":"))
gt_bl = hashlib.sha256(canon.encode()).hexdigest()
mismatch = sum(1 for t in tab
               if o.oracle_lookup(*t["config"], "anchor.json") != ("REPRODUCIBLE" if t["regime"] != "GREY" else "NON_REPRODUCIBLE"))
grey = sum(1 for t in tab if t["regime"] == "GREY")
ok = (rule_bl == spec["byte_locks"]["rule_byte_lock"] and gt_bl == spec["byte_locks"]["ground_truth_byte_lock"]
      and mismatch == 0 and 0 < grey < len(tab))
print(f"rule_byte_lock:         {rule_bl[:24]}  {'OK' if rule_bl == spec['byte_locks']['rule_byte_lock'] else 'MISMATCH!'}")
print(f"ground_truth_byte_lock: {gt_bl[:24]}  {'OK' if gt_bl == spec['byte_locks']['ground_truth_byte_lock'] else 'MISMATCH!'} ({stack.get('gpu')} / CUDA {stack.get('cuda')} / driver {stack.get('driver')})")
print(f"lookup vs tabla: {len(tab) - mismatch}/{len(tab)}  | partition GREY={grey}/{len(tab)}  non_degenerate={0 < grey < len(tab)}")
print("\nVERDICT:", "OK -- per-stack lookup-orakulum konzisztens (GPU NELKUL)" if ok else "HIBA")
sys.exit(0 if ok else 1)
