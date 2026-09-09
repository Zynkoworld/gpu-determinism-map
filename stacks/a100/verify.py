#!/usr/bin/env python3
"""verify.py — GPU-FREE verification of the per-stack bit-reproducibility oracle.

WHY THIS WAS REWRITTEN (2026-09-09). The previous version ran three checks; two were real, one was blind:
  - `rule_byte_lock` (the CODE of the portable rule)        -> real
  - `ground_truth_byte_lock` (the ROWS of the table)        -> real: flipping one `regime` produces MISMATCH
  - "lookup vs table"                                       -> **TAUTOLOGY**: the lookup reads the very table it
    is compared against, so both sides come from one source. It reports 216/216 even if every row is wrong.

NOW: that third check is replaced by an INDEPENDENT cross-check — the RULE (`oracle_reproducible`, a pure
function of (M, N, K) that does NOT read the table) against the MEASURED table. It can say no, and it does:
the measured disagreement is 8-25 configs depending on the stack.
The lookup count is still printed, but relabelled: it shows the LOADER is correct, it is not evidence.

THREE VERDICTS, NOT TWO:
  rc=0  OK          — the table discriminates, byte-locks match, the negative control passed
  rc=2  DEGENERATE  — the table has no GREY (or only GREY): the lane does NOT discriminate.
                      This is a NEGATIVE RESULT, not an error: a legitimate measurement outcome, but a
                      constant "yes" is not an oracle — there is nothing here to decide.
  rc=1  ERROR       — byte-lock mismatch, or the negative control failed (the measure is blind).
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, ".")
import gpu_determinism_oracle as o                                    # noqa: E402


def anchor_path():
    """Stacks differ in anchor filename (anchor.json / l4_anchor_s64.json) -> do not hard-code it."""
    if os.path.exists("anchor.json"):
        return "anchor.json"
    j = sorted(f for f in glob.glob("*.json") if "anchor" in f.lower())
    if not j:
        raise SystemExit("ERROR: no anchor file in the stack directory")
    return j[0]


def gt_byte_lock(stack, tab):
    canon = json.dumps({"stack": stack, "gt": sorted([[t["config"], t["regime"]] for t in tab])},
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def rule_vs_table(tab):
    """The INDEPENDENT cross-check: the rule (which does not read the table) vs the measured table."""
    agree = fp = fn = 0
    for t in tab:
        M, N, K = t["config"]
        measured = (t["regime"] != "GREY")
        ruled = o.oracle_reproducible(M, N, K)
        if ruled == measured:
            agree += 1
        elif ruled and not measured:
            fp += 1               # rule says reproducible, measurement says GREY -> the dangerous direction
        else:
            fn += 1
    return agree, fp, fn


def main():
    ap = anchor_path()
    tab = json.load(open(ap))["table"]
    spec = json.load(open("BYTE_LOCK_SPEC.json"))
    stack = spec["stack_HASHED"]
    n = len(tab)

    rule_bl = o.rule_byte_lock()
    gt_bl = gt_byte_lock(stack, tab)
    rule_ok = (rule_bl == spec["byte_locks"]["rule_byte_lock"])
    gt_ok = (gt_bl == spec["byte_locks"]["ground_truth_byte_lock"])

    agree, fp, fn = rule_vs_table(tab)
    lookup_agree = n - sum(1 for t in tab
                           if o.oracle_lookup(*t["config"], ap)
                           != ("REPRODUCIBLE" if t["regime"] != "GREY" else "NON_REPRODUCIBLE"))
    grey = sum(1 for t in tab if t["regime"] == "GREY")
    discriminates = 0 < grey < n

    # --- MANDATORY NEGATIVE CONTROL: flip one regime (IN MEMORY; the file is untouched) -----------
    forged = [dict(t) for t in tab]
    idx = next((i for i, t in enumerate(forged) if t["regime"] != "GREY"), 0)
    original = forged[idx]["regime"]
    forged[idx]["regime"] = "GREY" if original != "GREY" else "REPRO"
    nc_lock = (gt_byte_lock(stack, forged) != gt_bl)
    nc_rule = (rule_vs_table(forged) != (agree, fp, fn))
    nc_ok = nc_lock and nc_rule

    print("rule_byte_lock:         %s  %s" % (rule_bl[:24], "OK" if rule_ok else "MISMATCH!"))
    print("ground_truth_byte_lock: %s  %s (%s / CUDA %s / driver %s)"
          % (gt_bl[:24], "OK" if gt_ok else "MISMATCH!", stack.get("gpu"), stack.get("cuda"),
             stack.get("driver")))
    print()
    print("INDEPENDENT CROSS-CHECK -- RULE (does not read the table) vs MEASURED table:")
    print("   agree=%d/%d (%.3f)   FP=%d (rule: reproducible, measurement: GREY)   FN=%d"
          % (agree, n, agree / n, fp, fn))
    print("   NOTE: the TABLE is the oracle; the rule is a portable approximation with a known residual.")
    print("lookup self-consistency: %d/%d -- NOT evidence (the lookup reads the same table);"
          " table integrity is enforced by ground_truth_byte_lock" % (lookup_agree, n))
    print("partition: GREY=%d/%d  discriminates=%s" % (grey, n, discriminates))
    print()
    print("NEGATIVE CONTROL (in memory: %s -> %s at config %s; the file is untouched):"
          % (original, forged[idx]["regime"], forged[idx]["config"]))
    print("   ground_truth_byte_lock breaks:   %s" % ("YES" if nc_lock else "NO -- BLIND MEASURE!"))
    print("   rule cross-check also changes:   %s" % ("YES" if nc_rule else "NO"))
    print()

    if not (rule_ok and gt_ok and nc_ok):
        print("VERDICT: ERROR")
        if not rule_ok:
            print("   - rule_byte_lock mismatch")
        if not gt_ok:
            print("   - ground_truth_byte_lock mismatch (the table changed since it was pinned)")
        if not nc_ok:
            print("   - the negative control FAILED: the measure does not notice a forged row")
        return 1
    if not discriminates:
        print("VERDICT: DEGENERATE -- a NEGATIVE RESULT, not an error.")
        print("   GREY=%d/%d: on this stack every measured config falls in the same class." % (grey, n))
        print("   The byte-locks hold and the negative control passed -- the MEASUREMENT is sound.")
        print("   But a constant 'yes' is not an oracle: there is nothing to decide here, so this lane is")
        print("   not publishable as a decider. See NEGATIVE_RESULT.md.")
        return 2
    print("VERDICT: OK -- decides from an independent source, byte-locks match, negative control passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
