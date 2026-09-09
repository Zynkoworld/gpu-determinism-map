#!/usr/bin/env python3
"""verify_all.py — verifies all four stacks + the MANDATORY POSITIVE/NEGATIVE CONTROL PAIR.

Why the pair is needed: a verifier that can only say "OK" is not a measure. So each stack is run TWICE:
  N (control against forgery)      = UNTOUCHED table -> the stack's EXPECTED verdict (OK or DEGENERATE)
  P (control of the measure itself) = ONE `regime` FLIPPED on a TEMPORARY COPY -> MISMATCH, rc=1
The original files are NEVER modified: the flip happens on a copy in a temp directory.

EXPECTED VERDICTS (measured 2026-09-09):
  l4, rtx-4090  : OK         (GREY=25/216 -> the table discriminates)
  a100, rtx-3090: DEGENERATE (GREY=0/216  -> NEGATIVE RESULT: no grey zone, not an oracle)

Run: `python3 verify_all.py`   (rc=0 only if EVERY stack gives the expected result on BOTH controls)
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

GYOKER = os.path.dirname(os.path.abspath(__file__))
STACKS = os.path.join(GYOKER, "stacks")
EXPECTED = {"l4": 0, "rtx-4090": 0, "a100": 2, "rtx-3090": 2}
NAMES = {0: "OK", 1: "MISMATCH/ERROR", 2: "DEGENERATE"}


def run(directory):
    p = subprocess.run([sys.executable, "verify.py"], cwd=directory,
                       capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout


def main():
    failures = []
    for s in sorted(EXPECTED):
        d = os.path.join(STACKS, s)
        # --- N: untouched table -> the stack's expected verdict ---------------------------------
        rc, _ = run(d)
        ok = (rc == EXPECTED[s])
        print("  %-9s N (untouched)     rc=%d %-16s expected: %-16s %s"
              % (s, rc, NAMES.get(rc, "?"), NAMES[EXPECTED[s]], "PASS" if ok else "FAIL"))
        if not ok:
            failures.append("%s: untouched table rc=%d, expected %d" % (s, rc, EXPECTED[s]))

        # --- P: one regime flipped ON A COPY -> MISMATCH, rc=1 ----------------------------------
        with tempfile.TemporaryDirectory(prefix="gpumap_control_") as tmp:
            copy = os.path.join(tmp, s)
            shutil.copytree(d, copy, ignore=shutil.ignore_patterns("__pycache__"))
            anchors = [f for f in glob.glob(os.path.join(copy, "*.json"))
                       if "anchor" in os.path.basename(f).lower()]
            with open(anchors[0]) as f:
                dd = json.load(f)
            t0 = dd["table"][0]
            t0["regime"] = "GREY" if t0["regime"] != "GREY" else "REPRO"
            with open(anchors[0], "w") as f:
                json.dump(dd, f)
            rc2, out = run(copy)
            ok2 = (rc2 == 1 and "MISMATCH" in out)
            print("  %-9s P (regime flipped) rc=%d %-16s expected: MISMATCH/ERROR  %s"
                  % (s, rc2, NAMES.get(rc2, "?"), "PASS" if ok2 else "FAIL"))
            if not ok2:
                failures.append("%s: flipped table rc=%d (expected MISMATCH)" % (s, rc2))

    print()
    if failures:
        print("FAIL -- the control pair did not give the expected result:")
        for h in failures:
            print("   -", h)
        return 1
    print("ALL 4 STACKS PASSED BOTH CONTROLS.")
    print("  2 VALID oracles (l4, rtx-4090): the table discriminates")
    print("  2 NEGATIVE RESULTS (a100, rtx-3090): GREY=0/216 -> no grey zone, not an oracle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
