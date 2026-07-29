#!/usr/bin/env python3
"""Compare Loc-Bench conditions that differ only in the compiled-in embedding
table, on the instances all of them share.

The conditions run identical flags, prompts, harness, and driver model, so the
table (and its dimensionality) is the only variable. That makes the set a
factorial:

    prose@512 (sg-plain)  vs  prose@256 (sg-p256)   -> isolates DIMS
    prose@256 (sg-p256)   vs  code@256  (sg-code)   -> isolates MODEL
    prose@512 (sg-plain)  vs  code@256  (sg-code)   -> both at once

Reported on all shared instances and, separately, on the subset where every
condition actually invoked semgrep — whether the agent searches at all is
decided before any result returns, so the table cannot influence it, and
those runs only add driver variance.

Usage: python3 eval/locbench/table-ab.py [--metric file_acc@5]
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "locbench"
SOURCES = ["results-ab.jsonl", "results-model.jsonl", "results.jsonl"]
# rg/both come from the §7.1 pilot: same 50 instances, same driver model, but
# the semgrep-side description there was v1 where the sg-* conditions use v4.
# rg's own condition is identical across both, so it is a stable reference.
LABELS = {"rg": "ripgrep (pilot)", "both": "rg+semgrep (pilot)",
          "sg-plain": "prose@512 (shipped)", "sg-p256": "prose@256", "sg-code": "code@256"}


def load():
    by_cond = defaultdict(dict)
    for src in SOURCES:
        p = DATA / src
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("condition") in LABELS:
                by_cond[r["condition"]][r["instance_id"]] = r
    return by_cond


def exact_binomial(b, c):
    """Two-sided sign test on discordant pairs (McNemar, exact)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def row(d, ids, label):
    sem = [(d[i]["search"].get("n_invocations") or 0) for i in ids]
    z = sum(1 for x in sem if x == 0)
    fa = statistics.mean(d[i]["metrics"]["file_acc@5"] for i in ids)
    fn = statistics.mean(d[i]["metrics"]["func_acc@10_tol"] for i in ids)
    cost = [d[i]["agent"].get("total_cost_usd") or 0 for i in ids]
    print(f"  {label:<22} zero-search {z:>2} ({z/len(ids):>3.0%})  medSearch {statistics.median(sem):>2.0f}  "
          f"fileAcc@5 {fa:>4.0%}  fnAcc@10t {fn:>4.0%}  medCost ${statistics.median(cost):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="file_acc@5")
    args = ap.parse_args()

    by_cond = load()
    present = [c for c in LABELS if c in by_cond]
    missing = [c for c in LABELS if c not in by_cond]
    if missing:
        print(f"(missing conditions: {', '.join(missing)})\n")
    if len(present) < 2:
        print("need at least two conditions")
        return

    common = set.intersection(*(set(by_cond[c]) for c in present))
    common = sorted(common)
    searched = [i for i in common
                if all((by_cond[c][i]["search"].get("n_invocations") or 0) > 0 for c in present)]

    for ids, tag in ((common, "ALL SHARED"), (searched, "ALL ACTUALLY SEARCHED")):
        print(f"\n=== {tag} (n={len(ids)}) ===")
        for c in present:
            row(by_cond[c], ids, LABELS[c])

    print(f"\n=== PAIRWISE on {args.metric} (exact sign test over discordant pairs) ===")
    for ids, tag in ((common, "all"), (searched, "searched")):
        print(f"\n  [{tag}, n={len(ids)}]")
        for a in present:
            for b in present:
                if a >= b:
                    continue
                m = args.metric
                aw = sum(1 for i in ids if by_cond[a][i]["metrics"][m] and not by_cond[b][i]["metrics"][m])
                bw = sum(1 for i in ids if by_cond[b][i]["metrics"][m] and not by_cond[a][i]["metrics"][m])
                p = exact_binomial(aw, bw)
                print(f"  {LABELS[a]:<22} vs {LABELS[b]:<18} {aw:>2} - {bw:<2} wins   p={p:.3f}")


if __name__ == "__main__":
    main()
