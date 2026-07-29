#!/usr/bin/env python3
"""Stratify a paired Loc-Bench comparison by whether the issue text already
names the gold identifier.

This is the mechanism the tool's premise rests on. When an issue says
"`retry_backoff` returns the wrong delay", the agent can hand that token
straight to ripgrep and exact matching wins. When it says "retries happen too
fast", there is no token to match and ranked retrieval should separate. The
existing pilot mixes both cases, so a flat average hides whichever effect is
real.

Splits the shared instances into NAMED (the problem statement contains at
least one gold function's own name) and UNNAMED, then reports each condition
per stratum.

Usage: python3 eval/locbench/stratify.py [--a rg] [--b sg-p256]
"""
import argparse
import json
import re
import statistics
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "locbench"
SOURCES = ["results.jsonl", "results-ab.jsonl", "results-model.jsonl"]


def load(cond):
    out = {}
    for src in SOURCES:
        p = DATA / src
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") == cond and r.get("status") == "ok":
                out[r["instance_id"]] = r
    return out


def leaf_names(gold_functions):
    """'path/to/file.py:Class.method' -> {'method', 'Class'}; bare funcs too."""
    names = set()
    for g in gold_functions or []:
        qual = g.split(":", 1)[1] if ":" in g else g
        for part in qual.split("."):
            part = part.strip()
            if len(part) > 2:          # skip 'x', '__' noise
                names.add(part)
    return names


def names_identifier(problem_statement, names):
    """True if the issue text mentions a gold identifier as a whole word."""
    text = problem_statement or ""
    for n in names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", text):
            return True
    return False


def report(rows, ids, label):
    if not ids:
        print(f"  {label:<22} (no instances)")
        return
    fa = statistics.mean(rows[i]["metrics"]["file_acc@5"] for i in ids)
    fn = statistics.mean(rows[i]["metrics"]["func_acc@10_tol"] for i in ids)
    sr = [rows[i]["search"].get("n_invocations") or 0 for i in ids]
    print(f"  {label:<22} fileAcc@5 {fa:>4.0%}   fnAcc@10t {fn:>4.0%}   medSearch {statistics.median(sr):>2.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="rg")
    ap.add_argument("--b", default="sg-p256")
    args = ap.parse_args()

    ds = {}
    for line in (DATA / "dataset.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ds[r["instance_id"]] = r

    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B) & set(ds))

    named, unnamed = [], []
    for i in common:
        gold = A[i].get("gold_functions") or ds[i].get("edit_functions")
        (named if names_identifier(ds[i].get("problem_statement"), leaf_names(gold))
         else unnamed).append(i)

    print(f"paired instances: {len(common)}   "
          f"issue NAMES a gold identifier: {len(named)}   does NOT: {len(unnamed)}\n")

    for ids, tag in ((named, "NAMED — the identifier is handed over (grep's best case)"),
                     (unnamed, "UNNAMED — no token to match (ranked search's case)")):
        print(f"=== {tag}  (n={len(ids)}) ===")
        report(A, ids, args.a)
        report(B, ids, args.b)
        if ids:
            for m in ("file_acc@5", "func_acc@10_tol"):
                aw = sum(1 for i in ids if A[i]["metrics"][m] and not B[i]["metrics"][m])
                bw = sum(1 for i in ids if B[i]["metrics"][m] and not A[i]["metrics"][m])
                print(f"    {m:<18} {args.a} wins {aw}, {args.b} wins {bw}")
        print()


if __name__ == "__main__":
    main()
