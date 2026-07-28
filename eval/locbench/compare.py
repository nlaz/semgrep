#!/usr/bin/env python3
"""Compare two Loc-Bench result sets on the pre-registered nudge metrics.

Usage:
  python3 eval/locbench/compare.py eval/data/locbench/results.jsonl eval/data/locbench/results-guided.jsonl

For each (condition, file) cell: first-search-hit rate, function Acc@10
(tolerant), file Acc@5, %-of-first-queries-that-are-ranked, tool-choice
split, ranked-vs-exact semgrep style, median cost/searches. Only instances
present in BOTH files (per condition) are compared, so deltas are paired.
"""

import json
import pathlib
import statistics
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"

import scoring


def load(path):
    rows = {}
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            rows[(r["instance_id"], r["condition"])] = r
    return rows


def shim_rows(r):
    p = DATA / r["paths"]["shim_log"]
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return [x for x in sorted(out, key=lambda x: x["seq"]) if not x.get("blocked")]


def stats(rs):
    n = len(rs)
    if not n:
        return None
    fn = [r["metrics"].get("func_acc@10_tol") for r in rs]
    fa = [r["metrics"].get("file_acc@5") for r in rs]
    first_hit, ranked_first, n_rg, n_sg, sg_ranked, sg_exact = [], 0, 0, 0, 0, 0
    for r in rs:
        rows = shim_rows(r)
        first_hit.append(
            scoring.first_gold_hit_seq(
                DATA / r["paths"]["shim_log"],
                DATA / r["paths"]["shim_log"].replace("shim_log.jsonl", "searches"),
                r.get("gold_files") or [],
            )
        )
        # "search" is the same binary under the name-ablation alias.
        if rows and rows[0]["tool"] in ("semgrep", "search") and "-e" not in rows[0]["argv"]:
            ranked_first += 1
        for x in rows:
            if x["tool"] == "rg":
                n_rg += 1
            elif x["tool"] in ("semgrep", "search"):
                n_sg += 1
                if "-e" in x["argv"]:
                    sg_exact += 1
                else:
                    sg_ranked += 1
    pct = lambda xs: f"{100 * sum(x for x in xs if x) / len(xs):.0f}%"
    hit1 = [h == 1 for h in first_hit if h is not None]
    return {
        "n": n,
        "fnAcc@10t": pct(fn),
        "fAcc@5": pct(fa),
        "hit@1st": f"{100 * sum(hit1) / len(hit1):.0f}%" if hit1 else "—",
        "ranked-first": f"{100 * ranked_first / n:.0f}%",
        "tool rg/sg": f"{n_rg}/{n_sg}",
        "sg -e/ranked": f"{sg_exact}/{sg_ranked}",
        "med $": f"${statistics.median([r['agent']['total_cost_usd'] or 0 for r in rs]):.2f}",
        "med searches": statistics.median([r["search"]["n_invocations"] for r in rs]),
    }


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    conds = sorted({c for _, c in a} & {c for _, c in b})
    labels = [pathlib.Path(p).stem for p in sys.argv[1:3]]
    for cond in conds:
        common = {i for i, c in a if c == cond} & {i for i, c in b if c == cond}
        ra = [a[(i, cond)] for i in sorted(common)]
        rb = [b[(i, cond)] for i in sorted(common)]
        print(f"\n== condition: {cond} (paired n={len(common)}) ==")
        sa, sb = stats(ra), stats(rb)
        keys = list(sa.keys())
        w = max(len(k) for k in keys) + 2
        print(f"{'':{w}} {labels[0]:>22} {labels[1]:>22}")
        for k in keys:
            print(f"{k:{w}} {str(sa[k]):>22} {str(sb[k]):>22}")


if __name__ == "__main__":
    main()
