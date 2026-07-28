#!/usr/bin/env python3
"""Aggregate Loc-Bench results into markdown tables.

Usage:
  python3 eval/locbench/report.py [eval/data/locbench/results.jsonl]

Dedupes by (instance_id, condition, model) keeping the last row (re-runs
supersede, like bench/report.py). Writes report.md next to the input and
prints it. Accuracy is computed over rows with status=ok; efficiency
medians are additionally reported *conditioned on success* (file_acc@5=1)
to avoid the failed-fast artifact (RESEARCH.md §7 metric conventions).
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import scoring

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"
DEFAULT = DATA / "results.jsonl"


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[(r["instance_id"], r["condition"], r.get("model"))] = r
    out = list(rows.values())
    # Recompute searches-to-first-hit from the shim logs (1-based, blocked
    # invocations excluded) so rows emitted by any harness version agree.
    for r in out:
        shim_log = (r.get("paths") or {}).get("shim_log")
        if shim_log and r.get("metrics") is not None:
            r["metrics"]["first_hit_search_seq"] = scoring.first_gold_hit_seq(
                DATA / shim_log,
                DATA / shim_log.replace("shim_log.jsonl", "searches"),
                r.get("gold_files") or [],
            )
    return out


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "—"


def med(xs, fmt="{:.0f}"):
    xs = [x for x in xs if x is not None]
    return fmt.format(statistics.median(xs)) if xs else "—"


def size_bucket(r):
    n = (r.get("checkout") or {}).get("file_count")
    if n is None:
        return "?"
    return "<2k files" if n < 2000 else "2k–10k files" if n <= 10000 else ">10k files"


def acc_row(cond, rs):
    ok = [r for r in rs if r["status"] == "ok"]
    m = lambda key: [r["metrics"].get(key) for r in ok if r.get("metrics")]
    return [
        cond, len(rs), f"{100 * len(ok) / len(rs):.0f}%" if rs else "—",
        pct(m("file_acc@1")), pct(m("file_acc@3")), pct(m("file_acc@5")),
        med(m("file_recall@5"), "{:.2f}"),
        pct(m("func_acc@5_strict")), pct(m("func_acc@5_tol")), pct(m("func_acc@10_tol")),
        med(m("func_recall@10_tol"), "{:.2f}"),
    ]


ACC_HEADER = ["condition", "n", "ok", "fAcc@1", "fAcc@3", "fAcc@5", "fR@5",
              "fnAcc@5s", "fnAcc@5t", "fnAcc@10t", "fnR@10t"]


def eff_row(cond, rs, success_only=False):
    ok = [r for r in rs if r["status"] == "ok"]
    if success_only:
        ok = [r for r in ok if (r.get("metrics") or {}).get("file_acc@5") == 1]
    a = lambda key: [(r.get("agent") or {}).get(key) for r in ok]
    u = lambda key: [((r.get("agent") or {}).get("usage") or {}).get(key) for r in ok]
    s = lambda key: [(r.get("search") or {}).get(key) for r in ok]
    first_hit = [(r.get("metrics") or {}).get("first_hit_search_seq") for r in ok]
    hit1 = [h == 1 for h in first_hit if h is not None]
    idx_s = [(r.get("index") or {}).get("build_wall_s") for r in ok
             if (r.get("index") or {}).get("built")]
    return [
        cond, len(ok),
        med(a("total_cost_usd"), "${:.2f}"),
        med(a("num_turns")),
        med(u("input_tokens"), "{:,.0f}"), med(u("output_tokens"), "{:,.0f}"),
        med(u("cache_read_input_tokens"), "{:,.0f}"),
        med(s("n_invocations")),
        med(s("n_blocked")),
        med(s("total_wall_ms"), "{:,.0f}"),
        med(s("total_stdout_bytes"), "{:,.0f}"),
        med(first_hit),
        f"{100 * sum(hit1) / len(hit1):.0f}%" if hit1 else "—",
        med([(r.get("agent") or {}).get("duration_ms") for r in ok], "{:,.0f}"),
        med(idx_s) if idx_s else "—",
    ]


EFF_HEADER = ["condition", "n", "med $", "turns", "in tok", "out tok", "cache rd",
              "searches", "blocked", "search ms", "search bytes", "1st-hit#", "hit@1st",
              "agent ms", "idx build s"]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    rows = load(path)
    if not rows:
        sys.exit(f"no rows in {path}")
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    conds = sorted(by_cond, key=lambda c: ["rg", "semgrep", "both"].index(c)
                   if c in ("rg", "semgrep", "both") else 99)

    out = [f"# Loc-Bench localization results\n",
           f"{len(rows)} rows ({path}), models: "
           f"{sorted({r.get('model') for r in rows})}\n"]

    out.append("## Accuracy (status=ok rows)\n")
    out.append(table([acc_row(c, by_cond[c]) for c in conds], ACC_HEADER))

    out.append("\n\n## Efficiency (medians, all ok rows)\n")
    out.append(table([eff_row(c, by_cond[c]) for c in conds], EFF_HEADER))
    out.append("\n\n## Efficiency conditioned on success (file_acc@5 = 1)\n")
    out.append(table([eff_row(c, by_cond[c], success_only=True) for c in conds], EFF_HEADER))

    for strat_name, keyfn in (("category", lambda r: r.get("category") or "?"),
                              ("repo size", size_bucket)):
        out.append(f"\n\n## Accuracy by {strat_name}\n")
        groups = defaultdict(lambda: defaultdict(list))
        for r in rows:
            groups[keyfn(r)][r["condition"]].append(r)
        for g in sorted(groups):
            out.append(f"\n**{g}**\n")
            out.append(table([acc_row(c, groups[g][c]) for c in conds if groups[g][c]],
                             ACC_HEADER))

    # Paired deltas on the instance intersection — the headline comparison.
    if len(conds) > 1:
        common = set.intersection(*(
            {r["instance_id"] for r in by_cond[c] if r["status"] == "ok"} for c in conds
        ))
        out.append(f"\n\n## Paired comparison ({len(common)} instances in all conditions)\n")
        prows = []
        for c in conds:
            rs = [r for r in by_cond[c] if r["instance_id"] in common and r["status"] == "ok"]
            prows.append(acc_row(c, rs) + [
                med([(r.get("agent") or {}).get("total_cost_usd") for r in rs], "${:.2f}"),
                med([(r.get("search") or {}).get("n_invocations") for r in rs]),
                med([((r.get("agent") or {}).get("usage") or {}).get("output_tokens") for r in rs], "{:,.0f}"),
            ])
        out.append(table(prows, ACC_HEADER + ["med $", "searches", "out tok"]))

    n_bypass = sum(1 for r in rows if (r.get("search") or {}).get("shim_bypass_suspected"))
    n_forbidden = sum((r.get("agent") or {}).get("forbidden_tool_events") or 0 for r in rows)
    out.append(f"\n\nintegrity: shim-bypass suspected in {n_bypass} rows; "
               f"forbidden tool events (Grep/Task): {n_forbidden}\n")

    text = "\n".join(out)
    report = path.parent / "report.md"
    report.write_text(text)
    print(text)
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
