#!/usr/bin/env python3
"""Render bench/results/results.jsonl into a markdown report with
speed / peak-RSS / CPU tables per corpus."""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def fmt(v, suffix=""):
    return "—" if v is None else f"{v}{suffix}"


def table(rows, cols, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/results.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # rerun cells supersede earlier ones: keep the last record per cell key
    latest = {}
    for r in recs:
        latest[(r["corpus"], r["tool"], r.get("mode"), r.get("scenario"))] = r
    by_corpus = defaultdict(list)
    for r in latest.values():
        by_corpus[r["corpus"]].append(r)

    md = ["# semgrep benchmark report", ""]
    for corpus, rows in by_corpus.items():
        md += [f"## {corpus}", ""]

        builds = [r for r in rows if r["scenario"] == "index-build"]
        if builds:
            b = builds[0]
            md += [
                f"**Index build:** {b['wall_s']}s wall, "
                f"peak RSS {fmt(b.get('peak_rss_mb'), ' MB')}, "
                f"index size {fmt(b.get('index_mb'), ' MB')}, "
                f"CPU {fmt(b.get('cpu_util'), 'x')}",
                "",
            ]

        for section, pred in [
            ("Keyword (all tools)", lambda r: r["mode"] == "keyword"),
            ("NL-as-keywords fallback (grep family)", lambda r: r["mode"] == "nl-fallback"),
            ("Ranked modes (semgrep)", lambda r: r["mode"] in ("bm25", "semantic", "hybrid")),
        ]:
            sec = [r for r in rows if pred(r)]
            if not sec:
                continue
            md += [f"### {section}", ""]
            body = [
                (r["tool"], r["mode"], r["scenario"],
                 f"{r['wall_s']:.3f} ± {r.get('wall_stdev', 0):.3f}",
                 fmt(r.get("peak_rss_mb")), fmt(r.get("cpu_util")))
                for r in sorted(sec, key=lambda r: (r["scenario"], r["wall_s"]))
            ]
            md += [table(body, 6, ["tool", "mode", "scenario", "wall (s)", "peak RSS (MB)", "CPU util"]), ""]

    out = path.parent / "report.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")




# ---------------------------------------------------------------------------
# regression gate
#
# Perf numbers that live only in markdown go stale silently — the README's
# ranked-query table was wrong for a day after the dims change, and function
# chunking's 1.39x cold-index regression had to be found by hand. A gate turns
# both into a build failure.
# ---------------------------------------------------------------------------

# scenario substring -> (metric, max allowed ratio vs baseline)
BUDGETS = {
    "index-build": ("wall_s", 1.25),   # the cache design rests on cold being cheap
    "ranked": ("wall_s", 1.30),
    "keyword": ("wall_s", 1.15),       # this is ripgrep's own engine; it should not move
}
SIZE_BUDGET = 1.15                     # binary and index growth


def _key(r):
    return (r.get("corpus"), r.get("tool"), r.get("scenario"))


def gate(rows, baseline_rows, strict=False):
    """Compare the latest run against a baseline. Returns (violations, notes)."""
    latest, base = {}, {}
    for r in rows:
        latest[_key(r)] = r
    for r in baseline_rows:
        base[_key(r)] = r

    lp = next((r.get("provenance") for r in rows if r.get("provenance")), {}) or {}
    bp = next((r.get("provenance") for r in baseline_rows if r.get("provenance")), {}) or {}
    violations, notes = [], []
    if lp.get("machine") and bp.get("machine") and lp["machine"] != bp["machine"]:
        notes.append(f"machines differ ({bp['machine']} -> {lp['machine']}); "
                     f"timing comparisons are not meaningful")
    if lp.get("git_dirty"):
        notes.append("working tree is dirty; results may not match the recorded SHA")

    # binary size
    lb, bb = lp.get("binary_bytes"), bp.get("binary_bytes")
    if lb and bb:
        ratio = lb / bb
        line = f"binary {bb/1e6:.1f}MB -> {lb/1e6:.1f}MB ({ratio-1:+.1%} = x{ratio:.2f})"
        (violations if ratio > SIZE_BUDGET else notes).append(line)

    for k, r in sorted(latest.items(), key=lambda kv: str(kv[0])):
        b = base.get(k)
        if not b:
            continue
        scen = (r.get("scenario") or "")
        for pat, (metric, budget) in BUDGETS.items():
            if pat not in scen:
                continue
            lv, bv = r.get(metric), b.get(metric)
            if not lv or not bv:
                continue
            ratio = lv / bv
            desc = (f"{r.get('corpus')}/{r.get('tool')}/{scen} {metric} "
                    f"{bv:.3f} -> {lv:.3f} (x{ratio:.2f}, budget x{budget:.2f})")
            if ratio > budget:
                violations.append(desc)
            elif strict and ratio > 1.05:
                notes.append(desc)
        # index size
        li, bi = r.get("index_bytes"), b.get("index_bytes")
        if li and bi and li / bi > SIZE_BUDGET:
            violations.append(f"{r.get('corpus')} index {bi/1e6:.0f}MB -> {li/1e6:.0f}MB "
                              f"(x{li/bi:.2f}, budget x{SIZE_BUDGET:.2f})")
    return violations, notes


def load_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def gate_main():
    import argparse
    ap = argparse.ArgumentParser(description="regression gate for bench results")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results/results.jsonl"))
    ap.add_argument("--against", required=True, help="baseline results.jsonl to compare with")
    ap.add_argument("--strict", action="store_true", help="also note anything over +5%%")
    args = ap.parse_args()

    rows, base = load_rows(args.results), load_rows(args.against)
    violations, notes = gate(rows, base, args.strict)
    for n in notes:
        print(f"note:      {n}")
    for v in violations:
        print(f"REGRESSION {v}")
    if violations:
        print(f"\n{len(violations)} budget violation(s)")
        raise SystemExit(1)
    print("within budget")


if __name__ == "__main__":
    # `--against <baseline>` runs the regression gate; otherwise render the
    # usual markdown report.
    if "--against" in sys.argv:
        gate_main()
    else:
        main()
