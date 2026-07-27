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


if __name__ == "__main__":
    main()
