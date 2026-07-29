#!/usr/bin/env python3
"""Benchmark semgrep against grep/ripgrep/ugrep/ack on speed, memory, and CPU.

For every (corpus, tool, query) cell this runs the command N times via
`/usr/bin/time -l` (macOS) and records wall time, user+sys CPU, and peak RSS
per run, reporting medians. Warmup runs populate the FS cache first, so
numbers are warm-cache; note that in writeups.

Usage:
  python3 bench/run.py [--corpora linux,vscode,wikipedia] [--runs 5]
                       [--skip-index-build] [--out bench/results]

Outputs bench/results/results.jsonl (one record per cell) and prints progress.
Generate the report with bench/report.py.
"""

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SEMGREP = HERE.parent / "target/release/semgrep"

# Keyword competitors, invoked by absolute path (dev shells wrap `grep`).
# The greps get -E so all tools speak extended regex (BRE chokes on `+`).
TOOLS = {
    "grep-bsd": ["/usr/bin/grep", "-r", "-n", "-E"],
    "grep-gnu": ["/opt/homebrew/bin/ggrep", "-r", "-n", "-E"],
    "rg": ["/opt/homebrew/bin/rg", "--no-heading", "-n"],
    "ugrep": ["/opt/homebrew/bin/ugrep", "-r", "-n"],
    "ack": ["/opt/homebrew/bin/ack", "--nocolor"],
}

TIME_RE = {
    "wall": re.compile(r"([\d.]+)\s+real"),
    "user": re.compile(r"([\d.]+)\s+user"),
    "sys": re.compile(r"([\d.]+)\s+sys"),
    "rss": re.compile(r"(\d+)\s+maximum resident set size"),
}


def measure(cmd, cwd=None):
    """One measured run under /usr/bin/time -l. Returns metrics dict.

    stdout goes to a real temp file, not /dev/null — some tools (ugrep)
    short-circuit when they detect the null device, which would flatter them.
    """
    import tempfile
    t0 = time.monotonic()
    with tempfile.TemporaryFile(dir=str(HERE / "results")) as sink:
        proc = subprocess.run(
            ["/usr/bin/time", "-l"] + cmd,
            cwd=cwd,
            stdout=sink,
            stderr=subprocess.PIPE,
            text=True,
        )
    wall_fallback = time.monotonic() - t0
    out = {"exit": proc.returncode}
    for key, rx in TIME_RE.items():
        m = rx.search(proc.stderr)
        out[key] = float(m.group(1)) if m else None
    if out["wall"] is None:
        out["wall"] = wall_fallback
    # macOS `time -l` reports max RSS in bytes
    if out["rss"] is not None:
        out["rss_mb"] = out["rss"] / 1e6
    return out


def bench_cell(record, cmd, runs, warmup=1, cwd=None):
    for _ in range(warmup):
        subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    samples = [measure(cmd, cwd=cwd) for _ in range(runs)]
    ok = [s for s in samples if s["wall"] is not None]
    record["runs"] = len(ok)
    walls = sorted(s["wall"] for s in ok)
    record["wall_s"] = round(statistics.median(walls), 4)
    record["wall_stdev"] = round(statistics.pstdev(walls), 4)
    # An agent feels the tail, not the median: a p99 that is 5x the median is
    # a different product from one that tracks it, at the same median.
    def _pct(q):
        if not walls:
            return None
        return round(walls[min(len(walls) - 1, int(q * len(walls)))], 4)
    record["wall_p50"], record["wall_p90"], record["wall_p99"] = (
        _pct(0.50), _pct(0.90), _pct(0.99))
    record["wall_min"], record["wall_max"] = round(walls[0], 4), round(walls[-1], 4)
    record["user_s"] = round(statistics.median(s["user"] or 0 for s in ok), 4)
    record["sys_s"] = round(statistics.median(s["sys"] or 0 for s in ok), 4)
    record["cpu_util"] = round(
        (record["user_s"] + record["sys_s"]) / record["wall_s"], 2
    ) if record["wall_s"] else None
    rss = [s.get("rss_mb") for s in ok if s.get("rss_mb")]
    record["peak_rss_mb"] = round(statistics.median(rss), 1) if rss else None
    record["exit"] = samples[-1]["exit"]
    return record


def provenance():
    """Enough to know whether two result sets are comparable at all. A
    regression check across machines is meaningless, so record the machine."""
    def sh(*c):
        try:
            return subprocess.run(c, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None
    return {
        "git_sha": sh("git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"),
        "git_dirty": bool(sh("git", "-C", str(HERE.parent), "status", "--porcelain")),
        "binary_bytes": SEMGREP.stat().st_size if SEMGREP.exists() else None,
        "machine": f"{platform.system()}-{platform.machine()}",
        "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "cores": os.cpu_count(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def dir_bytes(p):
    p = Path(p)
    if not p.is_dir():
        return None
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="linux,vscode,wikipedia")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--skip-index-build", action="store_true")
    ap.add_argument("--sections", default="index,keyword,ranked,fallback",
                    help="comma list of cell groups to run")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    sections = set(args.sections.split(","))

    if not SEMGREP.exists():
        sys.exit(f"build first: cargo build --release   (missing {SEMGREP})")

    queries = json.loads((HERE / "queries.json").read_text())
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"
    results = []

    prov = provenance()
    print(f"build {prov['git_sha']}{'+dirty' if prov['git_dirty'] else ''} "
          f"binary {(prov['binary_bytes'] or 0)/1e6:.1f}MB on {prov['machine']} "
          f"({prov['cores']} cores)")

    def emit(rec):
        rec.setdefault("provenance", prov)
        # Index size is a first-class cost: it moved 1.3GB -> 946MB with the
        # dims change and would have moved again with function chunking.
        if rec.get("corpus"):
            rec.setdefault("index_bytes",
                           dir_bytes(HERE / "corpora" / rec["corpus"] / ".semgrep"))
        results.append(rec)
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"  {rec['tool']:>10} {rec.get('scenario', ''):<28}"
            f" {rec['wall_s']:>8.3f}s  rss {rec.get('peak_rss_mb') or 0:>7.1f}MB"
            f"  cpu {rec.get('cpu_util') or 0:>4}x"
        )

    for corpus in args.corpora.split(","):
        corpus_dir = HERE / "corpora" / corpus
        if not corpus_dir.is_dir():
            print(f"skipping {corpus}: {corpus_dir} missing (run fetch-corpora.sh)")
            continue
        print(f"== {corpus} ==")
        qset = queries[corpus]

        # -- index build cost (once, measured) --------------------------------
        semgrep_dir = corpus_dir / ".semgrep"
        if "index" in sections and not args.skip_index_build:
            if semgrep_dir.exists():
                shutil.rmtree(semgrep_dir)
            rec = {"corpus": corpus, "tool": "semgrep", "scenario": "index-build", "mode": "index"}
            rec = bench_cell(rec, [str(SEMGREP), "index", str(corpus_dir)], runs=1, warmup=0)
            rec["index_mb"] = round(
                sum(p.stat().st_size for p in semgrep_dir.rglob("*")) / 1e6, 1
            )
            emit(rec)

        # -- keyword queries: all tools ---------------------------------------
        for q in qset["keyword"] if "keyword" in sections else []:
            flags = q.get("flags", [])
            for tool, base in TOOLS.items():
                if not Path(base[0]).exists():
                    continue
                cmd = base + flags + [q["pattern"], str(corpus_dir)]
                emit(bench_cell(
                    {"corpus": corpus, "tool": tool, "mode": "keyword",
                     "scenario": f"kw:{q['name']}"},
                    cmd, args.runs))
            cmd = [str(SEMGREP), "--mode", "keyword"] + flags + [q["pattern"], str(corpus_dir)]
            emit(bench_cell(
                {"corpus": corpus, "tool": "semgrep", "mode": "keyword",
                 "scenario": f"kw:{q['name']}"},
                cmd, args.runs))

        # -- ranked queries: semgrep only, unindexed vs indexed -------------------
        for q in qset["nl"]:
            label = q["query"][:24]
            for mode in ["bm25", "semantic", "hybrid"] if "ranked" in sections else []:
                emit(bench_cell(
                    {"corpus": corpus, "tool": "semgrep", "mode": mode,
                     "scenario": f"cold:{label}"},
                    [str(SEMGREP), "--mode", mode, "--no-index", q["query"], str(corpus_dir)],
                    max(1, args.runs // 2), warmup=0))  # cold runs are slow; fewer reps
                emit(bench_cell(
                    {"corpus": corpus, "tool": "semgrep", "mode": mode,
                     "scenario": f"warm:{label}"},
                    [str(SEMGREP), "--mode", mode, q["query"], str(corpus_dir)],
                    args.runs))
            # grep-family NL fallback: agent-style keyword extraction
            for tool, base in TOOLS.items():
                if "fallback" not in sections or not Path(base[0]).exists():
                    continue
                emit(bench_cell(
                    {"corpus": corpus, "tool": tool, "mode": "nl-fallback",
                     "scenario": f"nlkw:{label}"},
                    base + ["-i", q["nl_keywords"].replace(" ", ".*"), str(corpus_dir)],
                    args.runs))

    print(f"\nwrote {len(results)} records to {results_path}")


if __name__ == "__main__":
    main()
