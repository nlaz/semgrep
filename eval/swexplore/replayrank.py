#!/usr/bin/env python3
"""Replay a misswhy bucket's own queries at deeper k to find gold's rank.

    python3 eval/swexplore/replayrank.py --misswhy misswhy-sg.jsonl \
        --run-id s32 --arm sub-sg --bucket NEVER_SURFACED_W -k 30 \
        --out replay-ranks.jsonl

For each missed region in the bucket whose repo checkout still exists, rerun
the session's recorded repo-wide queries with the campaign flags but a deeper
k, and record the rank at which the gold file first appears. Read the result
as three bands: 6-10 is a display/fold problem, 11-30 a rerank/fusion
problem, >k a vocabulary problem no display change reaches. Rank is
k-dependent (candidate pool is k*6 and MMR sees the pool), so a few percent
of regions rank <=5 here despite never surfacing at k=5 in-session — drift,
not a defect (§32.4).

Repo roots come from each cell's trace envelope (root_canonical), so this
needs no side files beyond the misswhy --json output. Checkouts are LRU'd by
the campaign; whatever survives is a convenience sample and should be
reported as one. Indexes are built into a throwaway cache and wiped per repo
— mind SEMGREP_CACHE_DIR if you point this at a live one.
"""
import argparse
import collections
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"
SG = HERE.parent.parent / "target" / "release" / "sg"
# Matches both hit-line forms: grep-style `path:41:text` (exact mode and
# old captures) and the §34 unit-view header `path:41-58` (ranked mode
# since the unit view shipped). group(1)/group(2) mean the same in both.
PATHLINE = re.compile(r"^([^\s:][^:]*):(\d+)(?::|-\d+$)", re.M)


def repo_root(run_id, iid, arm):
    t = DATA / "runs" / run_id / iid / arm / "trace.jsonl"
    if not t.exists():
        return None
    for line in open(t):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        rc = (e.get("input") or {}).get("root_canonical")
        if rc:
            return rc
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misswhy", required=True, help="misswhy.py --json output")
    ap.add_argument("--run-id", default="s32")
    ap.add_argument("--arm", default="sub-sg")
    ap.add_argument("--bucket", default="NEVER_SURFACED_W")
    ap.add_argument("-k", type=int, default=30)
    ap.add_argument("--flags", default="--chunking function --min-score 0.42")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.misswhy)]
    rows = [r for r in rows if r["bucket"] == args.bucket and r["wide_queries"]]
    by_repo = collections.defaultdict(list)
    skipped = 0
    for r in rows:
        root = repo_root(args.run_id, r["instance_id"], args.arm)
        if root and pathlib.Path(root).is_dir():
            by_repo[root].append(r)
        else:
            skipped += 1
    print(f"{sum(len(v) for v in by_repo.values())} regions replayable "
          f"across {len(by_repo)} surviving checkouts; {skipped} skipped "
          f"(checkout gone) — a convenience sample, report it as one")

    cache = pathlib.Path(tempfile.mkdtemp(prefix="replayrank-"))
    out = open(args.out, "w")
    n = 0
    try:
        for repo, rrows in sorted(by_repo.items()):
            env = dict(os.environ, SEMGREP_CACHE_DIR=str(cache))
            qcache = {}
            for q in sorted({q for r in rrows for q in r["wide_queries"]}):
                p = subprocess.run([str(SG), q, *args.flags.split(),
                                    "-k", str(args.k)],
                                   cwd=repo, env=env, capture_output=True,
                                   text=True, timeout=600)
                groups = []
                for block in p.stdout.split("\n\n"):
                    files = {m.group(1) for m in PATHLINE.finditer(block)}
                    if files:
                        groups.append(files)
                qcache[q] = (groups, p.returncode)
            for r in rrows:
                gp = r["region"][0]
                per_q = []
                for q in r["wide_queries"]:
                    groups, rc = qcache[q]
                    rank = next((i + 1 for i, fs in enumerate(groups)
                                 if gp in fs), None)
                    per_q.append({"query": q, "rank": rank, "rc": rc})
                ranks = [x["rank"] for x in per_q if x["rank"]]
                out.write(json.dumps({
                    "instance_id": r["instance_id"], "region": r["region"],
                    "best_rank": min(ranks) if ranks else None,
                    "per_query": per_q}) + "\n")
                n += 1
            for entry in cache.iterdir():
                shutil.rmtree(entry, ignore_errors=True)
            print(f"  {pathlib.Path(repo).name[:60]}: {len(rrows)} regions",
                  flush=True)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
        out.close()
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
