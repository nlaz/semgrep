#!/usr/bin/env python3
"""Calibrate the §29.2 score floor from replayed real agent queries.

For every harvested ranked query (kind=guess_ranked) in the corpus, run it
against its instance's worktree with the current binary and record the pair
the floor decision needs:

    top_score   the best hit's fine cosine (§29.1 — cross-query comparable)
    gold_rank   where the first gold file landed, None for a miss

The floor is then read off the joint distribution: the largest threshold
whose false-floor rate on gold-hitting queries stays under --false-floor
(default 2%), reported with the true-negative rate it buys — how many
gold-missing queries it would have turned into an honest "no matches".

    python3 eval/locbench/floorcal.py --limit-instances 3     # smoke
    python3 eval/locbench/floorcal.py                         # full
    python3 eval/locbench/floorcal.py --report                # analyze only

Scope policy is the invocation's own (orig), because that is what the floor
will see in production: a floor calibrated repo-wide would be tested against
scoped score distributions it was never fit to.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run as locbench  # noqa: E402
from replay import gold_files  # noqa: E402

DATA = HERE.parent / "data" / "locbench"


def rank_of_gold_hits(hits, golds, scope_rel):
    for i, h in enumerate(hits, 1):
        p = h["path"]
        if scope_rel not in (None, ".") and not p.startswith(scope_rel):
            p = f"{scope_rel.rstrip('/')}/{p}"
        if p in golds:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=HERE.parent / "queries" / "guesses-v1-descv9.jsonl")
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "floorcal.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit-instances", type=int, default=0)
    ap.add_argument("--false-floor", type=float, default=0.02,
                    help="max share of gold-hitting queries the floor may refuse")
    ap.add_argument("--report", action="store_true", help="analyze --out and exit")
    args = ap.parse_args()

    if args.report:
        return report(args)

    ds = {json.loads(l)["instance_id"]: json.loads(l)
          for l in open(args.dataset) if l.strip()}
    rows = [json.loads(l) for l in open(args.corpus) if l.strip()]
    ranked = [r for r in rows if r.get("kind") == "guess_ranked"]
    by_instance = defaultdict(list)
    for r in ranked:
        if r["instance_id"] in ds:
            by_instance[r["instance_id"]].append(r)
    instances = sorted(by_instance)
    if args.limit_instances:
        instances = instances[: args.limit_instances]

    done = set()
    if args.out.exists():
        done = {json.loads(l)["gid"] for l in open(args.out) if l.strip()}
    out_f = open(args.out, "a")
    tmp = tempfile.TemporaryDirectory(prefix="semgrep-floorcal-cache-")

    n = 0
    for i, inst_id in enumerate(instances, 1):
        inst = ds[inst_id]
        golds = gold_files(inst)
        if not golds:
            continue
        todo = [r for r in by_instance[inst_id]
                if f"{r['run_id']}/{r['instance_id']}/{r['condition']}/{r['seq']}"
                not in done]
        if not todo:
            continue
        try:
            tree, _ = locbench.ensure_worktree(inst["repo"], inst["base_commit"])
        except Exception as e:  # noqa: BLE001
            print(f"  skip {inst_id}: {type(e).__name__}: {e}")
            continue
        subprocess.run([str(locbench.SEMGREP), "index", str(tree)],
                       check=True, capture_output=True, timeout=600)
        for r in todo:
            scope_rel = (r.get("scopes_rel") or ["."])[0] or "."
            path = tree if scope_rel in (None, ".") else tree / scope_rel
            if not Path(path).exists():
                path, scope_rel = tree, "."
            cmd = [str(locbench.SEMGREP), "--json", "-k", str(args.k),
                   r["pattern"], str(path)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                   env={"SEMGREP_CACHE_DIR": tmp.name,
                                        "PATH": "/usr/bin:/bin"})
                hits = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
            except Exception:  # noqa: BLE001
                continue
            row = {
                "gid": f"{r['run_id']}/{r['instance_id']}/{r['condition']}/{r['seq']}",
                "instance_id": inst_id,
                "query": r["pattern"],
                "scope": scope_rel,
                "top_score": hits[0]["score"] if hits else None,
                "gold_rank": rank_of_gold_hits(hits, golds, scope_rel),
                "n_hits": len(hits),
            }
            out_f.write(json.dumps(row) + "\n")
            n += 1
        out_f.flush()
        print(f"  [{i}/{len(instances)}] {inst_id}: {len(todo)} queries")
    print(f"done: {n} new rows in {args.out}")
    return 0


def report(args):
    rows = [json.loads(l) for l in open(args.out) if l.strip()]
    scored = [r for r in rows if r["top_score"] is not None]
    hit = sorted(r["top_score"] for r in scored if r["gold_rank"] is not None)
    miss = sorted(r["top_score"] for r in scored if r["gold_rank"] is None)
    if not hit or not miss:
        print(f"n={len(scored)}: need both hits ({len(hit)}) and misses "
              f"({len(miss)}) to place a floor")
        return 1
    print(f"n={len(scored)} replayed queries: {len(hit)} gold-hitting, "
          f"{len(miss)} gold-missing")
    print(f"gold-hitting top-1 score:  p5={pct(hit, 5):.3f} p25={pct(hit, 25):.3f} "
          f"median={pct(hit, 50):.3f}")
    print(f"gold-missing top-1 score:  p50={pct(miss, 50):.3f} p75={pct(miss, 75):.3f} "
          f"p95={pct(miss, 95):.3f}")
    # The floor: largest threshold refusing <= false_floor of gold-hitting
    # queries. Swept over the observed hit scores, so it lands on a real
    # boundary rather than a round number with hidden slack.
    best = None
    for t in hit + miss:
        ff = sum(1 for s in hit if s < t) / len(hit)
        if ff <= args.false_floor:
            tn = sum(1 for s in miss if s < t) / len(miss)
            if best is None or t > best[0]:
                best = (t, ff, tn)
    t, ff, tn = best
    print(f"\nfloor at {t:.3f}: refuses {ff:.1%} of gold-hitting queries "
          f"(bound {args.false_floor:.0%}), turns {tn:.1%} of gold-missing "
          f"queries into an honest 'no matches'")
    print("Ship it with --min-score only if the true-negative rate is worth "
          "the false floors; a floor nobody hits is decoration.")
    return 0


def pct(sorted_vals, p):
    i = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[i]


if __name__ == "__main__":
    sys.exit(main())
