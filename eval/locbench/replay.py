#!/usr/bin/env python3
"""Replay the queries agents actually issued, offline, through arbitrary
engine conditions.

Why this exists (RESEARCH.md §11.5): the agent eval cannot resolve engine
differences below ~7pp, because 80-87% of its instances are decided by
something other than the search tool and only ~8% of instance pairs ever
disagree. Every agent search, though, was logged with its full argv. Those
queries are a far better sample than either the LLM-generated query sets
(which are guesses about what an agent would type) or the instances
themselves (which mostly do not discriminate):

  * deterministic — no agent stochasticity, so a delta is the engine
  * free — no model calls
  * ~5x the sample size of the run that produced them
  * the real query distribution, including the bad queries agents write

The metric is rank-of-first-gold-file: where in the ranked list does a file
the real fix touched first appear? That is the engine's actual job, measured
without the agent's decisions on top of it.

Usage:
  python3 eval/locbench/replay.py --conditions "plain:,maxsim:--maxsim"
  python3 eval/locbench/replay.py --conditions "plain:" --exact --limit 50
"""

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run as locbench  # noqa: E402  (ensure_worktree, SEMGREP, DATA)

DATA = HERE.parent / "data" / "locbench"


# ---------------------------------------------------------------------------
# harvest queries from the shim logs
# ---------------------------------------------------------------------------

def parse_argv(argv):
    """Split a logged semgrep argv into (is_exact, query, scopes).

    Flags that take a value are skipped with their value so a `-k 20` does not
    get mistaken for the query. The first bare token is the query; the rest
    are paths.
    """
    VALUED = {"-k", "-C", "--mode", "--sem-weight", "--mmr-lambda", "--window",
              "--overlap", "--prf", "--maxsim-pool", "--maxsim-blend", "-A", "-B"}
    is_exact = False
    query, scopes = None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-e", "--exact"):
            is_exact = True
        elif a in VALUED:
            i += 1
        elif a.startswith("-"):
            pass
        elif query is None:
            query = a
        else:
            scopes.append(a)
        i += 1
    return is_exact, query, scopes


def harvest(runs_dir, want_exact):
    """(instance_id, query, is_exact) for every non-blocked search, deduped."""
    seen, out = set(), []
    for log in Path(runs_dir).rglob("shim_log.jsonl"):
        instance = log.parent.parent.name
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("blocked") or e.get("tool") not in ("semgrep", "rg", "search"):
                continue
            is_exact, q, _ = parse_argv(e.get("argv") or [])
            if not q or (is_exact and not want_exact) or (not is_exact and want_exact):
                continue
            key = (instance, q, is_exact)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def gold_files(instance):
    return sorted({f.split(":", 1)[0] for f in instance.get("edit_functions") or []})


def rank_of_gold(hits, golds):
    """1-based rank of the first hit landing in a gold file, else None."""
    gset = set(golds)
    for i, h in enumerate(hits, 1):
        p = h.get("path", "")
        if p in gset or any(p.endswith("/" + g) or g.endswith("/" + p) for g in gset):
            return i
    return None


def run_query(tree, query, flags, k, is_exact):
    cmd = [str(locbench.SEMGREP), "--json", "-k", str(k)]
    if is_exact:
        cmd += ["-e"]
    cmd += flags + [query, str(tree)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    hits = []
    for line in p.stdout.splitlines():
        try:
            hits.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="plain:",
                    help="comma-separated name:flags, e.g. 'plain:,mx:--maxsim'")
    ap.add_argument("--runs", type=Path, default=DATA / "runs")
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="max queries (0 = all)")
    ap.add_argument("--exact", action="store_true",
                    help="replay -e exact queries instead of ranked ones")
    ap.add_argument("--keep-worktrees", action="store_true")
    ap.add_argument("--out", type=Path, default=DATA / "replay.jsonl")
    args = ap.parse_args()

    conds = []
    for spec in args.conditions.split(","):
        name, _, flags = spec.partition(":")
        conds.append((name, flags.split()))

    ds = {}
    for line in args.dataset.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ds[r["instance_id"]] = r

    queries = harvest(args.runs, args.exact)
    queries = [q for q in queries if q[0] in ds]
    if args.limit:
        queries = queries[: args.limit]
    by_instance = defaultdict(list)
    for inst, q, ex in queries:
        by_instance[inst].append((q, ex))
    print(f"{len(queries)} unique {'exact' if args.exact else 'ranked'} queries "
          f"across {len(by_instance)} instances; conditions: "
          f"{', '.join(n for n, _ in conds)}")

    rows, done = [], 0
    for inst_id, qs in sorted(by_instance.items()):
        inst = ds[inst_id]
        golds = gold_files(inst)
        if not golds:
            continue
        try:
            tree, _ = locbench.ensure_worktree(inst["repo"], inst["base_commit"])
        except Exception as e:  # noqa: BLE001 — a missing mirror shouldn't kill the run
            print(f"  skip {inst_id}: {e}")
            continue
        for q, is_exact in qs:
            row = {"instance_id": inst_id, "query": q, "exact": is_exact,
                   "n_gold": len(golds), "ranks": {}}
            for name, flags in conds:
                hits = run_query(tree, q, flags, args.k, is_exact)
                row["ranks"][name] = None if hits is None else rank_of_gold(hits, golds)
            rows.append(row)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(queries)} queries")
        if not args.keep_worktrees:
            locbench.remove_worktree(inst["repo"], inst["base_commit"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report(rows, [n for n, _ in conds], args.k)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


def report(rows, names, k):
    if not rows:
        print("no rows")
        return
    print(f"\n{'condition':<14} {'n':>5} {'hit@1':>7} {'hit@5':>7} "
          f"{f'hit@{k}':>7} {'MRR':>7}")
    for n in names:
        rk = [r["ranks"].get(n) for r in rows]
        tot = len(rk)
        h1 = sum(1 for x in rk if x == 1) / tot
        h5 = sum(1 for x in rk if x and x <= 5) / tot
        hk = sum(1 for x in rk if x) / tot
        mrr = sum(1 / x for x in rk if x) / tot
        print(f"{n:<14} {tot:>5} {h1:>7.3f} {h5:>7.3f} {hk:>7.3f} {mrr:>7.3f}")

    # Paired analysis. Only queries where the conditions disagree carry
    # information, and *at which cutoff* they disagree matters: two engines
    # can both find the gold file within k while ranking it very differently,
    # which is exactly the difference an agent feels.
    if len(names) > 1:
        import math
        import random
        print("\npaired comparisons (only disagreements carry signal):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                print(f"\n  {a} vs {b}")
                for cut in (1, 5, k):
                    hit = lambda x: bool(x and x <= cut)  # noqa: E731
                    aw = sum(1 for r in rows if hit(r["ranks"].get(a)) and not hit(r["ranks"].get(b)))
                    bw = sum(1 for r in rows if hit(r["ranks"].get(b)) and not hit(r["ranks"].get(a)))
                    n = aw + bw
                    p = (min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(aw, bw) + 1)) / 2 ** n)
                         if n else 1.0)
                    verdict = "significant" if p < 0.05 else ""
                    print(f"    hit@{cut:<3} {aw:>4} - {bw:<4} discordant={n:<4} p={p:.4f} {verdict}")
                # Bootstrap CI on the MRR delta — uses every query, not just
                # the discordant ones, so it sees rank shifts a sign test misses.
                ra = [1 / x if x else 0.0 for x in (r["ranks"].get(a) for r in rows)]
                rb = [1 / x if x else 0.0 for x in (r["ranks"].get(b) for r in rows)]
                n = len(ra)
                point = sum(ra) / n - sum(rb) / n
                rng = random.Random(1)
                ds = []
                for _ in range(2000):
                    idx = [rng.randrange(n) for _ in range(n)]
                    ds.append(sum(ra[j] for j in idx) / n - sum(rb[j] for j in idx) / n)
                ds.sort()
                lo, hi = ds[50], ds[1949]
                call = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "inconclusive")
                print(f"    MRR delta {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {call}")


if __name__ == "__main__":
    main()
