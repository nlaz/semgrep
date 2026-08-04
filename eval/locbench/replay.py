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
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import leakage  # noqa: E402  (identifier predicate, shared with run_eval)
import run as locbench  # noqa: E402  (ensure_worktree, ensure_index, SEMGREP)

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


DEFAULT_TOOLS = ("semgrep", "search", "sg")


def harvest(runs_dir, want_exact, tools=DEFAULT_TOOLS):
    """Every non-blocked search, deduped, as dicts carrying their provenance.

    `tools` defaults to the semgrep-shaped invocations and EXCLUDES `rg`.
    That matters more than it looks. The shim logs hold 2,006 semgrep, 570 rg
    and 163 search calls, and the rg ones are regexes — `csrf|CSRF|X-CSRF|wtf`,
    `def persist`, `requests\\.`. Feeding a regex to a ranked engine measures
    how BM25 tokenizes punctuation, not how the engine answers a query. The
    original filter took all three, so the first replay number would have been
    a fifth regex by construction. Widen with --tools when that is the question.

    `condition` is recorded too: the logs span several engine A/B conditions,
    and a query typed under one is not a random draw from the others.
    """
    seen, out = set(), []
    for log in Path(runs_dir).rglob("shim_log.jsonl"):
        condition = log.parent.name
        instance = log.parent.parent.name
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            tool = e.get("tool")
            if e.get("blocked") or tool not in tools:
                continue
            is_exact, q, _ = parse_argv(e.get("argv") or [])
            if not q or (is_exact and not want_exact) or (not is_exact and want_exact):
                continue
            key = (instance, q, is_exact)
            if key in seen:
                continue
            seen.add(key)
            out.append({"instance_id": instance, "query": q, "exact": is_exact,
                        "tool": tool, "condition": condition})
    return out


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def gold_files(instance):
    return sorted({f.split(":", 1)[0] for f in instance.get("edit_functions") or []})


def rank_of_gold(hits, golds):
    """1-based rank of the first hit landing in a gold file, else None.

    Paths compare by equality. semgrep emits corpus-relative paths, the same
    shape as the file part of `edit_functions`, so equality is exactly right.

    This used to fall back to a suffix match when equality failed — either
    path being a suffix of the other — which credited a hit on
    `tests/test_a.py` as having found `src/tests/test_a.py`, a different file.
    The clause was instrumented rather than argued about and it fired **0
    times in 2,217 scored queries** (497 ranked x 3 conditions, 726 exact),
    so it was over-crediting that never got the chance. Removed.
    """
    gset = set(golds)
    for i, h in enumerate(hits, 1):
        if h.get("path", "") in gset:
            return i
    return None


def run_query(tree, query, flags, k, is_exact, cache_dir=None):
    cmd = [str(locbench.SEMGREP), "--json", "-k", str(k)]
    if is_exact:
        cmd += ["-e"]
    cmd += flags + [query, str(tree)]
    env = dict(os.environ)
    if cache_dir is not None:
        # CLAUDE.md: "tests and the eval harness isolate it". Without this the
        # replay reads and writes the developer's own ~/.cache/semgrep, so
        # condition A can be answered from an entry condition B wrote — the
        # cross-contamination FIXES.md #10 was about, with the same shape.
        env["SEMGREP_CACHE_DIR"] = str(cache_dir)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return None
    hits = []
    for line in p.stdout.splitlines():
        try:
            hits.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return hits


# Flags that change what gets INDEXED rather than how a query is answered.
# Conditions here share one index per worktree, so an index flag would have
# every condition silently measured against whichever index was built first.
#
# `--embed-preproc` belongs here even though it is spelled as a search flag:
# the warm path renders the query with the *index's* stored `meta.embed_preproc`
# and ignores the flag entirely (search/indexed.rs). So a condition passing
# `--embed-preproc split` against an index built `none` is a silent no-op that
# reports parity — a wrong number with nothing to distinguish it from a real
# one. It cost §14.5 its gate: that section still asks for a replay this file
# cannot run. Compare index builds with eval/locbench/guessplay.py, which
# reindexes per config.
INDEX_FLAGS = {"--sif", "--sif-a", "--window", "--overlap", "--dims",
               "--embed-preproc"}


def check_conditions(conds):
    for name, flags in conds:
        bad = sorted(INDEX_FLAGS.intersection(flags))
        if bad:
            raise SystemExit(
                f"condition {name!r} uses index-affecting flag(s) {bad}. Replay "
                f"builds one index per worktree and shares it across conditions, "
                f"so this would measure every condition against whichever index "
                f"was built first. Use eval/levers.sh for index variants.")


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
    ap.add_argument("--tools", default=",".join(DEFAULT_TOOLS),
                    help="which logged tools to replay; 'rg' contributes regex "
                         "patterns, not queries, and is excluded by default")
    ap.add_argument("--dump-queries", type=Path, default=None,
                    help="write the harvested query distribution and exit "
                         "without replaying anything")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="SEMGREP_CACHE_DIR for the replay (default: a temp dir)")
    ap.add_argument("--out", type=Path, default=DATA / "replay.jsonl")
    args = ap.parse_args()

    conds = []
    for spec in args.conditions.split(","):
        name, _, flags = spec.partition(":")
        conds.append((name, flags.split()))
    check_conditions(conds)

    ds = {}
    for line in args.dataset.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ds[r["instance_id"]] = r

    tools = tuple(t for t in args.tools.split(",") if t)
    queries = harvest(args.runs, args.exact, tools)
    if args.dump_queries:
        dump_queries(queries, args.dump_queries)
        return
    queries = [q for q in queries if q["instance_id"] in ds]
    if args.limit:
        queries = queries[: args.limit]
    by_instance = defaultdict(list)
    for q in queries:
        by_instance[q["instance_id"]].append(q)
    print(f"{len(queries)} unique {'exact' if args.exact else 'ranked'} queries "
          f"across {len(by_instance)} instances; tools={','.join(tools)}; "
          f"conditions: {', '.join(n for n, _ in conds)}")

    tmp = None
    cache_dir = args.cache_dir
    if cache_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="semgrep-replay-cache-")
        cache_dir = Path(tmp.name)
    print(f"SEMGREP_CACHE_DIR={cache_dir}")

    rows, done, skipped = [], 0, 0
    try:
        for inst_id, qs in sorted(by_instance.items()):
            inst = ds[inst_id]
            golds = gold_files(inst)
            if not golds:
                continue
            try:
                # ensure_worktree returns (tree, meta DICT); ensure_index wants
                # the meta PATH, which run.py derives the same way at :475.
                tree, _meta = locbench.ensure_worktree(inst["repo"], inst["base_commit"])
                meta_path = (locbench.DATA / "repos" / "meta" /
                             f"{locbench.repo_key(inst['repo'], inst['base_commit'])}.json")
                # Build the index ONCE per worktree, before any condition runs.
                # Without this the first condition pays a cold streaming search
                # and later ones are answered warm, so the conditions differ by
                # index state as well as by their flags.
                locbench.ensure_index(tree, meta_path)
            except Exception as e:  # noqa: BLE001 — one bad instance shouldn't kill the run
                # Report what actually happened. An earlier version printed
                # "no mirror on disk" for every failure, which sent the first
                # run chasing a disk problem that did not exist.
                print(f"  skip {inst_id}: {type(e).__name__}: {e}")
                skipped += 1
                continue
            for q in qs:
                row = {"instance_id": inst_id, "query": q["query"],
                       "exact": q["exact"], "tool": q["tool"],
                       "condition_seen": q["condition"],
                       "n_gold": len(golds), "ranks": {}}
                for name, flags in conds:
                    hits = run_query(tree, q["query"], flags, args.k, q["exact"], cache_dir)
                    row["ranks"][name] = (None if hits is None else
                                          rank_of_gold(hits, golds))
                rows.append(row)
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(queries)} queries")
            if not args.keep_worktrees:
                locbench.remove_worktree(inst["repo"], inst["base_commit"])
    finally:
        if tmp is not None:
            tmp.cleanup()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if skipped:
        print(f"\nNOTE: {skipped}/{len(by_instance)} instances skipped (see the "
              f"reasons above) — the numbers below cover "
              f"{len(by_instance) - skipped} instances")
    report(rows, [n for n, _ in conds], args.k)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


def dump_queries(queries, out):
    """Write the real query distribution, and characterise it honestly."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(q) for q in queries) + "\n")

    words = sorted(len(q["query"].split()) for q in queries)
    n = len(words)
    idents = sum(1 for q in queries if leakage.identifiers(
        leakage.content_tokens(q["query"])))
    per_inst = Counter(q["instance_id"] for q in queries)
    top = per_inst.most_common(1)[0] if per_inst else ("-", 0)
    print(f"wrote {out} ({n} queries)")
    print(f"  instances        {len(per_inst)}")
    print(f"  words   median   {words[n // 2] if n else 0}")
    print(f"          mean     {sum(words) / n:.1f}" if n else "")
    print(f"          p90      {words[int(n * 0.9)] if n else 0}")
    print(f"  carry an identifier  {idents / n:.0%}" if n else "")
    print(f"  most queries from one instance: {top[1]} ({top[1] / n:.0%}) — {top[0]}")
    print("  NOTE: queries cluster by instance; any CI over them must resample")
    print("        INSTANCES, not queries, or it will be far too tight.")


def _mrr_scores(rows, name):
    return [1 / x if x else 0.0 for x in (r["ranks"].get(name) for r in rows)]


def bootstrap_ci(rows, a, b, clustered=True, n_resamples=2000, seed=1):
    """95% CI on the MRR delta, resampling INSTANCES rather than queries.

    Queries are not independent draws. They cluster hard by instance: one
    instance contributed 99 of 887 harvested queries (11%), and the median
    instance contributed 14. An agent that phrases things one way phrases
    them that way a dozen times in a row, against one repo, for one gold set.

    Resampling queries treats those 99 as 99 independent observations and
    reports a CI far tighter than the data supports — which is exactly how a
    replay result becomes a confidently wrong quotable number. Resampling
    instances (and taking all of their queries) respects the clustering.

    Both are computed so the inflation is visible rather than asserted.
    """
    rng = random.Random(seed)
    sa, sb = _mrr_scores(rows, a), _mrr_scores(rows, b)
    n = len(sa)
    if not n:
        return 0.0, 0.0, 0.0, 0
    point = sum(sa) / n - sum(sb) / n

    if clustered:
        idx_by_inst = defaultdict(list)
        for i, r in enumerate(rows):
            idx_by_inst[r["instance_id"]].append(i)
        clusters = list(idx_by_inst.values())
    else:
        clusters = [[i] for i in range(n)]

    deltas = []
    for _ in range(n_resamples):
        idx = []
        for _ in range(len(clusters)):
            idx.extend(clusters[rng.randrange(len(clusters))])
        m = len(idx)
        deltas.append(sum(sa[j] for j in idx) / m - sum(sb[j] for j in idx) / m)
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[min(int(0.975 * n_resamples), n_resamples - 1)]
    return point, lo, hi, len(clusters)


def report(rows, names, k):
    if not rows:
        print("no rows")
        return
    n_inst = len({r["instance_id"] for r in rows})
    print(f"\n{len(rows)} queries across {n_inst} instances "
          f"(queries cluster by instance; see the CI note below)")
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
        print("\npaired comparisons (only disagreements carry signal):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                print(f"\n  {a} vs {b}")
                for cut in (1, 5, k):
                    hit = lambda x: bool(x and x <= cut)  # noqa: E731
                    aw = [r for r in rows if hit(r["ranks"].get(a)) and not hit(r["ranks"].get(b))]
                    bw = [r for r in rows if hit(r["ranks"].get(b)) and not hit(r["ranks"].get(a))]
                    n = len(aw) + len(bw)
                    p = (min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(len(aw), len(bw)) + 1)) / 2 ** n)
                         if n else 1.0)
                    # The sign test assumes independent trials and the queries
                    # are not; report how many INSTANCES the disagreements come
                    # from, which is the honest denominator.
                    d_inst = len({r["instance_id"] for r in aw + bw})
                    verdict = "significant" if p < 0.05 else ""
                    print(f"    hit@{cut:<3} {len(aw):>4} - {len(bw):<4} "
                          f"discordant={n:<4} over {d_inst:>3} instances  "
                          f"p={p:.4f} {verdict}")
                    if n and d_inst < n / 3:
                        print(f"           ^ {n} disagreements from only {d_inst} "
                              f"instances — p is optimistic")
                # Bootstrap CI on the MRR delta — uses every query, not just
                # the discordant ones, so it sees rank shifts a sign test misses.
                pt, lo, hi, n_cl = bootstrap_ci(rows, a, b, clustered=True)
                _, nlo, nhi, _ = bootstrap_ci(rows, a, b, clustered=False)
                call = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "inconclusive")
                print(f"    MRR delta {pt:+.4f}")
                print(f"      clustered 95% CI [{lo:+.4f}, {hi:+.4f}] "
                      f"over {n_cl} instances  -> {call}")
                print(f"      naive     95% CI [{nlo:+.4f}, {nhi:+.4f}] "
                      f"(treats {len(rows)} queries as independent; too tight)")


if __name__ == "__main__":
    main()
