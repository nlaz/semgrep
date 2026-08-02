#!/usr/bin/env python3
"""Translate-and-compare: the §16 product-hypothesis test.

For every guess-group in the corpus (one exact/rg invocation's ladder; a
bare guess is a 1-rung ladder), replay three arms against the instance's
gold files:

  exact       the agent's actual pattern, verbatim, through `semgrep -e`
  exact-norm  dead `\\|` ladders re-run with `|` — what the agent MEANT
  t1 / t2     the ranked translations, under --mode bm25|semantic|hybrid

plus the agents' real ranked queries (kind=guess_ranked) re-scored under the
same build as the reference distribution. Scope policy: `orig` uses the
invocation's own first scope (93% of calls are scoped); `root` is the
sensitivity cut. Config `champion` rebuilds the worktree index with the §14
flags before its pass — measured because §15.9-B says SIF is mis-tuned for
token-poor queries, and guesses are token-poor.

    python3 eval/locbench/guessplay.py --limit-instances 3      # smoke
    python3 eval/locbench/guessplay.py                          # full

Output: one row per (group, arm, scope) appended to
eval/data/locbench/guessplay.jsonl (checkpointing: existing keys skipped);
convert to a results-board cell file with --emit-results.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ladder  # noqa: E402
import run as locbench  # noqa: E402
from replay import gold_files, rank_of_gold  # noqa: E402

DATA = HERE.parent / "data" / "locbench"
QUERIES = HERE.parent / "queries" / "guesses-v0.jsonl"
CHAMPION_INDEX_FLAGS = ["--embed-preproc", "split", "--sif"]
MODES = ("bm25", "semantic", "hybrid")


def gid(row):
    return f"{row['run_id']}/{row['instance_id']}/{row['condition']}/{row['seq']}"


def run_semgrep(tree, scope_rel, query, k, is_exact, mode, cache_dir):
    path = tree if scope_rel in (None, ".") else tree / scope_rel
    cmd = [str(locbench.SEMGREP), "--json", "-k", str(k)]
    if is_exact:
        cmd.append("-e")
    else:
        cmd += ["--mode", mode]
    cmd += [query, str(path)]
    env = dict(os.environ)
    env["SEMGREP_CACHE_DIR"] = str(cache_dir)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if p.returncode == 2:
        return None, "error"  # e.g. the agent's scope path doesn't exist
    hits = []
    for line in p.stdout.splitlines():
        try:
            hits.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return hits, None


def scope_of(row, policy):
    if policy == "root" or not row["scopes_rel"]:
        return "."
    return row["scopes_rel"][0]


def score(hits, golds, scope_rel):
    """Hits are scope-relative when scoped; golds are repo-relative."""
    if hits is None:
        return None
    if scope_rel not in (None, "."):
        prefix = scope_rel.rstrip("/") + "/"
        hits = [{**h, "path": prefix + h.get("path", "")} for h in hits]
    return rank_of_gold(hits, golds)


def arms_for(row):
    """(arm_name, query, is_exact, modes) tuples for one corpus row."""
    if row["kind"] == "guess_ranked":
        return [("ranked-own", row["pattern"], False, MODES)]
    lad = ladder.parse(row["patterns"])
    out = [("exact", row["pattern"], True, (None,))]
    if lad["engine_semantics_mismatch"]:
        out.append(("exact-norm", row["pattern"].replace("\\|", "|"), True, (None,)))
    t1 = ladder.translate_t1(lad)
    if t1:
        out.append(("t1", t1, False, MODES))
        t2 = ladder.translate_t2(lad)
        if t2 and t2 != t1.lower():
            out.append(("t2", t2, False, ("hybrid",)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=QUERIES)
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "guessplay.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--configs", default="default,champion")
    ap.add_argument("--scopes", default="orig,root")
    ap.add_argument("--limit-instances", type=int, default=0)
    ap.add_argument("--instances", default="")
    ap.add_argument("--keep-worktrees", action="store_true")
    ap.add_argument("--emit-results", type=Path, default=None,
                    help="aggregate an existing --out into a results-board "
                         "cell file and exit")
    args = ap.parse_args()

    if args.emit_results:
        emit_results(args.out, args.emit_results)
        return

    ds = {}
    for line in args.dataset.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ds[r["instance_id"]] = r

    rows = [json.loads(l) for l in args.corpus.read_text().splitlines()]
    rows = [r for r in rows if r["instance_id"] in ds]
    by_instance = defaultdict(list)
    for r in rows:
        by_instance[r["instance_id"]].append(r)
    instances = sorted(by_instance)
    if args.instances:
        want = set(args.instances.split(","))
        instances = [i for i in instances if i in want]
    if args.limit_instances:
        instances = instances[: args.limit_instances]

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                e = json.loads(line)
                done.add((e["gid"], e["arm"], e["mode"], e["config"], e["scope_policy"]))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"{len(instances)} instances, {sum(len(by_instance[i]) for i in instances)} "
          f"guess rows; {len(done)} arm-rows already done")

    configs = args.configs.split(",")
    scope_policies = args.scopes.split(",")
    tmp = tempfile.TemporaryDirectory(prefix="semgrep-guessplay-cache-")
    cache_dir = Path(tmp.name)

    out_f = open(args.out, "a")
    n_run = 0
    for n_i, inst_id in enumerate(instances, 1):
        inst = ds[inst_id]
        golds = gold_files(inst)
        if not golds:
            continue
        try:
            tree, _ = locbench.ensure_worktree(inst["repo"], inst["base_commit"])
            meta_path = (locbench.DATA / "repos" / "meta" /
                         f"{locbench.repo_key(inst['repo'], inst['base_commit'])}.json")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {inst_id}: {type(e).__name__}: {e}")
            continue
        for config in configs:
            # One index per (worktree, config); exact arms don't read it but
            # ranked arms must all see the same build within a config.
            flags = CHAMPION_INDEX_FLAGS if config == "champion" else []
            subprocess.run([str(locbench.SEMGREP), "index", str(tree), *flags],
                           capture_output=True, timeout=600)
            for row in by_instance[inst_id]:
                for policy in scope_policies:
                    if config == "champion" and policy != scope_policies[0]:
                        continue  # champion measured under primary policy only
                    sc = scope_of(row, policy)
                    for arm, query, is_exact, modes in arms_for(row):
                        if is_exact and config == "champion":
                            continue  # keyword path ignores the index
                        for mode in modes:
                            key = (gid(row), arm, mode or "-", config, policy)
                            if key in done:
                                continue
                            hits, err = run_semgrep(
                                tree, sc, query, args.k, is_exact, mode, cache_dir)
                            out_f.write(json.dumps({
                                "gid": gid(row), "instance_id": inst_id,
                                "kind": row["kind"], "condition": row["condition"],
                                "arm": arm, "mode": mode or "-", "config": config,
                                "scope_policy": policy, "scope": sc,
                                "n_rungs": ladder.parse(row["patterns"])["n_rungs"],
                                "dead": "\\|" in row["pattern"],
                                "query": query, "rank": score(hits, golds, sc),
                                "err": err,
                            }, sort_keys=True) + "\n")
                            out_f.flush()
                            n_run += 1
        if not args.keep_worktrees:
            locbench.remove_worktree(inst["repo"], inst["base_commit"])
        if n_i % 5 == 0:
            print(f"  {n_i}/{len(instances)} instances, {n_run} arm-rows", flush=True)
    out_f.close()
    print(f"done: {n_run} new arm-rows in {args.out}")


def emit_results(raw_path, out_path):
    """Aggregate guessplay rows into run_eval cell shape for the Guess board."""
    rows = [json.loads(l) for l in raw_path.read_text().splitlines()]
    cells = defaultdict(list)
    for r in rows:
        if r["scope_policy"] != "orig" or r["config"] != "default":
            continue
        mode = r["arm"] if r["arm"].startswith("exact") else f"{r['arm']}-{r['mode']}"
        cells[(mode, r["kind"])].append((r["instance_id"], r["rank"]))
    out = []
    for (mode, kind), pairs in sorted(cells.items()):
        ranks = [rk for _, rk in pairs]
        n = len(ranks)
        rec = lambda k: sum(1 for r in ranks if r and r <= k) / n
        mrr = sum(1 / r for r in ranks if r and r <= 10) / n
        out.append({
            "mode": mode, "kind": kind, "n": n,
            "recall@1": rec(1), "recall@5": rec(5), "recall@10": rec(10),
            "mrr@10": mrr,
            "ranks": ranks,
            "instances": [i for i, _ in pairs],
            "run": {"queries": "guesses-v0.jsonl", "corpus": "(locbench worktrees)",
                    "scope_policy": "orig", "config": "default"},
        })
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(out)} cells to {out_path}")


if __name__ == "__main__":
    main()
