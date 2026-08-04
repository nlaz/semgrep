#!/usr/bin/env python3
"""Build an equal-strata instance list by issue-naming tier (§19.5's frame).

The dataset is 62% `named` / 26% `partial` / 12% `blind`, and §19.2b's
mechanism predicts the entire description effect lives in `blind`: a query that
shares no vocabulary with the gold finds it 13% of the time as a description
and 50% as a name, while a query that already carries the gold's vocabulary is
indifferent. A random 560-instance frame therefore spends 62% of its money
where the predicted effect is zero and still yields only 68 blind pairs.

Equal strata buy the *same* 68 blind pairs for a third of the cost. What that
costs is pooled comparability: a pooled number over this frame sits on a harder
population than §16.9/§18 and must not be quoted beside them. Report per
stratum, and reweight to the true 348/144/68 population when a comparable
pooled figure is wanted.

    python3 eval/locbench/tierframe.py                  # 68/68/68, seed 1
    python3 eval/locbench/tierframe.py --per-tier 40    # a smaller frame
    python3 eval/locbench/tierframe.py --seed 2         # an independent draw

stdout is the comma-separated id list, for `--instances` / `INSTANCES=`.
The tier counts go to stderr so `INSTANCES=$(tierframe.py)` stays clean while a
miscount is still visible.

`run.py`'s `stratified_sample` cannot do this: it stratifies by *category* (Bug
Report, Feature Request…), not by naming tier. The sampling discipline is
copied from it deliberately, including the reason — shuffle within each repo,
then shuffle the repo *order*, then round-robin. Sorting after a shuffle is
what made seeds 1 and 2 share 37 of 40 instances in §18.6, because Python's
sort is stable and quietly undid the shuffle. `--seed` here does move the
sample — `--check-seeds` measures by how much, per stratum and against the
overlap two independent draws would give.

Read that output before claiming an independent replication. Overlap runs
around twice chance (partial 75% vs 47%, named 40% vs 20% at the default
frame) because round-robin takes one instance from each repo before a second
from any, so the repo set dominates the draw and a single-instance repo
contributes the same id whenever it is picked. That is far from §18.6's 37-of-40,
but it is not an independent sample either, and the blind stratum — the one
§19.5 cares about — is taken whole and cannot vary at all.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"
TIERS = ("blind", "partial", "named")


def interleaved_by_repo(instances, repo_of, rng):
    """Spread a stratum across repos so one repo cannot dominate it.

    Same shape as run.py's stratified_sample: shuffle within repo, shuffle the
    repo order, round-robin. Taking a prefix of the result is then a sample
    that is both seed-driven and repo-spread.
    """
    by_repo = defaultdict(list)
    for i in instances:
        by_repo[repo_of.get(i, "?")].append(i)
    for group in by_repo.values():
        group.sort()          # sort first so the shuffle, not dict/JSON order,
        rng.shuffle(group)    # is what the seed drives
    repos = sorted(by_repo)
    rng.shuffle(repos)
    out, i = [], 0
    while len(out) < len(instances):
        for repo in repos:
            if i < len(by_repo[repo]):
                out.append(by_repo[repo][i])
        i += 1
    return out


def build(tiers_path, dataset_path, per_tier, seed):
    tiers = json.loads(Path(tiers_path).read_text())
    repo_of = {}
    for line in Path(dataset_path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            repo_of[d["instance_id"]] = d.get("repo", "?")

    by_tier = defaultdict(list)
    for iid, meta in tiers.items():
        by_tier[meta.get("tier", "?")].append(iid)

    rng = random.Random(seed)
    picked, counts = [], {}
    for tier in TIERS:
        pool = by_tier.get(tier, [])
        # Ordered before truncation, so a stratum smaller than per_tier is
        # taken whole rather than silently sampled from a sorted prefix.
        ordered = interleaved_by_repo(pool, repo_of, rng)
        take = ordered[:per_tier] if per_tier else ordered
        counts[tier] = (len(take), len(pool))
        picked.extend(take)
    return picked, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=Path, default=DATA / "blind-instances.json")
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset.jsonl")
    ap.add_argument("--per-tier", type=int, default=68,
                    help="instances per tier; 0 takes every tier whole")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check-seeds", action="store_true",
                    help="report overlap between seed and seed+1, then exit")
    args = ap.parse_args()

    picked, counts = build(args.tiers, args.dataset, args.per_tier, args.seed)

    if args.check_seeds:
        other, _ = build(args.tiers, args.dataset, args.per_tier, args.seed + 1)
        tiers = json.loads(Path(args.tiers).read_text())
        tier_of = {i: m.get("tier", "?") for i, m in tiers.items()}
        print(f"seed {args.seed} vs {args.seed + 1}")
        # Per stratum, because a whole stratum is 100% shared by construction
        # and pooling it into one figure hides whether the seed did anything.
        # Reported against the overlap two *independent* draws would give
        # (k²/N), so "high" means high relative to chance rather than to 0.
        samplable = 0
        for tier in TIERS:
            took, pool = counts[tier]
            a = {i for i in picked if tier_of.get(i) == tier}
            b = {i for i in other if tier_of.get(i) == tier}
            shared = len(a & b)
            if took >= pool:
                print(f"  {tier:<8} {shared:>3}/{took:<3} whole stratum — "
                      f"shared by construction, not a seed failure")
                continue
            samplable += took
            chance = took * took / pool
            print(f"  {tier:<8} {shared:>3}/{took:<3} shared "
                  f"({100 * shared / took:>3.0f}%), chance ≈ {chance:.0f} "
                  f"({100 * chance / took:.0f}%)")
        if samplable == 0:
            print("  every stratum taken whole — --seed cannot move this frame")
        return 0

    for tier in TIERS:
        took, pool = counts[tier]
        note = "  (whole stratum)" if took == pool else ""
        print(f"{tier:<8} {took:>4} of {pool:>4}{note}", file=sys.stderr)
    print(f"{'total':<8} {len(picked):>4}", file=sys.stderr)
    print(",".join(picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
