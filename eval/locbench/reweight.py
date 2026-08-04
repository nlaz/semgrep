#!/usr/bin/env python3
"""Population-reweighted pooled delta for a stratified frame (§19.5).

§19.5 runs equal strata — 68 blind / 68 partial / 68 named — because §19.2b
predicts the entire description effect lives in `blind`, which is only 12% of
the dataset. Equal strata buy the same 68 blind pairs for a third of the price
of a random 560-instance frame.

What that costs is the pooled number. A mean over this frame is a mean over a
population that is one-third blind, where the real one is 12% blind, so it is
not comparable to §16.9/§18 and must not be quoted beside them. This script
recovers a comparable figure the standard way: compute the delta *within* each
stratum, then weight the strata by their true population shares.

    python3 eval/locbench/reweight.py --results <file> --a desc-v8 --b rg

The CI comes from a stratified bootstrap — resample instances within each
stratum, reweight, repeat — so it carries the uncertainty of the small strata
rather than pretending the weights are free. Expect it to be *wider* than the
unweighted pooled CI: the blind stratum gets down-weighted to 12% but stays the
noisiest, and honest reweighting shows that rather than hiding it.

This lives beside ab_analyze.py rather than inside it because that script is
deliberately narrow — "it computes [the pre-registered endpoints], it does not
explore" — and a reweighted pooled estimate is a different endpoint, registered
separately in §19.5.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from ab_analyze import load  # noqa: E402  (last-ok-wins, matching report.py)

DATA = HERE.parent / "data" / "locbench"
PRIMARY = "func_acc@10_tol"
CO_PRIMARY = "func_recall@10_tol"


def population_weights(tiers):
    """True share of each stratum in the full dataset — the thing being restored."""
    counts = defaultdict(int)
    for meta in tiers.values():
        counts[meta.get("tier", "?")] += 1
    total = sum(counts.values())
    return {t: n / total for t, n in counts.items()}, dict(counts)


def paired(by, a, b, metric):
    """[(instance, tier_key, a_value, b_value)] over instances both arms finished."""
    shared = sorted(set(by[a]) & set(by[b]))
    out = []
    for i in shared:
        va = by[a][i].get("metrics", {}).get(metric)
        vb = by[b][i].get("metrics", {}).get(metric)
        if va is None or vb is None:
            continue
        out.append((i, float(va), float(vb)))
    return out


def reweighted(rows_by_tier, weights):
    """Sum of within-stratum mean deltas, weighted by population share.

    Strata absent from the frame are dropped and their weight renormalised
    away, so the estimate is over the population the frame can speak to rather
    than silently treating a missing stratum as zero effect.
    """
    present = {t: v for t, v in rows_by_tier.items() if v}
    if not present:
        return None, {}
    wsum = sum(weights.get(t, 0) for t in present)
    if wsum <= 0:
        return None, {}
    est, parts = 0.0, {}
    for t, vals in present.items():
        d = sum(a - b for a, b in vals) / len(vals)
        w = weights.get(t, 0) / wsum
        parts[t] = (d, w, len(vals))
        est += w * d
    return est, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--a", required=True, help="treatment arm")
    ap.add_argument("--b", required=True, help="control arm")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--tiers", type=Path, default=DATA / "blind-instances.json")
    ap.add_argument("--metric", default=PRIMARY,
                    help=f"default {PRIMARY}; {CO_PRIMARY} is the co-primary")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tiers_raw = json.loads(args.tiers.read_text())
    tier_of = {k: v.get("tier", "?") for k, v in tiers_raw.items()}
    weights, counts = population_weights(tiers_raw)

    by = load(args.results, [args.a, args.b], args.model)
    rows = paired(by, args.a, args.b, args.metric)
    if not rows:
        print(f"no paired instances for {args.a} vs {args.b}")
        return 1

    rows_by_tier = defaultdict(list)
    for i, va, vb in rows:
        rows_by_tier[tier_of.get(i, "?")].append((va, vb))

    est, parts = reweighted(rows_by_tier, weights)
    if est is None:
        print("no stratum carries population weight; nothing to reweight")
        return 1

    rnd = random.Random(args.seed)
    boots = []
    for _ in range(args.boot):
        resampled = {}
        for t, vals in rows_by_tier.items():
            if vals:
                resampled[t] = [vals[rnd.randrange(len(vals))] for _ in range(len(vals))]
        e, _ = reweighted(resampled, weights)
        if e is not None:
            boots.append(e)
    boots.sort()
    lo = boots[int(.025 * len(boots))]
    hi = boots[int(.975 * len(boots))]

    unweighted = sum(a - b for _, a, b in rows) / len(rows)

    print(f"=== §19.5 reweighted: {args.a} (a) − {args.b} (b), {args.metric} ===")
    print(f"{len(rows)} paired instances\n")
    print(f"{'stratum':<10}{'n':>5}{'pop share':>11}{'weight':>9}{'Δ':>9}")
    for t in sorted(parts, key=lambda t: -parts[t][1]):
        d, w, n = parts[t]
        print(f"{t:<10}{n:>5}{100 * weights.get(t, 0):>10.0f}%{w:>8.2f}{d:>+9.3f}")
    print()
    print(f"  reweighted to population  Δ={est:+.3f}  CI[{lo:+.3f}, {hi:+.3f}]")
    print(f"  unweighted over the frame Δ={unweighted:+.3f}")
    print(f"\n  frame is {100 * len(rows_by_tier.get('blind', [])) / len(rows):.0f}% blind "
          f"against a population that is {100 * weights.get('blind', 0):.0f}% "
          f"({counts.get('blind', 0)} of {sum(counts.values())}) — which is the whole "
          f"reason the unweighted figure is not comparable to §16.9/§18.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
