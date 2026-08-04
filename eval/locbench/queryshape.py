#!/usr/bin/env python3
"""Query shape by condition — the §19 description A/B's first endpoint.

§19.3 registers query length as prediction 1, and registers it *first* on
purpose: it is the mechanism the micro-example is supposed to act through, it
is readable from the shim logs with no scoring and no gold files, and it
gates the rest. If the description does not move how agents write queries,
predictions 2-3 are void rather than negative — so this is the check to run
on the first chunk, before the frame is paid for.

Reads `harvest.py`'s rows rather than the shim logs directly: argv parsing
for a tool whose flags keep changing belongs in one place, and harvest already
has the reconciliation gate that catches a silent parse failure.

    python3 eval/locbench/queryshape.py                        # every condition
    python3 eval/locbench/queryshape.py --a desc-v7 --b desc-v5
    python3 eval/locbench/queryshape.py --runs ../data/locbench/runs/20260804-1200

Ranked queries only. Exact-mode patterns are regexes, where "words" is not a
meaningful unit and a long pattern means the opposite of what it means in a
ranked query.
"""

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from harvest import harvest  # noqa: E402

DATA = HERE.parent / "data" / "locbench"


def words(pattern):
    return len((pattern or "").split())


def summarize(lengths):
    """The distribution, plus the one number §19.1 reported: the short share."""
    n = len(lengths)
    if not n:
        return None
    return {
        "n": n,
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "short_share": sum(1 for w in lengths if w <= 2) / n,
        "hist": Counter(min(w, 6) for w in lengths),
    }


def print_table(by_cond):
    print(f"{'condition':<14} {'n':>6} {'mean':>6} {'median':>7} {'<=2 words':>10}")
    print("-" * 47)
    for cond, s in sorted(by_cond.items()):
        print(f"{cond:<14} {s['n']:>6} {s['mean']:>6.2f} {s['median']:>7.1f} "
              f"{100 * s['short_share']:>9.0f}%")
    print()
    print("word-count distribution (6 = 6 or more)")
    print(f"{'condition':<14}" + "".join(f"{w:>7}" for w in range(1, 7)))
    for cond, s in sorted(by_cond.items()):
        row = "".join(f"{100 * s['hist'][w] / s['n']:>6.0f}%" for w in range(1, 7))
        print(f"{cond:<14}{row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DATA / "runs")
    ap.add_argument("--a", default=None, help="treatment arm, for the two-arm delta")
    ap.add_argument("--b", default=None, help="control arm")
    args = ap.parse_args()

    rows, _, _ = harvest(args.runs)
    lengths = defaultdict(list)
    for r in rows:
        # Ranked only, and semgrep only: `rg` has no ranked mode, so including
        # it would compare a query language against a regex dialect.
        if r.get("tool") == "rg" or r.get("mode") != "ranked":
            continue
        if r.get("pattern"):
            lengths[r["condition"]].append(words(r["pattern"]))

    by_cond = {c: s for c, v in lengths.items() if (s := summarize(v))}
    if not by_cond:
        print("no ranked semgrep searches found under", args.runs)
        return 1
    print_table(by_cond)

    if args.a and args.b:
        missing = [c for c in (args.a, args.b) if c not in by_cond]
        if missing:
            print(f"\nno ranked searches for: {', '.join(missing)}")
            return 1
        a, b = by_cond[args.a], by_cond[args.b]
        d = a["mean"] - b["mean"]
        print(f"\n{args.a} − {args.b}: {d:+.2f} words/query "
              f"({a['mean']:.2f} vs {b['mean']:.2f}), "
              f"short share {100 * a['short_share']:.0f}% vs "
              f"{100 * b['short_share']:.0f}%")
        # Deliberately no p-value. These are per-search rows clustered within
        # instance and within session, so an unpaired test over them would
        # overstate its confidence by pretending 400 searches are 400
        # independent draws. §19.3 asks whether the description took, and a
        # direction with a visible n answers that; a significance claim about
        # query length would need the clustered treatment ab_analyze.py gives
        # the accuracy endpoints.
        print("(direction and magnitude only — these rows cluster within "
              "session, so no test is reported here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
