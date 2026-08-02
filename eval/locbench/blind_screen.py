#!/usr/bin/env python3
"""Screen Loc-Bench for blind search (RESEARCH.md §15, Phase 2).

Two screens against each instance's gold identifiers (the function names and
file stems of `edit_functions`):

1. **Instances**: does the issue text (`problem_statement`) name the gold?
   Instances whose issue carries no gold identifier are the real-world L3
   stratum — a person describing a symptom with zero knowledge of where the
   fix lives. Written to eval/data/locbench/blind-instances.json.
2. **Replayed agent queries**: same predicate per query, then the recorded
   per-mode ranks re-aggregated by stratum with the instance-clustered
   bootstrap (queries cluster hard by instance; see replay.py).

Tiers rather than a boolean, because without the gold file text there is no
prose-guard for subtokens (the §15.3 clause (c) analog):
    named   — a gold function name or file stem appears verbatim
    partial — only subtokens of gold names appear (len >= 4, non-stopword);
              common verbs land here, so it is reported, not folded into
              either side
    blind   — neither
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import leakage  # noqa: E402
from replay import bootstrap_ci  # noqa: E402

DATA = HERE.parent / "data" / "locbench"


def _subtokens(word):
    out = []
    for part in word.split("_"):
        for m in re.finditer(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", part):
            out.append(m.group(0).lower())
    return out


def gold_vocabulary(instance):
    """(verbatim names, subtokens) from edit_functions' files and functions."""
    names, subs = set(), set()
    for ef in instance["edit_functions"]:
        path, _, func = ef.partition(":")
        for name in filter(None, [func, Path(path).stem]):
            names.add(name.lower())
            for s in _subtokens(name):
                if len(s) >= leakage.MIN_SEGMENT and s not in leakage.STOPWORDS:
                    subs.add(s)
    return names, subs - names


def tier(text, names, subs):
    toks = {t.lower() for t in leakage.content_tokens(text)}
    hit_names = sorted(toks & names)
    if hit_names:
        return "named", hit_names
    hit_subs = sorted(toks & subs)
    if hit_subs:
        return "partial", hit_subs
    return "blind", []


def screen_instances(dataset):
    out = {}
    for inst in dataset:
        names, subs = gold_vocabulary(inst)
        t, hits = tier(inst["problem_statement"], names, subs)
        out[inst["instance_id"]] = {"tier": t, "hits": hits}
    return out


def hit_at(rank, k):
    return rank is not None and rank <= k


def report_replay(rows, dataset, which):
    by_id = {i["instance_id"]: i for i in dataset}
    strata = defaultdict(list)
    for r in rows:
        inst = by_id.get(r["instance_id"])
        if inst is None:
            continue
        names, subs = gold_vocabulary(inst)
        t, _ = tier(r["query"], names, subs)
        strata[t].append(r)

    modes = sorted({m for r in rows for m in r["ranks"]})
    print(f"\n=== replayed agent queries by {which} tier ===")
    print(f"{'tier':<8} {'n':>4} " + " ".join(f"{m + ' MRR':>13}" for m in modes)
          + "  " + " ".join(f"{m + ' h@5':>13}" for m in modes))
    for t in ("named", "partial", "blind"):
        rs = strata.get(t, [])
        if not rs:
            continue
        mrr = {m: sum(1 / r["ranks"][m] if r["ranks"].get(m) else 0 for r in rs) / len(rs)
               for m in modes}
        h5 = {m: sum(hit_at(r["ranks"].get(m), 5) for r in rs) / len(rs) for m in modes}
        print(f"{t:<8} {len(rs):>4} " + " ".join(f"{mrr[m]:>13.3f}" for m in modes)
              + "  " + " ".join(f"{h5[m]:>13.3f}" for m in modes))

    for a, b in (("hybrid", "bm25"), ("semantic", "bm25")):
        if a not in modes or b not in modes:
            continue
        print(f"\npaired {a} - {b} MRR, instance-clustered CI:")
        for t in ("named", "partial", "blind"):
            rs = strata.get(t, [])
            if len(rs) < 10:
                continue
            point, lo, hi, ncl = bootstrap_ci(rs, a, b)
            print(f"  {t:<8} n={len(rs):>4} clusters={ncl:>3} "
                  f"d={point:+.4f} CI[{lo:+.4f},{hi:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, default=DATA / "replay.jsonl",
                    help="replay results with per-mode ranks")
    args = ap.parse_args()

    dataset = [json.loads(l) for l in (DATA / "dataset.jsonl").read_text().splitlines() if l.strip()]
    screened = screen_instances(dataset)
    counts = defaultdict(int)
    for v in screened.values():
        counts[v["tier"]] += 1
    out = DATA / "blind-instances.json"
    out.write_text(json.dumps(screened, indent=1) + "\n")
    total = len(screened)
    print(f"instances: {total} — "
          + " ".join(f"{t}={counts[t]} ({counts[t] / total:.0%})"
                     for t in ("named", "partial", "blind")))
    print(f"wrote {out}")

    if args.replay.exists():
        rows = [json.loads(l) for l in args.replay.read_text().splitlines() if l.strip()]
        report_replay(rows, dataset, "query")


if __name__ == "__main__":
    main()
