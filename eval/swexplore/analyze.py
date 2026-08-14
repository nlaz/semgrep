#!/usr/bin/env python3
"""The §27 analysis: three contrasts, one convention.

    python3 eval/swexplore/analyze.py --run-id R2
    python3 eval/swexplore/analyze.py --run-id R2 --by language

Statistics are imported from `eval/locbench/ab_analyze.py` unchanged —
`boot_ci` (percentile CI on the paired mean difference, 4,000 resamples,
seed 1, resampling instances) and `mcnemar` (exact two-sided sign test over
discordant pairs). One bootstrap, one convention, across both harnesses; a
second copy here would drift from the one every prior section used.

Three contrasts, always printed together, because two of them only mean
something next to the third:

    cc-sg − cc      PRIMARY. Does enabling semgrep help an agent that
                    already has Grep?
    cc-rg − cc      The Bash confound. `cc-sg` gains a shell as well as a
                    tool; this is what the shell alone is worth. The §27.1
                    pilot found it flat on coverage and NOT flat on ctxEff,
                    which is the whole reason the third arm exists.
    cc-sg − cc-rg   Semgrep against ripgrep with Bash held constant.

Holm applies to the secondary family only. The primary and the co-primary cost
endpoint are registered singly and are not corrected — correcting a registered
primary against exploratory company is how a real effect gets buried.
"""

import argparse
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"
sys.path.insert(0, str(HERE.parent / "locbench"))
from ab_analyze import boot_ci, mcnemar  # noqa: E402

# Defaults reproduce §27 exactly; --arms / --contrasts override for §28.
# Arm names contain hyphens, so contrast pairs are separated by ':' — a '-'
# separator would split `cc-sg` in half.
ARMS = ("cc", "cc-rg", "cc-sg")
ALL_ARM_TOOL = {"cc": None, "cc-rg": "rg", "cc-sg": "sg",
                "sub-rg": "rg", "sub-sg": "sg"}
ARM_TOOL = {"cc-rg": "rg", "cc-sg": "sg"}
CONTRASTS = (("cc-sg", "cc", "PRIMARY  sg − cc"),
             ("cc-rg", "cc", "confound rg − cc"),
             ("cc-sg", "cc-rg", "held-Bash sg − rg"))
PRIMARY_PAIR = ("cc-sg", "cc")

PRIMARY = ("hit_region_rate", "hitRegion@5")
COPRIMARY = (("total_cost_usd", "cost $"), ("num_turns", "turns"))
SECONDARY = (("hit_file_rate", "hitFile@5"), ("context_efficiency", "ctxEff"),
             ("ndcg_at_500", "nDCG@500"), ("recall_at_100", "recall@100"),
             ("precision", "precision"))
# Reported with the bound rather than as findings: at these baselines the
# detectable difference is larger than any plausible effect (§27.1).
CEILING = {"ndcg_at_500", "first_useful_hit"}


def val(row, key):
    if key in ("total_cost_usd", "num_turns"):
        return (row.get("agent") or {}).get(key) or 0
    return row["metrics"].get(key)


def load(run_id):
    rows = {}
    for p in sorted((DATA / "results").glob(f"{run_id}-*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows[(r["instance_id"], r["explorer"])] = r
    by_arm = collections.defaultdict(dict)
    for (iid, arm), r in rows.items():
        by_arm[arm][iid] = r
    return by_arm


def pairs_for(by_arm, a, b, key, ids):
    out = []
    for i in ids:
        x, y = val(by_arm[a][i], key), val(by_arm[b][i], key)
        if x is not None and y is not None:
            out.append((x, y))
    return out


def line(label, pairs, note=""):
    d, lo, hi = boot_ci(pairs)
    w, l, p = mcnemar(pairs)
    star = " " if lo <= 0 <= hi else "*"
    sd = st.pstdev([x - y for x, y in pairs]) if len(pairs) > 1 else 0.0
    # MDE at 80% power for the n actually in hand, so a null is reported with
    # what it could have seen rather than as "no difference".
    mde = 2.80 * sd / math.sqrt(len(pairs)) if pairs else float("nan")
    # The two tests answer different questions and can disagree loudly. The
    # bootstrap is on the mean difference, so a few instances moving a long way
    # can carry it; the sign test counts how many instances moved at all. On
    # the §27.1 pilot, recall@100 held-Bash was w/l 10/12 — more losses than
    # wins — with a mean CI excluding zero. That is magnitude on a handful of
    # instances, not a consistent direction, and reporting only the starred CI
    # would have called it a win. Flagged rather than resolved: which one you
    # want depends on whether the claim is "usually better" or "better on
    # average", and both belong in the record.
    flag = ""
    if star == "*" and p > 0.10:
        flag = "  !! CI excludes 0 but sign test p>0.10 — mean-driven, not consistent"
    elif star == " " and p <= 0.05:
        flag = "  !! sign test significant but CI spans 0 — direction without magnitude"
    # An estimate landing at its own detection limit is the winner's-curse
    # setup: conditional on being detected, it is biased upward.
    if star == "*" and abs(d) < 1.25 * mde:
        flag += "  ~at MDE, expect regression to the mean"
    print(f"  {label:16s} {d:+8.4f}{star} [{lo:+.4f},{hi:+.4f}]  "
          f"w/l {w:3d}/{l:<3d} p={p:.3f}  mde={mde:.4f}{note}{flag}")
    return p


def holm(pvals):
    """Holm-Bonferroni, same inline form as ab_analyze.main():242-247."""
    out, m = {}, len(pvals)
    for rank, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        out[k] = min(1.0, p * (m - rank))
    return out


def invocation(by_arm, ids):
    print("\n=== tool invocation (the dilution factor, not a gate) ===")
    out = {}
    for arm, tool in sorted(ARM_TOOL.items()):
        rs = [by_arm[arm][i] for i in ids if i in by_arm[arm]]
        used = sum(1 for r in rs
                   if ((r.get("agent") or {}).get("n_by_tool") or {}).get(tool, 0) > 0)
        out[arm] = used / len(rs) if rs else 0.0
        print(f"  {arm:6s} {used:4d}/{len(rs)} used {tool} ({out[arm]:.0%})")
    print("  An effect measured over all instances is diluted by the "
          "non-invoking share; the treatment cannot act where it is not called.")
    return out


def main():
    ap = argparse.ArgumentParser()
    global ARMS, ARM_TOOL, CONTRASTS, PRIMARY_PAIR
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated arms to analyse")
    ap.add_argument("--contrasts", default="",
                    help="colon-paired contrasts, comma separated, first is "
                         "primary, e.g. 'sub-sg:sub-rg,sub-sg:cc,sub-rg:cc'")
    ap.add_argument("--by", default="", help="stratify: language | dataset")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    ARMS = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = [a for a in ARMS if a not in ALL_ARM_TOOL]
    if unknown:
        sys.exit(f"unknown arm(s): {unknown}; known: {sorted(ALL_ARM_TOOL)}")
    ARM_TOOL = {a: ALL_ARM_TOOL[a] for a in ARMS if ALL_ARM_TOOL[a]}
    if args.contrasts:
        CONTRASTS = []
        for i, spec in enumerate(args.contrasts.split(",")):
            a, b = spec.split(":")
            CONTRASTS.append((a.strip(), b.strip(),
                              f"{'PRIMARY ' if i == 0 else ''}{a.strip()} − {b.strip()}"))
        CONTRASTS = tuple(CONTRASTS)
        PRIMARY_PAIR = (CONTRASTS[0][0], CONTRASTS[0][1])
    missing = [a for c in CONTRASTS for a in c[:2] if a not in ARMS]
    if missing:
        sys.exit(f"contrast references arm(s) not in --arms: {sorted(set(missing))}")

    by_arm = load(args.run_id)
    missing = [a for a in ARMS if a not in by_arm]
    if missing:
        sys.exit(f"missing arms: {missing}")
    ids = sorted(set.intersection(*(set(by_arm[a]) for a in ARMS)))
    if not ids:
        sys.exit(f"no instances common to all of {list(ARMS)}")
    print(f"§27 — run {args.run_id}, {len(ids)} instances paired across "
          f"{', '.join(ARMS)}\n")

    means = {a: {k: st.mean(val(by_arm[a][i], k) for i in ids)
                 for k, _ in (PRIMARY,) + COPRIMARY + SECONDARY} for a in ARMS}
    print(f"{'endpoint':16s} " + " ".join(f"{a:>9}" for a in ARMS))
    for k, lbl in (PRIMARY,) + COPRIMARY + SECONDARY:
        print(f"  {lbl:16s} " + " ".join(f"{means[a][k]:9.4f}" for a in ARMS))

    results = {}
    for a, b, title in CONTRASTS:
        print(f"\n=== {title} ===")
        line(PRIMARY[1], pairs_for(by_arm, a, b, PRIMARY[0], ids),
             note="   <- primary" if (a, b) == PRIMARY_PAIR else "")
        for k, lbl in COPRIMARY:
            line(lbl, pairs_for(by_arm, a, b, k, ids), note="   <- co-primary")
        ps = {}
        for k, lbl in SECONDARY:
            note = "   (near ceiling)" if k in CEILING else ""
            ps[lbl] = line(lbl, pairs_for(by_arm, a, b, k, ids), note=note)
        adj = holm(ps)
        print("  Holm-adjusted (secondary family): " +
              ", ".join(f"{k} {v:.3f}" for k, v in sorted(adj.items(), key=lambda x: x[1])))
        results[title] = {"holm": adj}

    inv = invocation(by_arm, ids)

    if args.by:
        side = {json.loads(l)["instance_id"]: json.loads(l)
                for l in (DATA / "sidecar.jsonl").read_text().splitlines() if l.strip()}
        print(f"\n=== PRIMARY by {args.by} (exploratory; strata are not powered) ===")
        groups = collections.defaultdict(list)
        for i in ids:
            groups[side[i][args.by]].append(i)
        for g, gids in sorted(groups.items(), key=lambda x: -len(x[1])):
            p = pairs_for(by_arm, PRIMARY_PAIR[0], PRIMARY_PAIR[1], PRIMARY[0], gids)
            if len(p) < 5:
                print(f"  {g:12s} n={len(p):3d}  unpowered by construction — not reported")
                continue
            d, lo, hi = boot_ci(p)
            print(f"  {g:12s} n={len(p):3d}  {d:+.4f} [{lo:+.4f},{hi:+.4f}]")

    if args.json:
        args.json.write_text(json.dumps(
            {"run_id": args.run_id, "n": len(ids), "means": means,
             "invocation": inv, "contrasts": results}, indent=1) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
