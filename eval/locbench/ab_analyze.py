#!/usr/bin/env python3
"""Powered two-arm analysis for the §16.9 A/B (semantic vs exact-term).

The endpoints are pre-registered, so this script is deliberately narrow:
it computes them, it does not explore. Exact McNemar over discordant pairs
is the primary test (the metrics are all-or-nothing binaries per instance,
so a paired binary test is the right one); an instance-bootstrap CI on the
delta is reported beside it because a p-value alone hides the effect size.

    python3 eval/locbench/ab_analyze.py                       # defaults
    python3 eval/locbench/ab_analyze.py --emit-screen         # + the §11.5 screen

Reads eval/data/locbench/results-scale.jsonl by default, dedupes
last-wins on (instance_id, condition, model) exactly as report.py does,
and refuses to score an arm pair with no shared instances.
"""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"

PRIMARY = "func_acc@10_tol"
# Co-primary (§16.9a C1): the binary endpoint discards resolution on the
# ~96% of instances where both arms agree; recall is continuous.
CO_PRIMARY = "func_recall@10_tol"
SECONDARY = ["file_acc@5", "file_acc@1", "file_recall@5", "func_acc@10_strict"]


def load(path, conds, model=None):
    """-> {condition: {instance_id: row}}, last row wins (report.py:27-34)."""
    by = {c: {} for c in conds}
    dupes = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") != "ok" or r.get("condition") not in by:
            continue
        if model and r.get("model") != model:
            continue
        key = r["instance_id"]
        if key in by[r["condition"]]:
            dupes.append((r["condition"], key))
        by[r["condition"]][key] = r
    if dupes:
        print(f"note: {len(dupes)} (instance, condition) keys appear more than "
              f"once; the LAST ok row wins, matching report.py. Re-run chunks "
              f"can supersede earlier rows — check if this is unexpected.")
    return by


def mcnemar(pairs):
    """(b, c, p): b = a-only wins, c = b-only wins, exact two-sided."""
    b = sum(1 for x, y in pairs if x > y)
    c = sum(1 for x, y in pairs if y > x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return b, c, min(1.0, 2 * tail / 2 ** n)


def boot_ci(pairs, n_resamples=4000, seed=1):
    """Percentile CI on mean(a) - mean(b), resampling instances."""
    rng = random.Random(seed)
    n = len(pairs)
    if not n:
        return 0.0, 0.0, 0.0
    point = sum(x - y for x, y in pairs) / n
    deltas = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(pairs[i][0] - pairs[i][1] for i in idx) / n)
    deltas.sort()
    return (point, deltas[int(0.025 * n_resamples)],
            deltas[min(int(0.975 * n_resamples), n_resamples - 1)])


def metric_pairs(by, a, b, key, ids):
    out = []
    for i in ids:
        ma = (by[a][i].get("metrics") or {}).get(key)
        mb = (by[b][i].get("metrics") or {}).get(key)
        if ma is None or mb is None:
            continue
        out.append((float(ma), float(mb)))
    return out


def endpoint_line(label, pairs):
    if not pairs:
        return f"  {label:<22} (no paired rows)"
    ma = sum(x for x, _ in pairs) / len(pairs)
    mb = sum(y for _, y in pairs) / len(pairs)
    b, c, p = mcnemar(pairs)
    d, lo, hi = boot_ci(pairs)
    star = "*" if p < 0.05 else " "
    # A CI whose bound is pinned at 0 by a single discordant pair reads as
    # an effect to anyone skimming; say so instead.
    ci = (f"CI[{lo:+.3f},{hi:+.3f}]" if b + c >= 5
          else f"CI[{lo:+.3f},{hi:+.3f}](thin)")
    return (f"  {label:<22} n={len(pairs):>4}  a={ma:>5.3f} b={mb:>5.3f}  "
            f"Δ={d:+.3f} {ci}  discordant {b}/{c}  p={p:.4f}{star}")


def size_bucket(r):
    n = (r.get("checkout") or {}).get("file_count")
    if n is None:
        return "?"
    return "<2k" if n < 2000 else "2k-10k" if n <= 10000 else ">10k"


UNSHIMMED = ("find", "python3", "python", "awk", "sed", "perl", "ls", "cat", "wc")


def unshimmed_search(row, cond):
    """Bash calls that are content searches the shim cannot see.

    §16.9a A2: `find`, `python3 -c '...in line...'`, awk/sed are live and
    unlogged in both arms. They are counted here from the transcript so the
    ranked-share denominator and `first_hit_search_seq` can be read with the
    right caveat rather than silently believed."""
    rid = row.get("run_id")
    if not rid:
        return None
    t = DATA / "runs" / rid / row["instance_id"] / cond / "transcript.jsonl"
    if not t.exists():
        return None
    n_bash = n_uns = 0
    for line in t.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") or {}
        for blk in (msg.get("content") or []):
            if not isinstance(blk, dict) or blk.get("name") != "Bash":
                continue
            cmd = ((blk.get("input") or {}).get("command") or "").strip()
            if not cmd:
                continue
            n_bash += 1
            first = cmd.split()[0].split("/")[-1] if cmd.split() else ""
            if first in UNSHIMMED:
                n_uns += 1
    return n_bash, n_uns


def behavior(by, cond, ids):
    """Ranked/exact/rg shares and cost over the PAIRED instances only.

    Reporting each arm over its own row set silently compares different
    instance sets — on the §16.8 data that inverted the cost comparison
    (rg looked 10% more expensive; paired, it is 3% cheaper). Chunked runs
    guarantee the arms are out of sync at every boundary, so this must be
    the intersection."""
    rows = [by[cond][i] for i in ids]
    runs = DATA / "runs"
    ranked = exact = rg = 0
    n_bash = n_unshimmed = n_no_run_id = 0
    for r in rows:
        u = unshimmed_search(r, cond)
        if u:
            n_bash += u[0]
            n_unshimmed += u[1]
        rid = r.get("run_id")
        if not rid:
            n_no_run_id += 1
            continue
        log = runs / rid / r["instance_id"] / cond / "shim_log.jsonl"
        if not log.exists():
            continue
        for line in log.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("blocked") or e.get("tool") not in ("semgrep", "rg", "search"):
                continue
            if e["tool"] == "rg":
                rg += 1
            elif "-e" in (e.get("argv") or []) or "--exact" in (e.get("argv") or []):
                exact += 1
            else:
                ranked += 1
    tot = ranked + exact + rg
    costs = [(r.get("agent") or {}).get("total_cost_usd") or 0 for r in rows]
    searches = [(r.get("search") or {}).get("n_invocations") or 0 for r in rows]
    bypass = sum(1 for r in rows
                 if (r.get("search") or {}).get("shim_bypass_suspected"))
    return {
        "n_runs": len(rows), "ranked": ranked, "exact": exact, "rg": rg,
        "ranked_share": ranked / tot if tot else 0.0,
        "exact_share": exact / tot if tot else 0.0,
        "median_cost": statistics.median(costs) if costs else 0,
        "total_cost": sum(costs),
        "median_searches": statistics.median(searches) if searches else 0,
        "zero_search": sum(1 for s in searches if s == 0),
        "bypass": bypass,
        "bash": n_bash, "unshimmed": n_unshimmed,
        "unshimmed_share": n_unshimmed / n_bash if n_bash else 0.0,
        "no_run_id": n_no_run_id,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DATA / "results-scale.jsonl")
    ap.add_argument("--a", default="desc-v5", help="treatment arm")
    ap.add_argument("--b", default="rg", help="control arm")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--tiers", type=Path, default=DATA / "blind-instances.json")
    ap.add_argument("--emit-screen", type=Path, default=None,
                    help="write the discriminative-instance list (§11.5)")
    args = ap.parse_args()

    by = load(args.results, [args.a, args.b], args.model)
    ids = sorted(set(by[args.a]) & set(by[args.b]))
    if not ids:
        raise SystemExit(
            f"no instances completed under BOTH {args.a} and {args.b} in "
            f"{args.results} (model={args.model})")
    print(f"=== §16.9 A/B: {args.a} (a) vs {args.b} (b) ===")
    print(f"{len(ids)} paired instances "
          f"({len(by[args.a])} a-rows, {len(by[args.b])} b-rows)\n")

    print("endpoints (a − b):")
    print(endpoint_line(f"PRIMARY {PRIMARY}", metric_pairs(by, args.a, args.b, PRIMARY, ids)))
    print(endpoint_line(f"CO-PRIM {CO_PRIMARY}",
                        metric_pairs(by, args.a, args.b, CO_PRIMARY, ids)))
    sec = [(k, metric_pairs(by, args.a, args.b, k, ids)) for k in SECONDARY]
    raw_p = {k: (mcnemar(p)[2] if p else 1.0) for k, p in sec}
    holm = {}
    for rank, (k, p) in enumerate(sorted(raw_p.items(), key=lambda kv: kv[1])):
        holm[k] = min(1.0, p * (len(sec) - rank))
    for k, pairs in sec:
        print(endpoint_line(k, pairs) + f"  holm={holm[k]:.3f}")

    # --- strata ---
    tiers = {}
    if args.tiers.exists():
        tiers = {k: v.get("tier") for k, v in json.loads(args.tiers.read_text()).items()}

    def stratum_report(name, keyfn):
        groups = defaultdict(list)
        for i in ids:
            groups[keyfn(i)].append(i)
        print(f"\n{name} — EXPLORATORY, uncorrected (primary endpoint):")
        for g, gids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            pairs = metric_pairs(by, args.a, args.b, PRIMARY, gids)
            print(endpoint_line(str(g), pairs).replace("*", " "))

    if tiers:
        stratum_report("by issue-naming tier (§15.7)", lambda i: tiers.get(i, "?"))
    stratum_report("by category", lambda i: by[args.a][i].get("category") or "?")
    stratum_report("by repo size", lambda i: size_bucket(by[args.a][i]))
    # (The "by search usage" stratum is deliberately absent: partitioning on
    # whether the treatment arm chose to search conditions on an outcome of
    # treatment — collider stratification, §16.9a C4.)

    # --- behavior ---
    print("\nbehavior:")
    print(f"  {'arm':<10} {'runs':>5} {'ranked':>7} {'exact':>6} {'rg':>5} "
          f"{'ranked%':>8} {'medCost':>8} {'total$':>8} {'medSrch':>8} "
          f"{'0-srch':>7} {'bypass':>7} {'unshim%':>8}")
    for arm in (args.a, args.b):
        h = behavior(by, arm, ids)
        print(f"  {arm:<10} {h['n_runs']:>5} {h['ranked']:>7} {h['exact']:>6} "
              f"{h['rg']:>5} {h['ranked_share']:>8.0%} ${h['median_cost']:>7.2f} "
              f"${h['total_cost']:>7.2f} {h['median_searches']:>8.0f} "
              f"{h['zero_search']:>7} {h['bypass']:>7} {h['unshimmed_share']:>8.0%}")
    print("  unshim% = share of the arm's Bash calls that are find/python3/awk/"
          "sed content searches\n  the shim never sees (§16.9a A2). Ranked share "
          "and first-hit-seq are conditional on this.")

    # --- the §11.5 screen ---
    if args.emit_screen:
        disc = []
        for i in ids:
            ma = by[args.a][i].get("metrics") or {}
            mb = by[args.b][i].get("metrics") or {}
            if any(bool(ma.get(k)) != bool(mb.get(k))
                   for k in (PRIMARY, "file_acc@5")):
                disc.append(i)
        args.emit_screen.write_text(json.dumps({
            "arms": [args.a, args.b], "model": args.model,
            "n_paired": len(ids), "n_discriminative": len(disc),
            "note": "DISCORDANCE MAP OF THIS RUN, not a neutral screen "
                    "(§16.9a C5): instances selected by their own noisy "
                    "outcome here. Re-using it as a cheap frame for future "
                    "A/Bs inherits a winner's-curse bias; it is published as "
                    "a description of where these two arms differed.",
            "instances": disc,
        }, indent=1) + "\n")
        print(f"\nwrote {len(disc)} discriminative instances "
              f"({len(disc)/len(ids):.0%}) to {args.emit_screen}")


if __name__ == "__main__":
    main()
