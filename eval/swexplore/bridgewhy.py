#!/usr/bin/env python3
"""§33 mechanism analysis: did the effect land where the mechanism engaged?

    python3 eval/swexplore/bridgewhy.py --run-id s33 \
        --treatment sub-sgb --control sub-sg

A campaign's primary says *whether* something moved. This says *where*, by
stratifying the paired difference on whether bridge expansion actually fired
in that session — the within-campaign twin of guessplay's query-length dose
test (§33.1c: +0.010 at one word, +0.028 at three-to-four, because a
committee needs two covered query tokens to form at all).

The prediction it tests, registered in §33.1c before the data existed:
an effect concentrated in the fired stratum is a mechanism; an effect spread
evenly across fired and not-fired is a coincidence wearing a mechanism's
clothes, since a session where expansion never fired ran the control engine
in all but name.

EXPLORATORY. §33.1 binds the endpoints to one computation on the pooled 848;
this is a diagnostic on top of that, and its strata are chosen by an engine
behaviour rather than by the registration. Report it as such.
"""

import argparse
import collections
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"
sys.path.insert(0, str(HERE.parent / "locbench"))
from ab_analyze import boot_ci, mcnemar  # noqa: E402


def load(run_id, arm):
    out = {}
    p = DATA / "results" / f"{run_id}-{arm}.jsonl"
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        if ((r.get("agent") or {}).get("status") or "ok") == "ok":
            out[r["instance_id"]] = r
    return out


def fired_stats(run_id, iid, arm):
    """(n_searches, n_fired, total_terms) from this cell's trace envelopes."""
    t = DATA / "runs" / run_id / iid / arm / "trace.jsonl"
    n = fired = terms = 0
    if not t.exists():
        return n, fired, terms
    for line in open(t):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") != "search" or e.get("phase") != "primary":
            continue
        n += 1
        bt = (e.get("results") or {}).get("bridge_terms")
        if bt:
            fired += 1
            terms += len(bt)
    return n, fired, terms


def report(name, pairs):
    if len(pairs) < 2:
        print(f"  {name:28s} n={len(pairs):3d}  (too few to estimate)")
        return
    d, lo, hi = boot_ci(pairs)
    w, l, p = mcnemar(pairs)
    t = statistics.mean(x for x, _ in pairs)
    c = statistics.mean(y for _, y in pairs)
    sd = statistics.pstdev([x - y for x, y in pairs])
    mde = 2.80 * sd / len(pairs) ** 0.5 if pairs else float("nan")
    print(f"  {name:28s} n={len(pairs):3d}  {c:.3f} -> {t:.3f}   "
          f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]  w/l {w}/{l}  p={p:.3f}  "
          f"(MDE {mde:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s33")
    ap.add_argument("--treatment", default="sub-sgb")
    ap.add_argument("--control", default="sub-sg")
    ap.add_argument("--metric", default="hit_region_rate")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="exposure/fire-rate only, NO endpoint tables — the "
                         "flag that would have prevented §33.1f's accidental "
                         "interim look")
    ap.add_argument("--target-n", type=int, default=848,
                    help="registered paired n; a shorter run is flagged as "
                         "an interim look rather than reported bare")
    ap.add_argument("--also", default="hit_file_rate",
                    help="second metric to stratify, §33.1c predicts the "
                         "mechanism shows here rather than on the primary")
    args = ap.parse_args()

    trt, ctl = load(args.run_id, args.treatment), load(args.run_id, args.control)
    ids = sorted(set(trt) & set(ctl))
    if not ids:
        raise SystemExit("no paired instances")

    fire = {i: fired_stats(args.run_id, i, args.treatment) for i in ids}
    n_any = sum(1 for i in ids if fire[i][1] > 0)
    n_searched = sum(1 for i in ids if fire[i][0] > 0)
    all_terms = sum(fire[i][2] for i in ids)
    all_fired = sum(fire[i][1] for i in ids)
    print(f"{args.run_id}: {len(ids)} paired instances "
          f"({args.treatment} vs {args.control})")
    print(f"  sessions that searched at all : {n_searched} "
          f"({n_searched / len(ids):.0%})")
    print(f"  sessions where bridge fired   : {n_any} ({n_any / len(ids):.0%})")
    print(f"  searches expanded             : {all_fired}"
          + (f", mean {all_terms / all_fired:.1f} terms" if all_fired else ""))

    if len(ids) < args.target_n:
        print(f"\n*** INTERIM: {len(ids)} of {args.target_n} registered pairs. "
              f"Any endpoint below is an unregistered look and must be\n"
              f"    disclosed if acted on (RESEARCH.md §33.1b, §33.1f).")
    if args.diagnostics_only:
        print("\n(--diagnostics-only: endpoints withheld)")
        return

    for metric in (args.metric, args.also):
        print(f"\n== {metric} (paired, treatment − control)")
        strata = collections.OrderedDict()
        strata["ALL"] = ids
        strata["bridge fired >=1x"] = [i for i in ids if fire[i][1] > 0]
        strata["bridge never fired"] = [i for i in ids if fire[i][1] == 0]
        strata["  ...of which: searched"] = [
            i for i in ids if fire[i][1] == 0 and fire[i][0] > 0]
        strata["  ...of which: no search"] = [
            i for i in ids if fire[i][0] == 0]
        # Dose, by COUNT of expanded searches. Splitting on the *share* of a
        # session's searches that expanded is degenerate — measured on the
        # pilot, the median share is 1.00 because a session either expands
        # everything or nothing, so the split put 73 instances on one side
        # and 0 on the other. How many times the mechanism acted is the
        # dose; what fraction of the session it occupied is not.
        fired_ids = strata["bridge fired >=1x"]
        if len(fired_ids) >= 4:
            strata["fired once"] = [i for i in fired_ids if fire[i][1] == 1]
            strata["fired 2-3x"] = [i for i in fired_ids if 2 <= fire[i][1] <= 3]
            strata["fired 4x+"] = [i for i in fired_ids if fire[i][1] >= 4]
        for name, sub in strata.items():
            pairs = [((trt[i]["metrics"].get(metric) or 0.0),
                      (ctl[i]["metrics"].get(metric) or 0.0)) for i in sub]
            report(name, pairs)

    exposed = n_any / len(ids)
    print(f"\nReading: the effect belongs to the mechanism only if it "
          f"concentrates in the fired stratum. A session where expansion\n"
          f"never fired ran the control engine in all but name, so a "
          f"difference there is noise by construction — and a useful check\n"
          f"on the pairing.")
    print(f"\nDILUTION: only {exposed:.0%} of paired instances had ANY "
          f"exposure to the treatment ({len(ids) - n_any} of {len(ids)} never\n"
          f"fired, mostly because the agent never invoked sg at all — the "
          f"§32.1a availability-is-not-use problem, one layer in). The\n"
          f"registered primary is intention-to-treat, so it estimates roughly "
          f"{exposed:.2f}x the per-protocol effect: a per-protocol {0.018:.3f}\n"
          f"would surface as about {0.018 * exposed:+.4f} ITT. Read the "
          f"primary against that, not against the offline number.")


if __name__ == "__main__":
    main()
