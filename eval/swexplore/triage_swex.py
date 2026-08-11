#!/usr/bin/env python3
"""The §27 rung gate: `eval/locbench/triage.py` adapted, not forked.

Same four gates, same thresholds, same "exit nonzero so it stops a campaign
rather than describing one" contract. Only the row shape and two
locbench-specific assumptions are replaced, so a threshold loosened here is a
visible edit in one place rather than a second copy that drifts.

    python3 eval/swexplore/triage_swex.py --run-id R1
    python3 eval/swexplore/triage_swex.py --run-id R1 --json gate.json

Five adaptations, each of which was a real defect if left alone:

1. **`run_id` and `condition` do not exist** on a swexplore row (it carries
   `explorer`). `cond_dir` would `KeyError` on the first gate. Injected at load;
   the on-disk layout `<run>/<instance>/<arm>/` already matches.
2. **`load_results` collapses the arms.** Its key is
   `(instance_id, condition, model)`, which for un-normalised swexplore rows is
   `(iid, None, None)` — so all three arms of an instance dedupe to ONE and the
   gate silently inspects a third of the campaign. This is the reason the
   loader is reimplemented here rather than reused.
3. **`status` lives at `agent.status`**, so `check_harness`'s `r.get("status")`
   reads `None` for every row and the non-ok gate fails 100% of a healthy run.
   Lifted to the row.
4. **The leaked-worktree gate is meaningless here.** It walks
   `data/locbench/repos/trees`; we use `data/swexplore/repos` under an LRU that
   deletes by design, so "leaked" is not a defect. Worse, its `done` set is
   built from `status == "ok"` which never matched, so it would have passed
   **vacuously** — the 0/0 failure mode §18 exists to end. Replaced with a real
   check: the LRU is actually bounded.
5. **The arm split is `condition != "rg"`.** `cc-rg` is not literally `"rg"`, so
   the ripgrep arm would be fed into the semgrep tool gate and dilute every
   share it computes. Replaced with an explicit arm→tool map.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "swexplore"
sys.path.insert(0, str(HERE.parent / "locbench"))
import triage  # noqa: E402

# cond_dir(), the free-disk statvfs and everything else in triage that resolves
# paths reads this module global. Point it at swexplore before any gate runs.
triage.DATA = DATA

# Which binary each arm was told to type. `cc` has no Bash tool at all.
# cc*: additive (native Grep present). sub-*: substitutive (Grep removed).
ALL_ARM_TOOL = {"cc": None, "cc-rg": "rg", "cc-sg": "sg",
                "sub-rg": "rg", "sub-sg": "sg"}
# Set per invocation from --arms. A rung must gate exactly the arms it ran:
# the harness-health gates compare against the REGISTERED set, so leaving this
# as every known arm would fail "registered arms absent" on any partial run,
# and leaving it as one rung's set fails "unexpected arm labels" on the next.
ARM_TOOL = dict(ALL_ARM_TOOL)
SG_ARMS = []

# The LRU's own ceiling, so the replacement disk gate has something to check
# against rather than a number invented here.
CACHE_GB = float(os.environ.get("SWEXPLORE_CACHE_GB", "8"))


def load(paths, run_id):
    """Normalise swexplore result rows into the shape triage's gates expect.

    `eval_runner` writes one file per explorer per k, so the caller passes
    several. Dedupe is last-write-wins on the full (instance, arm) key — the
    bug named in the docstring is that dropping `arm` from that key silently
    keeps one row in three.
    """
    rows = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            arm = r.get("explorer")
            agent = r.get("agent") or {}
            r["condition"] = arm
            r["run_id"] = run_id
            r["model"] = agent.get("model") or "sonnet"
            # (3) lift agent.status; a row whose explorer never wrote telemetry
            # has no status at all, and that is itself a failure, not an "ok".
            r["status"] = agent.get("status") or ("no_telemetry" if not agent else "ok")
            rows[(r.get("instance_id"), arm)] = r
    return list(rows.values())


def check_distress_swex(rows, examples):
    """`triage.check_distress`, but a search the tool correctly REJECTED is not
    an empty result.

    The distress gate asks "did the tool stop working, judged by how the agent
    behaved". Its all-empty check counts any invocation returning zero bytes,
    and `classify_usage` already distinguishes the case where zero bytes is the
    tool being *right* — a path that does not exist at that commit — but the
    all-empty check never consulted it.

    R1 tripped on exactly that: `php-cs-fixer-8064` ran one search,
    `sg "basic fix yield" tests/.../SimpleToComplexStringVariableFixerTest.php`,
    against a file absent at that base commit. semgrep exited 2 with "no such
    file or directory". One correct rejection, and the instance counted as
    "every search empty".

    triage.py's own principle is that "a gate that punishes the tool for being
    right is a gate nobody can pass", so this applies that filter to the
    distress check too. Only for distress: `check_tool` must still *see* these
    rows, because classifying them is its whole job.
    """
    orig = triage.searches

    def without_correct_rejections(row):
        return [e for e in orig(row)
                if not (e.get("exit") == 2 and "(tool correct)" in
                        triage.classify_usage(e.get("argv") or []))]

    triage.searches = without_correct_rejections
    try:
        return triage.check_distress(rows, examples)
    finally:
        triage.searches = orig


def check_harness_swex(rows):
    """triage.check_harness with the worktree gate replaced (see adaptation 4)."""
    print("\n[4/4] harness health")
    st = Counter(r.get("status") for r in rows)
    for k, v in st.most_common():
        print(f"  ---   {v:4d}  {k}")
    non_ok = sum(v for k, v in st.items() if k != "ok")
    triage.gate("non-ok rows", non_ok, 0)

    free_gb = os.statvfs(DATA).f_bavail * os.statvfs(DATA).f_frsize / 1e9
    triage.gate("free disk (GB)", round(free_gb, 1), 2.0, worse="below",
                detail="a campaign that fills the disk loses the chunk, not the row")

    # Checkouts are supposed to be evicted, so their presence is not a leak.
    # What would be a defect is the LRU not bounding: 848 checkouts average
    # 32.1 MB and would be 26.6 GB unbounded, against ~21 GiB free.
    repos = DATA / "repos"
    live = list(repos.iterdir()) if repos.exists() else []
    used_gb = sum(f.stat().st_size for d in live for f in d.rglob("*")
                  if f.is_file()) / 1e9 if live else 0.0
    print(f"  ---   {len(live)} checkout(s) resident, {used_gb:.1f} GB")
    triage.gate("checkout cache (GB)", round(used_gb, 1), round(CACHE_GB * 1.5, 1),
                detail="the LRU is not bounding; 848 unbounded is 26.6 GB")

    # Every instance must carry every arm, or the paired analysis silently
    # drops it and the arms are compared on different instances (ab_analyze's
    # note: reporting each arm over its own row set inverted a cost comparison).
    #
    # Gated against the REGISTERED arm set, not the observed one. The first
    # version compared observed-to-observed, so a results file whose arm label
    # had been lost collapsed 93 rows to 31, reported `arms present: [None]`,
    # and PASSED — self-consistent and vacuous, the exact 0/0 failure the
    # worktree gate above was replaced for. Caught by running this gate against
    # a deliberately broken copy, which is the only way that class of bug shows.
    expected = set(ARM_TOOL)
    arms = {r["condition"] for r in rows}
    triage.gate("unexpected arm labels", len(arms - expected), 0,
                detail=f"observed {sorted(map(str, arms))}, registered {sorted(expected)}")
    triage.gate("registered arms absent", len(expected - arms), 0,
                detail=f"missing {sorted(expected - arms)}")
    per = defaultdict(set)
    for r in rows:
        if r.get("status") == "ok":
            per[r["instance_id"]].add(r["condition"])
    partial = sum(1 for i, a in per.items() if a != expected)
    triage.gate("instances missing an arm", partial, 0,
                detail=f"arms registered: {sorted(expected)}")

    cost = sum((r.get("agent") or {}).get("total_cost_usd") or 0 for r in rows)
    print(f"  ---   spend so far: ${cost:.2f} over {len(rows)} rows")
    return {"non_ok": non_ok, "free_gb": round(free_gb, 1),
            "cache_gb": round(used_gb, 1), "partial_instances": partial,
            "cost": round(cost, 2)}


def _shim_entries(row):
    """This run's sg shim entries, via locbench triage's own reader — one
    parser for the log format, not two that can drift."""
    return triage.searches(row)


def report_invocation(rows):
    """Not a gate — a dilution factor. The pre-registered 70% floor was wrong
    (§27.1 measured 45%/52%), and a floor nobody can pass is a gate that gets
    quietly loosened. Reported so the effect can be read against it."""
    print("\n[--] tool invocation (reported, not gated)")
    out = {}
    for arm, tool in sorted(ARM_TOOL.items()):
        rs = [r for r in rows if r["condition"] == arm]
        if not rs or tool is None:
            continue
        used = sum(1 for r in rs
                   if ((r.get("agent") or {}).get("n_by_tool") or {}).get(tool, 0) > 0)
        calls = sum(((r.get("agent") or {}).get("n_by_tool") or {}).get(tool, 0)
                    for r in rs)
        print(f"  ---   {arm:6s} {used:4d}/{len(rs)} sessions used {tool} "
              f"({used / len(rs):.0%}), {calls} calls total")
        out[f"invocation_{arm}"] = round(used / len(rs), 3)
        # §30.4's registered tripwire for desc-v11: the exact-mode share of sg
        # invocations. The v11 clause names `-e` for the first time since
        # §16.10 measured an unconditional mention collapsing ranked share
        # 98% → 7%; if agents collapse onto exact match, the variant dies and
        # the result is reported as §16.10 replicating. Reported per rung so
        # the collapse is visible at the 120-gate, before the 848 is funded.
        if tool == "sg":
            exact = ranked = 0
            for r in rs:
                for e in _shim_entries(r):
                    if e.get("tool") != "sg" or e.get("blocked"):
                        continue
                    if "-e" in e["argv"] or "--exact" in e["argv"]:
                        exact += 1
                    else:
                        ranked += 1
            total = exact + ranked
            if total:
                print(f"  ---   {arm:6s} exact-share {exact}/{total} "
                      f"({exact / total:.0%}) — §30.4 tripwire: collapse onto "
                      f"-e kills the description variant")
                out[f"exact_share_{arm}"] = round(exact / total, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="SWEXPLORE_RUN_ID of the rung")
    ap.add_argument("--arms", default="cc,cc-rg,cc-sg",
                    help="comma-separated arms this rung ran; gates are applied "
                         "against exactly this set")
    ap.add_argument("--results", type=Path, nargs="*", default=None,
                    help="result JSONLs (default: results/<run-id>-*.jsonl)")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    global ARM_TOOL, SG_ARMS
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in want if a not in ALL_ARM_TOOL]
    if unknown:
        sys.exit(f"unknown arm(s): {unknown}; known: {sorted(ALL_ARM_TOOL)}")
    ARM_TOOL = {a: ALL_ARM_TOOL[a] for a in want}
    SG_ARMS = [a for a, t in ARM_TOOL.items() if t == "sg"]

    # Read only this rung's arm files, so a results directory holding several
    # arm sets under one run id does not leak foreign arms into the gates.
    paths = args.results or [DATA / "results" / f"{args.run_id}-{a}.jsonl" for a in want]
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        sys.exit(f"no results files for run {args.run_id}")
    rows = load(paths, args.run_id)
    if not rows:
        sys.exit(f"no rows in {[str(p) for p in paths]}")

    conds = Counter(r["condition"] for r in rows)
    print(f"triage: {len(rows)} rows from {len(paths)} file(s), run {args.run_id} "
          f"({', '.join(f'{k}={v}' for k, v in conds.most_common())})")

    # (5) only the arm that actually has semgrep is fed the tool/distress gates.
    sg = [r for r in rows if r["condition"] in SG_ARMS]
    summary = {}
    if sg:
        summary |= triage.check_tool(sg, args.examples)
        summary |= check_distress_swex(sg, args.examples)
        summary |= triage.check_cache(sg)
    else:
        print("\n[1-3/4] skipped: no semgrep arm in these results")
    summary |= check_harness_swex(rows)
    summary |= report_invocation(rows)

    print()
    if triage.FAILURES:
        print(f"GATE FAILED — {len(triage.FAILURES)} check(s). Do not start the "
              f"next rung until these are understood:")
        for name, value, limit, detail in triage.FAILURES:
            print(f"  · {name}: {value} (limit {limit})"
                  f"{(' — ' + detail) if detail else ''}")
    else:
        print("GATE PASSED — the tool answered what the agents typed.")
    if args.json:
        args.json.write_text(json.dumps(
            {"summary": summary, "failures": triage.FAILURES,
             "passed": not triage.FAILURES}, indent=1))
        print(f"wrote {args.json}")
    sys.exit(1 if triage.FAILURES else 0)


if __name__ == "__main__":
    main()
