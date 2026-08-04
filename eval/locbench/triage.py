#!/usr/bin/env python3
"""Did the tool work, this time? A gate between campaign tiers (RESEARCH.md §18).

The §16.10 campaign spent $361 and 1,115 runs measuring a tool whose searches
silently returned nothing 47% of the time. Nothing in the harness noticed. The
bug was found afterwards, by a human reading four transcripts, because the only
per-search record was `(argv, exit, stdout_bytes)` — enough to see that a search
returned no bytes, not enough to see *why*, and nobody was looking.

This is the missing instrument. It reads the per-invocation trace envelopes
(`SEMGREP_TRACE_FILE`, set by run.py since §18) alongside the shim logs, and
reports the failure modes this project has actually been bitten by. Every check
below exists because something went wrong that no test could see.

    python3 eval/locbench/triage.py --results eval/data/locbench/results-tier1.jsonl
    python3 eval/locbench/triage.py --results ... --condition desc-v5 --examples 5

Exit codes: 0 all gates pass, 1 a gate failed (do not proceed to the next
tier), 2 the input could not be read.

Three families, and the reason each is here:

*Tool* — the search itself misbehaved. A ranked search over a readable scope
essentially cannot return zero results (top-k always has k candidates), so a
zero is a structural fact, not a vocabulary miss. `files_walked` and
`n_chunks_considered` together say which: no files at all (the caller scoped at
nothing) versus files that could not be read (the §16.11 signature).

*Agent distress* — the agent telling us the tool failed, in its own behavior.
This is the check that would have caught §16.11 on day one: 27 of 27 `--help`
probes in the campaign followed three or more consecutive empty searches. An
agent reads the manual when the tool has stopped working, and repeating one
query verbatim means it does not believe the answer.

*Harness* — rows that never produced data, and the disk that runs out halfway.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "locbench"

# Gate thresholds. Stated as constants so a run that trips one names the number
# it tripped, and so loosening a gate is a visible edit rather than a judgement
# call made while staring at a failing report.
MAX_UNKNOWN_FLAG = 0            # exit 2 the TOOL is answerable for (see classify_usage)
MAX_EMPTY_RANKED_SHARE = 0.02   # was 0.59 in §16.10
MAX_ALL_EMPTY_INSTANCES = 0     # was 18% of instances in §16.10
MAX_TOOL_DISTRESS = 0           # help probes / repeats caused by empty results
CONSECUTIVE_EMPTY = 3           # what counts as "the tool stopped working"
REPEATED_QUERY = 3              # same query this many times = disbelief

HELP_FLAGS = {"--help", "-h", "--version", "-V"}
FAILURES = []


def gate(name, value, limit, worse="above", detail="", pct=False):
    """Record a gate result. `pct` formats a share; without it a float is a
    quantity (a disk figure is not 580%)."""
    bad = value > limit if worse == "above" else value < limit
    mark = "FAIL" if bad else "ok  "
    fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:g}")
    shown, lim = fmt(value), fmt(limit)
    print(f"  {mark}  {name}: {shown} (limit {lim}){(' — ' + detail) if detail else ''}")
    if bad:
        FAILURES.append((name, shown, lim, detail))


# Every flag the CLI accepts, and which of them take a value. Classification
# works from argv alone: the shim records `stderr_bytes` but deliberately not
# the stderr text, because semgrep's footers coach mode choice and that is a
# treatment channel in an A/B (§16.9a A1).
VALUED_FLAGS = {"-k", "--top", "-C", "--context", "-A", "--after-context",
                "-B", "--before-context", "-g", "--glob", "--include",
                "--mode", "--sem-weight", "--window", "--overlap"}
KNOWN_FLAGS = VALUED_FLAGS | {
    "-e", "--exact", "-i", "--ignore-case", "-F", "--fixed-string", "--all",
    "--json", "--stats", "--stats-json", "--check-stale", "-l",
    "--files-with-matches", "-n", "--line-number", "-r", "--recursive",
    "-R", "--dereference-recursive", "-H", "--with-filename",
    "-h", "--help", "-V", "--version", "--no-index", "--"}


def classify_usage(argv):
    """Whose fault is an exit 2?

    Not every usage error is a defect. Gating on the raw count fails the run
    when the tool *correctly* rejects a path that does not exist — which is the
    single most useful error it emits, because an agent reads "no results" as
    "the code is not there" and a wrong path is the other explanation. A gate
    that punishes the tool for being right is a gate nobody can pass.

    So gate only on the category that can hide a real gap: an unrecognised
    flag. That is where grep muscle memory lands — §17 found `-n` was 88% of
    the flags typed at a grep-shaped tool and every one exited 2 — so a nonzero
    count there means the compat surface has gone short again. The rest are
    correct rejections: reported, not gated.
    """
    toks = [a for a in argv if a]
    if not toks:
        return "empty argv"
    # A trailing value-taking flag: `semgrep "q" tests -k` with no number.
    if toks[-1] in VALUED_FLAGS:
        return "missing flag value (tool correct)"
    # A query that begins with a dash reads as flags. The caller's mistake, but
    # the message is unhelpful without the `--` hint, so it is tracked apart.
    if toks[0].startswith("-") and " " in toks[0]:
        return "query starts with a dash (needs --)"
    # Any dash token the CLI does not define. Combined shorts (-rn) are
    # expanded so a known pair is not misread as unknown.
    for t in toks:
        if not t.startswith("-") or t in KNOWN_FLAGS:
            continue
        if re.fullmatch(r"-[A-Za-z]+", t) and all(f"-{c}" in KNOWN_FLAGS for c in t[1:]):
            continue
        if t.split("=", 1)[0] in KNOWN_FLAGS:
            continue
        return "unknown flag"
    # All flags known and well-formed, so the argv parsed; what failed was the
    # path. Cannot be re-verified offline — the worktree is gone — but it is
    # the only remaining explanation for an exit 2.
    return "bad path (tool correct)"


def load_results(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Last write wins: --resume re-runs a failed row and appends.
        rows[(r.get("instance_id"), r.get("condition"), r.get("model"))] = r
    return list(rows.values())


def read_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def cond_dir(row):
    return DATA / "runs" / row["run_id"] / row["instance_id"] / row["condition"]


def searches(row):
    """This run's semgrep invocations, in order, from the shim log."""
    return [e for e in read_jsonl(cond_dir(row) / "shim_log.jsonl")
            if not e.get("blocked") and e.get("tool") == "semgrep"]


def traces(row):
    """This run's engine envelopes. Empty when the run predates §18."""
    return [e for e in read_jsonl(cond_dir(row) / "trace.jsonl")
            if e.get("kind") == "search"]


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------

def check_tool(rows, examples):
    print("\n[1/4] the tool itself")
    n_search = 0
    usage, empty_ranked, unreadable, no_files = [], [], [], []
    traced = 0
    kinds = Counter()
    for r in rows:
        for e in searches(r):
            n_search += 1
            if e.get("exit") == 2:
                kind = classify_usage(e["argv"])
                kinds[kind] += 1
                usage.append((r["instance_id"], kind, e["argv"][:3]))
        for t in traces(r):
            traced += 1
            res, inp = t.get("results", {}), t.get("input", {})
            if inp.get("mode") == "keyword":
                continue  # exact mode may legitimately match nothing
            if res.get("n_hits", 0) == 0:
                empty_ranked.append((r["instance_id"], inp.get("query", "")[:40]))
                fw, ch = res.get("files_walked"), res.get("n_chunks_considered")
                if fw and not ch:
                    unreadable.append((r["instance_id"], inp.get("root", "")[-46:]))
                elif fw == 0:
                    no_files.append((r["instance_id"], inp.get("root", "")[-46:]))

    print(f"  {n_search} semgrep invocations, {traced} engine traces")
    if usage:
        print(f"  ---   {len(usage)} usage errors (exit 2), by cause:")
        for k, v in kinds.most_common():
            print(f"          {v:4d}  {k}")
    gate("usage errors the tool is answerable for", kinds["unknown flag"],
         MAX_UNKNOWN_FLAG,
         detail="a rejected path or a malformed argv is the tool being right; "
                "an unrecognised flag is the compat surface gone short")
    # A missing trace must fail, not pass quietly. Without envelopes the
    # empty-result share is 0/0, which formats as a clean 0% and would sail
    # through the single most important gate here while measuring nothing —
    # the same shape of silence this whole tool exists to end. Runs predating
    # §18 have no traces and are expected to trip this.
    if n_search and not traced:
        gate("engine traces present", 0, 1, worse="below",
             detail="no SEMGREP_TRACE_FILE envelopes; the empty-result "
                    "diagnosis cannot be made and the share below is vacuous")
    share = len(empty_ranked) / traced if traced else 0.0
    gate("ranked searches returning nothing", share, MAX_EMPTY_RANKED_SHARE,
         detail=f"{len(empty_ranked)}/{traced}", pct=True)
    # Not gated separately — they are a subset of the line above — but they are
    # the diagnosis, and the number that says whether §16.11 is really gone.
    print(f"  ---   of those: {len(unreadable)} had files but read none "
          f"(the §16.11 signature), {len(no_files)} had no files at all")
    for iid, kind, what in usage[:examples]:
        print(f"          {iid} · {kind} · {what}")
    for label, items in (("unreadable scope", unreadable), ("empty scope", no_files)):
        for iid, what in items[:examples]:
            print(f"          {label}: {iid} · {what}")
    return {"n_search": n_search, "traced": traced, "usage": len(usage),
            "usage_by_cause": dict(kinds),
            "empty_ranked": len(empty_ranked), "unreadable": len(unreadable),
            "no_files": len(no_files)}


# ---------------------------------------------------------------------------
# agent distress
# ---------------------------------------------------------------------------

def distress(evs):
    """Signals that an agent had stopped trusting the tool.

    Returns (signal, detail) pairs. `preceded_by_empty` is what separates "the
    tool failed" from "the agent was curious": a help probe after three dead
    searches is a symptom, the same probe as the first action is not.
    """
    out = []
    run = 0
    for i, e in enumerate(evs):
        if any(f in e["argv"] for f in HELP_FLAGS):
            out.append(("help probe", f"after {run} consecutive empty searches", run >= CONSECUTIVE_EMPTY))
            continue
        if (e.get("stdout_bytes") or 0) == 0:
            run += 1
            if run == CONSECUTIVE_EMPTY:
                out.append((f"{CONSECUTIVE_EMPTY} consecutive empty searches",
                            " ".join(evs[i]["argv"][:2])[:44], True))
        else:
            run = 0
    qs = Counter(" ".join(e["argv"]) for e in evs
                 if not any(f in e["argv"] for f in HELP_FLAGS))
    for q, n in qs.items():
        if n >= REPEATED_QUERY:
            # Only a symptom if the repeats were fruitless; an agent may
            # legitimately re-run a query after editing a file.
            fruitless = all((e.get("stdout_bytes") or 0) == 0
                            for e in evs if " ".join(e["argv"]) == q)
            out.append((f"query repeated {n}x", q[:44], fruitless))
    return out


def check_distress(rows, examples):
    print("\n[2/4] agent distress (the tool failing, seen in behavior)")
    kinds = Counter()
    tool_caused = []
    all_empty_instances = []
    for r in rows:
        evs = searches(r)
        if not evs:
            continue
        if all((e.get("stdout_bytes") or 0) == 0 for e in evs):
            all_empty_instances.append(r["instance_id"])
        for sig, detail, caused in distress(evs):
            kinds[sig] += 1
            if caused:
                tool_caused.append((r["instance_id"], sig, detail))
    for k, n in kinds.most_common():
        print(f"  ---   {n:4d}  {k}")
    if not kinds:
        print("  ---   none")
    gate("distress attributable to the tool", len(tool_caused), MAX_TOOL_DISTRESS)
    gate("instances where every search was empty", len(all_empty_instances),
         MAX_ALL_EMPTY_INSTANCES,
         detail=", ".join(all_empty_instances[:3]) if all_empty_instances else "")
    for iid, sig, detail in tool_caused[:examples]:
        print(f"          {iid} · {sig} · {detail}")
    return {"distress": sum(kinds.values()), "tool_caused": len(tool_caused),
            "all_empty": len(all_empty_instances)}


# ---------------------------------------------------------------------------
# cache and harness
# ---------------------------------------------------------------------------

def check_cache(rows):
    print("\n[3/4] cache behaviour")
    wrote = defaultdict(int)
    repairs = Counter()
    for r in rows:
        for t in traces(r):
            root = (t.get("input") or {}).get("root_canonical") or \
                   (t.get("input") or {}).get("root", "")
            if (t.get("resolution") or {}).get("wrote_cache"):
                wrote[(r["instance_id"], root)] += 1
            rep = t.get("repair")
            if rep:
                repairs[rep if isinstance(rep, str) else json.dumps(rep)[:40]] += 1
    churn = {k: v for k, v in wrote.items() if v > 1}
    print(f"  ---   {len(wrote)} scopes cached; {len(churn)} written more than once")
    for k, v in list(churn.items())[:3]:
        print(f"          churn {v}x: {k[0]} · {k[1][-46:]}")
    if repairs:
        print("  ---   repair outcomes: " +
              ", ".join(f"{k}={v}" for k, v in repairs.most_common(4)))
    # Not gated: churn is a performance smell, not a correctness failure, and a
    # threshold picked without a baseline would be noise. Reported so a jump
    # between tiers is visible.
    return {"cached_scopes": len(wrote), "churn": len(churn)}


def check_harness(rows, results_path):
    print("\n[4/4] harness health")
    st = Counter(r.get("status") for r in rows)
    for k, v in st.most_common():
        print(f"  ---   {v:4d}  {k}")
    non_ok = sum(v for k, v in st.items() if k != "ok")
    gate("non-ok rows", non_ok, 0)
    free_gb = os.statvfs(DATA).f_bavail * os.statvfs(DATA).f_frsize / 1e9
    gate("free disk (GB)", round(free_gb, 1), 2.0, worse="below",
         detail="a campaign that fills the disk loses the chunk, not the row")
    trees = DATA / "repos" / "trees"
    stray = len(list(trees.iterdir())) if trees.exists() else 0
    gate("leaked worktrees", stray, 0)
    cost = sum((r.get("agent") or {}).get("total_cost_usd") or 0 for r in rows)
    print(f"  ---   spend so far: ${cost:.2f} over {len(rows)} rows")
    return {"non_ok": non_ok, "free_gb": round(free_gb, 1), "cost": round(cost, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DATA / "results-tier1.jsonl")
    ap.add_argument("--condition", default="", help="restrict to one arm")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None, help="write the summary")
    args = ap.parse_args()

    if not args.results.exists():
        sys.exit(f"no such results file: {args.results}")
    rows = load_results(args.results)
    if args.condition:
        rows = [r for r in rows if r.get("condition") == args.condition]
    if not rows:
        sys.exit(f"no rows in {args.results}"
                 f"{f' for condition {args.condition}' if args.condition else ''}")

    conds = Counter(r.get("condition") for r in rows)
    print(f"triage: {len(rows)} rows from {args.results.name} "
          f"({', '.join(f'{k}={v}' for k, v in conds.most_common())})")

    summary = {}
    # Tool and distress checks only mean something for the arm that has the
    # tool; an rg row has no semgrep invocations and would dilute every share.
    sg = [r for r in rows if r.get("condition") != "rg"]
    summary |= check_tool(sg, args.examples)
    summary |= check_distress(sg, args.examples)
    summary |= check_cache(sg)
    summary |= check_harness(rows, args.results)

    print()
    if FAILURES:
        print(f"GATE FAILED — {len(FAILURES)} check(s). Do not start the next "
              f"tier until these are understood:")
        for name, value, limit, detail in FAILURES:
            print(f"  · {name}: {value} (limit {limit}){(' — ' + detail) if detail else ''}")
    else:
        print("GATE PASSED — the tool answered what the agents typed.")
    if args.json:
        args.json.write_text(json.dumps(
            {"summary": summary, "failures": FAILURES,
             "passed": not FAILURES}, indent=1))
        print(f"wrote {args.json}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
